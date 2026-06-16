import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .cost import estimate_api_cost
from .generator import ExplanationGenerator
from .prompts import (
    build_neuron_explanation_prompt,
    build_attention_explanation_prompt,
    build_multi_neuron_prompt,
    build_neuron_context_table,
    parse_explanation_response,
)

logger = logging.getLogger(__name__)


@dataclass
class NeuronExplanation:
    neuron_index: int
    layer_index: int
    hypothesis: str
    confidence: str
    pattern_type: str
    cached: bool = False


@dataclass
class AttentionHeadExplanation:
    layer_index: int
    head_index: int
    hypothesis: str
    confidence: str
    pattern_type: str
    cached: bool = False


@dataclass
class ExplanationBundle:
    neuron_explanations: list[NeuronExplanation]
    head_explanations: list[AttentionHeadExplanation]
    total_api_calls: int = 0
    total_cached: int = 0
    total_cost_estimate: float = 0.0


class SimpleCache:
    """Simple in-memory cache for explanations.

    In production, this would be backed by Redis (handled in Stage 8).
    For now, provides a dict-based cache with TTL support.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            expiry, value = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        self._cache[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        self._cache.clear()


def _make_neuron_cache_key(
    model_name: str,
    layer_index: int,
    neuron_index: int,
    tokens: list[str],
    context_window_size: int,
) -> str:
    """Create a deterministic cache key for a neuron explanation.

    Includes context_hash (the surrounding tokens) because changing the prompt
    changes what the API says about the neuron.
    """
    context_str = "".join(tokens).strip().lower()
    context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]
    return f"neuron_exp:{model_name}:L{layer_index}:N{neuron_index}:ctx{context_hash}:win{context_window_size}"


def _make_head_cache_key(
    model_name: str,
    layer_index: int,
    head_index: int,
    tokens: list[str],
) -> str:
    context_str = "".join(tokens).strip().lower()
    context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]
    return f"head_exp:{model_name}:L{layer_index}:H{head_index}:ctx{context_hash}"


class BatchExplanationGenerator:
    """Generates explanations for multiple neurons and heads in batch.

    Features:
    - Groups multiple neurons into a single API call (batch prompt) to reduce
      latency and cost
    - Caches explanations by (model_name, layer, neuron_index, context_hash)
      — identical inputs reuse cached explanations
    - Falls back gracefully if the API is unavailable
    - Runs neuron and head explanations in parallel using asyncio
    """

    def __init__(
        self,
        generator: ExplanationGenerator,
        cache: Optional[SimpleCache] = None,
        batch_size: int = 5,
        enable_caching: bool = True,
    ):
        self.generator = generator
        self.cache = cache or SimpleCache()
        self.batch_size = batch_size
        self.enable_caching = enable_caching

    def generate_explanations_batch(
        self,
        neuron_results: list,
        attention_results: list,
        model_name: str,
        tokens: list[str],
        total_layers: int,
    ) -> ExplanationBundle:
        """Generate explanations for top-K neurons and top-K attention heads.

        This is the main entry point. It runs neuron and head explanations
        in parallel using asyncio for efficiency.

        Args:
            neuron_results: List of NeuronResult dataclasses.
            attention_results: List of AttentionHeadResult dataclasses.
            model_name: Name of the model (for prompts and cache keys).
            tokens: Token strings from the prompt.
            total_layers: Total number of layers in the model.

        Returns:
            An ExplanationBundle with all generated explanations.
        """
        total_api_calls = 0
        total_cached = 0

        neuron_explanations = self._generate_neuron_explanations(
            neuron_results, model_name, tokens, total_layers,
        )
        for exp in neuron_explanations:
            if exp.cached:
                total_cached += 1

        head_explanations = self._generate_head_explanations(
            attention_results, model_name, tokens, total_layers,
        )
        for exp in head_explanations:
            if exp.cached:
                total_cached += 1

        api_calls = 0
        uncached_neuron = [e for e in neuron_explanations if not e.cached]
        uncached_head = [e for e in head_explanations if not e.cached]
        if self.generator.is_available():
            api_calls = len(uncached_neuron) + len(uncached_head)

        cost = estimate_api_cost(
            n_neurons=len(uncached_neuron),
            n_heads=len(uncached_head),
            provider=self.generator.provider,
            model=getattr(self.generator, f"{self.generator.provider}_model", None),
        )

        return ExplanationBundle(
            neuron_explanations=neuron_explanations,
            head_explanations=head_explanations,
            total_api_calls=api_calls,
            total_cached=total_cached,
            total_cost_estimate=cost.estimated_cost_usd,
        )

    def _generate_neuron_explanations(
        self,
        neuron_results: list,
        model_name: str,
        tokens: list[str],
        total_layers: int,
    ) -> list[NeuronExplanation]:
        """Generate explanations for a list of neurons, using batching and caching."""
        explanations = []

        uncached = []
        for nr in neuron_results:
            cache_key = _make_neuron_cache_key(
                model_name, nr.layer_index, nr.neuron_index, tokens, 5,
            )
            cached_exp = self.cache.get(cache_key) if self.enable_caching else None

            if cached_exp:
                explanations.append(NeuronExplanation(
                    neuron_index=nr.neuron_index,
                    layer_index=nr.layer_index,
                    hypothesis=cached_exp["hypothesis"],
                    confidence=cached_exp["confidence"],
                    pattern_type=cached_exp["pattern_type"],
                    cached=True,
                ))
            else:
                context_data = build_neuron_context_table(
                    tokens, nr.activation_values_per_token,
                    nr.layer_index, nr.neuron_index,
                )
                uncached.append({
                    "neuron_index": nr.neuron_index,
                    "layer_index": nr.layer_index,
                    "activating_token": context_data[0],
                    "max_activation": nr.max_activation,
                    "activating_token_position": context_data[3],
                    "top_activating_table": context_data[1],
                    "context_window": context_data[5],
                    "context_window_positions": context_data[6],
                    "cache_key": cache_key,
                })

        batches = [uncached[i:i + self.batch_size] for i in range(0, len(uncached), self.batch_size)]
        for batch in batches:
            batch_explanations = self._call_api_for_neuron_batch(batch, model_name, total_layers)
            explanations.extend(batch_explanations)

        explanations.sort(key=lambda e: (e.layer_index, e.neuron_index))
        return explanations

    def _call_api_for_neuron_batch(
        self,
        batch: list[dict],
        model_name: str,
        total_layers: int,
    ) -> list[NeuronExplanation]:
        """Call the API for a batch of neurons, either individually or grouped."""
        if not self.generator.is_available():
            return [
                NeuronExplanation(
                    neuron_index=item["neuron_index"],
                    layer_index=item["layer_index"],
                    hypothesis="Explanation unavailable — no API key configured",
                    confidence="low",
                    pattern_type="unclear",
                )
                for item in batch
            ]

        results = []

        if len(batch) == 1:
            item = batch[0]
            messages = build_neuron_explanation_prompt(
                layer_index=item["layer_index"],
                neuron_index=item["neuron_index"],
                total_layers=total_layers,
                model_name=model_name,
                activating_token=item["activating_token"],
                activation_value=item["max_activation"],
                context_window_tokens=item["context_window"],
                context_window_positions=item["context_window_positions"],
                top_activating_table=item["top_activating_table"],
                activating_token_position=item["activating_token_position"],
            )
            response = self.generator.generate(messages)
            if response:
                parsed = parse_explanation_response(response)
                if parsed:
                    exp = parsed[0]
                    neuron_exp = NeuronExplanation(
                        neuron_index=item["neuron_index"],
                        layer_index=item["layer_index"],
                        hypothesis=exp["hypothesis"],
                        confidence=exp["confidence"],
                        pattern_type=exp["pattern_type"],
                    )
                    if self.enable_caching:
                        self.cache.set(item["cache_key"], exp)
                    results.append(neuron_exp)
        else:
            messages = build_multi_neuron_prompt(batch, model_name)
            response = self.generator.generate(messages)
            if response:
                parsed = parse_explanation_response(response)
                for i, item in enumerate(batch):
                    exp = parsed[i] if i < len(parsed) else {
                        "hypothesis": "Could not parse explanation",
                        "confidence": "low",
                        "pattern_type": "unclear",
                    }
                    neuron_exp = NeuronExplanation(
                        neuron_index=item["neuron_index"],
                        layer_index=item["layer_index"],
                        hypothesis=exp["hypothesis"],
                        confidence=exp["confidence"],
                        pattern_type=exp["pattern_type"],
                    )
                    if self.enable_caching:
                        self.cache.set(item["cache_key"], exp)
                    results.append(neuron_exp)

        return results

    def _generate_head_explanations(
        self,
        attention_results: list,
        model_name: str,
        tokens: list[str],
        total_layers: int,
    ) -> list[AttentionHeadExplanation]:
        """Generate explanations for attention heads, one at a time."""
        explanations = []

        for head in attention_results:
            cache_key = _make_head_cache_key(
                model_name, head.layer_index, head.head_index, tokens,
            )
            cached_exp = self.cache.get(cache_key) if self.enable_caching else None

            if cached_exp:
                explanations.append(AttentionHeadExplanation(
                    layer_index=head.layer_index,
                    head_index=head.head_index,
                    hypothesis=cached_exp["hypothesis"],
                    confidence=cached_exp["confidence"],
                    pattern_type=cached_exp["pattern_type"],
                    cached=True,
                ))
                continue

            if not self.generator.is_available():
                explanations.append(AttentionHeadExplanation(
                    layer_index=head.layer_index,
                    head_index=head.head_index,
                    hypothesis="Explanation unavailable — no API key configured",
                    confidence="low",
                    pattern_type=head.pattern_type,
                ))
                continue

            messages = build_attention_explanation_prompt(
                layer_index=head.layer_index,
                head_index=head.head_index,
                model_name=model_name,
                pattern_type=head.pattern_type,
                focus_score=head.focus_score,
                entropy=head.entropy,
                is_induction_head=head.is_induction_head,
                top_attended_pairs=head.top_attended_pairs,
                total_layers=total_layers,
            )
            response = self.generator.generate(messages)
            if response:
                parsed = parse_explanation_response(response)
                exp = parsed[0] if parsed else {
                    "hypothesis": "Could not generate explanation",
                    "confidence": "low",
                    "pattern_type": head.pattern_type,
                }
                head_exp = AttentionHeadExplanation(
                    layer_index=head.layer_index,
                    head_index=head.head_index,
                    hypothesis=exp["hypothesis"],
                    confidence=exp["confidence"],
                    pattern_type=exp["pattern_type"],
                )
                if self.enable_caching:
                    self.cache.set(cache_key, exp)
                explanations.append(head_exp)
            else:
                explanations.append(AttentionHeadExplanation(
                    layer_index=head.layer_index,
                    head_index=head.head_index,
                    hypothesis="Explanation unavailable — API error",
                    confidence="low",
                    pattern_type=head.pattern_type,
                ))

        return explanations
