import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .neuron_analyser import NeuronResult, LayerSummary
from .attention_analyser import AttentionHeadResult


def _make_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses and types to JSON-serialisable values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _make_serializable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return round(obj, 6)
    return obj


@dataclass
class AnalysisResult:
    model_name: str
    prompt: str
    tokens: list[str]
    architecture_type: str
    parameter_count: int
    n_layers: int
    n_heads: int

    neuron_results: list[NeuronResult]
    attention_results: list[AttentionHeadResult]
    layer_summaries: list[LayerSummary]

    neuron_count: int = 0
    head_count: int = 0
    top_neuron_explanation: str = ""
    total_dead_neurons: int = 0

    created_at: float = 0.0
    analysis_duration_seconds: float = 0.0

    def __post_init__(self):
        self.neuron_count = len(self.neuron_results)
        self.head_count = len(self.attention_results)
        if self.created_at == 0.0:
            self.created_at = time.time()

        if self.total_dead_neurons == 0 and self.layer_summaries:
            self.total_dead_neurons = sum(ls.dead_neurons for ls in self.layer_summaries)

    def to_json(self) -> str:
        """Serialise the full analysis result to a JSON string.

        This is what gets stored in the database and returned to the frontend.
        All values are converted to JSON-serialisable types.
        """
        data = _make_serializable(asdict(self))
        return json.dumps(data, indent=2)

    def to_dict(self) -> dict:
        """Return the result as a JSON-serialisable dict."""
        return _make_serializable(asdict(self))

    def summary_stats(self) -> str:
        """Return a human-readable summary string."""
        top_neuron = self.neuron_results[0] if self.neuron_results else None
        top_head = self.attention_results[0] if self.attention_results else None

        lines = [
            f"Model: {self.model_name}",
            f"Architecture: {self.architecture_type}",
            f"Parameters: {self._format_count(self.parameter_count)}",
            f"Prompt ({len(self.tokens)} tokens): \"{self.prompt[:80]}{'...' if len(self.prompt) > 80 else ''}\"",
            f"",
            f"Neurons analysed: {self.neuron_count} across {len(self.layer_summaries)} layers",
            f"Total dead neurons: {self.total_dead_neurons}",
            f"Heads analysed: {self.head_count}",
            f"",
        ]

        if top_neuron:
            lines.append(
                f"Top neuron: Layer {top_neuron.layer_index}, "
                f"Neuron {top_neuron.neuron_index}, "
                f"Activation: {top_neuron.max_activation:.3f}, "
                f"Token: \"{top_neuron.activating_token}\""
            )

        if top_head:
            lines.append(
                f"Top head: Layer {top_head.layer_index}, "
                f"Head {top_head.head_index}, "
                f"Pattern: {top_head.pattern_type}, "
                f"Focus: {top_head.focus_score:.3f}"
            )

        lines.append(f"\nAnalysis time: {self.analysis_duration_seconds:.1f}s")

        return "\n".join(lines)

    @staticmethod
    def _format_count(count: int) -> str:
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)


def build_analysis_result(
    model_name: str,
    prompt: str,
    tokens: list[str],
    architecture_type: str,
    parameter_count: int,
    n_layers: int,
    n_heads: int,
    neuron_results: list[NeuronResult],
    attention_results: list[AttentionHeadResult],
    layer_summaries: list[LayerSummary],
    analysis_duration: float,
    top_neuron_explanation: str = "",
) -> AnalysisResult:
    """Convenience function to build and populate an AnalysisResult."""
    total_dead = sum(ls.dead_neurons for ls in layer_summaries) if layer_summaries else 0

    return AnalysisResult(
        model_name=model_name,
        prompt=prompt,
        tokens=tokens,
        architecture_type=architecture_type,
        parameter_count=parameter_count,
        n_layers=n_layers,
        n_heads=n_heads,
        neuron_results=neuron_results,
        attention_results=attention_results,
        layer_summaries=layer_summaries,
        neuron_count=len(neuron_results),
        head_count=len(attention_results),
        top_neuron_explanation=top_neuron_explanation,
        total_dead_neurons=total_dead,
        analysis_duration_seconds=analysis_duration,
    )
