import json
import pytest
from unittest.mock import MagicMock, patch

from src.explanations.prompts import (
    build_neuron_explanation_prompt,
    build_attention_explanation_prompt,
    build_multi_neuron_prompt,
    parse_explanation_response,
    build_neuron_context_table,
)
from src.explanations.generator import ExplanationGenerator, NullExplanationGenerator
from src.explanations.batch import (
    BatchExplanationGenerator,
    SimpleCache,
    NeuronExplanation,
    AttentionHeadExplanation,
    ExplanationBundle,
)
from src.explanations.cost import (
    estimate_api_cost,
    UsageTracker,
    RateLimiter,
    CostEstimate,
)


class TestNeuronPrompt:
    """Tests for neuron explanation prompt construction."""

    def test_build_neuron_prompt_contains_key_fields(self):
        messages = build_neuron_explanation_prompt(
            layer_index=3,
            neuron_index=42,
            total_layers=12,
            model_name="gpt2",
            activating_token="cat",
            activation_value=12.5,
            context_window_tokens=["the", " cat", " sat"],
            context_window_positions=[0, 1, 2],
            top_activating_table=[
                {"position": 1, "token": "cat", "activation": 12.5,
                 "left_context": "the ", "right_context": " sat"}
            ],
            activating_token_position=1,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert "gpt2" in content
        assert "Layer: 3" in content
        assert "Neuron index: 42" in content
        assert "cat" in content
        assert "12.500" in content
        assert "hypothesis" in content
        assert "confidence" in content

    def test_build_neuron_prompt_mentions_json_format(self):
        messages = build_neuron_explanation_prompt(
            layer_index=0, neuron_index=0, total_layers=6,
            model_name="distilbert", activating_token="the",
            activation_value=3.0, context_window_tokens=["the", " cat"],
            context_window_positions=[0, 1],
            top_activating_table=[], activating_token_position=0,
        )
        assert '"hypothesis"' in messages[1]["content"]
        assert '"confidence"' in messages[1]["content"]
        assert '"pattern_type"' in messages[1]["content"]
        assert '"high|medium|low"' in messages[1]["content"]


class TestAttentionPrompt:
    """Tests for attention head explanation prompt construction."""

    def test_build_attention_prompt_contains_key_fields(self):
        messages = build_attention_explanation_prompt(
            layer_index=5, head_index=3, model_name="gpt2",
            pattern_type="previous_token", focus_score=0.85,
            entropy=1.2, is_induction_head=True,
            top_attended_pairs=[
                {"query_token": "cat", "key_token": "the",
                 "query_position": 2, "key_position": 1, "weight": 0.9},
            ],
            total_layers=12,
        )
        content = messages[1]["content"]
        assert "gpt2" in content
        assert "Layer: 5" in content
        assert "Head: 3" in content
        assert "previous_token" in content
        assert "induction" in content.lower()
        assert "cat" in content
        assert "the" in content

    def test_non_induction_head_no_induction_note(self):
        messages = build_attention_explanation_prompt(
            layer_index=0, head_index=0, model_name="gpt2",
            pattern_type="diagonal", focus_score=0.5,
            entropy=2.0, is_induction_head=False,
            top_attended_pairs=[], total_layers=12,
        )
        assert "induction" not in messages[1]["content"].lower()


class TestMultiNeuronPrompt:
    """Tests for batch neuron prompt construction."""

    def test_multi_neuron_prompt_contains_all_neurons(self):
        neurons = [
            {"neuron_index": 1, "layer_index": 0, "activating_token": "cat",
             "max_activation": 5.0, "activating_token_position": 1,
             "top_activating_table": [], "context_window": ["the", " cat"],
             "context_window_positions": [0, 1],
             "cache_key": "key1"},
            {"neuron_index": 5, "layer_index": 2, "activating_token": "sat",
             "max_activation": 3.0, "activating_token_position": 2,
             "top_activating_table": [], "context_window": ["cat", " sat", " on"],
             "context_window_positions": [1, 2, 3],
             "cache_key": "key2"},
        ]
        messages = build_multi_neuron_prompt(neurons, "gpt2")
        content = messages[1]["content"]
        assert "Neuron 1" in content
        assert "Neuron 5" in content
        assert "Layer 0" in content
        assert "Layer 2" in content
        assert "JSON array" in content
        assert "gpt2" in content


class TestParseExplanationResponse:
    """Tests for parsing API responses."""

    def test_parse_valid_json_dict(self):
        response = '{"hypothesis": "detects nouns", "confidence": "high", "pattern_type": "semantic"}'
        result = parse_explanation_response(response)
        assert len(result) == 1
        assert result[0]["hypothesis"] == "detects nouns"
        assert result[0]["confidence"] == "high"
        assert result[0]["pattern_type"] == "semantic"

    def test_parse_valid_json_array(self):
        response = '[{"hypothesis": "detects nouns", "confidence": "high", "pattern_type": "semantic"}]'
        result = parse_explanation_response(response)
        assert len(result) == 1

    def test_parse_json_array_multiple(self):
        response = json.dumps([
            {"hypothesis": "detects nouns", "confidence": "high", "pattern_type": "semantic"},
            {"hypothesis": "detects verbs", "confidence": "medium", "pattern_type": "syntactic"},
        ])
        result = parse_explanation_response(response)
        assert len(result) == 2
        assert result[1]["hypothesis"] == "detects verbs"

    def test_parse_malformed_json_with_extra_text(self):
        response = 'Here is my analysis:\n\n{"hypothesis": "detects nouns", "confidence": "high", "pattern_type": "semantic"}\n\nHope this helps!'
        result = parse_explanation_response(response)
        assert len(result) == 1
        assert result[0]["hypothesis"] == "detects nouns"

    def test_parse_malformed_json_array(self):
        response = 'Here are the results:\n\n[{"hypothesis": "detects nouns", "confidence": "high", "pattern_type": "semantic"}]\n\nEnd'
        result = parse_explanation_response(response)
        assert len(result) == 1

    def test_parse_invalid_text_fallback(self):
        result = parse_explanation_response("This is not JSON at all")
        assert len(result) == 1
        assert result[0]["confidence"] == "low"
        assert result[0]["pattern_type"] == "unclear"

    def test_parse_empty_string(self):
        result = parse_explanation_response("")
        assert len(result) == 1
        assert result[0]["hypothesis"] is not None


class TestExplanationGenerator:
    """Tests for the ExplanationGenerator class."""

    def test_null_generator_not_available(self):
        gen = NullExplanationGenerator()
        assert gen.is_available() is False

    def test_null_generator_returns_none(self):
        gen = NullExplanationGenerator()
        assert gen.generate([{"role": "user", "content": "test"}]) is None

    def test_gemini_available_with_key(self):
        gen = ExplanationGenerator(provider="gemini", gemini_api_key="test-key")
        assert gen.is_available() is True

    def test_gemini_not_available_without_key(self):
        gen = ExplanationGenerator(provider="gemini", gemini_api_key=None)
        assert gen.is_available() is False

    def test_claude_available_with_key(self):
        gen = ExplanationGenerator(provider="claude", claude_api_key="test-key")
        assert gen.is_available() is True

    def test_claude_not_available_without_key(self):
        gen = ExplanationGenerator(provider="claude", claude_api_key=None)
        assert gen.is_available() is False

    def test_generate_returns_none_when_no_key(self):
        gen = ExplanationGenerator(provider="gemini", gemini_api_key=None)
        result = gen.generate([{"role": "user", "content": "test"}])
        assert result is None


class TestSimpleCache:
    """Tests for the in-memory explanation cache."""

    def test_cache_set_and_get(self):
        cache = SimpleCache()
        cache.set("test_key", {"hypothesis": "test"})
        result = cache.get("test_key")
        assert result["hypothesis"] == "test"

    def test_cache_miss(self):
        cache = SimpleCache()
        assert cache.get("nonexistent") is None

    def test_cache_clear(self):
        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_ttl_expiry(self):
        cache = SimpleCache()
        cache.set("key", "value", ttl_seconds=0)
        import time
        time.sleep(0.01)
        assert cache.get("key") is None


class TestBatchExplanationGenerator:
    """Tests for the BatchExplanationGenerator."""

    def test_no_explanations_when_no_key(self):
        gen = NullExplanationGenerator()
        batcher = BatchExplanationGenerator(gen)
        neurons = _make_mock_neuron_results(2)
        heads = _make_mock_head_results(1)
        bundle = batcher.generate_explanations_batch(
            neurons, heads, "gpt2", ["the", "cat"], 12,
        )
        assert len(bundle.neuron_explanations) == 2
        assert len(bundle.head_explanations) == 1
        for exp in bundle.neuron_explanations:
            assert "unavailable" in exp.hypothesis.lower()
        assert bundle.total_api_calls == 0

    def test_cache_hit_on_second_call(self):
        gen = NullExplanationGenerator()
        cache = SimpleCache()
        cache.set("neuron_exp:gpt2:L0:N0:ctx", {"hypothesis": "cached", "confidence": "high", "pattern_type": "semantic"})
        batcher = BatchExplanationGenerator(gen, cache=cache)
        neurons = _make_mock_neuron_results(1)
        bundle = batcher.generate_explanations_batch(
            neurons, [], "gpt2", ["the", "cat"], 12,
        )
        assert len(bundle.neuron_explanations) == 1
        assert bundle.total_cached >= 0


class TestCostEstimation:
    """Tests for API cost estimation."""

    def test_estimate_basic(self):
        estimate = estimate_api_cost(n_neurons=5, n_heads=2, provider="gemini")
        assert isinstance(estimate, CostEstimate)
        assert estimate.neuron_count == 5
        assert estimate.head_count == 2
        assert estimate.estimated_cost_usd >= 0

    def test_estimate_zero_neurons(self):
        estimate = estimate_api_cost(n_neurons=0, n_heads=0)
        assert estimate.estimated_cost_usd == 0

    def test_estimate_provider_costs_differ(self):
        gemini = estimate_api_cost(n_neurons=10, n_heads=5, provider="gemini")
        claude = estimate_api_cost(n_neurons=10, n_heads=5, provider="claude")
        assert gemini.estimated_cost_usd <= claude.estimated_cost_usd


class TestUsageTracker:
    """Tests for the UsageTracker."""

    def test_record_and_get_usage(self):
        tracker = UsageTracker()
        tracker.record_usage("job_1", 100, 50)
        usage = tracker.get_job_usage("job_1")
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50

    def test_missing_job_returns_none(self):
        tracker = UsageTracker()
        assert tracker.get_job_usage("nonexistent") is None

    def test_daily_cap_not_exceeded(self):
        tracker = UsageTracker()
        assert tracker.check_daily_cap(cap_usd=5.0) is True


class TestRateLimiter:
    """Tests for the RateLimiter."""

    def test_rate_limiter_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=3600)
        for _ in range(5):
            assert limiter.check() is True

    def test_rate_limiter_blocks_excess(self):
        limiter = RateLimiter(max_requests=3, window_seconds=3600)
        for _ in range(3):
            limiter.check()
        assert limiter.check() is False

    def test_rate_limiter_remaining(self):
        limiter = RateLimiter(max_requests=5, window_seconds=3600)
        assert limiter.remaining() == 5
        limiter.check()
        limiter.check()
        assert limiter.remaining() == 3

    def test_rate_limiter_reset(self):
        limiter = RateLimiter(max_requests=2, window_seconds=3600)
        limiter.check()
        limiter.check()
        limiter.reset()
        assert limiter.remaining() == 2


def _make_mock_neuron_results(n: int):
    """Create mock neuron results for testing."""
    results = []
    for i in range(n):
        mock = type("MockNeuron", (), {
            "layer_index": i,
            "neuron_index": i * 10,
            "max_activation": float(10 - i),
            "mean_activation": 5.0,
            "std_activation": 2.0,
            "fraction_active": 0.8,
            "activating_token": "test",
            "activating_token_position": 0,
            "context_window": ["test"],
            "context_window_positions": [0],
            "activation_values_per_token": [1.0, 2.0, 3.0],
            "is_dead": False,
            "z_score": 0.0,
            "rank": i + 1,
        })()
        results.append(mock)
    return results


def _make_mock_head_results(n: int):
    """Create mock attention head results for testing."""
    results = []
    for i in range(n):
        mock = type("MockHead", (), {
            "layer_index": i,
            "head_index": i * 2,
            "focus_score": 0.8 - i * 0.1,
            "entropy": 1.0 + i * 0.5,
            "pattern_type": "diagonal",
            "attention_matrix": [[1.0]],
            "top_attended_pairs": [],
            "is_induction_head": False,
            "max_attention_weight": 1.0,
            "attending_entropy": 1.0,
            "rank": i + 1,
        })()
        results.append(mock)
    return results
