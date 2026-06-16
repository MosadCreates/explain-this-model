import json

import pytest

from src.analysis.aggregator import (
    AnalysisResult,
    NeuronResult,
    AttentionHeadResult,
    LayerSummary,
    _make_serializable,
    build_analysis_result,
)


def make_sample_neuron(layer=0, neuron=0, act=1.0, dead=False):
    return NeuronResult(
        layer_index=layer,
        neuron_index=neuron,
        max_activation=act,
        mean_activation=act * 0.5,
        std_activation=act * 0.1,
        fraction_active=0.8 if not dead else 0.0,
        activating_token="test",
        activating_token_position=0,
        context_window=["test", "prompt"],
        context_window_positions=[0, 1],
        activation_values_per_token=[act, 0.0],
        is_dead=dead,
        z_score=1.0,
        rank=1,
    )


def make_sample_head(layer=0, head=0, pattern="diagonal"):
    return AttentionHeadResult(
        layer_index=layer,
        head_index=head,
        focus_score=0.8,
        entropy=0.5,
        pattern_type=pattern,
        attention_matrix=[[0.5, 0.5], [0.3, 0.7]],
        top_attended_pairs=[
            {"query_position": 0, "key_position": 1, "query_token": "test", "key_token": "prompt", "weight": 0.7}
        ],
        is_induction_head=False,
        max_attention_weight=0.7,
        attending_entropy=0.5,
        rank=1,
    )


def make_sample_summary(layer=0):
    return LayerSummary(
        layer_index=layer,
        total_neurons=100,
        dead_neurons=5,
        max_activation=2.0,
        mean_activation=0.3,
        fraction_dead=0.05,
    )


class TestAnalysisResultInit:
    def test_auto_counts_neurons_and_heads(self):
        neurons = [make_sample_neuron(0, i) for i in range(3)]
        heads = [make_sample_head(0, i) for i in range(2)]
        summaries = [make_sample_summary(0)]

        result = AnalysisResult(
            model_name="gpt2",
            prompt="test prompt",
            tokens=["test", "prompt"],
            architecture_type="gpt_style",
            parameter_count=100_000_000,
            n_layers=1,
            n_heads=2,
            neuron_results=neurons,
            attention_results=heads,
            layer_summaries=summaries,
        )
        assert result.neuron_count == 3
        assert result.head_count == 2

    def test_auto_sets_created_at(self):
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=[], architecture_type="gpt_style",
            parameter_count=0, n_layers=1, n_heads=1, neuron_results=[], attention_results=[], layer_summaries=[],
        )
        assert result.created_at > 0


class TestAnalysisResultToDict:
    def test_to_dict_contains_all_keys(self):
        neurons = [make_sample_neuron()]
        heads = [make_sample_head()]
        summaries = [make_sample_summary()]
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=["a", "b"],
            architecture_type="gpt_style", parameter_count=100,
            n_layers=1, n_heads=1, neuron_results=neurons,
            attention_results=heads, layer_summaries=summaries,
        )
        d = result.to_dict()
        assert d["model_name"] == "gpt2"
        assert d["prompt"] == "test"
        assert d["tokens"] == ["a", "b"]
        assert d["neuron_count"] == 1
        assert d["head_count"] == 1
        assert "neuron_results" in d
        assert "attention_results" in d
        assert "layer_summaries" in d
        assert "created_at" in d

    def test_to_dict_serializable_types(self):
        neurons = [make_sample_neuron()]
        heads = [make_sample_head()]
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=[],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=1, n_heads=1, neuron_results=neurons,
            attention_results=heads, layer_summaries=[make_sample_summary()],
        )
        d = result.to_dict()
        json.dumps(d)


class TestAnalysisResultToJson:
    def test_to_json_valid_string(self):
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=[],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=1, n_heads=1, neuron_results=[make_sample_neuron()],
            attention_results=[make_sample_head()],
            layer_summaries=[make_sample_summary()],
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "gpt2"

    def test_to_json_indented(self):
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=[],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=1, n_heads=1, neuron_results=[],
            attention_results=[], layer_summaries=[],
        )
        json_str = result.to_json()
        assert "\n" in json_str


class TestAnalysisResultSummaryStats:
    def test_summary_contains_model_name(self):
        result = AnalysisResult(
            model_name="gpt2-xl", prompt="test", tokens=["t"],
            architecture_type="gpt_style", parameter_count=1_500_000_000,
            n_layers=48, n_heads=25, neuron_results=[make_sample_neuron()],
            attention_results=[make_sample_head()],
            layer_summaries=[make_sample_summary()],
        )
        summary = result.summary_stats()
        assert "gpt2-xl" in summary
        assert "1.5B" in summary

    def test_summary_with_top_neuron(self):
        neuron = make_sample_neuron(layer=3, neuron=7, act=5.0)
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=["t"],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=1, n_heads=1, neuron_results=[neuron],
            attention_results=[make_sample_head()],
            layer_summaries=[make_sample_summary()],
        )
        summary = result.summary_stats()
        assert "Layer 3" in summary
        assert "Neuron 7" in summary
        assert "5.000" in summary

    def test_summary_with_top_head(self):
        head = make_sample_head(layer=1, head=3, pattern="induction")
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=["t"],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=1, n_heads=1, neuron_results=[],
            attention_results=[head],
            layer_summaries=[],
        )
        summary = result.summary_stats()
        assert "induction" in summary

    def test_format_count_billions(self):
        assert AnalysisResult._format_count(1_500_000_000) == "1.5B"

    def test_format_count_millions(self):
        assert AnalysisResult._format_count(12_000_000) == "12.0M"

    def test_format_count_thousands(self):
        assert AnalysisResult._format_count(500_000) == "500.0K"

    def test_format_count_small(self):
        assert AnalysisResult._format_count(42) == "42"


class TestBuildAnalysisResult:
    def test_build_with_all_fields(self):
        neurons = [make_sample_neuron()]
        heads = [make_sample_head()]
        summaries = [make_sample_summary()]
        result = build_analysis_result(
            model_name="gpt2", prompt="hello", tokens=["h", "e"],
            architecture_type="gpt_style", parameter_count=100,
            n_layers=1, n_heads=1,
            neuron_results=neurons, attention_results=heads,
            layer_summaries=summaries,
            analysis_duration=1.5,
            top_neuron_explanation="This neuron detects greetings",
        )
        assert result.model_name == "gpt2"
        assert result.neuron_count == 1
        assert result.head_count == 1
        assert result.top_neuron_explanation == "This neuron detects greetings"
        assert result.analysis_duration_seconds == 1.5

    def test_build_empty_lists(self):
        result = build_analysis_result(
            model_name="gpt2", prompt="test", tokens=[],
            architecture_type="gpt_style", parameter_count=0,
            n_layers=0, n_heads=0,
            neuron_results=[], attention_results=[], layer_summaries=[],
            analysis_duration=0.0,
        )
        assert result.neuron_count == 0
        assert result.head_count == 0
        assert result.total_dead_neurons == 0


class TestMakeSerializable:
    def test_nan_becomes_none(self):
        assert _make_serializable(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _make_serializable(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _make_serializable(float("-inf")) is None

    def test_float_rounded(self):
        result = _make_serializable(3.1415926535)
        assert result == 3.141593

    def test_list_of_dataclasses(self):
        results = [make_sample_neuron(), make_sample_neuron(layer=1)]
        serialized = _make_serializable(results)
        assert isinstance(serialized, list)
        assert len(serialized) == 2
        assert serialized[0]["layer_index"] == 0
        assert serialized[1]["layer_index"] == 1
