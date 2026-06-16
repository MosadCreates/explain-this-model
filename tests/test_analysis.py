import json
import pytest
import torch
import numpy as np

from src.models.registry import ModelRegistry
from src.models.hooks import ActivationHookManager
from src.analysis.neuron_analyser import NeuronAnalyser, NeuronResult, LayerSummary
from src.analysis.attention_analyser import (
    AttentionAnalyser,
    AttentionHeadResult,
    _classify_pattern,
    _detect_induction_head,
    compute_focus_score,
)
from src.analysis.aggregator import AnalysisResult, build_analysis_result


class TestNeuronAnalyserUnit:
    """Unit tests for NeuronAnalyser with controlled synthetic activations."""

    def test_basic_statistics(self):
        activations = [type("Mock", (), {"output_tensor": torch.randn(1, 5, 4)})()]
        tokens = ["the", "cat", "sat", "on", "mat"]
        analyser = NeuronAnalyser(activations, tokens)
        stats = analyser.compute_neuron_stats(0, 0)
        assert "max_activation" in stats
        assert "mean_activation" in stats
        assert "std_activation" in stats
        assert "fraction_active" in stats
        assert "is_dead" in stats

    def test_dead_neuron_detection(self):
        output = torch.zeros(1, 5, 4)
        output[:, :, 0] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        output[:, :, 1] = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["the", "cat", "sat", "on", "mat"]
        analyser = NeuronAnalyser(activations, tokens)
        stats0 = analyser.compute_neuron_stats(0, 0)
        stats1 = analyser.compute_neuron_stats(0, 1)
        assert stats0["is_dead"] is True
        assert stats1["is_dead"] is False

    def test_fraction_active(self):
        output = torch.zeros(1, 5, 2)
        output[:, :, 0] = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["a", "b", "c", "d", "e"]
        analyser = NeuronAnalyser(activations, tokens)
        stats = analyser.compute_neuron_stats(0, 0)
        assert stats["fraction_active"] == 0.6

    def test_top_activating_position(self):
        output = torch.zeros(1, 5, 2)
        output[:, :, 0] = torch.tensor([[1.0, 5.0, 2.0, 0.0, 3.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["the", "cat", "sat", "on", "mat"]
        analyser = NeuronAnalyser(activations, tokens)
        pos, val = analyser.get_top_activating_position(0, 0)
        assert pos == 1
        assert val == 5.0

    def test_context_window(self):
        activations = [type("Mock", (), {"output_tensor": torch.randn(1, 7, 2)})()]
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "dog"]
        analyser = NeuronAnalyser(activations, tokens, context_window_size=2)
        window_tokens, window_pos = analyser.get_context_window(3)
        assert window_tokens == ["quick", "brown", "fox", "jumps", "over"]
        assert window_pos == [1, 2, 3, 4, 5]

    def test_context_window_at_boundary(self):
        activations = [type("Mock", (), {"output_tensor": torch.randn(1, 5, 2)})()]
        tokens = ["the", "cat", "sat", "on", "mat"]
        analyser = NeuronAnalyser(activations, tokens, context_window_size=2)
        window_tokens, _ = analyser.get_context_window(0)
        assert window_tokens == ["the", "cat", "sat"]
        window_tokens, _ = analyser.get_context_window(4)
        assert window_tokens == ["sat", "on", "mat"]

    def test_rank_top_k_returns_correct_order(self):
        output = torch.zeros(1, 5, 4)
        output[:, :, 0] = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0]])
        output[:, :, 1] = torch.tensor([[5.0, 1.0, 0.0, 0.0, 1.0]])
        output[:, :, 2] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        output[:, :, 3] = torch.tensor([[2.0, 1.0, 4.0, 0.0, 1.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["a", "b", "c", "d", "e"]
        analyser = NeuronAnalyser(activations, tokens)
        top_k = analyser.rank_top_k(k=3)
        assert len(top_k) == 3
        assert top_k[0].neuron_index == 1
        assert top_k[0].max_activation == 5.0
        assert top_k[1].neuron_index == 3
        assert top_k[2].neuron_index == 0

    def test_layer_summaries(self):
        output = torch.zeros(1, 5, 4)
        output[:, :, 0] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        output[:, :, 1] = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0]])
        output[:, :, 2] = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        output[:, :, 3] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        activations = [
            type("Mock", (), {"output_tensor": output})(),
            type("Mock", (), {"output_tensor": torch.randn(1, 5, 4)})(),
        ]
        tokens = ["a", "b", "c", "d", "e"]
        analyser = NeuronAnalyser(activations, tokens)
        summaries = analyser.compute_layer_summaries()
        assert len(summaries) == 2
        assert summaries[0].dead_neurons == 1
        assert summaries[0].total_neurons == 4

    def test_dead_neuron_count(self):
        output = torch.zeros(1, 5, 4)
        output[:, :, 0] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        output[:, :, 1] = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0]])
        output[:, :, 2] = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        output[:, :, 3] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["a", "b", "c", "d", "e"]
        analyser = NeuronAnalyser(activations, tokens)
        dead_count = analyser.get_dead_neuron_count(0)
        assert dead_count == 2
        output = torch.zeros(1, 4, 2)
        output[:, :, 0] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        activations = [type("Mock", (), {"output_tensor": output})()]
        tokens = ["a", "b", "c", "d"]
        analyser = NeuronAnalyser(activations, tokens)
        heatmap = analyser.get_activation_heatmap(0, 0)
        assert heatmap == [1.0, 2.0, 3.0, 4.0]


class TestNeuronAnalyserIntegration:
    """Integration tests using real GPT-2 activations."""

    @pytest.mark.slow
    def test_neuron_analyser_on_gpt2(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        prompt = "The cat sat on the mat"
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
            mlp_acts = manager.get_mlp_activations()
            tokens = [loaded.tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

        analyser = NeuronAnalyser(mlp_acts, tokens)
        top_neurons = analyser.rank_top_k(k=5)
        assert len(top_neurons) == 5
        for neuron in top_neurons:
            assert isinstance(neuron, NeuronResult)
            assert neuron.max_activation > 0
            assert neuron.layer_index >= 0
            assert neuron.neuron_index >= 0
            assert len(neuron.activation_values_per_token) == 6
            assert len(neuron.context_window) > 0

    @pytest.mark.slow
    def test_neuron_analyser_layer_summaries_gpt2(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
            mlp_acts = manager.get_mlp_activations()
            tokens = [loaded.tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

        analyser = NeuronAnalyser(mlp_acts, tokens)
        summaries = analyser.compute_layer_summaries()
        assert len(summaries) == 12
        for ls in summaries:
            assert isinstance(ls, LayerSummary)
            assert ls.total_neurons > 0


class TestAttentionAnalyserUnit:
    """Unit tests for AttentionAnalyser with controlled synthetic patterns."""

    def test_diagonal_pattern(self):
        seq_len = 8
        matrix = np.eye(seq_len) * 0.8
        matrix += np.random.uniform(0, 0.01, (seq_len, seq_len))
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        result = _classify_pattern(matrix, seq_len)
        assert result == "diagonal"

    def test_previous_token_pattern(self):
        seq_len = 8
        matrix = np.zeros((seq_len, seq_len))
        for i in range(1, seq_len):
            matrix[i, i - 1] = 0.9
        matrix[0, 0] = 0.9
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        result = _classify_pattern(matrix, seq_len)
        assert result == "previous_token"

    def test_first_token_pattern(self):
        seq_len = 8
        matrix = np.zeros((seq_len, seq_len))
        matrix[:, 0] = 0.9
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        result = _classify_pattern(matrix, seq_len)
        assert result == "first_token"

    def test_diffuse_pattern(self):
        seq_len = 8
        matrix = np.ones((seq_len, seq_len)) / seq_len
        result = _classify_pattern(matrix, seq_len)
        assert result == "diffuse"

    def test_content_based_pattern(self):
        seq_len = 8
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0.0, 1.0, (seq_len, seq_len))
        for i in range(seq_len):
            matrix[i] = matrix[i] * (1.0 + rng.uniform(-0.5, 0.5, seq_len))
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        result = _classify_pattern(matrix, seq_len)
        assert result == "content_based"

    def test_focus_score_perfect_diagonal(self):
        seq_len = 8
        matrix = np.eye(seq_len)
        score = compute_focus_score(matrix)
        assert score > 0.9

    def test_focus_score_diffuse(self):
        seq_len = 8
        matrix = np.ones((seq_len, seq_len)) / seq_len
        score = compute_focus_score(matrix)
        assert score < 0.1

    def test_induction_head_detection(self):
        seq_len = 6
        matrix = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            matrix[i, i - 1] = 0.9 if i > 0 else 0.0
            matrix[i, i] = 0.1
        matrix[0, 0] = 1.0
        tokens = ["A", "B", "A", "B", "C", "D"]
        assert _detect_induction_head(matrix, tokens, threshold=0.5) == True

    def test_no_induction_head(self):
        matrix = np.eye(6) * 0.9
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        tokens = ["A", "B", "C", "D", "E", "F"]
        assert _detect_induction_head(matrix, tokens) == False

    def test_analyse_head_native_attentions(self):
        seq_len = 5
        n_heads = 2
        n_layers = 1
        native = [torch.randn(1, n_heads, seq_len, seq_len).softmax(dim=-1)]
        mock_activations = [type("Mock", (), {})() for _ in range(n_layers)]
        tokens = ["the", "cat", "sat", "on", "mat"]

        analyser = AttentionAnalyser(
            mock_activations, tokens, native_attentions=native, n_heads=n_heads
        )
        result = analyser.analyse_head(0, 0)
        assert result is not None
        assert isinstance(result, AttentionHeadResult)
        assert 0 <= result.focus_score <= 1.0
        assert result.pattern_type in ["diagonal", "previous_token", "first_token", "last_token", "diffuse", "content_based"]
        assert len(result.attention_matrix) == seq_len
        assert len(result.top_attended_pairs) > 0

    def test_top_k_ranking(self):
        seq_len = 5
        n_heads = 4
        n_layers = 2
        native = []
        for _ in range(n_layers):
            attn = torch.zeros(1, n_heads, seq_len, seq_len)
            for h in range(n_heads):
                attn[0, h] = torch.eye(seq_len) * (0.5 + h * 0.1)
                attn[0, h] = attn[0, h] / attn[0, h].sum(dim=-1, keepdim=True)
            native.append(attn)

        mock_activations = [type("Mock", (), {})() for _ in range(n_layers)]
        tokens = ["a", "b", "c", "d", "e"]
        analyser = AttentionAnalyser(mock_activations, tokens, native_attentions=native, n_heads=n_heads)
        top_heads = analyser.rank_top_k(k=3)
        assert len(top_heads) == 3
        assert top_heads[0].focus_score >= top_heads[1].focus_score


class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass and serialisation."""

    def test_empty_result(self):
        result = AnalysisResult(
            model_name="gpt2",
            prompt="test",
            tokens=["test"],
            architecture_type="gpt_style",
            parameter_count=100,
            n_layers=1,
            n_heads=1,
            neuron_results=[],
            attention_results=[],
            layer_summaries=[],
        )
        assert result.neuron_count == 0
        assert result.head_count == 0

    def test_to_json_serialisation(self):
        neurons = [NeuronResult(
            layer_index=0, neuron_index=0, max_activation=1.0, mean_activation=0.5,
            std_activation=0.3, fraction_active=0.5, activating_token="test",
            activating_token_position=0, context_window=["test"], context_window_positions=[0],
            activation_values_per_token=[1.0], is_dead=False,
        )]
        layers = [LayerSummary(layer_index=0, total_neurons=10, dead_neurons=1,
                               max_activation=1.0, mean_activation=0.5, fraction_dead=0.1)]
        heads = [AttentionHeadResult(
            layer_index=0, head_index=0, focus_score=0.8, entropy=0.5,
            pattern_type="diagonal", attention_matrix=[[1.0]], top_attended_pairs=[],
            is_induction_head=False, max_attention_weight=1.0, attending_entropy=0.5,
        )]
        result = build_analysis_result(
            model_name="gpt2", prompt="test", tokens=["test"],
            architecture_type="gpt_style", parameter_count=100,
            n_layers=1, n_heads=1,
            neuron_results=neurons, attention_results=heads, layer_summaries=layers,
            analysis_duration=1.0, top_neuron_explanation="detects test tokens",
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["model_name"] == "gpt2"
        assert data["prompt"] == "test"
        assert data["neuron_count"] == 1
        assert data["head_count"] == 1
        assert data["top_neuron_explanation"] == "detects test tokens"
        assert len(data["neuron_results"]) == 1
        assert len(data["attention_results"]) == 1
        assert len(data["layer_summaries"]) == 1

    def test_to_dict_structure(self):
        result = AnalysisResult(
            model_name="gpt2", prompt="test", tokens=["test"],
            architecture_type="gpt_style", parameter_count=100,
            n_layers=1, n_heads=1,
            neuron_results=[], attention_results=[], layer_summaries=[],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d
        assert "neuron_results" in d
        assert "attention_results" in d

    def test_summary_stats_contains_key_info(self):
        neurons = [NeuronResult(
            layer_index=0, neuron_index=5, max_activation=3.5, mean_activation=1.0,
            std_activation=0.5, fraction_active=0.8, activating_token="hello",
            activating_token_position=0, context_window=["hello"], context_window_positions=[0],
            activation_values_per_token=[3.5], is_dead=False,
        )]
        heads = [AttentionHeadResult(
            layer_index=0, head_index=3, focus_score=0.9, entropy=0.3,
            pattern_type="diagonal", attention_matrix=[[1.0]], top_attended_pairs=[],
            is_induction_head=False, max_attention_weight=1.0, attending_entropy=0.3,
        )]
        result = build_analysis_result(
            model_name="gpt2", prompt="hello world", tokens=["hello", "world"],
            architecture_type="gpt_style", parameter_count=124_000_000,
            n_layers=1, n_heads=1,
            neuron_results=neurons, attention_results=heads,
            layer_summaries=[LayerSummary(0, 10, 0, 3.5, 1.0, 0.0)],
            analysis_duration=2.5,
        )
        summary = result.summary_stats()
        assert "gpt2" in summary
        assert "124.0M" in summary
        assert "Neuron 5" in summary
        assert "Layer 0" in summary
        assert "2.5s" in summary
