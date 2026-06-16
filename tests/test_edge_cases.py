import pytest

from src.analysis.neuron_analyser import NeuronAnalyser, NeuronResult
from src.analysis.attention_analyser import AttentionAnalyser, AttentionHeadResult
from src.analysis.aggregator import AnalysisResult, build_analysis_result
from src.explanations.batch import SimpleCache, BatchExplanationGenerator
from src.explanations.generator import NullExplanationGenerator
from src.explanations.cost import estimate_api_cost, CostEstimate
from src.database import create_job, get_job, update_job_status


class TestNeuronAnalyserEdgeCases:
    def test_empty_activations_list(self):
        with pytest.raises(IndexError):
            analyser = NeuronAnalyser(mlp_activations=[], tokens=["hello"])
            analyser.compute_all_neurons_for_layer(0)

    def test_single_token_prompt(self):
        class MockCapture:
            output_tensor = None
        mock_capture = MockCapture()
        import torch
        mock_capture.output_tensor = torch.randn(1, 1, 64)

        analyser = NeuronAnalyser(mlp_activations=[mock_capture], tokens=["single"])
        results = analyser.compute_all_neurons_for_layer(0)
        assert len(results) == 64
        assert results[0].context_window == ["single"]
        assert results[0].context_window_positions == [0]


class TestAttentionAnalyserEdgeCases:
    def test_empty_activations(self):
        analyser = AttentionAnalyser(attention_activations=[], tokens=["hello"])
        results = analyser.rank_top_k(k=5)
        assert results == []

    def test_single_token_attention(self):
        import torch
        class MockCapture:
            output_tensor = torch.randn(1, 1, 768)
        analyser = AttentionAnalyser(
            attention_activations=[MockCapture()],
            tokens=["single"],
            native_attentions=[torch.randn(1, 12, 1, 1)],
            n_heads=12,
        )
        result = analyser.analyse_head(0, 0)
        assert result is not None
        assert result.pattern_type in ("diffuse", "content_based")


class TestAnalysisResultEdgeCases:
    def test_result_with_empty_lists(self):
        result = AnalysisResult(
            model_name="test", prompt="", tokens=[],
            architecture_type="unknown", parameter_count=0,
            n_layers=0, n_heads=0,
            neuron_results=[], attention_results=[], layer_summaries=[],
        )
        assert result.neuron_count == 0
        assert result.head_count == 0
        assert result.total_dead_neurons == 0

    def test_result_with_only_dead_neurons(self):
        dead = NeuronResult(
            layer_index=0, neuron_index=0,
            max_activation=0.0, mean_activation=0.0,
            std_activation=0.0, fraction_active=0.0,
            activating_token="[PAD]", activating_token_position=0,
            context_window=["[PAD]"], context_window_positions=[0],
            activation_values_per_token=[0.0],
            is_dead=True, z_score=0.0, rank=0,
        )
        result = AnalysisResult(
            model_name="test", prompt="x", tokens=["x"],
            architecture_type="test", parameter_count=0,
            n_layers=1, n_heads=0,
            neuron_results=[dead], attention_results=[], layer_summaries=[],
        )
        assert result.total_dead_neurons == 0
        assert result.neuron_results[0].is_dead is True


class TestCacheEdgeCases:
    def test_cache_empty_key(self):
        cache = SimpleCache()
        cache.set("", "value")
        assert cache.get("") == "value"

    def test_cache_special_chars_in_key(self):
        cache = SimpleCache()
        cache.set("key/with/slashes:and:colons", {"data": 1})
        assert cache.get("key/with/slashes:and:colons")["data"] == 1

    def test_cache_unicode_key(self):
        cache = SimpleCache()
        cache.set("résumé", "data")
        assert cache.get("résumé") == "data"

    def test_cache_very_large_value(self):
        cache = SimpleCache()
        large = {"data": "x" * 100_000}
        cache.set("large", large)
        assert len(cache.get("large")["data"]) == 100_000


class TestBatchGeneratorEdgeCases:
    def test_generator_with_no_neurons(self):
        gen = NullExplanationGenerator()
        batch = BatchExplanationGenerator(generator=gen)
        bundle = batch.generate_explanations_batch(
            neuron_results=[], attention_results=[],
            model_name="gpt2", tokens=["hello"], total_layers=1,
        )
        assert len(bundle.neuron_explanations) == 0
        assert len(bundle.head_explanations) == 0

    def test_generator_with_non_available_api(self):
        gen = NullExplanationGenerator()
        batch = BatchExplanationGenerator(generator=gen)

        class MockNeuron:
            layer_index = 0
            neuron_index = 0
            max_activation = 1.0
            mean_activation = 0.5
            std_activation = 0.1
            fraction_active = 1.0
            activating_token = "test"
            activating_token_position = 0
            context_window = ["test"]
            context_window_positions = [0]
            activation_values_per_token = [1.0]
            is_dead = False

        bundle = batch.generate_explanations_batch(
            neuron_results=[MockNeuron()],
            attention_results=[],
            model_name="gpt2", tokens=["test"], total_layers=1,
        )
        assert len(bundle.neuron_explanations) == 1
        assert "unavailable" in bundle.neuron_explanations[0].hypothesis.lower()


class TestCostEstimationEdgeCases:
    def test_zero_neurons_zero_heads(self):
        cost = estimate_api_cost(n_neurons=0, n_heads=0)
        assert cost.estimated_cost_usd == 0.0
        assert cost.neuron_count == 0
        assert cost.head_count == 0

    def test_large_number_of_neurons(self):
        cost = estimate_api_cost(n_neurons=1000, n_heads=100)
        assert cost.estimated_cost_usd > 0
        assert cost.neuron_count == 1000
        assert cost.head_count == 100

    def test_unknown_provider_returns_zero_cost(self):
        cost = estimate_api_cost(n_neurons=10, n_heads=5, provider="unknown")
        assert cost.estimated_cost_usd == 0.0


class TestDatabaseEdgeCases:
    def test_create_and_fetch_with_long_model_name(self, monkeypatch):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        monkeypatch.setenv("DATABASE_PATH", db_path)
        from src import database
        database._engine = None
        database._SessionLocal = None
        database.init_db()

        long_name = "huggingface/very-long-model-name-that-might-cause-issues" * 5
        job_id = create_job(long_name, "test prompt")
        job = get_job(job_id)
        assert job is not None
        assert job.model_name == long_name

        if database._engine is not None:
            database._engine.dispose()
        database._engine = None
        database._SessionLocal = None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_create_job_with_unicode_prompt(self, monkeypatch):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        monkeypatch.setenv("DATABASE_PATH", db_path)
        from src import database
        database._engine = None
        database._SessionLocal = None
        database.init_db()

        unicode_prompt = "你好世界 🎉 こんにちは"
        job_id = create_job("gpt2", unicode_prompt)
        job = get_job(job_id)
        assert job is not None
        assert job.prompt == unicode_prompt

        if database._engine is not None:
            database._engine.dispose()
        database._engine = None
        database._SessionLocal = None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_update_nonexistent_job_does_not_crash(self, monkeypatch):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        monkeypatch.setenv("DATABASE_PATH", db_path)
        from src import database
        database._engine = None
        database._SessionLocal = None
        database.init_db()

        update_job_status("nonexistent-job-id", "completed")
        assert get_job("nonexistent-job-id") is None

        if database._engine is not None:
            database._engine.dispose()
        database._engine = None
        database._SessionLocal = None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
