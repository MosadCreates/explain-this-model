import json
import tempfile
import os
from unittest.mock import patch, MagicMock

import pytest

from src.analysis.aggregator import build_analysis_result
from src.analysis.neuron_analyser import NeuronAnalyser, NeuronResult
from src.analysis.attention_analyser import AttentionAnalyser, AttentionHeadResult
from src.database import create_job, get_job, store_result, get_result, init_db
from src.explanations.batch import BatchExplanationGenerator, NeuronExplanation, AttentionHeadExplanation, ExplanationBundle
from src.explanations.generator import NullExplanationGenerator


class TestFullPipelineMock:
    """Integration-style tests that exercise the full analysis pipeline with mocks."""

    def test_full_analysis_flow(self):
        """Simulate the full analysis pipeline: activations → analyse → aggregate → explain → store."""
        import torch

        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        seq_len = len(tokens)
        d_mlp = 64

        mock_mlp_hook = MagicMock()
        mock_mlp_hook.output_tensor = torch.randn(1, seq_len, d_mlp)
        mock_mlp_hook.input_tensor = torch.randn(1, seq_len, 768)
        mock_mlp_hook.module_name = "transformer.h.0.mlp"

        mock_attn_hook = MagicMock()
        mock_attn_hook.output_tensor = torch.randn(1, seq_len, 768)
        mock_attn_hook.module_name = "transformer.h.0.attention"

        mlp_captures = [mock_mlp_hook]
        attn_captures = [mock_attn_hook]
        native_attentions = [torch.randn(1, 12, seq_len, seq_len)]
        n_heads = 12
        n_layers = 1

        neuron_analyser = NeuronAnalyser(mlp_captures, tokens, context_window_size=3)
        all_neurons = neuron_analyser.compute_all_neurons_for_layer(0)
        ranked_neurons = sorted(all_neurons, key=lambda r: r.max_activation, reverse=True)[:5]
        layer_summaries = neuron_analyser.compute_layer_summaries()

        assert len(ranked_neurons) == 5
        assert len(layer_summaries) == 1
        assert layer_summaries[0].layer_index == 0
        assert layer_summaries[0].total_neurons == d_mlp

        attn_analyser = AttentionAnalyser(attn_captures, tokens, native_attentions, n_heads=n_heads)
        ranked_heads = attn_analyser.rank_top_k(k=3)
        assert len(ranked_heads) <= 3

        gen = NullExplanationGenerator()
        batch_gen = BatchExplanationGenerator(generator=gen)

        bundle = batch_gen.generate_explanations_batch(
            neuron_results=ranked_neurons,
            attention_results=ranked_heads,
            model_name="gpt2",
            tokens=tokens,
            total_layers=n_layers,
        )

        assert len(bundle.neuron_explanations) == len(ranked_neurons)
        assert len(bundle.head_explanations) == len(ranked_heads)

        result = build_analysis_result(
            model_name="gpt2",
            prompt="the cat sat on the mat",
            tokens=tokens,
            architecture_type="gpt_style",
            parameter_count=100_000_000,
            n_layers=n_layers,
            n_heads=n_heads,
            neuron_results=ranked_neurons,
            attention_results=ranked_heads,
            layer_summaries=layer_summaries,
            analysis_duration=0.5,
        )

        assert result.model_name == "gpt2"
        assert result.neuron_count == 5
        assert result.head_count <= 3
        assert result.total_dead_neurons >= 0

        result_dict = result.to_dict()
        result_dict["explanations"] = {
            "neurons": [
                {"layer_index": e.layer_index, "neuron_index": e.neuron_index,
                 "hypothesis": e.hypothesis, "confidence": e.confidence,
                 "pattern_type": e.pattern_type, "cached": e.cached}
                for e in bundle.neuron_explanations
            ],
            "heads": [
                {"layer_index": e.layer_index, "head_index": e.head_index,
                 "hypothesis": e.hypothesis, "confidence": e.confidence,
                 "pattern_type": e.pattern_type, "cached": e.cached}
                for e in bundle.head_explanations
            ],
        }

        assert "neuron_results" in result_dict
        assert "explanations" in result_dict
        assert "neurons" in result_dict["explanations"]
        assert "heads" in result_dict["explanations"]

    def test_end_to_end_db_storage(self):
        """Test that analysis results can be stored and retrieved from DB."""
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        os.environ["DATABASE_PATH"] = db_path

        from src import database as db_mod
        db_mod._engine = None
        db_mod._SessionLocal = None
        db_mod.init_db()

        job_id = db_mod.create_job("gpt2", "hello world")
        assert job_id is not None

        job = db_mod.get_job(job_id)
        assert job.status == "pending"

        result_data = {
            "model_name": "gpt2",
            "prompt": "hello world",
            "tokens": ["hello", "world"],
            "architecture_type": "gpt_style",
            "parameter_count": 100_000_000,
            "n_layers": 12,
            "n_heads": 12,
            "neuron_results": [],
            "attention_results": [],
            "layer_summaries": [],
            "neuron_count": 0,
            "head_count": 0,
            "total_dead_neurons": 0,
            "analysis_duration_seconds": 0.1,
            "explanations": {
                "neurons": [],
                "heads": [],
                "total_api_calls": 0,
                "total_cached": 0,
                "explanation_duration_seconds": 0.0,
            },
        }

        result_id = db_mod.store_result(job_id, result_data)
        assert result_id is not None

        job = db_mod.get_job(job_id)
        assert job.status == "completed"
        assert job.result_id is not None

        retrieved = db_mod.get_result(job.result_id)
        assert retrieved is not None
        assert retrieved["model_name"] == "gpt2"
        assert retrieved["prompt"] == "hello world"
        assert "explanations" in retrieved

        if db_mod._engine is not None:
            db_mod._engine.dispose()
        db_mod._engine = None
        db_mod._SessionLocal = None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_parallel_layer_analysis_consistency(self):
        """Test that ThreadPoolExecutor-based analysis produces same results as sequential."""
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed

        torch.manual_seed(42)
        tokens = ["a", "b", "c", "d"]
        seq_len = len(tokens)
        d_mlp = 32

        captures = []
        for layer in range(4):
            hook = MagicMock()
            hook.output_tensor = torch.randn(1, seq_len, d_mlp)
            hook.input_tensor = torch.randn(1, seq_len, 128)
            hook.module_name = f"layer.{layer}.mlp"
            captures.append(hook)

        analyser = NeuronAnalyser(captures, tokens)

        sequential = []
        for layer_idx in range(len(captures)):
            sequential.extend(analyser.compute_all_neurons_for_layer(layer_idx))
        sequential = sorted(sequential, key=lambda r: r.max_activation, reverse=True)[:10]

        parallel = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(analyser.compute_all_neurons_for_layer, i): i for i in range(len(captures))}
            for f in as_completed(futures):
                parallel.extend(f.result())
        parallel = sorted(parallel, key=lambda r: r.max_activation, reverse=True)[:10]

        assert len(sequential) == len(parallel)
        for s, p in zip(sequential, parallel):
            assert s.layer_index == p.layer_index
            assert s.neuron_index == p.neuron_index
            assert abs(s.max_activation - p.max_activation) < 1e-5

    def test_pipeline_error_propagation(self):
        """Test that errors in the pipeline are properly caught and reported."""
        from src.explanations.batch import BatchExplanationGenerator

        gen = NullExplanationGenerator()
        batch = BatchExplanationGenerator(generator=gen)

        class BadNeuron:
            @property
            def layer_index(self):
                raise RuntimeError("Simulated error")

        try:
            bundle = batch.generate_explanations_batch(
                neuron_results=[BadNeuron()],
                attention_results=[],
                model_name="gpt2", tokens=["test"], total_layers=1,
            )
            assert False, "Should have raised an error"
        except (RuntimeError, Exception) as e:
            assert "Simulated" in str(e)
