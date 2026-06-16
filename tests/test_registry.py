import pytest
import torch
from src.models.registry import (
    ModelRegistry,
    infer_architecture_type,
    _count_parameters,
    format_parameter_count,
)


class TestInferArchitectureType:
    """Tests for architecture classification from model config."""

    def test_gpt2_is_gpt_style(self):
        """GPT-2 should be classified as gpt_style."""
        from transformers import GPT2Config
        config = GPT2Config()
        assert infer_architecture_type(config) == "gpt_style"

    def test_distilbert_is_bert_style(self):
        """DistilBERT should be classified as bert_style."""
        from transformers import DistilBertConfig
        config = DistilBertConfig()
        assert infer_architecture_type(config) == "bert_style"

    def test_bert_is_bert_style(self):
        """BERT should be classified as bert_style."""
        from transformers import BertConfig
        config = BertConfig()
        assert infer_architecture_type(config) == "bert_style"

    def test_t5_is_encoder_decoder(self):
        """T5 should be classified as encoder_decoder."""
        from transformers import T5Config
        config = T5Config()
        config.is_encoder_decoder = True
        assert infer_architecture_type(config) == "encoder_decoder"


class TestFormatParameterCount:
    """Tests for human-readable parameter count formatting."""

    def test_billions(self):
        assert format_parameter_count(7_000_000_000) == "7.0B"

    def test_millions(self):
        assert format_parameter_count(124_000_000) == "124.0M"

    def test_thousands(self):
        assert format_parameter_count(5_000) == "5.0K"

    def test_small(self):
        assert format_parameter_count(42) == "42"


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_registry_empty_on_init(self):
        registry = ModelRegistry(max_size=3)
        assert registry.cache_size() == 0
        assert registry.list_cached() == []

    def test_registry_clear(self):
        registry = ModelRegistry(max_size=3)
        registry._add_to_cache("test", None)
        assert registry.cache_size() == 1
        registry.clear()
        assert registry.cache_size() == 0

    def test_lru_eviction(self):
        registry = ModelRegistry(max_size=2)
        registry._add_to_cache("a", "model_a")
        registry._add_to_cache("b", "model_b")
        assert registry.cache_size() == 2
        registry._add_to_cache("c", "model_c")
        assert registry.cache_size() == 2
        assert registry.get_model("a") is None
        assert registry.get_model("c") == "model_c"

    def test_lru_reorder_on_access(self):
        registry = ModelRegistry(max_size=2)
        registry._add_to_cache("a", "model_a")
        registry._add_to_cache("b", "model_b")
        registry.get_model("a")
        registry._add_to_cache("c", "model_c")
        assert registry.get_model("b") is None
        assert registry.get_model("a") == "model_a"
        assert registry.get_model("c") == "model_c"


@pytest.mark.slow
class TestRealModelLoading:
    """Integration tests that download actual models from HuggingFace hub."""

    def test_load_gpt2(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        assert loaded.model is not None
        assert loaded.tokenizer is not None
        assert loaded.architecture_type == "gpt_style"
        assert loaded.parameter_count > 0
        assert loaded.model_name == "gpt2"
        assert loaded.device == "cpu"

    def test_load_distilbert(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("distilbert-base-uncased")
        assert loaded.model is not None
        assert loaded.architecture_type == "bert_style"

    def test_cached_model_returned(self):
        registry = ModelRegistry(max_size=3)
        first = registry.load_model("gpt2")
        second = registry.load_model("gpt2")
        assert first is second
        assert registry.cache_size() == 1

    def test_force_reload_bypasses_cache(self):
        registry = ModelRegistry(max_size=3)
        registry.load_model("gpt2")
        loaded = registry.load_model("gpt2", force_reload=True)
        assert loaded.model_name == "gpt2"

    def test_model_runs_forward_pass(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        inputs = loaded.tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = loaded.model(**inputs)
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == 2
