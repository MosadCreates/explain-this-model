import pytest
import torch

from src.models.hooks import ActivationHookManager, run_with_hooks, hook_context
from src.models.registry import ModelRegistry


class TestActivationHookManager:
    """Unit tests for the ActivationHookManager."""

    def test_hook_pattern_matches_mlp(self):
        manager = ActivationHookManager.__new__(ActivationHookManager)
        assert manager._matches_pattern("GPT2MLP", ["MLP", "FFN"])
        assert manager._matches_pattern("LlamaMLP", ["MLP"])
        assert manager._matches_pattern("FFN", ["MLP", "FFN"])
        assert manager._matches_pattern("BertIntermediate", ["MLP"]) is False

    def test_hook_pattern_matches_attention(self):
        manager = ActivationHookManager.__new__(ActivationHookManager)
        assert manager._matches_pattern("GPT2Attention", ["Attention"])
        assert manager._matches_pattern("BertSelfAttention", ["Attention"])
        assert manager._matches_pattern("T5Attention", ["Attention"])
        assert manager._matches_pattern("MLP", ["Attention"]) is False

    def test_manager_starts_inactive(self):
        model = torch.nn.Linear(10, 10)
        manager = ActivationHookManager(model)
        assert manager.is_active is False

    def test_cleanup_on_exit(self):
        """Hooks should be removed after context manager exits."""
        import transformers
        config = transformers.GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4)
        model = transformers.GPT2Model(config)
        with ActivationHookManager(model) as manager:
            assert manager.is_active
            dummy = torch.randint(0, 100, (1, 5))
            model(dummy)
            activations = manager.get_activations()
            assert len(activations["mlp"]) > 0 or len(activations["attention"]) > 0
        assert manager.is_active is False
        assert len(manager._hooks) == 0


@pytest.mark.slow
class TestRealHookExecution:
    """Integration tests for hooks on real models."""

    def test_hooks_fire_on_gpt2(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer("The cat sat on the mat", return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
            mlp_acts = manager.get_mlp_activations()
            attn_acts = manager.get_attention_activations()
            assert len(mlp_acts) > 0
            assert len(attn_acts) > 0

    def test_hooks_fire_on_distilbert(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("distilbert-base-uncased")
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer("The cat sat on the mat", return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
            mlp_acts = manager.get_mlp_activations()
            attn_acts = manager.get_attention_activations()
            assert len(mlp_acts) > 0
            assert len(attn_acts) > 0

    def test_activation_shapes_match_gpt2(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        prompt = "Hello world"
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
            for cap in manager.get_mlp_activations():
                assert cap.output_tensor.dim() == 3
                seq_len = cap.output_tensor.shape[1]
                assert seq_len == 2
                assert cap.module_name is not None
                assert cap.hook_type == "mlp"

    def test_hook_cleanup_after_context(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        hook_count_before = len(loaded.model._forward_hooks)
        with ActivationHookManager(loaded.model) as manager:
            inputs = loaded.tokenizer("test", return_tensors="pt")
            with torch.no_grad():
                loaded.model(**inputs)
        hook_count_after = len(loaded.model._forward_hooks)
        assert hook_count_after == hook_count_before

    def test_deterministic_activations(self):
        """Running the same prompt twice should give identical activations."""
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        prompt = "The future of AI"
        torch.manual_seed(42)

        def get_activations(prompt_text):
            with ActivationHookManager(loaded.model) as manager:
                inputs = loaded.tokenizer(prompt_text, return_tensors="pt")
                with torch.no_grad():
                    loaded.model(**inputs)
                return [cap.output_tensor.clone() for cap in manager.get_mlp_activations()]

        acts1 = get_activations(prompt)
        acts2 = get_activations(prompt)
        for a1, a2 in zip(acts1, acts2):
            assert torch.allclose(a1, a2)

    def test_run_with_hooks_function(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        manager = ActivationHookManager(loaded.model)
        result = run_with_hooks(loaded.model, loaded.tokenizer, "Hello world", manager)
        assert "logits" in result
        assert "mlp_activations" in result
        assert "attention_activations" in result
        assert "tokens" in result
        assert len(result["tokens"]) == 2
        assert result["mlp_activations"][0].hook_type == "mlp"

    def test_hook_context_convenience(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        inputs = loaded.tokenizer("test", return_tensors="pt")
        with hook_context(loaded.model) as manager:
            with torch.no_grad():
                loaded.model(**inputs)
            mlp_acts = manager.get_mlp_activations()
            assert len(mlp_acts) > 0

    def test_no_hook_leak_on_exception(self):
        registry = ModelRegistry(max_size=3)
        loaded = registry.load_model("gpt2")
        hook_count_before = len(loaded.model._forward_hooks)
        try:
            with ActivationHookManager(loaded.model) as manager:
                raise RuntimeError("deliberate error")
        except RuntimeError:
            pass
        hook_count_after = len(loaded.model._forward_hooks)
        assert hook_count_after == hook_count_before
