import pytest
import torch
import torch.nn as nn

from src.models.quirks import (
    ArchitectureQuirks,
    _check_gated_mlp,
    _detect_activation,
    _detect_layer_norm_placement,
    _find_mlp_module,
    format_quirks_summary,
)


class TestArchitectureQuirksDataclass:
    def test_default_values(self):
        q = ArchitectureQuirks()
        assert q.mlp_activation_dim == 0
        assert q.n_layers == 0
        assert q.n_heads == 0
        assert q.uses_gated_mlp is False
        assert q.activation_function == "unknown"
        assert q.has_bias is True
        assert q.layer_norm_placement == "pre"

    def test_custom_values(self):
        q = ArchitectureQuirks(
            n_layers=12,
            n_heads=16,
            d_model=768,
            uses_gated_mlp=True,
            activation_function="silu",
        )
        assert q.n_layers == 12
        assert q.n_heads == 16
        assert q.d_model == 768
        assert q.uses_gated_mlp is True
        assert q.activation_function == "silu"


class TestFormatQuirksSummary:
    def test_summary_contains_key_info(self):
        q = ArchitectureQuirks(n_layers=6, d_model=512, mlp_activation_dim=2048, n_heads=8, head_dim=64)
        summary = format_quirks_summary(q)
        assert "Layers: 6" in summary
        assert "d_model: 512" in summary
        assert "d_mlp: 2048" in summary
        assert "8" in summary
        assert "64" in summary
        assert "Gate:" in summary

    def test_gated_mlp_in_summary(self):
        q = ArchitectureQuirks(uses_gated_mlp=True)
        summary = format_quirks_summary(q)
        assert "yes" in summary

    def test_non_gated_mlp_in_summary(self):
        q = ArchitectureQuirks(uses_gated_mlp=False)
        summary = format_quirks_summary(q)
        assert "no" in summary


class TestFindMlpModule:
    def test_finds_module_by_name(self):
        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(10, 10)
            def forward(self, x):
                return self.mlp(x)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.ModuleList([TransformerBlock()])
            def forward(self, x):
                return self.h[0](x)
        model = Model()
        found = _find_mlp_module(model, "h.0.mlp")
        assert found is not None
        assert isinstance(found, nn.Linear)

    def test_returns_none_for_missing(self):
        model = nn.Linear(10, 10)
        found = _find_mlp_module(model, "nonexistent.module")
        assert found is None


class TestCheckGatedMlp:
    def test_standard_mlp_has_two_linears(self):
        mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        assert _check_gated_mlp(mlp, 256, 64) is False

    def test_gated_mlp_has_three_or_more_linears(self):
        class GatedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 256)
                self.up_proj = nn.Linear(64, 256)
                self.down_proj = nn.Linear(256, 64)
                self.act = nn.SiLU()
            def forward(self, x):
                return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        mlp = GatedMLP()
        assert _check_gated_mlp(mlp, 256, 64) is True

    def test_empty_module(self):
        mlp = nn.Sequential()
        assert _check_gated_mlp(mlp, 0, 0) is False


class TestDetectActivation:
    def test_detects_gelu(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.GELU())
        assert _detect_activation(mlp) == "gelu"

    def test_detects_relu(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        assert _detect_activation(mlp) == "relu"

    def test_detects_silu(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.SiLU())
        assert _detect_activation(mlp) == "silu"

    def test_detects_sigmoid(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.Sigmoid())
        assert _detect_activation(mlp) == "sigmoid"

    def test_detects_tanh(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        assert _detect_activation(mlp) == "tanh"

    def test_unknown_activation(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.1))
        assert _detect_activation(mlp) == "unknown"

    def test_no_activation_layers(self):
        mlp = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        assert _detect_activation(mlp) == "unknown"


class TestDetectLayerNormPlacement:
    def make_pre_ln_block(self):
        return nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
        )

    def test_detects_pre_ln(self):
        model = nn.Sequential(self.make_pre_ln_block())
        result = _detect_layer_norm_placement(model)
        assert result in ("pre", "post")

    def test_no_norm_found(self):
        model = nn.Sequential(nn.Linear(10, 10))
        result = _detect_layer_norm_placement(model)
        assert result == "pre"


class TestInferQuirksRealModel:
    @pytest.mark.slow
    def test_infer_quirks_on_gpt2(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        from src.models.quirks import infer_quirks
        quirks = infer_quirks(model, tokenizer)
        assert quirks.n_layers > 0
        assert quirks.mlp_activation_dim > 0
        assert quirks.activation_function in ("gelu", "unknown")
        assert quirks.layer_norm_placement in ("pre", "post")

    @pytest.mark.slow
    def test_infer_quirks_on_distilbert(self):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        from src.models.quirks import infer_quirks
        quirks = infer_quirks(model, tokenizer)
        assert quirks.n_layers > 0
        assert quirks.d_model > 0
        assert quirks.activation_function in ("gelu", "relu", "unknown")
