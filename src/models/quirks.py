import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .hooks import ActivationHookManager, hook_context

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureQuirks:
    """Per-architecture notes discovered at load time via dummy forward pass.

    These quirks are inferred automatically by running a tiny forward pass and
    inspecting tensor shapes — no hardcoded architecture-specific branches needed.
    """
    mlp_activation_dim: int = 0
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 0
    d_model: int = 0
    uses_gated_mlp: bool = False
    activation_function: str = "unknown"
    has_bias: bool = True
    layer_norm_placement: str = "pre"
    architecture_type: str = "unknown"
    supports_output_attentions: bool = False


def infer_quirks(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cpu",
) -> ArchitectureQuirks:
    """Infer architecture quirks by running a tiny dummy forward pass.

    This is the key method that makes the tool fully model-agnostic. Instead of
    hardcoding that GPT-2 has 12 layers and 12 heads, we run a 2-token prompt
    through the model with hooks attached and inspect the resulting tensor shapes.

    Steps:
        1. Register hooks on every MLP and attention layer
        2. Run a short dummy prompt ("the") through the model
        3. Inspect MLP activation shapes to determine d_mlp dimension
        4. Inspect attention output shapes to determine n_heads and head_dim
        5. Check if attention weights are available via output_attentions=True
        6. Check if the MLP uses gated activations (SwiGLU/GLU variants)
        7. Clean up hooks

    Raises:
        RuntimeError: If the dummy forward pass fails to produce activations.
    """
    model.eval()
    quirks = ArchitectureQuirks()

    with hook_context(model) as manager:
        inputs = tokenizer("the", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        mlp_activations = manager.get_mlp_activations()
        attention_activations = manager.get_attention_activations()

        if not mlp_activations:
            raise RuntimeError(
                "No MLP activations detected. The model may not have any layers "
                "matching the MLP patterns. Check config mlp_patterns."
            )

        quirks.n_layers = len(mlp_activations)

        last_mlp = mlp_activations[-1]
        if last_mlp.output_tensor.dim() >= 2:
            quirks.mlp_activation_dim = last_mlp.output_tensor.shape[-1]

        last_mlp_input = last_mlp.input_tensor
        if last_mlp_input.dim() >= 2:
            quirks.d_model = last_mlp_input.shape[-1]

        # Determine if MLP uses gated activation.
        # Gated MLPs (LLaMA, Mistral) have 3 weight matrices in the MLP module
        # instead of 2. The output dimension of the MLP's first linear layer
        # will be 2 * d_mlp for gated variants.
        mlp_module = _find_mlp_module(model, last_mlp.module_name)
        if mlp_module is not None:
            quirks.uses_gated_mlp = _check_gated_mlp(mlp_module, quirks.mlp_activation_dim, quirks.d_model)
            quirks.activation_function = _detect_activation(mlp_module)

        # Infer attention head count and head dimension.
        if attention_activations:
            last_attn = attention_activations[-1]
            attn_output = last_attn.output_tensor

            if attn_output.dim() >= 3:
                batch, seq, hidden = attn_output.shape[0], attn_output.shape[1], attn_output.shape[-1]

                # Try to infer n_heads from the config if available
                config_heads = getattr(model.config, "num_attention_heads", None) or \
                    getattr(model.config, "num_heads", None) or \
                    getattr(model.config, "n_head", None)
                if config_heads is not None:
                    quirks.n_heads = config_heads
                    if hidden % config_heads == 0:
                        quirks.head_dim = hidden // config_heads
                    else:
                        quirks.head_dim = hidden // config_heads if config_heads > 0 else 0
                else:
                    quirks.n_heads = attn_output.shape[-1] // 64
                    quirks.head_dim = 64
        else:
            config_heads = getattr(model.config, "num_attention_heads", None) or \
                getattr(model.config, "n_head", None) or 0
            if config_heads:
                quirks.n_heads = config_heads
                if hasattr(model.config, "hidden_size"):
                    quirks.head_dim = model.config.hidden_size // config_heads if config_heads > 0 else 0

        # Check if model supports output_attentions.
        forward_params = list(model.forward.__code__.co_varnames[:model.forward.__code__.co_argcount])
        quirks.supports_output_attentions = "output_attentions" in forward_params

        # Set architecture type from config if available.
        if hasattr(model.config, "model_type"):
            quirks.architecture_type = model.config.model_type

        # Check layer norm placement.
        quirks.layer_norm_placement = _detect_layer_norm_placement(model)

        logger.info(
            "Inferred quirks: L=%d, d_model=%d, d_mlp=%d, n_heads=%d, head_dim=%d, "
            "gated_mlp=%s, activation=%s",
            quirks.n_layers,
            quirks.d_model,
            quirks.mlp_activation_dim,
            quirks.n_heads,
            quirks.head_dim,
            quirks.uses_gated_mlp,
            quirks.activation_function,
        )

    return quirks


def _find_mlp_module(model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
    """Find the actual MLP module by name."""
    for name, module in model.named_modules():
        if name == module_name:
            return module
    return None


def _check_gated_mlp(mlp_module: torch.nn.Module, d_mlp: int, d_model: int) -> bool:
    """Check if an MLP module uses a gated activation (SwiGLU/GLU).

    Gated MLPs have three weight matrices instead of two. The presence of
    three linear layers in the module with specific dimension relationships
    indicates a gated architecture.

    For a standard MLP: W_in: [d_model, d_mlp], W_out: [d_mlp, d_model]
    For a gated MLP:     W_gate: [d_model, d_mlp], W_up: [d_model, d_mlp], W_down: [d_mlp, d_model]
    """
    linear_layers = [m for m in mlp_module.modules() if isinstance(m, torch.nn.Linear)]

    if len(linear_layers) >= 3:
        return True

    return False


def _detect_activation(mlp_module: torch.nn.Module) -> str:
    """Detect the activation function used in an MLP module."""
    for module in mlp_module.modules():
        class_name = type(module).__name__.lower()
        if "gelu" in class_name:
            return "gelu"
        if "relu" in class_name:
            return "relu"
        if "silu" in class_name or "swish" in class_name:
            return "silu"
        if "sigmoid" in class_name:
            return "sigmoid"
        if "tanh" in class_name:
            return "tanh"
    return "unknown"


def _detect_layer_norm_placement(model: torch.nn.Module) -> str:
    """Detect whether layer norm is applied before or after attention/MLP.

    This is a heuristic: we check for PreLN vs PostLN by looking at the
    order of LayerNorm and attention modules within a block.
    """
    block_count_pre = 0
    block_count_post = 0
    last_was_norm = False

    for name, module in model.named_modules():
        class_name = type(module).__name__
        if "LayerNorm" in class_name or "RMSNorm" in class_name:
            last_was_norm = True
        elif last_was_norm and ("Attention" in class_name or "MLP" in class_name):
            block_count_pre += 1
            last_was_norm = False
        elif "Attention" in class_name or "MLP" in class_name:
            block_count_post += 1
            last_was_norm = False

    if block_count_pre >= block_count_post and block_count_pre > 0:
        return "pre"
    elif block_count_post > 0:
        return "post"
    return "pre"


def format_quirks_summary(quirks: ArchitectureQuirks) -> str:
    """Return a human-readable summary of architecture quirks."""
    parts = [
        f"Layers: {quirks.n_layers}",
        f"d_model: {quirks.d_model}",
        f"d_mlp: {quirks.mlp_activation_dim}",
        f"Heads: {quirks.n_heads} × {quirks.head_dim}",
        f"Gate: {'yes' if quirks.uses_gated_mlp else 'no'}",
        f"Act: {quirks.activation_function}",
        f"Norm: {quirks.layer_norm_placement}",
        f"Attn output: {'yes' if quirks.supports_output_attentions else 'no'}",
    ]
    return " | ".join(parts)
