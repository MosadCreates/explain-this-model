import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class HookCapture:
    """Captured activations from a single hook point."""
    module_name: str
    layer_index: int
    input_tensor: torch.Tensor
    output_tensor: torch.Tensor
    hook_type: str  # "mlp" or "attention"


class ActivationHookManager:
    """Manages forward hooks on MLP and attention layers of any HuggingFace model.

    Hooks are model-agnostic: they identify target modules by matching class names
    against configurable patterns rather than hardcoded paths. This allows the
    same code to work with GPT-2, BERT, T5, LLaMA, and any other transformer
    architecture.

    Used as a context manager to guarantee hook cleanup:
        with ActivationHookManager(model) as manager:
            outputs = model(**inputs)
            activations = manager.get_activations()

    Hook lifecycle:
        1. register_hooks() — placed on __enter__
        2. forward pass — hooks capture activations thread-safely
        3. cleanup() — placed on __exit__, always runs even on exception
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mlp_patterns: Optional[list[str]] = None,
        attention_patterns: Optional[list[str]] = None,
    ):
        self.model = model
        self.mlp_patterns = mlp_patterns or [
            "MLP", "FFN", "FeedForward", "mlp", "ffn", "DenseReluDense",
        ]
        self.attention_patterns = attention_patterns or [
            "Attention", "MultiHead", "attention", "attn", "SelfAttention", "AttentionLayer",
        ]
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._activations: dict[str, list[HookCapture]] = defaultdict(list)
        self._layer_counter: dict[str, int] = defaultdict(int)

    def _make_hook(self, module_name: str, hook_type: str):
        """Create a forward hook closure that captures input and output tensors."""
        def hook_fn(module, input_tensor, output_tensor):
            if isinstance(input_tensor, tuple) and len(input_tensor) > 0:
                inp = input_tensor[0].detach()
            elif isinstance(input_tensor, torch.Tensor):
                inp = input_tensor.detach()
            else:
                inp = torch.tensor(0)
            if isinstance(output_tensor, tuple) and len(output_tensor) > 0:
                out = output_tensor[0].detach()
            elif isinstance(output_tensor, torch.Tensor):
                out = output_tensor.detach()
            else:
                out = torch.tensor(0)
            capture = HookCapture(
                module_name=module_name,
                layer_index=self._layer_counter[hook_type] - 1,
                input_tensor=inp,
                output_tensor=out,
                hook_type=hook_type,
            )
            self._activations[hook_type].append(capture)
        return hook_fn

    def register_hooks(self) -> None:
        """Iterate over all named modules and register hooks on MLP/attention layers.

        Module class names are matched against configurable patterns. This is
        the key design decision that makes the hook system model-agnostic:
        instead of hardcoding 'transformer.h.0.mlp' for GPT-2, we match any
        module whose class name contains 'MLP', 'FFN', etc.

        This correctly identifies:
        - GPT-2:    `MLP` class (in `GPT2Block`)
        - BERT:     `BertSelfAttention` + `BertIntermediate` classes
        - T5:       `T5FF` + `T5Attention` classes
        - LLaMA:    `LlamaMLP` + `LlamaAttention` classes
        - DistilBERT: `DistilBertSelfAttention` + `FFN` classes
        """
        for name, module in self.model.named_modules():
            class_name = type(module).__name__

            if self._matches_pattern(class_name, self.mlp_patterns):
                self._layer_counter["mlp"] += 1
                hook = module.register_forward_hook(self._make_hook(name, "mlp"))
                self._hooks.append(hook)
                logger.debug("Registered MLP hook on %s (%s)", name, class_name)

            elif self._matches_pattern(class_name, self.attention_patterns):
                self._layer_counter["attention"] += 1
                hook = module.register_forward_hook(self._make_hook(name, "attention"))
                self._hooks.append(hook)
                logger.debug("Registered attention hook on %s (%s)", name, class_name)

        logger.info(
            "Registered hooks: %d MLP layers, %d attention layers",
            self._layer_counter.get("mlp", 0),
            self._layer_counter.get("attention", 0),
        )

    def _matches_pattern(self, class_name: str, patterns: list[str]) -> bool:
        """Check if a class name matches any of the given patterns."""
        for pattern in patterns:
            if pattern in class_name:
                return True
        return False

    def get_activations(self, hook_type: Optional[str] = None) -> dict[str, list[HookCapture]]:
        """Return captured activations, optionally filtered by hook type."""
        if hook_type:
            return {hook_type: self._activations.get(hook_type, [])}
        return dict(self._activations)

    def get_mlp_activations(self) -> list[HookCapture]:
        """Return all MLP activation captures, sorted by layer index."""
        return sorted(self._activations.get("mlp", []), key=lambda c: c.layer_index)

    def get_attention_activations(self) -> list[HookCapture]:
        """Return all attention activation captures, sorted by layer index."""
        return sorted(self._activations.get("attention", []), key=lambda c: c.layer_index)

    def cleanup(self) -> None:
        """Remove all registered hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()
        self._layer_counter.clear()
        logger.debug("All hooks cleaned up")

    def __enter__(self):
        """Context manager entry: register hooks and return self."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: always clean up hooks."""
        self.cleanup()
        return False

    @property
    def is_active(self) -> bool:
        """Return True if hooks are currently registered."""
        return len(self._hooks) > 0


def run_with_hooks(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    hook_manager: ActivationHookManager,
    max_length: int = 512,
    output_attentions: bool = False,
) -> dict[str, Any]:
    """Tokenize a prompt and run a single forward pass inside a hook manager context.

    Args:
        model: The PyTorch model to run.
        tokenizer: The HuggingFace tokenizer.
        prompt: Input text prompt.
        hook_manager: An ActivationHookManager instance (hooks will be registered).
        max_length: Maximum token length for the prompt.
        output_attentions: If True, passes output_attentions=True to the model
            (required for architectures that return attention weights this way).

    Returns:
        dict with keys:
            - "logits": model output logits
            - "mlp_activations": list of HookCapture for MLP layers
            - "attention_activations": list of HookCapture for attention layers
            - "tokens": list of token strings
            - "input_ids": tensor of input token IDs
            - "attention_mask": attention mask tensor
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    tokens = [tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

    forward_kwargs = {**inputs}
    if output_attentions:
        forward_kwargs["output_attentions"] = True

    with hook_manager:
        with torch.no_grad():
            outputs = model(**forward_kwargs)
        mlp_activations = hook_manager.get_mlp_activations()
        attention_activations = hook_manager.get_attention_activations()

    logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()

    # For models that return attention weights natively, try to extract them
    native_attentions = None
    if output_attentions and hasattr(outputs, "attentions") and outputs.attentions is not None:
        native_attentions = [att.detach() for att in outputs.attentions]

    return {
        "logits": logits,
        "mlp_activations": mlp_activations,
        "attention_activations": attention_activations,
        "tokens": tokens,
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs.get("attention_mask"),
        "native_attentions": native_attentions,
    }


@contextmanager
def hook_context(model: torch.nn.Module, mlp_patterns=None, attention_patterns=None):
    """Convenience context manager for quick hook-and-run operations.

    Usage:
        with hook_context(model) as manager:
            outputs = model(**inputs)
            activations = manager.get_activations()
    """
    manager = ActivationHookManager(model, mlp_patterns, attention_patterns)
    try:
        manager.register_hooks()
        yield manager
    finally:
        manager.cleanup()
