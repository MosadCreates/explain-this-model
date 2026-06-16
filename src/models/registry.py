import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    config: AutoConfig
    architecture_type: str
    parameter_count: int
    model_name: str
    device: str = "cpu"


def infer_architecture_type(model_config) -> str:
    """Classify a HuggingFace model into gpt_style, bert_style, or encoder_decoder.

    Uses the model's config class name and architecture attributes to determine
    the architecture type without any model-specific hardcoding.
    """
    config_class_name = type(model_config).__name__.lower()
    model_type = getattr(model_config, "model_type", "").lower()
    is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False)

    if is_encoder_decoder:
        return "encoder_decoder"

    if any(keyword in config_class_name for keyword in ["causal", "gpt", "llama", "mistral", "falcon", "opt"]):
        return "gpt_style"

    if any(keyword in model_type for keyword in ["gpt", "llama", "mistral", "falcon", "opt", "pythia", "neo", "jamba"]):
        return "gpt_style"

    if any(keyword in config_class_name for keyword in ["bert", "roberta", "electra", "albert", "deberta", "distilbert"]):
        return "bert_style"

    if any(keyword in model_type for keyword in ["bert", "roberta", "electra", "albert", "deberta", "distilbert"]):
        return "bert_style"

    if hasattr(model_config, "architectures") and model_config.architectures:
        for arch in model_config.architectures:
            if any(keyword in arch.lower() for keyword in ["forcausallm", "forconditionalgeneration"]):
                return "gpt_style"
            if any(keyword in arch.lower() for keyword in ["formaskedlm", "forsequenceclassification"]):
                return "bert_style"

    if is_encoder_decoder:
        return "encoder_decoder"

    logger.warning("Could not determine architecture type for %s, defaulting to gpt_style", config_class_name)
    return "gpt_style"


def _count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _estimate_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB based on parameter count and dtype."""
    param_count = _count_parameters(model)
    dtype_size = 4.0
    if next(model.parameters()).dtype == torch.bfloat16:
        dtype_size = 2.0
    elif next(model.parameters()).dtype == torch.float16:
        dtype_size = 2.0
    return (param_count * dtype_size) / (1024 * 1024)


def _get_auto_class(architecture_type: str):
    """Return the appropriate AutoModel class based on architecture type."""
    if architecture_type == "encoder_decoder":
        return AutoModelForSeq2SeqLM
    elif architecture_type == "bert_style":
        return AutoModelForMaskedLM
    else:
        return AutoModelForCausalLM


class ModelRegistry:
    """LRU-cached registry for loaded HuggingFace models.

    Maintains an in-memory cache of loaded models to avoid redundant
    downloading and deserialisation. The cache is bounded by max_size.
    """

    def __init__(self, max_size: int = 3, max_model_size_mb: int = 1024):
        self.max_size = max_size
        self.max_model_size_mb = max_model_size_mb
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()

    def load_model(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        force_reload: bool = False,
    ) -> LoadedModel:
        """Load a model from HuggingFace hub or local path.

        Args:
            model_name_or_path: HuggingFace model ID (e.g. "gpt2") or local path.
            device: Device to place the model on ("cpu" or "cuda").
            force_reload: If True, bypass cache and reload from disk/hub.

        Returns:
            A LoadedModel dataclass with model, tokenizer, config, and metadata.

        Raises:
            ValueError: If the model exceeds the size limit.
            OSError: If the model cannot be found or loaded.
        """
        if not force_reload and model_name_or_path in self._cache:
            logger.info("Model '%s' found in cache", model_name_or_path)
            return self._cache[model_name_or_path]

        logger.info("Loading model '%s'...", model_name_or_path)

        config = AutoConfig.from_pretrained(model_name_or_path)
        architecture_type = infer_architecture_type(config)
        auto_class = _get_auto_class(architecture_type)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        model = auto_class.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch.float32,
        )

        model_size_mb = _estimate_model_size_mb(model)
        if model_size_mb > self.max_model_size_mb:
            raise ValueError(
                f"Model '{model_name_or_path}' is {model_size_mb:.0f} MB, "
                f"which exceeds the maximum allowed size of {self.max_model_size_mb} MB."
            )

        model.eval()
        model.to(device)

        param_count = _count_parameters(model)
        logger.info(
            "Loaded '%s' (%s, %dM params, %.0f MB)",
            model_name_or_path,
            architecture_type,
            param_count // 1_000_000,
            model_size_mb,
        )

        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            architecture_type=architecture_type,
            parameter_count=param_count,
            model_name=model_name_or_path,
            device=device,
        )

        self._add_to_cache(model_name_or_path, loaded)
        return loaded

    def _add_to_cache(self, key: str, loaded: LoadedModel) -> None:
        """Add a model to the LRU cache, evicting the oldest if necessary."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self.max_size:
            evicted_key, evicted = self._cache.popitem(last=False)
            evicted_size = _estimate_model_size_mb(evicted.model) if hasattr(evicted, 'model') else 0
            logger.info("Evicted model '%s' from cache (%.0f MB)", evicted_key, evicted_size)
        self._cache[key] = loaded

    def get_model(self, model_name_or_path: str) -> Optional[LoadedModel]:
        """Retrieve a cached model without loading it (updates LRU order)."""
        if model_name_or_path in self._cache:
            self._cache.move_to_end(model_name_or_path)
            return self._cache[model_name_or_path]
        return None

    def clear(self) -> None:
        """Clear all cached models from memory."""
        self._cache.clear()
        logger.info("Model cache cleared")

    def cache_size(self) -> int:
        """Return the number of models currently cached."""
        return len(self._cache)

    def list_cached(self) -> list[str]:
        """Return list of model names currently in cache."""
        return list(self._cache.keys())


def format_parameter_count(count: int) -> str:
    """Format parameter count into human-readable string."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
