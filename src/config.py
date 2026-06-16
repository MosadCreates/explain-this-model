import os
import threading
from typing import Any

import yaml

_config: dict[str, Any] = None
_lock = threading.Lock()
_config_path: str = None


def _load_default_config() -> dict[str, Any]:
    return {
        "analysis": {
            "top_k_neurons": 20,
            "top_k_heads": 10,
            "max_prompt_tokens": 512,
            "context_window_size": 5,
        },
        "explanation": {
            "provider": "gemini",
            "gemini_model": "gemini-2.0-flash",
            "claude_model": "claude-3-haiku-20240307",
            "max_explanations_per_job": 30,
            "batch_size": 5,
        },
        "rate_limiting": {
            "max_jobs_per_hour": 10,
            "window_seconds": 3600,
        },
        "cache": {
            "model_cache_size": 3,
            "activation_ttl_seconds": 3600,
            "explanation_ttl_seconds": 86400,
        },
        "model": {
            "max_size_mb": 1024,
            "supported_architectures": ["gpt_style", "bert_style", "encoder_decoder"],
            "mlp_patterns": ["MLP", "FFN", "FeedForward", "mlp", "ffn", "DenseReluDense"],
            "attention_patterns": ["Attention", "MultiHead", "attention", "attn", "SelfAttention", "AttentionLayer"],
        },
    }


def set_config_path(path: str):
    global _config_path
    _config_path = path


def get_config() -> dict[str, Any]:
    global _config, _config_path

    if _config is not None:
        return _config

    with _lock:
        if _config is not None:
            return _config

        cfg = _load_default_config()

        path = _config_path or os.environ.get("CONFIG_PATH", "")
        if not path:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(base_dir, "configs", "default.yaml")

        if os.path.exists(path):
            with open(path, "r") as f:
                file_cfg = yaml.safe_load(f) or {}
                for section in cfg:
                    if section in file_cfg:
                        cfg[section].update(file_cfg[section])

        _config = cfg
        return _config


def reload_config():
    global _config
    with _lock:
        _config = None
    return get_config()
