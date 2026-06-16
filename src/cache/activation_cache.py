import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ActivationCache:
    """Disk-backed cache for model activations keyed by (model_name, prompt_hash).

    Stores serialised activation tensors as JSON files on disk.
    Since activations can be large, this avoids re-running the forward
    pass for identical (model, prompt) pairs.

    The cache is bounded by max_entries; oldest entries are evicted first.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: int = 50,
        ttl_seconds: int = 3600,
    ):
        if cache_dir is None:
            cache_dir = os.environ.get(
                "ACTIVATION_CACHE_DIR",
                os.path.join(tempfile.gettempdir(), "etc_activation_cache"),
            )
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index_path = os.path.join(self.cache_dir, "_index.json")
        self._index: dict[str, dict] = self._load_index()
        logger.info("ActivationCache initialised at %s (max=%d, ttl=%ds)", cache_dir, max_entries, ttl_seconds)

    def _load_index(self) -> dict[str, dict]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load activation cache index: %s", e)
        return {}

    def _save_index(self) -> None:
        try:
            with open(self._index_path, "w") as f:
                json.dump(self._index, f)
        except OSError as e:
            logger.warning("Failed to save activation cache index: %s", e)

    @staticmethod
    def _make_key(model_name: str, prompt: str) -> str:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        return f"{model_name}:{prompt_hash}"

    def _cache_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def get(self, model_name: str, prompt: str) -> Optional[dict[str, Any]]:
        key = self._make_key(model_name, prompt)
        entry = self._index.get(key)
        if entry is None:
            return None

        if time.time() - entry.get("cached_at", 0) > self.ttl_seconds:
            logger.debug("Activation cache expired for %s", key)
            self._evict(key)
            return None

        path = self._cache_path(key)
        if not os.path.exists(path):
            self._evict(key)
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read activation cache %s: %s", key, e)
            self._evict(key)
            return None

    def set(self, model_name: str, prompt: str, data: dict[str, Any]) -> None:
        key = self._make_key(model_name, prompt)
        path = self._cache_path(key)

        while len(self._index) >= self.max_entries:
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k].get("cached_at", 0))
            self._evict(oldest_key)

        try:
            with open(path, "w") as f:
                json.dump(data, f, default=str)
        except OSError as e:
            logger.warning("Failed to write activation cache %s: %s", key, e)
            return

        self._index[key] = {
            "model_name": model_name,
            "cached_at": time.time(),
            "ttl": self.ttl_seconds,
        }
        self._save_index()
        logger.debug("Activation cached: %s", key)

    def _evict(self, key: str) -> None:
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        self._index.pop(key, None)
        self._save_index()

    def clear(self) -> None:
        for key in list(self._index.keys()):
            self._evict(key)
        self._index = {}
        self._save_index()
        logger.info("Activation cache cleared")

    def __len__(self) -> int:
        return len(self._index)
