import logging
import os
from typing import Optional

from .activation_cache import ActivationCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

_explanation_cache: Optional[RedisCache] = None
_activation_cache: Optional[ActivationCache] = None


def get_explanation_cache(
    redis_url: Optional[str] = None,
    default_ttl: int = 86400,
) -> RedisCache:
    global _explanation_cache
    if _explanation_cache is None:
        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _explanation_cache = RedisCache(
            redis_url=url,
            default_ttl_seconds=default_ttl,
            prefix="etc:exp:",
        )
        logger.info("Explanation cache initialised (redis=%s)", _explanation_cache.using_redis)
    return _explanation_cache


def get_activation_cache(
    cache_dir: Optional[str] = None,
    max_entries: int = 50,
    ttl_seconds: int = 3600,
) -> ActivationCache:
    global _activation_cache
    if _activation_cache is None:
        _activation_cache = ActivationCache(
            cache_dir=cache_dir,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )
    return _activation_cache


def clear_all_caches() -> None:
    global _explanation_cache, _activation_cache
    if _explanation_cache is not None:
        _explanation_cache.clear()
    if _activation_cache is not None:
        _activation_cache.clear()
    logger.info("All caches cleared")


__all__ = [
    "RedisCache",
    "ActivationCache",
    "get_explanation_cache",
    "get_activation_cache",
    "clear_all_caches",
]
