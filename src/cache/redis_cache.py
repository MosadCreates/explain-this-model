import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-backed cache with TTL, implementing the same interface as SimpleCache.

    Falls back to a dict-based in-memory cache if Redis is unavailable,
    so the application never crashes due to cache unavailability.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl_seconds: int = 86400,
        prefix: str = "etc:",
    ):
        self.default_ttl = default_ttl_seconds
        self.prefix = prefix
        self._redis = None
        self._fallback: dict[str, tuple[float, Any]] = {}
        self._initialised = False

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._try_connect(url)

    def _try_connect(self, url: str) -> None:
        try:
            import redis as redis_mod
            self._redis = redis_mod.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            self._redis.ping()
            logger.info("Connected to Redis at %s", url)
        except Exception as e:
            logger.warning("Redis unavailable at %s, using in-memory fallback: %s", url, e)
            self._redis = None
        self._initialised = True

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _serialise(self, value: Any) -> str:
        return json.dumps(value, default=str)

    def _deserialise(self, raw: str) -> Any:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def get(self, key: str) -> Optional[Any]:
        prefixed = self._make_key(key)
        if self._redis is not None:
            try:
                raw = self._redis.get(prefixed)
                if raw is not None:
                    logger.debug("Redis cache HIT for key=%s", key)
                    return self._deserialise(raw)
                logger.debug("Redis cache MISS for key=%s", key)
                return None
            except Exception as e:
                logger.warning("Redis get failed, falling back: %s", e)
                self._redis = None

        if prefixed in self._fallback:
            expiry, value = self._fallback[prefixed]
            if time.time() < expiry:
                return value
            del self._fallback[prefixed]
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        prefixed = self._make_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        if self._redis is not None:
            try:
                raw = self._serialise(value)
                self._redis.setex(prefixed, ttl, raw)
                return
            except Exception as e:
                logger.warning("Redis set failed, falling back: %s", e)
                self._redis = None

        self._fallback[prefixed] = (time.time() + ttl, value)

    def clear(self) -> None:
        if self._redis is not None:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{self.prefix}*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Redis cache cleared (prefix=%s)", self.prefix)
                return
            except Exception as e:
                logger.warning("Redis clear failed: %s", e)
                self._redis = None

        self._fallback.clear()

    def get_many(self, keys: list[str]) -> dict[str, Optional[Any]]:
        result = {}
        for key in keys:
            result[key] = self.get(key)
        return result

    def set_many(self, mapping: dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        for key, value in mapping.items():
            self.set(key, value, ttl_seconds)

    @property
    def using_redis(self) -> bool:
        return self._redis is not None
