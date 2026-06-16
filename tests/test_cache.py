import json
import os
import tempfile

import pytest

from src.cache import ActivationCache, RedisCache, get_activation_cache, get_explanation_cache, clear_all_caches
from src.cache.activation_cache import ActivationCache as ActivationCacheDirect
from src.cache.redis_cache import RedisCache as RedisCacheDirect


class TestRedisCache:
    def test_init_uses_fallback_when_redis_unavailable(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        assert cache.using_redis is False

    def test_get_set_roundtrip(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        cache.set("test_key", {"hello": "world"})
        result = cache.get("test_key")
        assert result == {"hello": "world"}

    def test_get_miss_returns_none(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        assert cache.get("nonexistent_key") is None

    def test_ttl_expiry(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0", default_ttl_seconds=0)
        cache.set("expire_key", "value")
        import time
        time.sleep(0.01)
        assert cache.get("expire_key") is None

    def test_clear_removes_all(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_prefix_isolation(self):
        cache_a = RedisCache(redis_url="redis://localhost:16379/0", prefix="a:")
        cache_b = RedisCache(redis_url="redis://localhost:16379/0", prefix="b:")
        cache_a.set("key", "value_a")
        cache_b.set("key", "value_b")
        assert cache_a.get("key") == "value_a"
        assert cache_b.get("key") == "value_b"

    def test_get_many(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        cache.set("x", 10)
        cache.set("y", 20)
        results = cache.get_many(["x", "y", "z"])
        assert results["x"] == 10
        assert results["y"] == 20
        assert results["z"] is None

    def test_set_many(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        cache.set_many({"a": 1, "b": 2})
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_complex_serialisation(self):
        cache = RedisCache(redis_url="redis://localhost:16379/0")
        data = {
            "list": [1, 2, 3],
            "nested": {"a": 1},
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        cache.set("complex", data)
        assert cache.get("complex") == data


class TestActivationCache:
    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.cache_dir = tmpdir
            yield

    def test_init_creates_directory(self):
        path = os.path.join(tempfile.gettempdir(), "test_etc_cache")
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        cache = ActivationCache(cache_dir=path)
        assert os.path.exists(path)
        cache.clear()
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def test_get_set_roundtrip(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        data = {"tokens": ["hello", "world"], "activations": [0.1, 0.2]}
        cache.set("gpt2", "hello world", data)
        result = cache.get("gpt2", "hello world")
        assert result is not None
        assert result["tokens"] == ["hello", "world"]

    def test_get_miss_returns_none(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        assert cache.get("nonexistent", "prompt") is None

    def test_different_prompts_different_cache(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        cache.set("gpt2", "prompt A", {"data": "A"})
        cache.set("gpt2", "prompt B", {"data": "B"})
        assert cache.get("gpt2", "prompt A")["data"] == "A"
        assert cache.get("gpt2", "prompt B")["data"] == "B"

    def test_different_models_different_cache(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        cache.set("gpt2", "test", {"model": "gpt2"})
        cache.set("bert", "test", {"model": "bert"})
        assert cache.get("gpt2", "test")["model"] == "gpt2"
        assert cache.get("bert", "test")["model"] == "bert"

    def test_clear_removes_all(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        cache.set("gpt2", "p1", {"v": 1})
        cache.set("bert", "p2", {"v": 2})
        cache.clear()
        assert cache.get("gpt2", "p1") is None
        assert cache.get("bert", "p2") is None
        assert len(cache) == 0

    def test_max_entries_eviction(self):
        cache = ActivationCache(cache_dir=self.cache_dir, max_entries=3)
        cache.set("m1", "p1", {"v": 1})
        cache.set("m2", "p2", {"v": 2})
        cache.set("m3", "p3", {"v": 3})
        assert len(cache) == 3
        cache.set("m4", "p4", {"v": 4})
        assert len(cache) <= 3

    def test_empty_prompt(self):
        cache = ActivationCache(cache_dir=self.cache_dir)
        cache.set("gpt2", "", {"empty": True})
        assert cache.get("gpt2", "") is not None

    def test_persistent_across_instances(self):
        cache1 = ActivationCache(cache_dir=self.cache_dir)
        cache1.set("gpt2", "test prompt", {"persist": True})
        cache2 = ActivationCache(cache_dir=self.cache_dir)
        result = cache2.get("gpt2", "test prompt")
        assert result is not None
        assert result["persist"] is True

    def test_ttl_expiry(self):
        cache = ActivationCache(cache_dir=self.cache_dir, ttl_seconds=0)
        cache.set("gpt2", "test", {"data": 1})
        import time
        time.sleep(0.01)
        assert cache.get("gpt2", "test") is None


class TestCacheFactory:
    def test_get_explanation_cache_returns_instance(self):
        cache = get_explanation_cache()
        assert cache is not None

    def test_get_explanation_cache_singleton(self):
        cache1 = get_explanation_cache()
        cache2 = get_explanation_cache()
        assert cache1 is cache2

    def test_get_activation_cache_returns_instance(self):
        cache = get_activation_cache()
        assert cache is not None

    def test_get_activation_cache_singleton(self):
        cache1 = get_activation_cache()
        cache2 = get_activation_cache()
        assert cache1 is cache2

    def test_clear_all_caches(self):
        exp_cache = get_explanation_cache()
        act_cache = get_activation_cache()
        exp_cache.set("clear_test", 1)
        act_cache.set("gpt2", "test_clear", {"data": True})
        clear_all_caches()
        assert exp_cache.get("clear_test") is None
        assert act_cache.get("gpt2", "test_clear") is None
