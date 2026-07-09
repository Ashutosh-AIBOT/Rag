"""
Redis-based caching layer for query results, BM25 index, embeddings, etc.
Falls back gracefully to in-memory dict if Redis is unavailable.
"""
import json
import logging
from typing import Any, Optional
from functools import wraps

logger = logging.getLogger("cache")

_redis_client = None
_memory_cache: dict = {}
USE_REDIS = False


def init_cache(redis_url: str = "redis://localhost:6379/0"):
    global _redis_client, USE_REDIS
    try:
        import redis
        _redis_client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        USE_REDIS = True
        logger.info("Redis cache connected: %s", redis_url)
    except Exception as e:
        USE_REDIS = False
        logger.warning("Redis unavailable, using in-memory cache: %s", e)


def get_cache(key: str) -> Optional[Any]:
    if USE_REDIS and _redis_client:
        try:
            val = _redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
    # Return JSON-deserialized copy so memory and Redis caches return
    # identical types (dict/list, not raw Pydantic/model objects).
    raw = _memory_cache.get(key)
    if raw is None:
        return None
    try:
        return json.loads(json.dumps(raw, default=str))
    except (TypeError, ValueError):
        return raw


def set_cache(key: str, value: Any, ttl_seconds: int = 300):
    serialized = json.dumps(value, default=str)
    if USE_REDIS and _redis_client:
        try:
            _redis_client.setex(key, ttl_seconds, serialized)
            return
        except Exception:
            pass
    _memory_cache[key] = value


def delete_cache(key: str):
    if USE_REDIS and _redis_client:
        try:
            _redis_client.delete(key)
        except Exception:
            pass
    _memory_cache.pop(key, None)


def clear_pattern(pattern: str):
    if USE_REDIS and _redis_client:
        try:
            keys = _redis_client.keys(pattern)
            if keys:
                _redis_client.delete(*keys)
        except Exception:
            pass
    keys_to_delete = [k for k in _memory_cache if k.startswith(pattern.replace("*", ""))]
    for k in keys_to_delete:
        del _memory_cache[k]


def cached(prefix: str, ttl: int = 300):
    """Decorator that caches function results by prefix + args."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{prefix}:{hash((args, tuple(sorted(kwargs.items()))))}"
            result = get_cache(cache_key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            if result is not None:
                set_cache(cache_key, result, ttl)
            return result
        return wrapper
    return decorator


def query_result_key(query: str, strategy: str, filters_hash: str) -> str:
    import hashlib
    q_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:16]
    return f"query:{strategy}:{q_hash}:{filters_hash}"
