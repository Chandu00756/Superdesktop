# Compatibility shim for projects importing `aioredis`.
# Tries to import the old aioredis package first; if not found,
# provides a thin alias to `redis.asyncio` (modern replacement).
# This avoids ModuleNotFoundError when code imports `aioredis`.

try:
    import aioredis as _aioredis
    globals().update({k: getattr(_aioredis, k) for k in dir(_aioredis) if not k.startswith('_')})
except Exception:
    try:
        import redis.asyncio as _ra

        # Map common aioredis API to redis.asyncio equivalents
        def from_url(url, *args, **kwargs):
            return _ra.from_url(url, *args, **kwargs)

        Redis = _ra.Redis
        ConnectionPool = getattr(_ra, 'ConnectionPool', None)

        # Expose names expected by existing code
        globals().update({
            'from_url': from_url,
            'Redis': Redis,
            'ConnectionPool': ConnectionPool,
            'module': _ra,
        })
    except Exception:
        # If neither is available, raise ModuleNotFoundError to preserve original behavior
        raise
