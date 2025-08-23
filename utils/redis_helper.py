import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

class InMemoryRedisStub:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def close(self):
        return


async def get_redis_client(url: str) -> Optional[object]:
    """Try to obtain a Redis client in order: aioredis, redis.asyncio, or in-memory stub."""
    try:
        import aioredis
        if hasattr(aioredis, 'from_url'):
            logger.info("Using installed aioredis.from_url")
            return await aioredis.from_url(url)
    except Exception:
        pass

    try:
        import redis.asyncio as redis_asyncio
        if hasattr(redis_asyncio, 'from_url'):
            logger.info("Falling back to redis.asyncio.from_url")
            return await redis_asyncio.from_url(url)
    except Exception:
        pass

    logger.warning("Redis not available; using in-memory stub for degraded mode")
    # Return an instance of the in-memory stub
    return InMemoryRedisStub()
