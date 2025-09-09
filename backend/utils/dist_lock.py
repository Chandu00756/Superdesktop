"""
Redis-backed distributed lock with fencing tokens.
"""
import os
import uuid
from typing import Optional

import asyncio

try:
    import redis.asyncio as aioredis  # redis>=4.2
    _HAS_REDIS = True
except Exception:  # pragma: no cover - optional dependency path
    _HAS_REDIS = False


class DistLock:
    def __init__(self, redis_url: str, key: str, ttl: int = 10):
        if not _HAS_REDIS:
            raise RuntimeError("redis-py asyncio not available")
        self.redis = aioredis.from_url(redis_url)
        self.key = key
        self.ttl = ttl
        self.token = f"{uuid.uuid4()}:{os.getpid()}"

    async def acquire(self) -> bool:
        return bool(await self.redis.set(self.key, self.token, ex=self.ttl, nx=True))

    async def refresh(self) -> bool:
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('PEXPIRE', KEYS[1], ARGV[2])
        else return 0 end
        """
        return bool(
            await self.redis.eval(script, numkeys=1, keys=[self.key], args=[self.token, int(self.ttl * 1000)])
        )

    async def release(self) -> None:
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else return 0 end
        """
        await self.redis.eval(script, numkeys=1, keys=[self.key], args=[self.token])
