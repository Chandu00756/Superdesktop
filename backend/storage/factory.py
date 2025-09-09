"""Storage adapter factory for cache/kv backends.

Env vars:
- OMEGA_STORE_BACKEND: memory (default) | minio
"""
from typing import Any
import os

from .memory_adapter import MemoryStore
try:
    from .minio_adapter import MinioKVStore  # type: ignore
except Exception:
    MinioKVStore = None  # type: ignore


class StoreHandle:
    def __init__(self, impl: Any):
        self.impl = impl

    # Convenience sync wrappers for simple use sites
    async def get(self, key: str):
        return await self.impl.get(key)

    async def set(self, key: str, value: bytes, ttl: float | None = None):
        return await self.impl.set(key, value, ttl)

    async def delete(self, key: str):
        return await self.impl.delete(key)

    async def ping(self) -> bool:
        p = getattr(self.impl, "ping", None)
        if p:
            return await p()
        return True


def build_default_store(max_items: int = 100_000) -> StoreHandle:
    backend = os.getenv("OMEGA_STORE_BACKEND", "memory").lower()
    # Default to memory
    impl: Any = MemoryStore(max_items=max_items)
    if backend in ("minio", "s3"):  # s3 alias resolves to MinIO client if available
        if MinioKVStore is not None:
            try:
                cand = MinioKVStore()
                # Basic health check, fall back on failure
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                ok = loop.run_until_complete(cand.ping())
                if ok:
                    impl = cand
            except Exception:
                # keep memory
                pass
    return StoreHandle(impl)


__all__ = ["build_default_store", "StoreHandle"]
