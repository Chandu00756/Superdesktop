"""
Unified in-memory storage adapter with async API, LRU eviction and TTL.
"""
import asyncio
import time
from collections import OrderedDict
from typing import Optional


class _Item:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: bytes, ttl: Optional[float]):
        self.value = value
        self.expires_at = (time.monotonic() + ttl) if ttl else None


class MemoryStore:
    def __init__(self, max_items: int = 100_000):
        self._data: OrderedDict[str, _Item] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max = max_items

    async def get(self, key: str) -> Optional[bytes]:
        async with self._lock:
            it = self._data.get(key)
            if not it:
                return None
            if it.expires_at and it.expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return it.value

    async def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> None:
        async with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = _Item(value, ttl)
            if len(self._data) > self._max:
                self._data.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def mget(self, keys: list[str]) -> list[Optional[bytes]]:
        async with self._lock:
            now = time.monotonic()
            out = []
            for k in keys:
                it = self._data.get(k)
                if not it or (it.expires_at and it.expires_at < now):
                    if it:
                        self._data.pop(k, None)
                    out.append(None)
                else:
                    self._data.move_to_end(k)
                    out.append(it.value)
            return out

    async def scan(self, prefix: str) -> list[str]:
        async with self._lock:
            now = time.monotonic()
            keys = []
            for k, it in list(self._data.items()):
                if it.expires_at and it.expires_at < now:
                    self._data.pop(k, None)
                    continue
                if k.startswith(prefix):
                    keys.append(k)
            return keys
