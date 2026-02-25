"""
UFO-PGAS: PGAS Caching Layer
Writethrough/Writeback cache wrapper designed to intercept `fetch` and `store` calls 
to higher-latency Tier 2 (RDMA/TCP) devices, backing them in local Tier 0 (DRAM).
"""

import time
import logging
from typing import Dict, Tuple
from pgas_manager import PGASDevice

logger = logging.getLogger(__name__)

class PGASCachingLayer:
    """
    Wraps a remote PGASDevice. Buffers reads and writes locally using Python dictionaries
    or bytearrays, dramatically hiding network latency for repetitive accesses.
    """
    def __init__(self, backend_device: PGASDevice, max_cache_size_bytes: int = 10 * 1024 * 1024):
        self.backend = backend_device
        self.capacity_bytes = max_cache_size_bytes
        self.current_usage_bytes = 0
        
        # Mapping: offset -> (data: bytes, last_access: float, dirty: bool)
        self.cache: Dict[int, Tuple[bytes, float, bool]] = {}

    def _evict_lru(self, needed_bytes: int):
        """Simple LRU eviction to make space."""
        while self.current_usage_bytes + needed_bytes > self.capacity_bytes and self.cache:
            # Sort by last_access timestamp
            oldest_offset = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[0]
            data, _, dirty = self.cache.pop(oldest_offset)
            
            # If we were using write-back, we would have to flush dirty pages here.
            # But the backend `store` keeps us writethrough for consistency, so `dirty` is always False.
            
            self.current_usage_bytes -= len(data)
            logger.debug(f"Cache Eviction: Removed {len(data)} bytes at offset {oldest_offset}")

    async def allocate(self, size_bytes: int) -> int:
        return await self.backend.allocate(size_bytes)
        
    async def deallocate(self, offset: int, size_bytes: int) -> None:
        if offset in self.cache:
            data, _, _ = self.cache.pop(offset)
            self.current_usage_bytes -= len(data)
        await self.backend.deallocate(offset, size_bytes)

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        """Read from Cache or fetch from Backend on miss."""
        if offset in self.cache:
            data, _, dirty = self.cache[offset]
            if len(data) >= size_bytes:
                # Cache Hit!
                self.cache[offset] = (data, time.time(), dirty)
                return data[:size_bytes]
                
        # Cache Miss
        data = await self.backend.fetch(offset, size_bytes)
        
        # Populate Cache
        self._evict_lru(len(data))
        if self.current_usage_bytes + len(data) <= self.capacity_bytes:
            self.cache[offset] = (data, time.time(), False)
            self.current_usage_bytes += len(data)
            
        return data

    async def store(self, offset: int, data: bytes) -> None:
        """
        Writethrough caching approach. Write to backend, and if successful, update local cache.
        """
        # Await the remote transmission ensuring global state is accurate
        await self.backend.store(offset, data)
        
        # Update cache buffer
        if offset in self.cache:
            old_data, _, _ = self.cache.pop(offset)
            self.current_usage_bytes -= len(old_data)
            
        self._evict_lru(len(data))
        if self.current_usage_bytes + len(data) <= self.capacity_bytes:
            self.cache[offset] = (data, time.time(), False)
            self.current_usage_bytes += len(data)
