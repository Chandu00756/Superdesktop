"""
UFO-PGAS: CXL Allocator & Page Hotness Tracker
Tracks memory access patterns and migrates "hot" objects from Tier 2 
(emulated TCP) into Tier 0 (DRAM) or Tier 1 (CXL) hardware.
"""

import time
import struct
import logging
from collections import deque
from typing import Dict, Deque, Tuple

from pgas_manager import PGASManager, UfoGlobalPointer

logger = logging.getLogger(__name__)

# LRU / Hotness Metrics
MIGRATION_THRESHOLD = 5  # Number of accesses before promoting to Tier 0
HOTNESS_WINDOW_SEC = 2.0


class CxlAllocator:
    """
    Manages Tier promotion.
    Interprets Tier 2 memory requests and decides when a Page must
    move to Tier 1 / Tier 0.
    """
    def __init__(self, pgas_manager: PGASManager):
        self.pgas_manager = pgas_manager
        
        # ptr -> (access_count, last_access_timestamp)
        self.access_history: Dict[UfoGlobalPointer, Tuple[int, float]] = {}
        
        # LRU eviction queue tracking objects built in local Tier 0
        self.tier0_objects: Deque[UfoGlobalPointer] = deque()

    def record_access(self, ptr: UfoGlobalPointer) -> bool:
        """
        Record a read/write access.
        Return True if object was marked 'hot' and needs migration.
        """
        now = time.time()
        
        if ptr not in self.access_history:
            self.access_history[ptr] = (1, now)
        else:
            count, last_ts = self.access_history[ptr]
            
            # Reset count if it's been untouched for too long
            if now - last_ts > HOTNESS_WINDOW_SEC:
                count = 1
            else:
                count += 1
                
            self.access_history[ptr] = (count, now)
            
            if count >= MIGRATION_THRESHOLD:
                # Reset after migration event to prevent spam
                self.access_history[ptr] = (0, now)
                return True
                
        return False

    async def migrate_to_local(self, ptr: UfoGlobalPointer, size_bytes: int) -> UfoGlobalPointer:
        """
        Migrate an object from a remote tier to local Tier 0 Memory.
        Steps:
         1. Allocate new pointer locally.
         2. Fetch data from remote pointer.
         3. Store data into local pointer.
         4. Deallocate remote pointer.
         5. Return the new local pointer.
        """
        if ptr.node_id == self.pgas_manager.local_node_id and ptr.tier_id == 0:
            return ptr  # Already optimal
            
        logger.info(f"Page Hotness Triggered. Migrating {ptr} to Local Tier 0.")
        
        # Step 1
        new_ptr = await self.pgas_manager.allocate(size_bytes, tier_preference=0)
        
        # Step 2
        data = await self.pgas_manager.fetch(ptr, size_bytes)
        
        # Step 3
        await self.pgas_manager.store(new_ptr, data)
        
        # Step 4
        # Note: In a true PGAS, you'd send an RPC to `ptr.node_id` to deallocate.
        # For local test environments, we simply assume it's freed if it belonged to us.
        if ptr.node_id == self.pgas_manager.local_node_id:
            await self.pgas_manager.deallocate(ptr)
            
        self.tier0_objects.append(new_ptr)
        return new_ptr
        
    async def evict_lru(self, target_tier: int = 2) -> UfoGlobalPointer | None:
        """
        Demote the least recently used Tier 0 object into Tier 2 (ethernet).
        """
        if not self.tier0_objects:
            return None
            
        old_ptr = self.tier0_objects.popleft()
        
        # Get its size from PGAS
        if old_ptr not in self.pgas_manager.allocations:
            return None
            
        size_bytes = self.pgas_manager.allocations[old_ptr].size_bytes
        logger.info(f"Evicting {old_ptr} from local memory to Tier {target_tier}")
        
        # 1. New alloc
        demoted_ptr = await self.pgas_manager.allocate(size_bytes, tier_preference=target_tier)
        
        # 2. Fetch local data
        data = await self.pgas_manager.fetch(old_ptr, size_bytes)
        
        # 3. Store remote
        await self.pgas_manager.store(demoted_ptr, data)
        
        # 4. Deallocate local
        await self.pgas_manager.deallocate(old_ptr)
        
        return demoted_ptr
