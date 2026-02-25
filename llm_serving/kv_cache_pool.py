"""
UFO-LLM: PGAS-Backed KV Cache Pool
Demonstrates storing intermediate Key-Value tensors from LLM inference
directly into Disaggregated Shared Memory over the PGAS.
"""

from typing import Dict, List, Tuple
import struct
import numpy as np

from memory_fabric.pgas_manager import UfoGlobalPointer, PGASManager

class KVCacheBlock:
    """
    Metadata for a single contiguous block of KV cache representing
    a layer's attention states for a specific token range.
    """
    def __init__(self, block_id: int, layer_idx: int, tokens: Tuple[int, int], ptr: UfoGlobalPointer, max_seq_len: int, head_dim: int):
        self.block_id = block_id
        self.layer_idx = layer_idx
        self.tokens = tokens  # (start_token, end_token)
        self.ptr = ptr
        self.is_hot = False
        
        # Sizing
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        # 1 float16 = 2 bytes. Shape: [2 (K+V), seq_len, head_dim]
        self.size_bytes = 2 * (self.tokens[1] - self.tokens[0]) * head_dim * 2


class PGASKVCachePool:
    """
    Manages allocations of KVCacheBlocks.
    """
    def __init__(self, pgas_manager: PGASManager):
        self.pgas_manager = pgas_manager
        
        # Auto-negotiate CXL Tier 1 capability before falling back to Ethernet Tier 2
        if 1 in pgas_manager.local_devices:
            self.tier_preference = 1
        else:
            self.tier_preference = 2 
            
        self.blocks: Dict[int, KVCacheBlock] = {}
        self.next_block = 0
        
    async def allocate_block(self, layer_idx: int, num_tokens: int, head_dim: int) -> KVCacheBlock:
        """Reserve a chunk of the global address space for KV states."""
        # Calculate size needed in bytes (FP16 assumed)
        size_bytes = 2 * num_tokens * head_dim * 2
        
        ptr = await self.pgas_manager.allocate(size_bytes, tier_preference=self.tier_preference)
        
        block = KVCacheBlock(
            block_id=self.next_block,
            layer_idx=layer_idx,
            tokens=(0, num_tokens),
            ptr=ptr,
            max_seq_len=num_tokens,
            head_dim=head_dim
        )
        
        self.blocks[self.next_block] = block
        self.next_block += 1
        return block

    async def store_tensors(self, block: KVCacheBlock, keys: np.ndarray, values: np.ndarray) -> None:
        """
        Pushes NumPy arrays (K,V) directly to the memory fabric.
        """
        assert keys.dtype == np.float16 and values.dtype == np.float16
        # Interleave K, V or append them (appending for simplicity)
        combined = np.concatenate([keys.flatten(), values.flatten()])
        data_bytes = combined.tobytes()
        
        if len(data_bytes) > block.size_bytes:
            raise ValueError("Tensor is larger than allocated block!")
            
        await self.pgas_manager.store(block.ptr, data_bytes)

    async def fetch_tensors(self, block: KVCacheBlock) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pulls NumPy arrays out of the fabric.
        """
        data_bytes = await self.pgas_manager.fetch(block.ptr, block.size_bytes)
        combined = np.frombuffer(data_bytes, dtype=np.float16)
        
        # Split back into keys and values (simulated)
        midpoint = len(combined) // 2
        keys = combined[:midpoint].reshape((block.tokens[1]-block.tokens[0], block.head_dim))
        values = combined[midpoint:].reshape((block.tokens[1]-block.tokens[0], block.head_dim))
        
        return keys, values
