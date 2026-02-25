"""
UFO-LLM: Disaggregated Serving Demo
Executes Prefill mapping and Decode mapping utilizing the UFO-Task graph
and UFO-PGAS memory fabric.
"""

from typing import List, Tuple
import logging
import time

from memory_fabric.pgas_manager import PGASManager, UfoGlobalPointer
from llm_serving.kv_cache_pool import PGASKVCachePool, KVCacheBlock
from compute_node.task_graph import UfoTaskDescriptor, UfoTaskGraph, TaskCapability
from compute_node.scheduler import UfoScheduler

logger = logging.getLogger(__name__)

class DisaggregatedServingManager:
    """
    Simulates the separation of a Prefill cluster from a Decode cluster
    by creating a Task Graph mapped to PGAS pointers.
    """
    def __init__(self, pgas_manager: PGASManager, scheduler: UfoScheduler):
        self.pgas_manager = pgas_manager
        self.scheduler = scheduler
        self.kv_pool = PGASKVCachePool(pgas_manager, tier_preference=2) # Store KV cache on ethernet
        
    async def create_inference_job(self, prompt: str, num_layers: int = 4) -> Tuple[UfoTaskGraph, List[KVCacheBlock]]:
        """
        Subdivides the LLM job into a Prefill task and a set of Decode tasks.
        Uses the `UfoTaskGraph` API and PGAS to link their states physically.
        """
        graph = UfoTaskGraph(name=f"llm_inference_{int(time.time())}")
        num_tokens = len(prompt.split()) # Mock tokenization
        head_dim = 128
        
        # 1. Allocate KV Blocks across the Fabric for all layers
        kv_blocks = []
        for i in range(num_layers):
            block = await self.kv_pool.allocate_block(layer_idx=i, num_tokens=num_tokens, head_dim=head_dim)
            kv_blocks.append(block)
            
        kv_pointers = [b.ptr for b in kv_blocks]
            
        # 2. Define the Prefill Task
        # Prefill requires High-Bandwidth GPU locally to process massive prompt concurrently.
        prefill_task = UfoTaskDescriptor(
            name="Prefill_Stage",
            required_capabilities=[TaskCapability.GPU_TENSOR],
            latency_sensitive=False,  # Throughput focus
            ufo_out=kv_pointers       # Writes KVs to the fabric
        )
        graph.add_task(prefill_task)
        
        # 3. Define Decode Tasks
        # Decode requires low latency but can handle low bandwidth since it reads 1 token at a time.
        decode_task = UfoTaskDescriptor(
            name="Decode_Stage",
            required_capabilities=[TaskCapability.GPU_CUDA, TaskCapability.NPU_EDGE], # Fallback edge inference
            latency_sensitive=True,   # Latency focus
            ufo_in=kv_pointers        # Reads KVs back from the fabric
        )
        decode_task.add_dependency(prefill_task.task_id)
        graph.add_task(decode_task)
        
        logger.info(f"Built Task Graph mapping Prefill -> {len(kv_blocks)} KV Cache Blocks -> Decode Stage")
        return graph, kv_blocks
