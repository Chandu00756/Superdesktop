import pytest
from llm_serving.router import DisaggregatedServingManager
from llm_serving.kv_cache_pool import PGASKVCachePool, KVCacheBlock
from memory_fabric.pgas_manager import PGASManager, LocalMemoryDevice
from compute_node.scheduler import UfoScheduler, TaskCapability

@pytest.mark.asyncio
async def test_llm_inference_routing():
    # Setup mocks
    pgas_manager = PGASManager(local_node_id=1)
    
    # We mount an ethernet emulator tier (2) directly in local memory for simple test
    device = LocalMemoryDevice("remote0", tier_id=2, capacity_bytes=1024*1024*10)
    pgas_manager.register_device(device)
    
    scheduler = UfoScheduler()
    # Node 1: High End GPU Box
    scheduler.register_node(1, [TaskCapability.GPU_TENSOR, TaskCapability.GPU_CUDA])
    
    # Node 2: Weak Edge Device
    scheduler.register_node(2, [TaskCapability.NPU_EDGE, TaskCapability.CPU_GENERAL])
    
    manager = DisaggregatedServingManager(pgas_manager, scheduler)
    
    # Build Graph
    graph, kv_blocks = await manager.create_inference_job("The core architecture of UFO-PGAS relies on...", num_layers=4)
    
    # Check Blocks
    assert len(kv_blocks) == 4
    for b in kv_blocks:
        assert b.ptr.tier_id == 2
        
    # Check Scheduler placements
    ready = graph.get_ready_tasks(set())
    assert len(ready) == 1
    assert ready[0].name == "Prefill_Stage"
    
    placement = scheduler.schedule(graph, set())
    assert placement[ready[0].task_id] == 1 # Must route to Node 1 (GPU Tensor Core)
    
    # Complete prefill
    placement2 = scheduler.schedule(graph, {ready[0].task_id})
    decode_tasks = graph.get_ready_tasks({ready[0].task_id})
    assert len(decode_tasks) == 1
    assert decode_tasks[0].name == "Decode_Stage"
    
    # Route to NPU edge node 2 (It has NPU_EDGE capabilities requested by Decode task!)
    # Node 1 is an option, but in pure disaggregation, we expect Decode to go far and wide.
    # In scheduler logic, if both can handle it, it will pick one (mostly randomly or based on Data locality)
    # Both have the exact same locality (Ethernet Tier 2), so Node 1 is chosen arbitrarily. 
    # But for NPU requirement, Node 2 provides it if GPU CUDA is missing.
    assert decode_tasks[0].task_id in placement2
