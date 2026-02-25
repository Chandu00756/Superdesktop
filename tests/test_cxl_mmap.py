import os
import pytest
import numpy as np
import time

from memory_fabric.cxl_device import CXLMemoryDevice
from llm_serving.kv_cache_pool import PGASKVCachePool
from memory_fabric.pgas_manager import PGASManager

@pytest.fixture
def cxl_env():
    backing_file = "test_cxl.mem"
    capacity = 50 * 1024 * 1024 # 50 MB
    
    device = CXLMemoryDevice("cxl0", tier_id=1, backing_file=backing_file, capacity_bytes=capacity)
    
    yield device
    
    device.close()
    if os.path.exists(backing_file):
        os.remove(backing_file)

@pytest.mark.asyncio
async def test_cxl_zero_copy_operations(cxl_env):
    device = cxl_env
    
    # 1. Allocate 10 MB
    size = 10 * 1024 * 1024
    offset = await device.allocate(size)
    assert offset == 0
    
    # 2. Benchmark Write Speed (MMAP should be nearly instantaneous vs Socket)
    data = b"\xAA" * size
    
    t0 = time.perf_counter()
    for _ in range(10):
        await device.store(offset, data)
    write_time = time.perf_counter() - t0
    
    # 3. Read Verification
    t1 = time.perf_counter()
    for _ in range(10):
        fetched = await device.fetch(offset, size)
    read_time = time.perf_counter() - t1
    
    assert fetched == data
    
    print(f"\\nCXL MMAP Latency (10MB x 10 Writes): {write_time:.5f}s")
    print(f"CXL MMAP Latency (10MB x 10 Reads):  {read_time:.5f}s")
    
    await device.deallocate(offset, size)

@pytest.mark.asyncio
async def test_kv_pool_tier_negotiation(cxl_env):
    manager = PGASManager(local_node_id=1)
    
    # No CXL device mapped initially
    pool_eth = PGASKVCachePool(manager)
    assert pool_eth.tier_preference == 2  # Defaults to Ethernet
    
    # Map the CXL device (Tier 1)
    manager.register_device(cxl_env)
    
    # Ensure KV pool binds strictly to CXL when available
    pool_cxl = PGASKVCachePool(manager)
    assert pool_cxl.tier_preference == 1  # Successfully negotiated Tier 1 CXL Fabric
