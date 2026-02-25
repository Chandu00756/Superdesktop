import pytest
from pgas_manager import UfoGlobalPointer, PGASManager, LocalMemoryDevice

def test_pointer_serialization():
    # Node 42, Tier 1, Offset 1024
    ptr = UfoGlobalPointer(42, 1, 1024)
    serialized = ptr.serialize()
    
    assert len(serialized) == 8, "Serialization must be exactly 8 bytes."
    
    recovered = UfoGlobalPointer.deserialize(serialized)
    assert recovered.node_id == 42
    assert recovered.tier_id == 1
    assert recovered.offset == 1024
    assert ptr == recovered

@pytest.mark.asyncio
async def test_pgas_local_alloc():
    manager = PGASManager(local_node_id=1)
    device = LocalMemoryDevice("ram0", tier_id=0, capacity_bytes=1024*1024)
    manager.register_device(device)
    
    # Test successful alloc
    ptr = await manager.allocate(256, tier_preference=0)
    assert ptr.node_id == 1
    assert ptr.tier_id == 0
    assert ptr.offset == 0
    
    # Test valid operations
    data = b"Hello, Disaggregated World!"
    await manager.store(ptr, data)
    
    fetched = await manager.fetch(ptr, len(data))
    assert fetched == data
    
    # Test out of bounds / capacity
    with pytest.raises(Exception):
        await manager.allocate(1024*1024 * 2) # Exceeds capacity
