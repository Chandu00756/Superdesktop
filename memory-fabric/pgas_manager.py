"""
UFO-PGAS: Partitioned Global Address Space Manager
Core controller allocating global virtual addresses mapped to physical blocks across the cluster.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging
import uuid
import struct

logger = logging.getLogger(__name__)

# Constants for address space partitioning
NODE_ID_BITS = 16
TIER_ID_BITS = 8
OFFSET_BITS = 40

MAX_NODE_ID = (1 << NODE_ID_BITS) - 1
MAX_TIER_ID = (1 << TIER_ID_BITS) - 1
MAX_OFFSET = (1 << OFFSET_BITS) - 1


class UfoGlobalPointer:
    """
    64-bit global address representation.
    Layout: 
      [ Node ID: 16 bits ] [ Tier ID: 8 bits ] [ Offset/Address: 40 bits ]
    """
    __slots__ = ['node_id', 'tier_id', 'offset']

    def __init__(self, node_id: int, tier_id: int, offset: int):
        if not (0 <= node_id <= MAX_NODE_ID):
            raise ValueError(f"Node ID {node_id} out of bounds (0-{MAX_NODE_ID})")
        if not (0 <= tier_id <= MAX_TIER_ID):
            raise ValueError(f"Tier ID {tier_id} out of bounds (0-{MAX_TIER_ID})")
        if not (0 <= offset <= MAX_OFFSET):
            raise ValueError(f"Offset {offset} out of bounds (0-{MAX_OFFSET})")
            
        self.node_id = node_id
        self.tier_id = tier_id
        self.offset = offset

    @classmethod
    def from_int(cls, address: int) -> 'UfoGlobalPointer':
        """De-serialize a 64-bit integer into a pointer."""
        offset = address & MAX_OFFSET
        tier_id = (address >> OFFSET_BITS) & MAX_TIER_ID
        node_id = (address >> (OFFSET_BITS + TIER_ID_BITS)) & MAX_NODE_ID
        return cls(node_id, tier_id, offset)

    def to_int(self) -> int:
        """Serialize into a 64-bit integer."""
        address = (self.node_id << (OFFSET_BITS + TIER_ID_BITS))
        address |= (self.tier_id << OFFSET_BITS)
        address |= self.offset
        return address

    def serialize(self) -> bytes:
        """Serialize to 8 bytes for network transmission."""
        return struct.pack("!Q", self.to_int())

    @classmethod
    def deserialize(cls, data: bytes) -> 'UfoGlobalPointer':
        """Deserialize from 8 bytes."""
        address = struct.unpack("!Q", data)[0]
        return cls.from_int(address)
        
    def __repr__(self) -> str:
        return f"<UfoPtr node={self.node_id} tier={self.tier_id} offset=0x{self.offset:010x}>"
        
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UfoGlobalPointer):
            return False
        return (self.node_id == other.node_id and 
                self.tier_id == other.tier_id and 
                self.offset == other.offset)
                
    def __hash__(self) -> int:
        return hash(self.to_int())


class PGASDevice:
    """Abstract base class for different memory backend tiers."""
    def __init__(self, device_id: str, tier_id: int, capacity_bytes: int):
        self.device_id = device_id
        self.tier_id = tier_id
        self.capacity_bytes = capacity_bytes
        self.allocated_bytes = 0
        
    async def allocate(self, size_bytes: int) -> int:
        """Reserve a block of size_bytes. Return its local offset."""
        raise NotImplementedError
        
    async def deallocate(self, offset: int, size_bytes: int) -> None:
        """Release a previously reserved block."""
        raise NotImplementedError

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        """Read data from the device."""
        raise NotImplementedError

    async def store(self, offset: int, data: bytes) -> None:
        """Write data to the device."""
        raise NotImplementedError


class LocalMemoryDevice(PGASDevice):
    """Tier 0/1 local memory device using a simple Python dictionary backing."""
    def __init__(self, device_id: str, tier_id: int, capacity_bytes: int):
        super().__init__(device_id, tier_id, capacity_bytes)
        self.memory_store: Dict[int, bytes] = {}
        self.next_offset = 0

    async def allocate(self, size_bytes: int) -> int:
        if self.allocated_bytes + size_bytes > self.capacity_bytes:
            raise MemoryError(f"Device {self.device_id} out of memory.")
            
        offset = self.next_offset
        self.next_offset += size_bytes
        self.allocated_bytes += size_bytes
        
        # Initialize with zeros
        self.memory_store[offset] = b'\x00' * size_bytes
        return offset
        
    async def deallocate(self, offset: int, size_bytes: int) -> None:
        if offset in self.memory_store:
            del self.memory_store[offset]
            self.allocated_bytes -= size_bytes

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        if offset not in self.memory_store:
            raise ValueError(f"Invalid memory offset: {offset}")
        
        data = self.memory_store[offset]
        if len(data) < size_bytes:
            raise ValueError(f"Requested {size_bytes}b but only {len(data)}b available at offset {offset}")
            
        return data[:size_bytes]

    async def store(self, offset: int, data: bytes) -> None:
        if offset not in self.memory_store:
            raise ValueError(f"Invalid memory offset: {offset}")
        
        current_region = self.memory_store[offset]
        if len(data) > len(current_region):
            raise ValueError(f"Data size {len(data)} exceeds region size {len(current_region)}")
            
        # If we overwrite partially, splice (simplification for prototype)
        if len(data) == len(current_region):
            self.memory_store[offset] = data
        else:
            self.memory_store[offset] = bytearray(data) + bytearray(current_region[len(data):])


@dataclass
class AllocationRecord:
    ptr: UfoGlobalPointer
    size_bytes: int
    metadata: Dict[str, Any]


class PGASManager:
    """Core resource manager for the unified memory fabric."""
    def __init__(self, local_node_id: int):
        self.local_node_id = local_node_id
        
        # Mapping tier_id -> Device instance
        self.local_devices: Dict[int, PGASDevice] = {}
        
        # Remote node directory
        self.remote_nodes: Dict[int, Any] = {} # Mock routing table
        
        # Tracking local allocations
        self.allocations: Dict[UfoGlobalPointer, AllocationRecord] = {}

    def register_device(self, device: PGASDevice) -> None:
        """Mount a local storage tier/device."""
        if device.tier_id in self.local_devices:
            raise ValueError(f"Tier {device.tier_id} already registered.")
        self.local_devices[device.tier_id] = device
        logger.info(f"Registered device {device.device_id} to tier {device.tier_id}")

    async def allocate(self, size_bytes: int, tier_preference: int = 0) -> UfoGlobalPointer:
        """Allocate a chunk on the specified tier of the local node."""
        if tier_preference not in self.local_devices:
            raise RuntimeError(f"Requested tier {tier_preference} not available locally.")
            
        device = self.local_devices[tier_preference]
        offset = await device.allocate(size_bytes)
        
        ptr = UfoGlobalPointer(self.local_node_id, tier_preference, offset)
        self.allocations[ptr] = AllocationRecord(ptr, size_bytes, {})
        
        logger.debug(f"Allocated {size_bytes}b at {ptr}")
        return ptr

    async def deallocate(self, ptr: UfoGlobalPointer) -> None:
        """Release memory on the local node."""
        if ptr.node_id != self.local_node_id:
            raise ValueError(f"Cannot deallocate remote pointer: {ptr}")
            
        if ptr not in self.allocations:
            raise ValueError(f"Pointer {ptr} not found in local allocation table.")
            
        record = self.allocations.pop(ptr)
        device = self.local_devices[ptr.tier_id]
        
        await device.deallocate(ptr.offset, record.size_bytes)
        logger.debug(f"Deallocated {ptr}")

    async def fetch(self, ptr: UfoGlobalPointer, size_bytes: int) -> bytes:
        """Read data from the PGAS. Route automatically if remote."""
        if ptr.node_id == self.local_node_id:
            if ptr.tier_id not in self.local_devices:
                raise ValueError(f"Device tier {ptr.tier_id} not mounted.")
            return await self.local_devices[ptr.tier_id].fetch(ptr.offset, size_bytes)
        else:
            # Route to remote node (to be orchestrated by EmulationTCP/Network layer)
            return await self._route_fetch_remote(ptr, size_bytes)

    async def store(self, ptr: UfoGlobalPointer, data: bytes) -> None:
        """Write data to the PGAS. Route automatically if remote."""
        if ptr.node_id == self.local_node_id:
            if ptr.tier_id not in self.local_devices:
                raise ValueError(f"Device tier {ptr.tier_id} not mounted.")
            await self.local_devices[ptr.tier_id].store(ptr.offset, data)
        else:
            # Route to remote node
            await self._route_store_remote(ptr, data)

    async def _route_fetch_remote(self, ptr: UfoGlobalPointer, size_bytes: int) -> bytes:
        """Stub for fetching from another node over network."""
        raise NotImplementedError("Network routing not yet linked.")

    async def _route_store_remote(self, ptr: UfoGlobalPointer, data: bytes) -> None:
        """Stub for pushing to another node over network."""
        raise NotImplementedError("Network routing not yet linked.")
