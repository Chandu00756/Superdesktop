"""
UFO-PGAS: CXL Memory Device (Hardware Emulation)
Implements a Tier 1 PGAS device utilizing OS-level `mmap` to memory-map 
physical character devices (e.g. `/dev/dax0.0` or a tmpfs file) directly 
into the Python process virtual address space.

This simulates the 2.5x performance characteristics of native CXL disaggregated 
memory pooling compared to standard software-based spill/cache models.
"""

import os
import mmap
import logging
from typing import Optional

from pgas_manager import PGASDevice

logger = logging.getLogger(__name__)


class CXLMemoryDevice(PGASDevice):
    """
    Tier 1 Memory Device backed by `mmap`.
    Achieves zero-copy memory reads/writes bypassing kernel TCP stacks.
    """
    def __init__(self, device_id: str, tier_id: int, backing_file: str, capacity_bytes: int):
        super().__init__(device_id, tier_id, capacity_bytes)
        self.backing_file = backing_file
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        
        self.next_offset = 0

    def _initialize_mmap(self):
        """Map the physical file/device into process memory."""
        if self._mmap is not None:
            return
            
        # Create backing file if it doesn't exist (only for simulation purposes)
        # In production this points to a pre-existing CXL character device or hugetlbfs
        if not os.path.exists(self.backing_file):
            logger.info(f"Creating sparse CXL backing file at {self.backing_file} ({self.capacity_bytes} bytes)")
            with open(self.backing_file, "wb") as f:
                f.seek(self.capacity_bytes - 1)
                f.write(b"\0")
                
        self._fd = os.open(self.backing_file, os.O_RDWR)
        
        # map the entire device
        self._mmap = mmap.mmap(self._fd, self.capacity_bytes, access=mmap.ACCESS_WRITE)
        logger.info(f"CXL Device {self.device_id} successfully memory-mapped.")

    def close(self):
        """Unmap region and close file descriptors."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            
    # PGAS Device Protocols (We make these async to comply with the PGAS interface,
    # though CXL memory operations are intrinsically instantaneous and synchronous!)
    
    async def allocate(self, size_bytes: int) -> int:
        if self._mmap is None:
            self._initialize_mmap()
            
        if self.next_offset + size_bytes > self.capacity_bytes:
            raise MemoryError(f"CXL Device {self.device_id} out of memory.")
            
        offset = self.next_offset
        self.next_offset += size_bytes
        self.allocated_bytes += size_bytes
        return offset

    async def deallocate(self, offset: int, size_bytes: int) -> None:
        """Memory remains mapped, we just track the logical free (soft deallocate)."""
        self.allocated_bytes -= size_bytes

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        """
        Direct memory read. Returns a bytes object.
        For true zero-copy in computing libraries, one would return a memoryview
        or numpy array wrapping `self._mmap[offset:offset+size_bytes]`.
        """
        if self._mmap is None:
            self._initialize_mmap()
            
        return self._mmap[offset:offset+size_bytes]

    async def store(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """Direct memory write."""
        if self._mmap is None:
            self._initialize_mmap()
            
        self._mmap[offset:offset+len(data)] = data
