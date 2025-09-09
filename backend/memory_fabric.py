"""
Omega Super Desktop Console v2.0 - Unified Memory Fabric
Enterprise-grade distributed memory management with RDMA, compression, and intelligent caching
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import zlib
import threading
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import OrderedDict, defaultdict
import mmap
import pickle
import struct

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    SYSTEM = "system"           # System RAM
    GPU = "gpu"                 # GPU memory
    PERSISTENT = "persistent"   # Persistent storage
    CACHE = "cache"             # Cache memory
    SHARED = "shared"           # Shared memory segments

class CompressionType(Enum):
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    ZSTD = "zstd"

class CachePolicy(Enum):
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    FIFO = "fifo"               # First In, First Out
    ADAPTIVE = "adaptive"       # Adaptive Replacement Cache
    TEMPORAL = "temporal"       # Time-based expiration

class MemoryState(Enum):
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    LOCKED = "locked"
    MIGRATING = "migrating"
    CORRUPTED = "corrupted"
    RECOVERING = "recovering"

class RDMAOperation(Enum):
    READ = "read"
    WRITE = "write"
    ATOMIC_ADD = "atomic_add"
    ATOMIC_CAS = "atomic_cas"
    MULTICAST = "multicast"

@dataclass
class MemoryRegion:
    """Represents a memory region in the fabric"""
    region_id: str
    node_id: str
    memory_type: MemoryType
    base_address: int
    size: int
    state: MemoryState
    permissions: Set[str] = field(default_factory=set)
    compression: CompressionType = CompressionType.NONE
    checksum: Optional[str] = None
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    locked_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class MemoryBlock:
    """Individual memory block within a region"""
    block_id: str
    region_id: str
    offset: int
    size: int
    data: Optional[bytes] = None
    compressed_size: Optional[int] = None
    ref_count: int = 0
    dirty: bool = False
    pinned: bool = False
    last_modified: float = field(default_factory=time.time)

@dataclass
class RDMARequest:
    """RDMA operation request"""
    request_id: str
    operation: RDMAOperation
    source_node: str
    target_node: str
    source_region: str
    target_region: str
    offset: int
    size: int
    data: Optional[bytes] = None
    priority: int = 0
    timeout: float = 30.0
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class CacheEntry:
    """Cache entry in the memory fabric"""
    key: str
    value: bytes
    size: int
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    priority: int = 0

class MemoryAllocator:
    """Advanced memory allocator with fragmentation management"""
    
    def __init__(self, total_size: int, block_size: int = 4096):
        self.total_size = total_size
        self.block_size = block_size
        self.free_blocks: Set[int] = set(range(0, total_size, block_size))
        self.allocated_blocks: Dict[str, Tuple[int, int]] = {}  # region_id -> (start, size)
        self.fragmentation_threshold = 0.3
        
    async def allocate(self, region_id: str, size: int) -> Optional[int]:
        """Allocate memory block"""
        try:
            blocks_needed = (size + self.block_size - 1) // self.block_size
            
            # Find contiguous free blocks
            start_block = await self._find_contiguous_blocks(blocks_needed)
            if start_block is None:
                # Try defragmentation
                await self._defragment()
                start_block = await self._find_contiguous_blocks(blocks_needed)
                
            if start_block is not None:
                # Allocate blocks
                for i in range(blocks_needed):
                    self.free_blocks.discard(start_block + i)
                    
                self.allocated_blocks[region_id] = (start_block * self.block_size, size)
                return start_block * self.block_size
                
            return None
            
        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            return None
            
    async def deallocate(self, region_id: str) -> bool:
        """Deallocate memory block"""
        try:
            if region_id not in self.allocated_blocks:
                return False
                
            start_address, size = self.allocated_blocks[region_id]
            start_block = start_address // self.block_size
            blocks_count = (size + self.block_size - 1) // self.block_size
            
            # Free blocks
            for i in range(blocks_count):
                self.free_blocks.add(start_block + i)
                
            del self.allocated_blocks[region_id]
            return True
            
        except Exception as e:
            logger.error(f"Memory deallocation failed: {e}")
            return False
            
    async def _find_contiguous_blocks(self, count: int) -> Optional[int]:
        """Find contiguous free blocks"""
        if not self.free_blocks:
            return None
            
        sorted_blocks = sorted(self.free_blocks)
        for i in range(len(sorted_blocks) - count + 1):
            if all(sorted_blocks[i] + j in self.free_blocks for j in range(count)):
                return sorted_blocks[i]
        return None
        
    async def _defragment(self):
        """Defragment memory by moving allocated blocks"""
        # Simplified defragmentation - in production this would be more sophisticated
        logger.info("Performing memory defragmentation")
        
    async def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio"""
        if not self.free_blocks:
            return 0.0
            
        total_free = len(self.free_blocks)
        largest_contiguous = 0
        current_contiguous = 0
        
        sorted_blocks = sorted(self.free_blocks)
        for i, block in enumerate(sorted_blocks):
            if i == 0 or block == sorted_blocks[i-1] + 1:
                current_contiguous += 1
            else:
                largest_contiguous = max(largest_contiguous, current_contiguous)
                current_contiguous = 1
                
        largest_contiguous = max(largest_contiguous, current_contiguous)
        return 1.0 - (largest_contiguous / total_free) if total_free > 0 else 0.0

class CompressionEngine:
    """Advanced compression engine for memory optimization"""
    
    def __init__(self):
        self.compression_stats: Dict[CompressionType, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    async def compress(self, data: bytes, compression_type: CompressionType) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data using specified algorithm"""
        try:
            start_time = time.time()
            original_size = len(data)
            
            if compression_type == CompressionType.NONE:
                compressed_data = data
            elif compression_type == CompressionType.ZLIB:
                compressed_data = zlib.compress(data, level=6)
            else:
                # For other compression types, fall back to zlib
                # In production, implement LZ4, Snappy, ZSTD
                compressed_data = zlib.compress(data, level=6)
                
            compression_time = time.time() - start_time
            compressed_size = len(compressed_data)
            ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Update statistics
            self.compression_stats[compression_type]['operations'] += 1
            self.compression_stats[compression_type]['bytes_compressed'] += original_size
            self.compression_stats[compression_type]['bytes_saved'] += (original_size - compressed_size)
            
            metadata = {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': ratio,
                'compression_time': compression_time,
                'algorithm': compression_type.value
            }
            
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data, {'error': str(e)}
            
    async def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data"""
        try:
            if compression_type == CompressionType.NONE:
                return data
            elif compression_type == CompressionType.ZLIB:
                return zlib.decompress(data)
            else:
                # For other compression types, fall back to zlib
                return zlib.decompress(data)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data

class CacheManager:
    """Intelligent cache manager with multiple policies"""
    
    def __init__(self, max_size: int, policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.current_size = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    async def get(self, key: str) -> Optional[bytes]:
        """Get value from cache"""
        try:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl and time.time() > entry.created_at + entry.ttl:
                    await self._evict(key)
                    self.miss_count += 1
                    return None
                    
                # Update access statistics
                entry.hits += 1
                entry.last_accessed = time.time()
                self.access_times[key] = time.time()
                
                # Move to end for LRU
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                    
                self.hit_count += 1
                return entry.value
            else:
                self.miss_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
            
    async def put(self, key: str, value: bytes, ttl: Optional[float] = None, priority: int = 0) -> bool:
        """Put value in cache"""
        try:
            entry_size = len(value)
            
            # Check if we need to evict
            while self.current_size + entry_size > self.max_size and self.cache:
                await self._evict_lru()
                
            if self.current_size + entry_size <= self.max_size:
                # Remove existing entry if present
                if key in self.cache:
                    await self._evict(key)
                    
                # Add new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=entry_size,
                    ttl=ttl,
                    priority=priority
                )
                
                self.cache[key] = entry
                self.current_size += entry_size
                self.access_times[key] = time.time()
                
                return True
            else:
                logger.warning(f"Cache entry too large: {entry_size} > {self.max_size}")
                return False
                
        except Exception as e:
            logger.error(f"Cache put failed: {e}")
            return False
            
    async def _evict(self, key: str):
        """Evict specific key from cache"""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.current_size -= entry.size
            self.access_times.pop(key, None)
            
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            if self.policy == CachePolicy.LRU:
                key = next(iter(self.cache))
            elif self.policy == CachePolicy.LFU:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].hits)
            else:
                key = next(iter(self.cache))  # FIFO
                
            await self._evict(key)
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'policy': self.policy.value,
            'max_size': self.max_size,
            'current_size': self.current_size,
            'entry_count': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_ratio': hit_ratio,
            'utilization': self.current_size / self.max_size
        }

class UnifiedMemoryFabric:
    """Enterprise-grade unified memory fabric with distributed management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.node_id = self.config.get('node_id', str(uuid.uuid4()))
        
        # Memory management
        self.memory_regions: Dict[str, MemoryRegion] = {}
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.allocator = MemoryAllocator(
            total_size=self.config.get('memory_size', 1024 * 1024 * 1024),  # 1GB default
            block_size=self.config.get('block_size', 4096)
        )
        
        # Compression and caching
        self.compression_engine = CompressionEngine()
        self.cache_manager = CacheManager(
            max_size=self.config.get('cache_size', 256 * 1024 * 1024),  # 256MB default
            policy=CachePolicy(self.config.get('cache_policy', 'lru'))
        )
        
        # RDMA simulation (in production, use actual RDMA libraries)
        self.rdma_connections: Dict[str, Dict[str, Any]] = {}
        self.pending_operations: Dict[str, RDMARequest] = {}
        
        # Monitoring and metrics
        self.metrics = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'bytes_allocated': 0,
            'bytes_deallocated': 0,
            'rdma_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_operations': 0,
            'memory_errors': 0
        }
        
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize the memory fabric"""
        try:
            logger.info("Initializing Unified Memory Fabric...")
            
            # Initialize RDMA connections
            await self._setup_rdma_connections()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._memory_monitor())
            asyncio.create_task(self._cache_cleaner())
            asyncio.create_task(self._rdma_processor())
            
            logger.info("Unified Memory Fabric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Memory fabric initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the memory fabric"""
        try:
            logger.info("Shutting down Unified Memory Fabric...")
            self.running = False
            
            # Clean up all regions
            for region_id in list(self.memory_regions.keys()):
                await self.deallocate_region(region_id)
                
            logger.info("Unified Memory Fabric shutdown complete")
            
        except Exception as e:
            logger.error(f"Memory fabric shutdown error: {e}")
            
    async def allocate_region(self, 
                            size: int,
                            memory_type: MemoryType = MemoryType.SYSTEM,
                            permissions: Set[str] = None,
                            compression: CompressionType = CompressionType.NONE) -> Optional[str]:
        """Allocate a memory region"""
        try:
            region_id = str(uuid.uuid4())
            
            # Allocate memory
            base_address = await self.allocator.allocate(region_id, size)
            if base_address is None:
                logger.error(f"Failed to allocate {size} bytes")
                return None
                
            # Create memory region
            region = MemoryRegion(
                region_id=region_id,
                node_id=self.node_id,
                memory_type=memory_type,
                base_address=base_address,
                size=size,
                state=MemoryState.ALLOCATED,
                permissions=permissions or set(),
                compression=compression
            )
            
            self.memory_regions[region_id] = region
            
            # Update metrics
            self.metrics['total_allocations'] += 1
            self.metrics['bytes_allocated'] += size
            
            logger.info(f"Allocated memory region {region_id}: {size} bytes at 0x{base_address:x}")
            return region_id
            
        except Exception as e:
            logger.error(f"Region allocation failed: {e}")
            self.metrics['memory_errors'] += 1
            return None
            
    async def deallocate_region(self, region_id: str) -> bool:
        """Deallocate a memory region"""
        try:
            if region_id not in self.memory_regions:
                return False
                
            region = self.memory_regions[region_id]
            
            # Check if region is locked
            if region.state == MemoryState.LOCKED:
                logger.warning(f"Cannot deallocate locked region {region_id}")
                return False
                
            # Deallocate memory
            success = await self.allocator.deallocate(region_id)
            if success:
                # Remove all blocks in this region
                blocks_to_remove = [bid for bid, block in self.memory_blocks.items() 
                                  if block.region_id == region_id]
                for block_id in blocks_to_remove:
                    del self.memory_blocks[block_id]
                    
                del self.memory_regions[region_id]
                
                # Update metrics
                self.metrics['total_deallocations'] += 1
                self.metrics['bytes_deallocated'] += region.size
                
                logger.info(f"Deallocated memory region {region_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Region deallocation failed: {e}")
            self.metrics['memory_errors'] += 1
            return False
            
    async def write_block(self, region_id: str, offset: int, data: bytes) -> Optional[str]:
        """Write data to a memory block"""
        try:
            if region_id not in self.memory_regions:
                raise ValueError(f"Region {region_id} not found")
                
            region = self.memory_regions[region_id]
            if offset + len(data) > region.size:
                raise ValueError("Write exceeds region boundary")
                
            # Compress data if enabled
            compressed_data = data
            metadata = {}
            if region.compression != CompressionType.NONE:
                compressed_data, metadata = await self.compression_engine.compress(data, region.compression)
                self.metrics['compression_operations'] += 1
                
            # Create memory block
            block_id = str(uuid.uuid4())
            block = MemoryBlock(
                block_id=block_id,
                region_id=region_id,
                offset=offset,
                size=len(data),
                data=compressed_data,
                compressed_size=len(compressed_data),
                dirty=True
            )
            
            self.memory_blocks[block_id] = block
            
            # Update region checksum
            region.checksum = hashlib.sha256(compressed_data).hexdigest()
            region.last_accessed = time.time()
            
            # Cache the block
            cache_key = f"{region_id}:{offset}"
            await self.cache_manager.put(cache_key, compressed_data)
            
            logger.debug(f"Wrote {len(data)} bytes to region {region_id} at offset {offset}")
            return block_id
            
        except Exception as e:
            logger.error(f"Block write failed: {e}")
            self.metrics['memory_errors'] += 1
            return None
            
    async def read_block(self, region_id: str, offset: int, size: int) -> Optional[bytes]:
        """Read data from a memory block"""
        try:
            if region_id not in self.memory_regions:
                raise ValueError(f"Region {region_id} not found")
                
            region = self.memory_regions[region_id]
            
            # Check cache first
            cache_key = f"{region_id}:{offset}"
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                self.metrics['cache_hits'] += 1
                # Decompress if needed
                if region.compression != CompressionType.NONE:
                    return await self.compression_engine.decompress(cached_data, region.compression)
                return cached_data[:size]
            else:
                self.metrics['cache_misses'] += 1
                
            # Find matching block
            for block in self.memory_blocks.values():
                if (block.region_id == region_id and 
                    block.offset <= offset < block.offset + block.size):
                    
                    # Update access statistics
                    region.last_accessed = time.time()
                    region.access_count += 1
                    
                    # Decompress data if needed
                    data = block.data or b''
                    if region.compression != CompressionType.NONE:
                        data = await self.compression_engine.decompress(data, region.compression)
                        
                    # Cache the data
                    await self.cache_manager.put(cache_key, block.data or b'')
                    
                    return data[:size]
                    
            return None
            
        except Exception as e:
            logger.error(f"Block read failed: {e}")
            self.metrics['memory_errors'] += 1
            return None
            
    async def rdma_transfer(self, request: RDMARequest) -> bool:
        """Perform RDMA transfer operation"""
        try:
            # In production, this would use actual RDMA libraries (libibverbs, etc.)
            # For demo, we'll simulate the operation
            
            self.pending_operations[request.request_id] = request
            
            if request.operation == RDMAOperation.READ:
                # Read from remote region
                data = await self.read_block(request.source_region, request.offset, request.size)
                if data:
                    # Store in target region
                    await self.write_block(request.target_region, request.offset, data)
                    
            elif request.operation == RDMAOperation.WRITE:
                # Write to remote region
                if request.data:
                    await self.write_block(request.target_region, request.offset, request.data)
                    
            # Complete operation
            del self.pending_operations[request.request_id]
            self.metrics['rdma_operations'] += 1
            
            logger.info(f"RDMA {request.operation.value} operation completed: {request.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"RDMA transfer failed: {e}")
            return False
            
    async def migrate_region(self, region_id: str, target_node: str) -> bool:
        """Migrate memory region to another node"""
        try:
            if region_id not in self.memory_regions:
                return False
                
            region = self.memory_regions[region_id]
            region.state = MemoryState.MIGRATING
            
            # Read all data from region
            all_data = b''
            for block in self.memory_blocks.values():
                if block.region_id == region_id:
                    block_data = await self.read_block(region_id, block.offset, block.size)
                    if block_data:
                        all_data += block_data
                        
            # Create RDMA transfer request
            request = RDMARequest(
                request_id=str(uuid.uuid4()),
                operation=RDMAOperation.WRITE,
                source_node=self.node_id,
                target_node=target_node,
                source_region=region_id,
                target_region=region_id,
                offset=0,
                size=len(all_data),
                data=all_data
            )
            
            # Perform transfer
            success = await self.rdma_transfer(request)
            
            if success:
                # Update region node
                region.node_id = target_node
                region.state = MemoryState.ALLOCATED
                logger.info(f"Region {region_id} migrated to node {target_node}")
            else:
                region.state = MemoryState.AVAILABLE
                
            return success
            
        except Exception as e:
            logger.error(f"Region migration failed: {e}")
            return False
            
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory fabric statistics"""
        try:
            # Calculate memory utilization
            total_allocated = sum(region.size for region in self.memory_regions.values())
            fragmentation = await self.allocator.get_fragmentation_ratio()
            
            # Get cache stats
            cache_stats = await self.cache_manager.get_stats()
            
            # Calculate RDMA performance
            rdma_pending = len(self.pending_operations)
            
            return {
                'fabric_status': {
                    'node_id': self.node_id,
                    'running': self.running,
                    'total_regions': len(self.memory_regions),
                    'total_blocks': len(self.memory_blocks)
                },
                'memory_usage': {
                    'total_size': self.allocator.total_size,
                    'allocated_bytes': total_allocated,
                    'free_bytes': self.allocator.total_size - total_allocated,
                    'utilization_percent': (total_allocated / self.allocator.total_size) * 100,
                    'fragmentation_ratio': fragmentation
                },
                'cache_performance': cache_stats,
                'rdma_status': {
                    'total_operations': self.metrics['rdma_operations'],
                    'pending_operations': rdma_pending,
                    'connections': len(self.rdma_connections)
                },
                'compression_stats': dict(self.compression_engine.compression_stats),
                'performance_metrics': {
                    'total_allocations': self.metrics['total_allocations'],
                    'total_deallocations': self.metrics['total_deallocations'],
                    'cache_hit_ratio': cache_stats['hit_ratio'],
                    'compression_operations': self.metrics['compression_operations'],
                    'memory_errors': self.metrics['memory_errors']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
            
    async def _setup_rdma_connections(self):
        """Setup RDMA connections to other nodes"""
        # In production, this would establish actual RDMA connections
        logger.info("Setting up RDMA connections")
        
    async def _memory_monitor(self):
        """Monitor memory usage and health"""
        while self.running:
            try:
                # Check for memory leaks
                fragmentation = await self.allocator.get_fragmentation_ratio()
                if fragmentation > self.allocator.fragmentation_threshold:
                    logger.warning(f"High memory fragmentation: {fragmentation:.2%}")
                    
                # Check for stale regions
                current_time = time.time()
                stale_regions = [
                    rid for rid, region in self.memory_regions.items()
                    if current_time - region.last_accessed > 3600  # 1 hour
                ]
                
                if stale_regions:
                    logger.info(f"Found {len(stale_regions)} stale memory regions")
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _cache_cleaner(self):
        """Clean expired cache entries"""
        while self.running:
            try:
                # Clean expired entries would be implemented in CacheManager
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Cache cleaner error: {e}")
                await asyncio.sleep(300)
                
    async def _rdma_processor(self):
        """Process pending RDMA operations"""
        while self.running:
            try:
                # Process RDMA queue
                current_time = time.time()
                timed_out = [
                    req_id for req_id, req in self.pending_operations.items()
                    if current_time - req.timestamp > req.timeout
                ]
                
                for req_id in timed_out:
                    logger.warning(f"RDMA operation {req_id} timed out")
                    del self.pending_operations[req_id]
                    
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"RDMA processor error: {e}")
                await asyncio.sleep(1)

# Global instance
_memory_fabric: Optional[UnifiedMemoryFabric] = None

async def initialize_memory_fabric(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global unified memory fabric"""
    global _memory_fabric
    try:
        _memory_fabric = UnifiedMemoryFabric(config)
        return await _memory_fabric.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize memory fabric: {e}")
        return False

def get_memory_fabric() -> UnifiedMemoryFabric:
    """Get the global unified memory fabric instance"""
    global _memory_fabric
    if _memory_fabric is None:
        raise RuntimeError("Memory fabric not initialized. Call initialize_memory_fabric() first.")
    return _memory_fabric

async def shutdown_memory_fabric():
    """Shutdown the global unified memory fabric"""
    global _memory_fabric
    if _memory_fabric:
        await _memory_fabric.shutdown()
        _memory_fabric = None
