"""
Omega Super Desktop Console - Memory Fabric Service
Advanced Memory Management and Fabric Orchestration Service

This service manages distributed memory resources across the cluster using
advanced memory fabric technologies like CXL 3.0, NVLink, and Infinity Fabric.
It provides intelligent memory allocation, NUMA optimization, and real-time
memory performance monitoring.

Key Features:
- CXL 3.0 memory fabric management
- NUMA-aware memory allocation
- Real-time memory bandwidth optimization
- Memory compression and deduplication
- Cross-node memory sharing
- Memory security and encryption
"""

import asyncio
import json
import logging
import time
import mmap
import struct
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import os
import psutil
import threading
import numpy as np

# Core Dependencies
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import aioredis
import asyncpg

# System monitoring
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from utils.metrics import create_counter, create_gauge, create_histogram

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus Metrics
memory_allocations_total = create_counter('memory_allocations_total', 'Total memory allocations', ['fabric_type', 'allocation_type'])
memory_bandwidth_utilization = create_gauge('memory_bandwidth_utilization_gbps', 'Memory bandwidth utilization', ['fabric_id', 'direction'])
memory_fabric_latency = create_histogram('memory_fabric_latency_ns', 'Memory fabric access latency in nanoseconds')
memory_compression_ratio = create_gauge('memory_compression_ratio', 'Memory compression ratio', ['fabric_id'])
memory_deduplication_ratio = create_gauge('memory_deduplication_ratio', 'Memory deduplication ratio', ['fabric_id'])
numa_migrations_total = create_counter('numa_migrations_total', 'Total NUMA migrations', ['source_node', 'target_node'])
memory_encryption_overhead = create_gauge('memory_encryption_overhead_percent', 'Memory encryption overhead percentage')

# Memory Fabric Types and Technologies
class FabricType(Enum):
    CXL_3_0 = "cxl_3.0"
    CXL_2_0 = "cxl_2.0"
    NVLINK = "nvlink"
    INFINITY_FABRIC = "infinity_fabric"
    DDR5 = "ddr5"
    HBM3 = "hbm3"
    OPTANE = "optane"

class MemoryType(Enum):
    SYSTEM_RAM = "system_ram"
    GPU_MEMORY = "gpu_memory"
    STORAGE_CLASS_MEMORY = "storage_class_memory"
    PERSISTENT_MEMORY = "persistent_memory"
    CACHE_MEMORY = "cache_memory"

class AllocationStrategy(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    NUMA_LOCAL = "numa_local"
    BANDWIDTH_OPTIMIZED = "bandwidth_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"

class CompressionAlgorithm(Enum):
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    BROTLI = "brotli"

@dataclass
class MemoryFabricNode:
    """Memory fabric node configuration"""
    node_id: str
    fabric_type: FabricType
    total_capacity_gb: int
    available_capacity_gb: int
    bandwidth_read_gbps: float
    bandwidth_write_gbps: float
    latency_ns: int
    numa_domain: int
    
    # Advanced Features
    supports_compression: bool
    supports_deduplication: bool
    supports_encryption: bool
    supports_cache_coherency: bool
    
    # Physical Characteristics
    channel_count: int
    rank_count: int
    frequency_mhz: int
    bus_width: int
    
    # Current State
    current_utilization_percent: float
    temperature_celsius: int
    error_count: int
    health_status: str

@dataclass
class MemoryAllocation:
    """Memory allocation record"""
    allocation_id: str
    session_id: str
    fabric_node_id: str
    size_bytes: int
    numa_node: int
    allocation_strategy: AllocationStrategy
    
    # Memory Characteristics
    access_pattern: str  # sequential, random, mixed
    priority_level: int
    compression_enabled: bool
    encryption_enabled: bool
    
    # Performance Tracking
    allocated_at: datetime
    last_accessed: datetime
    access_count: int
    read_bandwidth_mbps: float
    write_bandwidth_mbps: float
    average_latency_ns: float
    
    # Optimization
    migration_candidate: bool
    optimization_score: float

@dataclass
class NumaTopology:
    """NUMA topology information"""
    total_nodes: int
    nodes: Dict[int, Dict[str, Any]]
    distances: Dict[Tuple[int, int], int]
    cpu_to_node_map: Dict[int, int]
    memory_to_node_map: Dict[str, int]

class MemoryFabricController:
    """Controls memory fabric hardware and operations"""
    
    def __init__(self):
        self.fabric_nodes: Dict[str, MemoryFabricNode] = {}
        self.numa_topology: Optional[NumaTopology] = None
        self.fabric_interconnects: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize memory fabric discovery and configuration"""
        try:
            # Discover NUMA topology
            await self._discover_numa_topology()
            
            # Discover memory fabric nodes
            await self._discover_fabric_nodes()
            
            # Initialize fabric interconnects
            await self._initialize_interconnects()
            
            logger.info(
                "Memory fabric controller initialized",
                fabric_nodes=len(self.fabric_nodes),
                numa_nodes=self.numa_topology.total_nodes if self.numa_topology else 0
            )
            
        except Exception as e:
            logger.error("Failed to initialize memory fabric controller", error=str(e))
            raise
    
    async def _discover_numa_topology(self):
        """Discover system NUMA topology"""
        try:
            # Use psutil to get basic CPU info
            cpu_count = psutil.cpu_count(logical=False)
            
            # Mock NUMA topology for demonstration
            numa_nodes = {}
            distances = {}
            cpu_to_node_map = {}
            memory_to_node_map = {}
            
            # Create mock NUMA topology with 2 nodes
            for node_id in range(2):
                numa_nodes[node_id] = {
                    'memory_gb': 64,
                    'cpu_cores': list(range(node_id * 8, (node_id + 1) * 8)),
                    'memory_controllers': 2,
                    'pci_domains': [node_id]
                }
                
                # Map CPUs to NUMA nodes
                for cpu in numa_nodes[node_id]['cpu_cores']:
                    cpu_to_node_map[cpu] = node_id
                
                # Mock memory regions
                memory_to_node_map[f"ddr5_bank_{node_id}"] = node_id
            
            # Mock NUMA distances (lower is better)
            for i in range(2):
                for j in range(2):
                    if i == j:
                        distances[(i, j)] = 10  # Local access
                    else:
                        distances[(i, j)] = 20  # Remote access
            
            self.numa_topology = NumaTopology(
                total_nodes=2,
                nodes=numa_nodes,
                distances=distances,
                cpu_to_node_map=cpu_to_node_map,
                memory_to_node_map=memory_to_node_map
            )
            
            logger.info("NUMA topology discovered", numa_nodes=2)
            
        except Exception as e:
            logger.error("Failed to discover NUMA topology", error=str(e))
            # Create minimal topology
            self.numa_topology = NumaTopology(
                total_nodes=1,
                nodes={0: {'memory_gb': 64, 'cpu_cores': list(range(8))}},
                distances={(0, 0): 10},
                cpu_to_node_map={i: 0 for i in range(8)},
                memory_to_node_map={'default': 0}
            )
    
    async def _discover_fabric_nodes(self):
        """Discover available memory fabric nodes"""
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total // (1024**3)
            available_memory_gb = memory_info.available // (1024**3)
            
            # Create fabric nodes based on NUMA topology
            if self.numa_topology:
                for numa_node_id, numa_info in self.numa_topology.nodes.items():
                    node_memory_gb = numa_info.get('memory_gb', total_memory_gb // len(self.numa_topology.nodes))
                    
                    # DDR5 fabric node
                    ddr5_node = MemoryFabricNode(
                        node_id=f"ddr5_node_{numa_node_id}",
                        fabric_type=FabricType.DDR5,
                        total_capacity_gb=node_memory_gb,
                        available_capacity_gb=int(node_memory_gb * 0.8),
                        bandwidth_read_gbps=76.8,  # DDR5-4800
                        bandwidth_write_gbps=76.8,
                        latency_ns=80,
                        numa_domain=numa_node_id,
                        supports_compression=True,
                        supports_deduplication=True,
                        supports_encryption=True,
                        supports_cache_coherency=True,
                        channel_count=2,
                        rank_count=2,
                        frequency_mhz=4800,
                        bus_width=64,
                        current_utilization_percent=np.random.uniform(20, 60),
                        temperature_celsius=np.random.randint(35, 55),
                        error_count=0,
                        health_status="healthy"
                    )
                    
                    self.fabric_nodes[ddr5_node.node_id] = ddr5_node
                    
                    # CXL 3.0 expansion memory (if available)
                    if numa_node_id == 0:  # Only on first NUMA node for demo
                        cxl_node = MemoryFabricNode(
                            node_id=f"cxl_node_{numa_node_id}",
                            fabric_type=FabricType.CXL_3_0,
                            total_capacity_gb=128,
                            available_capacity_gb=120,
                            bandwidth_read_gbps=256.0,  # CXL 3.0 theoretical
                            bandwidth_write_gbps=256.0,
                            latency_ns=150,
                            numa_domain=numa_node_id,
                            supports_compression=True,
                            supports_deduplication=True,
                            supports_encryption=True,
                            supports_cache_coherency=True,
                            channel_count=8,
                            rank_count=4,
                            frequency_mhz=6400,
                            bus_width=256,
                            current_utilization_percent=np.random.uniform(10, 40),
                            temperature_celsius=np.random.randint(40, 60),
                            error_count=0,
                            health_status="healthy"
                        )
                        
                        self.fabric_nodes[cxl_node.node_id] = cxl_node
            
            logger.info(f"Discovered {len(self.fabric_nodes)} memory fabric nodes")
            
        except Exception as e:
            logger.error("Failed to discover fabric nodes", error=str(e))
    
    async def _initialize_interconnects(self):
        """Initialize fabric interconnect mapping"""
        # Create interconnect bandwidth/latency matrix
        for node1_id in self.fabric_nodes:
            self.fabric_interconnects[node1_id] = {}
            for node2_id in self.fabric_nodes:
                if node1_id == node2_id:
                    # Local access
                    self.fabric_interconnects[node1_id][node2_id] = {
                        'bandwidth_gbps': self.fabric_nodes[node1_id].bandwidth_read_gbps,
                        'latency_ns': self.fabric_nodes[node1_id].latency_ns
                    }
                else:
                    # Cross-fabric access
                    node1 = self.fabric_nodes[node1_id]
                    node2 = self.fabric_nodes[node2_id]
                    
                    # Calculate cross-fabric bandwidth (usually lower)
                    cross_bandwidth = min(node1.bandwidth_read_gbps, node2.bandwidth_read_gbps) * 0.7
                    
                    # Calculate cross-fabric latency (usually higher)
                    numa_distance = 1
                    if self.numa_topology and node1.numa_domain != node2.numa_domain:
                        numa_distance = self.numa_topology.distances.get(
                            (node1.numa_domain, node2.numa_domain), 20
                        ) / 10
                    
                    cross_latency = max(node1.latency_ns, node2.latency_ns) * numa_distance
                    
                    self.fabric_interconnects[node1_id][node2_id] = {
                        'bandwidth_gbps': cross_bandwidth,
                        'latency_ns': cross_latency
                    }
    
    async def get_optimal_allocation_node(
        self, 
        size_bytes: int, 
        strategy: AllocationStrategy,
        numa_preference: Optional[int] = None
    ) -> Optional[str]:
        """Find optimal fabric node for memory allocation"""
        
        size_gb = size_bytes / (1024**3)
        candidates = []
        
        # Filter nodes with sufficient capacity
        for node_id, node in self.fabric_nodes.items():
            if node.available_capacity_gb >= size_gb:
                candidates.append(node_id)
        
        if not candidates:
            return None
        
        # Apply allocation strategy
        if strategy == AllocationStrategy.NUMA_LOCAL and numa_preference is not None:
            # Prefer nodes in the same NUMA domain
            numa_candidates = [
                node_id for node_id in candidates
                if self.fabric_nodes[node_id].numa_domain == numa_preference
            ]
            if numa_candidates:
                candidates = numa_candidates
        
        elif strategy == AllocationStrategy.BANDWIDTH_OPTIMIZED:
            # Sort by available bandwidth
            candidates.sort(
                key=lambda node_id: self.fabric_nodes[node_id].bandwidth_read_gbps,
                reverse=True
            )
        
        elif strategy == AllocationStrategy.LATENCY_OPTIMIZED:
            # Sort by lowest latency
            candidates.sort(
                key=lambda node_id: self.fabric_nodes[node_id].latency_ns
            )
        
        elif strategy == AllocationStrategy.BEST_FIT:
            # Find node with smallest sufficient capacity
            candidates.sort(
                key=lambda node_id: self.fabric_nodes[node_id].available_capacity_gb
            )
        
        # Return the best candidate
        return candidates[0] if candidates else None
    
    async def allocate_memory(
        self,
        allocation: MemoryAllocation
    ) -> bool:
        """Allocate memory on specified fabric node"""
        
        if allocation.fabric_node_id not in self.fabric_nodes:
            return False
        
        node = self.fabric_nodes[allocation.fabric_node_id]
        size_gb = allocation.size_bytes / (1024**3)
        
        if node.available_capacity_gb < size_gb:
            return False
        
        # Update node capacity
        node.available_capacity_gb -= size_gb
        node.current_utilization_percent = (
            (node.total_capacity_gb - node.available_capacity_gb) / 
            node.total_capacity_gb * 100
        )
        
        # Update metrics
        memory_allocations_total.labels(
            fabric_type=node.fabric_type.value,
            allocation_type='allocation'
        ).inc()
        
        return True
    
    async def deallocate_memory(self, allocation: MemoryAllocation) -> bool:
        """Deallocate memory from fabric node"""
        
        if allocation.fabric_node_id not in self.fabric_nodes:
            return False
        
        node = self.fabric_nodes[allocation.fabric_node_id]
        size_gb = allocation.size_bytes / (1024**3)
        
        # Update node capacity
        node.available_capacity_gb += size_gb
        node.current_utilization_percent = (
            (node.total_capacity_gb - node.available_capacity_gb) / 
            node.total_capacity_gb * 100
        )
        
        # Update metrics
        memory_allocations_total.labels(
            fabric_type=node.fabric_type.value,
            allocation_type='deallocation'
        ).inc()
        
        return True
    
    async def update_fabric_metrics(self):
        """Update fabric performance metrics"""
        for node_id, node in self.fabric_nodes.items():
            # Update bandwidth utilization (simulated)
            read_utilization = np.random.uniform(0.1, 0.8) * node.bandwidth_read_gbps
            write_utilization = np.random.uniform(0.1, 0.6) * node.bandwidth_write_gbps
            
            memory_bandwidth_utilization.labels(
                fabric_id=node_id,
                direction='read'
            ).set(read_utilization)
            
            memory_bandwidth_utilization.labels(
                fabric_id=node_id,
                direction='write'
            ).set(write_utilization)
            
            # Update compression and deduplication ratios
            if node.supports_compression:
                compression_ratio = np.random.uniform(1.2, 2.5)
                memory_compression_ratio.labels(fabric_id=node_id).set(compression_ratio)
            
            if node.supports_deduplication:
                dedup_ratio = np.random.uniform(1.1, 1.8)
                memory_deduplication_ratio.labels(fabric_id=node_id).set(dedup_ratio)

class MemoryFabricService:
    """Main memory fabric management service"""
    
    def __init__(self):
        self.fabric_controller = MemoryFabricController()
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_history: List[MemoryAllocation] = []
        
        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocket] = set()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.total_allocations = 0
        self.failed_allocations = 0
        self.migration_count = 0
    
    async def initialize(self):
        """Initialize the memory fabric service"""
        try:
            # Initialize fabric controller
            await self.fabric_controller.initialize()
            
            # Redis connection
            try:
                from utils.redis_helper import get_redis_client
                self.redis_client = await get_redis_client(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            except Exception as e:
                logger.warning(f"Redis helper failed, using in-memory stub: {e}")
                class _InMemoryRedisStub:
                    def __init__(self):
                        self._store = {}
                    async def hset(self, key, mapping=None, **kwargs):
                        self._store[key] = mapping or kwargs
                    async def delete(self, key):
                        self._store.pop(key, None)
                    async def hgetall(self, key):
                        return self._store.get(key, {})
                self.redis_client = _InMemoryRedisStub()
            
            # PostgreSQL connection
            self.postgres_pool = await asyncpg.create_pool(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                user=os.getenv('POSTGRES_USER', 'omega'),
                password=os.getenv('POSTGRES_PASSWORD', 'omega_secure_2025'),
                database=os.getenv('POSTGRES_DB', 'omega_sessions'),
                min_size=5,
                max_size=20
            )
            
            # Start background tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            logger.info("Memory fabric service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize memory fabric service", error=str(e))
            raise
    
    async def allocate_memory(
        self,
        session_id: str,
        size_bytes: int,
        allocation_strategy: AllocationStrategy = AllocationStrategy.NUMA_LOCAL,
        numa_preference: Optional[int] = None,
        access_pattern: str = "mixed",
        priority_level: int = 1,
        compression_enabled: bool = True,
        encryption_enabled: bool = True
    ) -> Optional[MemoryAllocation]:
        """Allocate memory with specified requirements"""
        
        try:
            start_time = time.time()
            
            # Find optimal fabric node
            optimal_node = await self.fabric_controller.get_optimal_allocation_node(
                size_bytes, allocation_strategy, numa_preference
            )
            
            if not optimal_node:
                self.failed_allocations += 1
                logger.warning(
                    "Memory allocation failed - no suitable node",
                    session_id=session_id,
                    size_gb=size_bytes / (1024**3)
                )
                return None
            
            # Create allocation record
            allocation = MemoryAllocation(
                allocation_id=f"alloc-{session_id}-{int(time.time())}",
                session_id=session_id,
                fabric_node_id=optimal_node,
                size_bytes=size_bytes,
                numa_node=self.fabric_controller.fabric_nodes[optimal_node].numa_domain,
                allocation_strategy=allocation_strategy,
                access_pattern=access_pattern,
                priority_level=priority_level,
                compression_enabled=compression_enabled,
                encryption_enabled=encryption_enabled,
                allocated_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                read_bandwidth_mbps=0.0,
                write_bandwidth_mbps=0.0,
                average_latency_ns=0.0,
                migration_candidate=False,
                optimization_score=1.0
            )
            
            # Perform allocation
            success = await self.fabric_controller.allocate_memory(allocation)
            
            if success:
                self.allocations[allocation.allocation_id] = allocation
                self.allocation_history.append(allocation)
                self.total_allocations += 1
                
                # Record latency
                allocation_latency = (time.time() - start_time) * 1_000_000_000  # nanoseconds
                memory_fabric_latency.observe(allocation_latency)
                
                # Persist allocation
                await self._persist_allocation(allocation)
                
                # Notify via WebSocket
                await self._broadcast_fabric_update("allocation_created", allocation.allocation_id)
                
                logger.info(
                    "Memory allocated successfully",
                    allocation_id=allocation.allocation_id,
                    session_id=session_id,
                    fabric_node=optimal_node,
                    size_gb=size_bytes / (1024**3),
                    numa_node=allocation.numa_node
                )
                
                return allocation
            else:
                self.failed_allocations += 1
                return None
            
        except Exception as e:
            self.failed_allocations += 1
            logger.error("Memory allocation failed", error=str(e))
            return None
    
    async def deallocate_memory(self, allocation_id: str) -> bool:
        """Deallocate memory"""
        try:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            # Perform deallocation
            success = await self.fabric_controller.deallocate_memory(allocation)
            
            if success:
                del self.allocations[allocation_id]
                
                # Update allocation record in history
                for hist_alloc in self.allocation_history:
                    if hist_alloc.allocation_id == allocation_id:
                        # Mark as deallocated (could add deallocated_at field)
                        break
                
                # Notify via WebSocket
                await self._broadcast_fabric_update("allocation_deleted", allocation_id)
                
                logger.info(
                    "Memory deallocated successfully",
                    allocation_id=allocation_id,
                    session_id=allocation.session_id
                )
                
                return True
            else:
                return False
            
        except Exception as e:
            logger.error("Memory deallocation failed", error=str(e))
            return False
    
    async def migrate_allocation(
        self,
        allocation_id: str,
        target_fabric_node: str
    ) -> bool:
        """Migrate memory allocation to different fabric node"""
        try:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            source_node = allocation.fabric_node_id
            
            # Check if target node can accommodate the allocation
            if target_fabric_node not in self.fabric_controller.fabric_nodes:
                return False
            
            target_node = self.fabric_controller.fabric_nodes[target_fabric_node]
            size_gb = allocation.size_bytes / (1024**3)
            
            if target_node.available_capacity_gb < size_gb:
                return False
            
            # Simulate migration process
            await asyncio.sleep(0.1)  # Migration overhead
            
            # Deallocate from source
            await self.fabric_controller.deallocate_memory(allocation)
            
            # Update allocation record
            allocation.fabric_node_id = target_fabric_node
            allocation.numa_node = target_node.numa_domain
            
            # Allocate on target
            success = await self.fabric_controller.allocate_memory(allocation)
            
            if success:
                self.migration_count += 1
                
                # Update metrics
                source_numa = self.fabric_controller.fabric_nodes[source_node].numa_domain
                target_numa = target_node.numa_domain
                
                numa_migrations_total.labels(
                    source_node=str(source_numa),
                    target_node=str(target_numa)
                ).inc()
                
                logger.info(
                    "Memory migration completed",
                    allocation_id=allocation_id,
                    source_node=source_node,
                    target_node=target_fabric_node
                )
                
                return True
            else:
                # Migration failed, restore original allocation
                allocation.fabric_node_id = source_node
                await self.fabric_controller.allocate_memory(allocation)
                return False
            
        except Exception as e:
            logger.error("Memory migration failed", error=str(e))
            return False
    
    async def get_fabric_status(self) -> Dict[str, Any]:
        """Get comprehensive fabric status"""
        try:
            fabric_nodes_status = {}
            
            for node_id, node in self.fabric_controller.fabric_nodes.items():
                fabric_nodes_status[node_id] = {
                    "fabric_type": node.fabric_type.value,
                    "total_capacity_gb": node.total_capacity_gb,
                    "available_capacity_gb": node.available_capacity_gb,
                    "utilization_percent": node.current_utilization_percent,
                    "bandwidth_read_gbps": node.bandwidth_read_gbps,
                    "bandwidth_write_gbps": node.bandwidth_write_gbps,
                    "latency_ns": node.latency_ns,
                    "numa_domain": node.numa_domain,
                    "health_status": node.health_status,
                    "temperature_celsius": node.temperature_celsius,
                    "error_count": node.error_count
                }
            
            # Calculate total statistics
            total_capacity = sum(node.total_capacity_gb for node in self.fabric_controller.fabric_nodes.values())
            total_available = sum(node.available_capacity_gb for node in self.fabric_controller.fabric_nodes.values())
            overall_utilization = ((total_capacity - total_available) / total_capacity * 100) if total_capacity > 0 else 0
            
            return {
                "fabric_nodes": fabric_nodes_status,
                "numa_topology": asdict(self.fabric_controller.numa_topology) if self.fabric_controller.numa_topology else None,
                "total_capacity_gb": total_capacity,
                "total_available_gb": total_available,
                "overall_utilization_percent": overall_utilization,
                "active_allocations": len(self.allocations),
                "total_allocations": self.total_allocations,
                "failed_allocations": self.failed_allocations,
                "migration_count": self.migration_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get fabric status", error=str(e))
            return {"error": str(e)}
    
    async def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize memory allocations for better performance"""
        try:
            optimization_results = {
                "migrations_performed": 0,
                "migrations_recommended": 0,
                "optimizations": []
            }
            
            # Analyze current allocations for optimization opportunities
            for allocation_id, allocation in self.allocations.items():
                current_node = self.fabric_controller.fabric_nodes[allocation.fabric_node_id]
                
                # Check if allocation should be migrated
                migration_candidate = False
                migration_reason = None
                
                # High utilization on current node
                if current_node.current_utilization_percent > 85:
                    migration_candidate = True
                    migration_reason = "high_utilization"
                
                # NUMA optimization
                elif allocation.numa_node != current_node.numa_domain:
                    migration_candidate = True
                    migration_reason = "numa_optimization"
                
                # Bandwidth optimization for high-bandwidth workloads
                elif (allocation.read_bandwidth_mbps > 1000 and 
                      current_node.bandwidth_read_gbps < 100):
                    migration_candidate = True
                    migration_reason = "bandwidth_optimization"
                
                if migration_candidate:
                    # Find better node
                    better_node = await self._find_better_node(allocation, migration_reason)
                    
                    if better_node:
                        optimization_results["migrations_recommended"] += 1
                        optimization_results["optimizations"].append({
                            "allocation_id": allocation_id,
                            "current_node": allocation.fabric_node_id,
                            "recommended_node": better_node,
                            "reason": migration_reason
                        })
                        
                        # Perform migration if benefit is significant
                        if await self._should_migrate(allocation, better_node):
                            success = await self.migrate_allocation(allocation_id, better_node)
                            if success:
                                optimization_results["migrations_performed"] += 1
            
            return optimization_results
            
        except Exception as e:
            logger.error("Allocation optimization failed", error=str(e))
            return {"error": str(e)}
    
    async def _find_better_node(self, allocation: MemoryAllocation, reason: str) -> Optional[str]:
        """Find a better fabric node for an allocation"""
        current_node_id = allocation.fabric_node_id
        size_gb = allocation.size_bytes / (1024**3)
        
        candidates = []
        
        for node_id, node in self.fabric_controller.fabric_nodes.items():
            if node_id == current_node_id:
                continue
            
            if node.available_capacity_gb < size_gb:
                continue
            
            # Score based on optimization reason
            score = 0.0
            
            if reason == "high_utilization":
                # Prefer nodes with lower utilization
                score = 100 - node.current_utilization_percent
            
            elif reason == "numa_optimization":
                # Prefer nodes in the correct NUMA domain
                if node.numa_domain == allocation.numa_node:
                    score = 100
                else:
                    score = 50
            
            elif reason == "bandwidth_optimization":
                # Prefer nodes with higher bandwidth
                score = node.bandwidth_read_gbps
            
            candidates.append((node_id, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    async def _should_migrate(self, allocation: MemoryAllocation, target_node_id: str) -> bool:
        """Determine if migration is beneficial"""
        current_node = self.fabric_controller.fabric_nodes[allocation.fabric_node_id]
        target_node = self.fabric_controller.fabric_nodes[target_node_id]
        
        # Calculate migration benefit score
        benefit_score = 0.0
        
        # Utilization improvement
        util_improvement = current_node.current_utilization_percent - target_node.current_utilization_percent
        benefit_score += util_improvement * 0.4
        
        # Bandwidth improvement
        bandwidth_improvement = target_node.bandwidth_read_gbps - current_node.bandwidth_read_gbps
        benefit_score += bandwidth_improvement * 0.3
        
        # Latency improvement
        latency_improvement = current_node.latency_ns - target_node.latency_ns
        benefit_score += latency_improvement * 0.3
        
        # Only migrate if benefit is significant
        return benefit_score > 20.0
    
    async def _monitoring_loop(self):
        """Continuous monitoring of fabric performance"""
        while True:
            try:
                # Update fabric metrics
                await self.fabric_controller.update_fabric_metrics()
                
                # Update node health
                for node in self.fabric_controller.fabric_nodes.values():
                    # Simulate health monitoring
                    if node.error_count > 10:
                        node.health_status = "degraded"
                    elif node.current_utilization_percent > 95:
                        node.health_status = "overloaded"
                    else:
                        node.health_status = "healthy"
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30.0)
    
    async def _optimization_loop(self):
        """Continuous optimization of memory allocations"""
        while True:
            try:
                # Run optimization every 5 minutes
                optimization_results = await self.optimize_allocations()
                
                if optimization_results.get("migrations_performed", 0) > 0:
                    logger.info(
                        "Memory optimization completed",
                        migrations_performed=optimization_results["migrations_performed"],
                        migrations_recommended=optimization_results["migrations_recommended"]
                    )
                
                await asyncio.sleep(300.0)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error("Error in optimization loop", error=str(e))
                await asyncio.sleep(600.0)
    
    async def _persist_allocation(self, allocation: MemoryAllocation):
        """Persist allocation to database"""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO memory_allocations (
                            allocation_id, session_id, fabric_node_id, size_bytes,
                            numa_node, allocation_strategy, access_pattern, priority_level,
                            compression_enabled, encryption_enabled, allocated_at,
                            allocation_data
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                        )
                        ON CONFLICT (allocation_id) DO UPDATE SET
                            allocation_data = $12
                        """,
                        allocation.allocation_id, allocation.session_id,
                        allocation.fabric_node_id, allocation.size_bytes,
                        allocation.numa_node, allocation.allocation_strategy.value,
                        allocation.access_pattern, allocation.priority_level,
                        allocation.compression_enabled, allocation.encryption_enabled,
                        allocation.allocated_at, json.dumps(asdict(allocation))
                    )
            except Exception as e:
                logger.error("Failed to persist allocation", error=str(e))
    
    async def _broadcast_fabric_update(self, event_type: str, allocation_id: str):
        """Broadcast fabric updates to WebSocket clients"""
        if self.websocket_connections:
            message = {
                "type": event_type,
                "allocation_id": allocation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
            
            self.websocket_connections -= disconnected
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection"""
        self.websocket_connections.add(websocket)
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        logger.info("Memory fabric service cleanup completed")

# FastAPI Application
app = FastAPI(
    title="Omega Memory Fabric Service",
    description="Advanced memory management and fabric orchestration service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory fabric service instance
memory_fabric_service = MemoryFabricService()

@app.on_event("startup")
async def startup_event():
    """Initialize the memory fabric service"""
    await memory_fabric_service.initialize()
    
    # Start Prometheus metrics server
    start_http_server(8004)
    logger.info("Memory fabric service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await memory_fabric_service.cleanup()

# API Endpoints
@app.post("/allocate", response_model=Dict[str, Any])
async def allocate_memory(request: Dict[str, Any]) -> Dict[str, Any]:
    """Allocate memory with specified requirements"""
    try:
        allocation = await memory_fabric_service.allocate_memory(
            session_id=request['session_id'],
            size_bytes=request['size_bytes'],
            allocation_strategy=AllocationStrategy(request.get('allocation_strategy', 'numa_local')),
            numa_preference=request.get('numa_preference'),
            access_pattern=request.get('access_pattern', 'mixed'),
            priority_level=request.get('priority_level', 1),
            compression_enabled=request.get('compression_enabled', True),
            encryption_enabled=request.get('encryption_enabled', True)
        )
        
        if allocation:
            return asdict(allocation)
        else:
            raise HTTPException(status_code=503, detail="Memory allocation failed")
        
    except Exception as e:
        logger.error("Memory allocation request failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/allocate/{allocation_id}")
async def deallocate_memory(allocation_id: str):
    """Deallocate memory"""
    success = await memory_fabric_service.deallocate_memory(allocation_id)
    
    if success:
        return {"message": "Memory deallocated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Allocation not found")

@app.post("/migrate/{allocation_id}")
async def migrate_allocation(allocation_id: str, request: Dict[str, Any]):
    """Migrate memory allocation to different fabric node"""
    try:
        success = await memory_fabric_service.migrate_allocation(
            allocation_id, request['target_fabric_node']
        )
        
        if success:
            return {"message": "Migration completed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Migration failed")
        
    except Exception as e:
        logger.error("Migration request failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status")
async def get_fabric_status() -> Dict[str, Any]:
    """Get comprehensive fabric status"""
    return await memory_fabric_service.get_fabric_status()

@app.post("/optimize")
async def optimize_allocations() -> Dict[str, Any]:
    """Optimize memory allocations"""
    return await memory_fabric_service.optimize_allocations()

@app.get("/allocations")
async def list_allocations() -> Dict[str, Any]:
    """List all active memory allocations"""
    return {
        "allocations": [
            asdict(allocation) 
            for allocation in memory_fabric_service.allocations.values()
        ]
    }

@app.get("/allocations/{session_id}")
async def get_session_allocations(session_id: str) -> Dict[str, Any]:
    """Get memory allocations for a specific session"""
    session_allocations = [
        asdict(allocation) 
        for allocation in memory_fabric_service.allocations.values()
        if allocation.session_id == session_id
    ]
    
    return {"allocations": session_allocations}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time fabric updates"""
    await websocket.accept()
    await memory_fabric_service.add_websocket_connection(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        await memory_fabric_service.remove_websocket_connection(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "fabric_nodes": len(memory_fabric_service.fabric_controller.fabric_nodes),
        "active_allocations": len(memory_fabric_service.allocations),
        "total_allocations": memory_fabric_service.total_allocations,
        "failed_allocations": memory_fabric_service.failed_allocations,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=False,
        access_log=True
    )
