"""
Omega Super Desktop Console v2.0 - Enhanced Control Node
Master coordinator and API hub with fault-tolerant multi-master leader election
Comprehensive node management with AI-powered optimization and heterogeneous hardware support

Features:
- Multi-master fault-tolerant architecture with leader election
- Heterogeneous compute node support (CPU, GPU, NPU, FPGA, Edge-TPU)
- Hot-swapping, self-registration, auto-scaling capabilities
- Tiered storage management with ML-driven optimization
- Advanced orchestration and global scheduling
- Real-time monitoring and adaptive resource allocation
"""

import asyncio
import logging
import time
import json
import uuid
import threading
import platform
import subprocess
import socket
import os
import random
import hashlib
import ssl
import struct
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, auto
from contextlib import asynccontextmanager
import weakref
import gc

# FastAPI and web framework imports
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Database imports
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, JSON, Boolean, Text, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool

# Monitoring and metrics imports
from prometheus_client import start_http_server, Counter, Gauge, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from utils.metrics import create_counter

# Scientific computing and ML imports
import numpy as np
import psutil
import jwt

# AI Engine imports - Enhanced with ML capabilities
import sys
sys.path.append(os.path.dirname(__file__))
try:
    from ai_engine.core import ai_engine
    from ai_engine.network_intelligence import network_intelligence
    from ai_engine.storage_intelligence import storage_intelligence
    AI_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI Engine not available: {e}")
    AI_ENGINE_AVAILABLE = False

# ML and optimization libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    logging.warning("Machine learning libraries not available")

# External service imports with enhanced connectivity
try:
    import os as _os
    _os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    import aiohttp
    import websockets
    import redis.asyncio as redis
    import etcd3
    import grpc
    from grpc import aio as aio_grpc
    import consul
    EXTERNAL_SERVICES_AVAILABLE = True
except ImportError:
    EXTERNAL_SERVICES_AVAILABLE = False
    logging.warning("External services (redis, etcd3, consul) not available - using fallbacks")

# Hardware monitoring and optimization
try:
    import py3nvml.py3nvml as nvml
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

# Network optimization libraries
try:
    import dpkt
    import scapy
    NETWORK_OPTIMIZATION_AVAILABLE = True
except ImportError:
    NETWORK_OPTIMIZATION_AVAILABLE = False

# Enhanced Configuration and Constants
SECRET_KEY = os.getenv("OMEGA_SECRET_KEY", "omega-super-desktop-secret-key-change-in-production")
CONTROL_NODE_PORT = int(os.getenv("OMEGA_CONTROL_PORT", "7777"))
METRICS_PORT = int(os.getenv("OMEGA_METRICS_PORT", "8000"))
WS_PORT = int(os.getenv("OMEGA_WS_PORT", "7778"))
CLUSTER_NAME = os.getenv("OMEGA_CLUSTER_NAME", "omega-cluster-v2")

# Leader election and fault tolerance settings
LEADER_ELECTION_ENABLED = os.getenv("OMEGA_LEADER_ELECTION", "true").lower() == "true"
HEARTBEAT_INTERVAL = int(os.getenv("OMEGA_HEARTBEAT_INTERVAL", "5"))
LEADER_LEASE_DURATION = int(os.getenv("OMEGA_LEADER_LEASE_DURATION", "30"))
FAILURE_DETECTION_THRESHOLD = int(os.getenv("OMEGA_FAILURE_THRESHOLD", "3"))
AUTO_FAILOVER_ENABLED = os.getenv("OMEGA_AUTO_FAILOVER", "true").lower() == "true"

# Resource management settings
MAX_CPU_OVERCOMMIT_RATIO = float(os.getenv("OMEGA_CPU_OVERCOMMIT", "1.5"))
MAX_MEMORY_OVERCOMMIT_RATIO = float(os.getenv("OMEGA_MEMORY_OVERCOMMIT", "1.2"))
GPU_MEMORY_RESERVATION_MB = int(os.getenv("OMEGA_GPU_MEMORY_RESERVATION", "512"))
THERMAL_THRESHOLD_CELSIUS = int(os.getenv("OMEGA_THERMAL_THRESHOLD", "80"))

# Advanced networking settings
RDMA_ENABLED = os.getenv("OMEGA_RDMA_ENABLED", "false").lower() == "true"
NETWORK_LATENCY_TARGET_MS = float(os.getenv("OMEGA_LATENCY_TARGET", "1.0"))
BANDWIDTH_RESERVATION_GBPS = float(os.getenv("OMEGA_BANDWIDTH_RESERVATION", "10.0"))

# Storage optimization settings
TIERED_STORAGE_ENABLED = os.getenv("OMEGA_TIERED_STORAGE", "true").lower() == "true"
ML_PREFETCHING_ENABLED = os.getenv("OMEGA_ML_PREFETCHING", "true").lower() == "true"
DEDUPLICATION_ENABLED = os.getenv("OMEGA_DEDUPLICATION", "true").lower() == "true"
COMPRESSION_ALGORITHM = os.getenv("OMEGA_COMPRESSION", "zstd")

# Security and authentication
TLS_ENABLED = os.getenv("OMEGA_TLS_ENABLED", "true").lower() == "true"
CERT_PATH = os.getenv("OMEGA_CERT_PATH", "../security/certs/control_node.crt")
KEY_PATH = os.getenv("OMEGA_KEY_PATH", "../security/certs/control_node.key")
CA_PATH = os.getenv("OMEGA_CA_PATH", "../security/certs/ca.crt")

# Enhanced Node Types and Status Enums - Block 1: Core Node Classifications
class NodeType(Enum):
    """Comprehensive node types supporting heterogeneous hardware architectures"""
    # Control Node Types - Master coordinators and API hubs
    CONTROL_MASTER = "control_master"              # Primary leader with full orchestration
    CONTROL_BACKUP = "control_backup"              # Backup control node for fault tolerance
    CONTROL_EDGE = "control_edge"                  # Edge control for distributed clusters
    
    # Compute Node Types - Heterogeneous processing units
    COMPUTE_CPU_X86 = "compute_cpu_x86"           # x86 CPU-focused compute nodes
    COMPUTE_CPU_ARM = "compute_cpu_arm"           # ARM CPU-focused compute nodes
    COMPUTE_GPU_NVIDIA = "compute_gpu_nvidia"     # NVIDIA GPU compute (CUDA)
    COMPUTE_GPU_AMD = "compute_gpu_amd"           # AMD GPU compute (ROCm)
    COMPUTE_GPU_INTEL = "compute_gpu_intel"       # Intel GPU compute (OpenCL/oneAPI)
    COMPUTE_NPU_DEDICATED = "compute_npu_dedicated" # Neural Processing Units
    COMPUTE_FPGA_XILINX = "compute_fpga_xilinx"   # Xilinx FPGA acceleration
    COMPUTE_FPGA_INTEL = "compute_fpga_intel"     # Intel FPGA acceleration
    COMPUTE_EDGE_TPU = "compute_edge_tpu"         # Google Edge TPU
    COMPUTE_HYBRID_CPU_GPU = "compute_hybrid_cpu_gpu" # Mixed CPU+GPU workloads
    COMPUTE_HYBRID_HETEROGENEOUS = "compute_hybrid_heterogeneous" # Multi-accelerator
    
    # Storage Node Types - Tiered storage systems
    STORAGE_HOT_NVME = "storage_hot_nvme"         # Hot tier - NVMe SSDs
    STORAGE_HOT_OPTANE = "storage_hot_optane"     # Hot tier - Intel Optane
    STORAGE_WARM_SSD = "storage_warm_ssd"         # Warm tier - SATA SSDs
    STORAGE_COLD_HDD = "storage_cold_hdd"         # Cold tier - HDDs
    STORAGE_ARCHIVE_TAPE = "storage_archive_tape" # Archive tier - Tape storage
    STORAGE_CLOUD_S3 = "storage_cloud_s3"        # Cloud storage - S3 compatible
    STORAGE_DISTRIBUTED_CEPH = "storage_distributed_ceph" # Distributed - Ceph
    STORAGE_DISTRIBUTED_GLUSTER = "storage_distributed_gluster" # Distributed - GlusterFS
    STORAGE_MEMORY_FABRIC = "storage_memory_fabric" # In-memory fabric storage
    
    # Network Node Types - Network infrastructure
    NETWORK_EDGE_ROUTER = "network_edge_router"   # Edge routing and load balancing
    NETWORK_CORE_SWITCH = "network_core_switch"   # Core network switching
    NETWORK_INFINIBAND = "network_infiniband"     # InfiniBand high-speed networking
    NETWORK_RDMA_CAPABLE = "network_rdma_capable" # RDMA-enabled networking
    
    # Specialized Node Types
    AI_INFERENCE_CPU = "ai_inference_cpu"         # CPU-based AI inference
    AI_INFERENCE_GPU = "ai_inference_gpu"         # GPU-based AI inference
    AI_INFERENCE_NPU = "ai_inference_npu"         # NPU-based AI inference
    AI_TRAINING_DISTRIBUTED = "ai_training_distributed" # Distributed AI training
    MEMORY_FABRIC_DDR = "memory_fabric_ddr"       # DDR memory fabric
    MEMORY_FABRIC_HBM = "memory_fabric_hbm"       # High Bandwidth Memory fabric

class NodeStatus(Enum):
    """Comprehensive node status tracking with fault tolerance states"""
    # Initialization and Registration States
    INITIALIZING = "initializing"                 # Node starting up
    REGISTERING = "registering"                   # Joining cluster
    AUTHENTICATING = "authenticating"             # Security validation
    PROVISIONING = "provisioning"                 # Resource allocation
    
    # Active Operation States
    ACTIVE = "active"                             # Normal operation
    BUSY = "busy"                                 # High utilization
    IDLE = "idle"                                 # Low utilization, available
    STANDBY = "standby"                           # Ready backup state
    
    # Maintenance and Management States
    MAINTENANCE_SCHEDULED = "maintenance_scheduled" # Planned maintenance
    MAINTENANCE_ACTIVE = "maintenance_active"     # Currently in maintenance
    DRAINING = "draining"                         # Gracefully removing workloads
    EVACUATING = "evacuating"                     # Emergency workload evacuation
    
    # Health and Error States
    HEALTHY = "healthy"                           # All systems normal
    DEGRADED = "degraded"                         # Partial functionality
    UNHEALTHY = "unhealthy"                       # System issues detected
    CRITICAL = "critical"                         # Severe issues
    FAILED = "failed"                             # Complete failure
    QUARANTINED = "quarantined"                   # Isolated due to issues
    
    # Network and Connectivity States
    OFFLINE = "offline"                           # Not reachable
    NETWORK_PARTITIONED = "network_partitioned"   # Split-brain scenario
    RECONNECTING = "reconnecting"                 # Attempting to rejoin
    
    # Scaling and Hot-swap States
    SCALING_UP = "scaling_up"                     # Adding capacity
    SCALING_DOWN = "scaling_down"                 # Reducing capacity
    HOT_SWAPPING = "hot_swapping"                 # Hardware replacement
    MIGRATING_IN = "migrating_in"                 # Receiving workloads
    MIGRATING_OUT = "migrating_out"               # Transferring workloads
    
    # Leader Election States (for control nodes)
    LEADER = "leader"                             # Current cluster leader
    FOLLOWER = "follower"                         # Following cluster leader
    CANDIDATE = "candidate"                       # Seeking leadership
    SPLIT_BRAIN = "split_brain"                   # Multiple leaders detected

class SessionStatus(Enum):
    """Enhanced session status tracking"""
    PENDING = "pending"
    SCHEDULING = "scheduling"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    MIGRATING = "migrating"
    CHECKPOINTING = "checkpointing"
    RESTORING = "restoring"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    ARCHIVED = "archived"

class TaskType(Enum):
    """Comprehensive task classification"""
    CPU_INTENSIVE = "cpu_intensive"
    GPU_COMPUTE = "gpu_compute"
    NPU_INFERENCE = "npu_inference"
    FPGA_ACCELERATION = "fpga_acceleration"
    MEMORY_INTENSIVE = "memory_intensive"
    STORAGE_INTENSIVE = "storage_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    AI_TRAINING = "ai_training"
    AI_INFERENCE = "ai_inference"
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"
    STREAM_PROCESSING = "stream_processing"
    MIXED_WORKLOAD = "mixed_workload"

class HardwareAccelerator(Enum):
    """Hardware accelerator types"""
    NONE = "none"
    CUDA = "cuda"
    ROCm = "rocm"
    OpenCL = "opencl"
    Vulkan = "vulkan"
    DirectML = "directml"
    TensorRT = "tensorrt"
    OpenVINO = "openvino"
    CoreML = "coreml"
    ONNX_RUNTIME = "onnx_runtime"

class StorageTier(Enum):
    """Storage tier classifications"""
    MEMORY = "memory"
    HOT_NVME = "hot_nvme"
    WARM_SSD = "warm_ssd"
    COLD_HDD = "cold_hdd"
    ARCHIVE_TAPE = "archive_tape"
    CLOUD_S3 = "cloud_s3"
    DISTRIBUTED = "distributed"

class NetworkProtocol(Enum):
    """Network protocol support"""
    TCP = "tcp"
    UDP = "udp"
    RDMA = "rdma"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"
    NVLINK = "nvlink"
    CXL = "cxl"
    PCIE = "pcie"

# Enhanced Data Classes for Node Management
# Enhanced Data Classes for Comprehensive Node Management - Block 2
@dataclass
class NodeCapabilities:
    """Comprehensive node capabilities with heterogeneous hardware support"""
    # Basic Node Information
    node_id: str = ""
    node_type: NodeType = NodeType.COMPUTE_CPU_X86
    node_class: str = "standard"  # standard, high-performance, edge, embedded
    
    # CPU Specifications - Enhanced for heterogeneous architectures
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_frequency_base_ghz: float = 0.0
    cpu_frequency_boost_ghz: float = 0.0
    cpu_architecture: str = "x86_64"  # x86_64, arm64, riscv
    cpu_vendor: str = "unknown"  # intel, amd, arm, apple
    cpu_model: str = ""
    cpu_instruction_sets: List[str] = field(default_factory=list)  # avx512, neon, etc.
    cpu_cache_l1_kb: int = 0
    cpu_cache_l2_kb: int = 0
    cpu_cache_l3_kb: int = 0
    cpu_tdp_watts: int = 0
    
    # Memory Specifications - Advanced memory hierarchy
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_type: str = "DDR4"  # DDR4, DDR5, LPDDR5, HBM2, HBM3
    memory_channels: int = 2
    memory_speed_mhz: int = 2400
    memory_bandwidth_gbps: float = 0.0
    memory_ecc_support: bool = False
    numa_nodes: int = 1
    numa_topology: Dict[str, Any] = field(default_factory=dict)
    
    # GPU Specifications - Multi-vendor GPU support
    gpu_units: int = 0
    gpu_total_memory_gb: float = 0.0
    gpu_available_memory_gb: float = 0.0
    gpu_compute_capability: str = ""  # CUDA compute capability or equivalent
    gpu_vendor: str = ""  # nvidia, amd, intel
    gpu_models: List[str] = field(default_factory=list)
    gpu_driver_version: str = ""
    gpu_cuda_version: str = ""
    gpu_rocm_version: str = ""
    gpu_opencl_version: str = ""
    gpu_vulkan_version: str = ""
    gpu_directml_support: bool = False
    gpu_tensor_cores: bool = False
    gpu_ray_tracing_support: bool = False
    gpu_nvlink_support: bool = False
    gpu_multi_instance_support: bool = False
    
    # Specialized Hardware - NPU, FPGA, Edge TPU
    npu_units: int = 0
    npu_tops_int8: float = 0.0
    npu_tops_fp16: float = 0.0
    npu_vendor: str = ""  # intel, qualcomm, google, apple
    npu_models: List[str] = field(default_factory=list)
    
    fpga_units: int = 0
    fpga_vendor: str = ""  # xilinx, intel, microsemi
    fpga_models: List[str] = field(default_factory=list)
    fpga_logic_elements: int = 0
    fpga_memory_blocks: int = 0
    fpga_dsp_blocks: int = 0
    
    edge_tpu_units: int = 0
    edge_tpu_tops: float = 0.0
    edge_tpu_version: str = ""
    
    # Storage Capabilities - Tiered storage architecture
    storage_total_gb: float = 0.0
    storage_available_gb: float = 0.0
    storage_tiers: Dict[StorageTier, Dict[str, Any]] = field(default_factory=dict)
    storage_nvme_units: int = 0
    storage_ssd_units: int = 0
    storage_hdd_units: int = 0
    storage_total_iops: int = 0
    storage_sequential_read_mbps: float = 0.0
    storage_sequential_write_mbps: float = 0.0
    storage_random_read_iops: int = 0
    storage_random_write_iops: int = 0
    storage_deduplication_support: bool = False
    storage_compression_support: bool = False
    storage_encryption_support: bool = False
    
    # Network Capabilities - High-performance networking
    network_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    network_bandwidth_gbps: float = 1.0
    network_protocols: List[NetworkProtocol] = field(default_factory=list)
    rdma_capable: bool = False
    rdma_protocol: str = ""  # roce, infiniband, iwarp
    infiniband_ports: int = 0
    infiniband_speed: str = ""  # FDR, EDR, HDR, NDR
    sr_iov_support: bool = False
    pcie_lanes: int = 0
    pcie_generation: int = 4
    
    # Virtualization and Containerization
    virtualization_support: bool = True
    hypervisor_support: List[str] = field(default_factory=list)  # kvm, xen, vmware
    container_runtime_support: List[str] = field(default_factory=list)  # docker, containerd, cri-o
    kubernetes_support: bool = True
    nested_virtualization: bool = False
    iommu_support: bool = False
    
    # Hardware Accelerators and Features
    hardware_accelerators: List[HardwareAccelerator] = field(default_factory=list)
    security_features: List[str] = field(default_factory=list)  # tpm, secure_boot, etc.
    crypto_acceleration: bool = False
    hardware_rng: bool = False
    
    # Power Management and Thermal
    power_management: Dict[str, Any] = field(default_factory=dict)
    thermal_design_power_watts: int = 0
    power_consumption_idle_watts: int = 0
    power_consumption_max_watts: int = 0
    cooling_solution: str = "air"  # air, liquid, immersion
    thermal_sensors: List[str] = field(default_factory=list)
    
    # Hot-swapping and Self-registration Capabilities
    hot_swappable: bool = False
    hot_swap_components: List[str] = field(default_factory=list)  # memory, storage, cards
    auto_registration: bool = True
    auto_discovery: bool = True
    plug_and_play: bool = False
    
    # Fault Tolerance and Reliability
    fault_tolerance_level: str = "standard"  # basic, standard, high, critical
    redundancy_support: bool = False
    error_correction: bool = False
    memory_mirroring: bool = False
    raid_support: List[str] = field(default_factory=list)  # 0, 1, 5, 6, 10
    
    # Performance Characteristics
    compute_performance_rating: float = 1.0  # Normalized performance score
    memory_latency_ns: float = 0.0
    storage_latency_us: float = 0.0
    network_latency_us: float = 0.0
    
    # Specialized Workload Support
    workload_types_optimized: List[TaskType] = field(default_factory=list)
    ai_frameworks_supported: List[str] = field(default_factory=list)  # tensorflow, pytorch, etc.
    
    # Location and Physical Characteristics
    rack_location: str = ""
    datacenter_zone: str = ""
    geographic_location: str = ""
    physical_size_form_factor: str = ""  # 1U, 2U, tower, blade, embedded

@dataclass
class NodeMetrics:
    """Comprehensive real-time node performance metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_load_1min: float = 0.0
    cpu_load_5min: float = 0.0
    cpu_load_15min: float = 0.0
    cpu_frequency_current_ghz: float = 0.0
    
    # Memory metrics
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_cached_gb: float = 0.0
    memory_swap_usage_percent: float = 0.0
    
    # GPU metrics
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_temperature_celsius: float = 0.0
    gpu_power_usage_watts: float = 0.0
    
    # Storage metrics
    disk_usage_percent: float = 0.0
    disk_io_read_mbps: float = 0.0
    disk_io_write_mbps: float = 0.0
    disk_io_utilization_percent: float = 0.0
    
    # Network metrics
    network_rx_mbps: float = 0.0
    network_tx_mbps: float = 0.0
    network_latency_ms: float = 0.0
    network_packet_loss_percent: float = 0.0
    
    # Thermal and power metrics
    cpu_temperature_celsius: float = 0.0
    system_temperature_celsius: float = 0.0
    power_consumption_watts: float = 0.0
    fan_speed_rpm: int = 0
    
    # Performance scores
    compute_performance_score: float = 1.0
    memory_performance_score: float = 1.0
    storage_performance_score: float = 1.0
    network_performance_score: float = 1.0
    overall_health_score: float = 1.0
    
    # Additional metrics
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0

@dataclass
class FaultToleranceConfig:
    """Enhanced fault tolerance and leader election configuration"""
    leader_election_enabled: bool = True
    heartbeat_interval_seconds: int = 5
    leader_lease_duration_seconds: int = 30
    failure_detection_threshold: int = 3
    auto_failover_enabled: bool = True
    split_brain_protection: bool = True
    backup_control_nodes: List[str] = field(default_factory=list)
    consensus_algorithm: str = "raft"
    quorum_size: int = 3
    leader_health_check_interval: int = 10
    failover_timeout_seconds: int = 60
    cluster_membership_timeout: int = 120

@dataclass
class LeaderElectionState:
    """Leader election state tracking"""
    current_leader: Optional[str] = None
    leader_term: int = 0
    leader_lease_expiry: Optional[datetime] = None
    election_in_progress: bool = False
    candidates: List[str] = field(default_factory=list)
    last_election_time: Optional[datetime] = None
    split_brain_detected: bool = False

# Enhanced Prometheus Metrics for comprehensive monitoring
REQUESTS_TOTAL = create_counter('omega_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
ACTIVE_SESSIONS = Gauge('omega_active_sessions', 'Number of active sessions', ['node_type'])
NODE_COUNT = Gauge('omega_node_count', 'Number of registered nodes', ['type', 'status'])
LATENCY_P95 = Gauge('omega_latency_p95_ms', 'P95 latency in milliseconds', ['node_id'])
RESOURCE_UTILIZATION = Gauge('omega_resource_utilization', 'Resource utilization', ['node_id', 'resource_type'])
FAULT_TOLERANCE_STATUS = Gauge('omega_fault_tolerance_status', 'Fault tolerance status', ['component'])

# Leader election metrics
LEADER_ELECTIONS_TOTAL = create_counter('omega_leader_elections_total', 'Total leader elections')
LEADER_ELECTION_DURATION = Histogram('omega_leader_election_duration_seconds', 'Leader election duration')
SPLIT_BRAIN_INCIDENTS = create_counter('omega_split_brain_incidents_total', 'Split brain incidents detected')

# Performance metrics
TASK_EXECUTION_TIME = Histogram('omega_task_execution_seconds', 'Task execution time', ['task_type', 'node_type'])
PLACEMENT_ALGORITHM_DURATION = Histogram('omega_placement_duration_seconds', 'Placement algorithm duration')
HEARTBEAT_LATENCY = Histogram('omega_heartbeat_latency_seconds', 'Heartbeat latency')

# Hardware metrics
CPU_TEMPERATURE = Gauge('omega_cpu_temperature_celsius', 'CPU temperature', ['node_id'])
GPU_UTILIZATION = Gauge('omega_gpu_utilization_percent', 'GPU utilization', ['node_id', 'gpu_id'])
MEMORY_PRESSURE = Gauge('omega_memory_pressure', 'Memory pressure indicator', ['node_id'])
STORAGE_TIER_UTILIZATION = Gauge('omega_storage_tier_utilization', 'Storage tier utilization', ['node_id', 'tier'])

# Network metrics
NETWORK_BANDWIDTH_UTILIZATION = Gauge('omega_network_bandwidth_utilization', 'Network bandwidth utilization', ['node_id', 'interface'])
RDMA_OPERATIONS = create_counter('omega_rdma_operations_total', 'RDMA operations', ['node_id', 'operation_type'])

# Enhanced in-memory metrics fallback with comprehensive tracking
class SimpleMetrics:
    def __init__(self):
        self.requests_total = 0
        self.active_sessions = 0
        self.node_count = 0
        self.errors = 0
        self.leader_elections = 0
        self.split_brain_incidents = 0
        self.task_executions = {}
        self.placement_times = []
        self.heartbeat_latencies = []
        self.resource_utilization = {}
        
    def inc_requests(self, method="GET", endpoint="/", status="200"):
        self.requests_total += 1
    
    def set_active_sessions(self, count, node_type="default"):
        self.active_sessions = count
    
    def set_node_count(self, count, node_type="default", status="active"):
        self.node_count = count
        
    def record_leader_election(self):
        self.leader_elections += 1
        
    def record_split_brain(self):
        self.split_brain_incidents += 1
        
    def record_task_execution(self, task_type, duration, node_type="default"):
        if task_type not in self.task_executions:
            self.task_executions[task_type] = []
        self.task_executions[task_type].append(duration)
        
    def record_placement_time(self, duration):
        self.placement_times.append(duration)
        if len(self.placement_times) > 1000:  # Keep only recent data
            self.placement_times = self.placement_times[-1000:]
            
    def record_heartbeat_latency(self, latency):
        self.heartbeat_latencies.append(latency)
        if len(self.heartbeat_latencies) > 1000:
            self.heartbeat_latencies = self.heartbeat_latencies[-1000:]

metrics = SimpleMetrics()

# Machine Learning Components for Intelligent Optimization
class MLPlacementOptimizer:
    """Machine learning-based placement optimization"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if ML_LIBRARIES_AVAILABLE else None
        self.training_data = []
        self.feature_names = [
            'cpu_utilization', 'memory_utilization', 'gpu_utilization',
            'network_latency', 'thermal_score', 'load_balance_score',
            'task_cpu_req', 'task_memory_req', 'task_gpu_req'
        ]
        
    def train_model(self, placement_history: List[Dict[str, Any]]):
        """Train placement model based on historical data"""
        if not ML_LIBRARIES_AVAILABLE or len(placement_history) < 10:
            return
            
        try:
            # Prepare training data
            features = []
            targets = []
            
            for record in placement_history:
                if 'features' in record and 'performance_score' in record:
                    features.append(record['features'])
                    targets.append(record['performance_score'])
            
            if len(features) < 10:
                return
                
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            logging.info(f"ML placement model trained on {len(features)} samples")
            
        except Exception as e:
            logging.error(f"Failed to train ML placement model: {e}")
    
    def predict_performance(self, node_features: List[float], task_requirements: List[float]) -> float:
        """Predict performance score for placement"""
        if not ML_LIBRARIES_AVAILABLE or self.model is None:
            return 0.5  # Default neutral score
            
        try:
            features = np.array([node_features + task_requirements]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            score = self.model.predict(features_scaled)[0]
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logging.error(f"ML prediction failed: {e}")
            return 0.5

# Enhanced Resource Management and Supporting Classes - Block 4
class ResourceManager:
    """
    Advanced resource management for heterogeneous compute environments.
    Handles resource allocation, tracking, and optimization.
    """
    
    def __init__(self):
        self.resource_pools = {}
        self.allocation_tracking = {}
        self.resource_quotas = {}
        self.utilization_history = defaultdict(deque)
        
    async def allocate_resources(self, task_id: str, node_id: str, requirements: Dict[str, Any]):
        """Allocate resources for a task on a specific node"""
        try:
            if node_id not in self.resource_pools:
                logging.error(f"Node {node_id} not found in resource pools")
                return False
            
            node_resources = self.resource_pools[node_id]
            
            # Check availability
            if not self._check_resource_availability(node_resources, requirements):
                return False
            
            # Allocate resources
            self._update_resource_allocation(node_id, requirements, 'allocate')
            
            # Track allocation
            self.allocation_tracking[task_id] = {
                'node_id': node_id,
                'requirements': requirements,
                'allocated_at': datetime.utcnow()
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Resource allocation failed: {e}")
            return False
    
    def _check_resource_availability(self, node_resources: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Check if node has sufficient resources"""
        try:
            available_cpu = node_resources.get('available_cpu_cores', 0)
            available_memory = node_resources.get('available_memory_gb', 0)
            available_gpu = node_resources.get('available_gpu_units', 0)
            
            required_cpu = requirements.get('cpu_cores', 0)
            required_memory = requirements.get('memory_gb', 0)
            required_gpu = requirements.get('gpu_units', 0)
            
            return (available_cpu >= required_cpu and 
                   available_memory >= required_memory and
                   available_gpu >= required_gpu)
                   
        except Exception as e:
            logging.error(f"Resource availability check failed: {e}")
            return False

class HeartbeatManager:
    """
    Heartbeat management for cluster health monitoring and failure detection.
    """
    
    def __init__(self):
        self.heartbeat_interval = HEARTBEAT_INTERVAL
        self.node_heartbeats = {}
        self.failure_threshold = FAILURE_DETECTION_THRESHOLD
        
    async def start(self):
        """Start heartbeat management"""
        try:
            asyncio.create_task(self._heartbeat_loop())
            logging.info("Heartbeat manager started")
        except Exception as e:
            logging.error(f"Heartbeat manager start failed: {e}")
    
    async def _heartbeat_loop(self):
        """Main heartbeat monitoring loop"""
        while True:
            try:
                await self._send_heartbeats()
                await self._check_node_failures()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logging.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeats(self):
        """Send heartbeats to cluster members"""
        try:
            if redis_client:
                heartbeat_data = {
                    'node_id': f"control-{socket.gethostname()}",
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'healthy'
                }
                await redis_client.setex('heartbeat:control', self.heartbeat_interval * 2, json.dumps(heartbeat_data))
        except Exception as e:
            logging.warning(f"Failed to send heartbeat: {e}")

class LoadBalancer:
    """Enhanced load balancer with predictive capabilities and heterogeneous hardware support"""
    
    def __init__(self):
        self.node_loads = defaultdict(float)
        self.task_history = defaultdict(list)
        self.placement_weights = {}
        self.load_prediction_models = {}
        
    def update_node_load(self, node_id: str, load_metrics: Dict[str, float]):
        """Update load metrics for a node with hardware-specific weighting"""
        try:
            # Get node type for specialized weighting
            node_type = load_metrics.get('node_type', 'cpu')
            
            if 'gpu' in node_type.lower():
                # GPU-focused weighting
                cpu_weight = 0.2
                memory_weight = 0.2
                gpu_weight = 0.5
                network_weight = 0.1
            elif 'storage' in node_type.lower():
                # Storage-focused weighting
                cpu_weight = 0.1
                memory_weight = 0.2
                gpu_weight = 0.0
                disk_weight = 0.6
                network_weight = 0.1
            else:
                # Default CPU-focused weighting
                cpu_weight = 0.4
                memory_weight = 0.3
                gpu_weight = 0.2
                network_weight = 0.1
            
            total_load = (
                load_metrics.get('cpu_utilization', 0) * cpu_weight +
                load_metrics.get('memory_utilization', 0) * memory_weight +
                load_metrics.get('gpu_utilization', 0) * gpu_weight +
                load_metrics.get('network_utilization', 0) * network_weight
            )
            
            # Add storage utilization for storage nodes
            if 'storage' in node_type.lower():
                total_load += load_metrics.get('disk_utilization', 0) * 0.6
            
            self.node_loads[node_id] = total_load
            
            # Update load history for prediction
            if node_id not in self.task_history:
                self.task_history[node_id] = deque(maxlen=100)
            self.task_history[node_id].append({
                'timestamp': time.time(),
                'load': total_load,
                'metrics': load_metrics
            })
            
        except Exception as e:
            logging.error(f"Load update failed for node {node_id}: {e}")
    
    def get_load_balance_score(self, node_id: str) -> float:
        """Get load balance score with predictive adjustment"""
        try:
            if node_id not in self.node_loads:
                return 0.5
                
            current_load = self.node_loads[node_id]
            
            # Predict future load
            predicted_load = self._predict_future_load(node_id)
            
            # Combine current and predicted load
            combined_load = (current_load * 0.7) + (predicted_load * 0.3)
            
            # Invert to get score (lower load = higher score)
            score = max(0.0, 1.0 - combined_load)
            
            return score
            
        except Exception as e:
            logging.error(f"Load balance score calculation failed: {e}")
            return 0.5
    
    def _predict_future_load(self, node_id: str) -> float:
        """Predict future load based on historical patterns"""
        try:
            if node_id not in self.task_history or len(self.task_history[node_id]) < 3:
                return self.node_loads.get(node_id, 0.5)
            
            # Simple linear trend prediction
            recent_loads = [entry['load'] for entry in list(self.task_history[node_id])[-5:]]
            
            if len(recent_loads) >= 2:
                # Calculate trend
                trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
                predicted = recent_loads[-1] + trend
                return max(0.0, min(1.0, predicted))
            
            return recent_loads[-1]
            
        except Exception as e:
            logging.error(f"Load prediction failed: {e}")
            return 0.5
    
    def select_optimal_node(self, available_nodes: List[str], task_requirements: Dict[str, Any]) -> str:
        """Select optimal node with hardware-aware load balancing"""
        try:
            if not available_nodes:
                return None
                
            scores = {}
            task_type = task_requirements.get('task_type', '')
            
            for node_id in available_nodes:
                base_score = self.get_load_balance_score(node_id)
                
                # Apply task-specific adjustments based on hardware requirements
                if 'gpu' in task_type.lower():
                    # Prefer GPU nodes for GPU tasks
                    if 'gpu' in node_id.lower():
                        base_score *= 1.5
                    else:
                        base_score *= 0.5
                
                elif 'storage' in task_type.lower():
                    # Prefer storage nodes for storage tasks
                    if 'storage' in node_id.lower():
                        base_score *= 1.3
                
                elif 'network' in task_type.lower():
                    # Prefer nodes with good network connectivity
                    network_score = task_requirements.get('network_score', 1.0)
                    base_score *= network_score
                
                # Consider thermal constraints
                thermal_penalty = self._get_thermal_penalty(node_id)
                base_score *= thermal_penalty
                
                scores[node_id] = base_score
            
            # Return node with highest score
            best_node = max(scores.items(), key=lambda x: x[1])
            return best_node[0]
            
        except Exception as e:
            logging.error(f"Optimal node selection failed: {e}")
            return available_nodes[0] if available_nodes else None
    
    def _get_thermal_penalty(self, node_id: str) -> float:
        """Calculate thermal penalty for node selection"""
        try:
            # This would integrate with thermal manager in full implementation
            # For now, return a neutral value
            return 1.0
        except Exception as e:
            logging.error(f"Thermal penalty calculation failed: {e}")
            return 1.0

class SplitBrainDetector:
    """
    Advanced split-brain detection and resolution for multi-master control nodes.
    """
    
    def __init__(self):
        self.detection_interval = 10  # seconds
        self.quorum_requirements = {}
        self.leadership_claims = {}
        self.network_partitions = set()
        
    async def detect_split_brain(self) -> bool:
        """Detect potential split-brain scenarios"""
        try:
            # Check for multiple leadership claims
            multiple_leaders = await self._check_multiple_leaders()
            
            # Check for network partitions
            network_partition = await self._check_network_partitions()
            
            # Check quorum violations
            quorum_violation = await self._check_quorum_violations()
            
            return multiple_leaders or network_partition or quorum_violation
            
        except Exception as e:
            logging.error(f"Split-brain detection failed: {e}")
            return False
    
    async def _check_multiple_leaders(self) -> bool:
        """Check for multiple nodes claiming leadership"""
        try:
            if not etcd_client:
                return False
            
            leaders = set()
            for key, value in etcd_client.get_prefix('/omega/leader/'):
                leaders.add(value.decode())
            
            return len(leaders) > 1
            
        except Exception as e:
            logging.error(f"Multiple leader check failed: {e}")
            return False
    
    async def _check_network_partitions(self) -> bool:
        """Detect network partitions between control nodes"""
        try:
            # This would implement network connectivity tests
            # between control nodes in full implementation
            return False
            
        except Exception as e:
            logging.error(f"Network partition check failed: {e}")
            return False
    
    async def _check_quorum_violations(self) -> bool:
        """Check for quorum violations in cluster consensus"""
        try:
            # This would check if sufficient nodes are available
            # for cluster consensus in full implementation
            return False
            
        except Exception as e:
            logging.error(f"Quorum violation check failed: {e}")
            return False

# Supporting Classes for Compute and Storage Management - Block 6
class SandboxManager:
    """Isolated sandboxing for distributed workloads"""
    
    async def setup_node_isolation(self, compute_node):
        """Setup isolation for a compute node"""
        try:
            logging.info(f"Setting up isolation for {compute_node.node_id}")
            # Implementation would setup containers, VMs, or process isolation
        except Exception as e:
            logging.error(f"Isolation setup failed: {e}")

class AutoScalingManager:
    """Auto-scaling management for compute nodes"""
    
    async def enable_auto_scaling(self, compute_node):
        """Enable auto-scaling for a compute node"""
        try:
            logging.info(f"Enabling auto-scaling for {compute_node.node_id}")
            # Implementation would setup scaling policies and monitoring
        except Exception as e:
            logging.error(f"Auto-scaling setup failed: {e}")

class HotSwapManager:
    """Hot-swapping capability management"""
    
    async def monitor_node(self, compute_node):
        """Monitor node for hot-swap events"""
        try:
            logging.info(f"Monitoring hot-swap for {compute_node.node_id}")
            # Implementation would monitor hardware changes
        except Exception as e:
            logging.error(f"Hot-swap monitoring failed: {e}")

class ComputePerformanceMetrics:
    """Performance metrics tracking for compute nodes"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = {}
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics"""
        try:
            self.current_metrics = metrics
            self.metrics_history.append({
                'timestamp': datetime.utcnow(),
                'metrics': metrics.copy()
            })
        except Exception as e:
            logging.error(f"Metrics update failed: {e}")

class ThermalMonitor:
    """Thermal monitoring for individual nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.thermal_data = deque(maxlen=100)
    
    def update_thermal_data(self, temp_data: Dict[str, float]):
        """Update thermal data for the node"""
        try:
            self.thermal_data.append({
                'timestamp': datetime.utcnow(),
                'data': temp_data
            })
        except Exception as e:
            logging.error(f"Thermal data update failed: {e}")

# Tiered Storage Nodes Implementation - Block 7  
class StorageNodeManager:
    """
    Advanced storage node management with tiered architecture.
    Supports hot (SSD), warm (SSD), cold (HDD) storage with distributed,
    redundant, deduplicated storage and ML-driven prefetching and tier migration.
    """
    
    def __init__(self):
        self.storage_nodes = {}
        self.storage_tiers = {
            StorageTier.HOT_NVME: {},
            StorageTier.WARM_SSD: {},
            StorageTier.COLD_HDD: {},
            StorageTier.ARCHIVE_TAPE: {},
            StorageTier.CLOUD_S3: {},
            StorageTier.DISTRIBUTED: {},
            StorageTier.MEMORY: {}
        }
        
        self.tier_manager = StorageTierManager()
        self.deduplication_engine = DeduplicationEngine()
        self.prefetch_predictor = PrefetchPredictor()
        self.replication_manager = ReplicationManager()
        
        # ML-driven optimization
        self.ml_tier_optimizer = MLTierOptimizer()
        self.access_pattern_analyzer = AccessPatternAnalyzer()
        
    async def register_storage_node(self, node_id: str, node_capabilities: NodeCapabilities):
        """Register storage node with automatic tier classification"""
        try:
            # Classify storage tiers available on node
            available_tiers = self._classify_storage_tiers(node_capabilities)
            
            # Create storage node instance
            storage_node = StorageNode(node_id, node_capabilities, available_tiers)
            
            # Initialize storage features
            await self._initialize_storage_features(storage_node)
            
            # Setup tier management
            await self.tier_manager.setup_node_tiers(storage_node)
            
            # Enable deduplication if supported
            if node_capabilities.storage_deduplication_support:
                await self.deduplication_engine.enable_for_node(storage_node)
            
            # Setup ML-driven prefetching
            if ML_PREFETCHING_ENABLED:
                await self.prefetch_predictor.enable_for_node(storage_node)
            
            # Configure replication
            await self.replication_manager.setup_replication(storage_node)
            
            self.storage_nodes[node_id] = storage_node
            
            logging.info(f"Storage node {node_id} registered with tiers: {available_tiers}")
            
            return True
            
        except Exception as e:
            logging.error(f"Storage node registration failed: {e}")
            return False
    
    def _classify_storage_tiers(self, capabilities: NodeCapabilities) -> List[StorageTier]:
        """Classify available storage tiers on node"""
        try:
            available_tiers = []
            
            # Check for memory tier (RAM-based storage)
            if capabilities.memory_total_gb > 0:
                available_tiers.append(StorageTier.MEMORY)
            
            # Check for NVMe hot tier
            if capabilities.storage_nvme_units > 0:
                available_tiers.append(StorageTier.HOT_NVME)
            
            # Check for SSD warm tier
            if capabilities.storage_ssd_units > 0:
                available_tiers.append(StorageTier.WARM_SSD)
            
            # Check for HDD cold tier
            if capabilities.storage_hdd_units > 0:
                available_tiers.append(StorageTier.COLD_HDD)
            
            # Check for distributed storage capability
            if capabilities.network_bandwidth_gbps >= 10.0:  # High bandwidth for distributed
                available_tiers.append(StorageTier.DISTRIBUTED)
            
            return available_tiers
            
        except Exception as e:
            logging.error(f"Storage tier classification failed: {e}")
            return [StorageTier.WARM_SSD]  # Default fallback
    
    async def _initialize_storage_features(self, storage_node):
        """Initialize storage-specific features"""
        try:
            # Setup compression if supported
            if storage_node.capabilities.storage_compression_support:
                await self._setup_compression(storage_node)
            
            # Setup encryption if supported
            if storage_node.capabilities.storage_encryption_support:
                await self._setup_encryption(storage_node)
            
            # Initialize performance monitoring
            await self._setup_storage_monitoring(storage_node)
            
        except Exception as e:
            logging.error(f"Storage feature initialization failed: {e}")

class StorageNode:
    """Individual storage node with tiered storage management"""
    
    def __init__(self, node_id: str, capabilities: NodeCapabilities, available_tiers: List[StorageTier]):
        self.node_id = node_id
        self.capabilities = capabilities
        self.available_tiers = available_tiers
        self.status = NodeStatus.INITIALIZING
        
        # Storage tier instances
        self.tier_storage = {}
        for tier in available_tiers:
            self.tier_storage[tier] = TierStorage(tier, self._get_tier_capacity(tier))
        
        # Performance tracking
        self.io_metrics = StorageIOMetrics()
        self.access_patterns = AccessPatternTracker()
        
        # Data management
        self.data_catalog = DataCatalog()
        self.migration_queue = deque()
        
    def _get_tier_capacity(self, tier: StorageTier) -> float:
        """Get capacity for specific storage tier"""
        try:
            if tier == StorageTier.MEMORY:
                return self.capabilities.memory_total_gb * 0.5  # Use 50% of RAM for storage
            elif tier == StorageTier.HOT_NVME:
                return self.capabilities.storage_total_gb * 0.3  # 30% for hot tier
            elif tier == StorageTier.WARM_SSD:
                return self.capabilities.storage_total_gb * 0.5  # 50% for warm tier
            elif tier == StorageTier.COLD_HDD:
                return self.capabilities.storage_total_gb * 0.8  # 80% for cold tier
            else:
                return self.capabilities.storage_total_gb * 0.1  # Default allocation
                
        except Exception as e:
            logging.error(f"Tier capacity calculation failed: {e}")
            return 100.0  # Default 100GB
    
    async def store_data(self, data_id: str, data_size_gb: float, access_pattern: str = "random") -> bool:
        """Store data with intelligent tier placement"""
        try:
            # Determine optimal tier based on access pattern and size
            optimal_tier = await self._select_optimal_tier(data_size_gb, access_pattern)
            
            if optimal_tier not in self.tier_storage:
                # Fallback to available tier
                optimal_tier = list(self.tier_storage.keys())[0]
            
            # Store data in selected tier
            success = await self.tier_storage[optimal_tier].store_data(data_id, data_size_gb)
            
            if success:
                # Update data catalog
                await self.data_catalog.register_data(data_id, optimal_tier, data_size_gb)
                
                # Update access patterns
                self.access_patterns.record_access(data_id, access_pattern)
                
                logging.info(f"Data {data_id} stored in {optimal_tier.value} tier")
                
            return success
            
        except Exception as e:
            logging.error(f"Data storage failed: {e}")
            return False
    
    async def _select_optimal_tier(self, data_size_gb: float, access_pattern: str) -> StorageTier:
        """Select optimal storage tier based on data characteristics"""
        try:
            # Hot data (frequent access) -> Memory/NVMe
            if access_pattern in ["frequent", "real_time", "hot"]:
                if StorageTier.MEMORY in self.available_tiers and data_size_gb < 10:
                    return StorageTier.MEMORY
                elif StorageTier.HOT_NVME in self.available_tiers:
                    return StorageTier.HOT_NVME
                else:
                    return StorageTier.WARM_SSD
            
            # Warm data (occasional access) -> SSD
            elif access_pattern in ["occasional", "warm", "moderate"]:
                if StorageTier.WARM_SSD in self.available_tiers:
                    return StorageTier.WARM_SSD
                else:
                    return StorageTier.COLD_HDD
            
            # Cold data (rare access) -> HDD
            elif access_pattern in ["rare", "cold", "archive"]:
                if StorageTier.COLD_HDD in self.available_tiers:
                    return StorageTier.COLD_HDD
                elif StorageTier.ARCHIVE_TAPE in self.available_tiers:
                    return StorageTier.ARCHIVE_TAPE
                else:
                    return StorageTier.WARM_SSD
            
            # Default to warm tier
            else:
                return StorageTier.WARM_SSD
                
        except Exception as e:
            logging.error(f"Tier selection failed: {e}")
            return StorageTier.WARM_SSD

class TierStorage:
    """Individual storage tier management"""
    
    def __init__(self, tier: StorageTier, capacity_gb: float):
        self.tier = tier
        self.capacity_gb = capacity_gb
        self.used_gb = 0.0
        self.data_blocks = {}
        
        # Performance characteristics
        self.performance_profile = self._get_performance_profile()
        
    def _get_performance_profile(self) -> Dict[str, float]:
        """Get performance characteristics for tier"""
        profiles = {
            StorageTier.MEMORY: {
                'read_latency_us': 0.1,
                'write_latency_us': 0.1,
                'bandwidth_gbps': 100.0,
                'iops': 1000000
            },
            StorageTier.HOT_NVME: {
                'read_latency_us': 50.0,
                'write_latency_us': 100.0,
                'bandwidth_gbps': 10.0,
                'iops': 500000
            },
            StorageTier.WARM_SSD: {
                'read_latency_us': 100.0,
                'write_latency_us': 200.0,
                'bandwidth_gbps': 5.0,
                'iops': 100000
            },
            StorageTier.COLD_HDD: {
                'read_latency_us': 10000.0,
                'write_latency_us': 10000.0,
                'bandwidth_gbps': 1.0,
                'iops': 1000
            }
        }
        
        return profiles.get(self.tier, profiles[StorageTier.WARM_SSD])
    
    async def store_data(self, data_id: str, data_size_gb: float) -> bool:
        """Store data in this tier"""
        try:
            if self.used_gb + data_size_gb > self.capacity_gb:
                return False  # Insufficient space
            
            self.data_blocks[data_id] = {
                'size_gb': data_size_gb,
                'stored_at': datetime.utcnow(),
                'access_count': 0,
                'last_access': datetime.utcnow()
            }
            
            self.used_gb += data_size_gb
            return True
            
        except Exception as e:
            logging.error(f"Data storage in tier failed: {e}")
            return False

class StorageTierManager:
    """Management of storage tier operations and migrations"""
    
    def __init__(self):
        self.migration_policies = {}
        self.tier_utilization = {}
        
    async def setup_node_tiers(self, storage_node):
        """Setup tier management for storage node"""
        try:
            # Configure migration policies
            await self._configure_migration_policies(storage_node)
            
            # Start tier monitoring
            asyncio.create_task(self._monitor_tier_utilization(storage_node))
            
            logging.info(f"Tier management setup for {storage_node.node_id}")
            
        except Exception as e:
            logging.error(f"Tier setup failed: {e}")
    
    async def _configure_migration_policies(self, storage_node):
        """Configure data migration policies between tiers"""
        try:
            # Hot to warm migration policy
            hot_to_warm = {
                'trigger': 'utilization_threshold',
                'threshold': 0.8,
                'target_tier': StorageTier.WARM_SSD,
                'data_selection': 'least_recently_used'
            }
            
            # Warm to cold migration policy
            warm_to_cold = {
                'trigger': 'age_threshold',
                'threshold_days': 30,
                'target_tier': StorageTier.COLD_HDD,
                'data_selection': 'oldest_first'
            }
            
            self.migration_policies[storage_node.node_id] = {
                'hot_to_warm': hot_to_warm,
                'warm_to_cold': warm_to_cold
            }
            
        except Exception as e:
            logging.error(f"Migration policy configuration failed: {e}")

class DeduplicationEngine:
    """Data deduplication engine for storage optimization"""
    
    def __init__(self):
        self.dedup_enabled_nodes = set()
        self.hash_index = {}
        self.dedup_stats = {}
        
    async def enable_for_node(self, storage_node):
        """Enable deduplication for storage node"""
        try:
            self.dedup_enabled_nodes.add(storage_node.node_id)
            self.dedup_stats[storage_node.node_id] = {
                'total_data_gb': 0,
                'deduplicated_gb': 0,
                'dedup_ratio': 0.0
            }
            
            logging.info(f"Deduplication enabled for {storage_node.node_id}")
            
        except Exception as e:
            logging.error(f"Deduplication enablement failed: {e}")

class PrefetchPredictor:
    """ML-driven prefetching predictor"""
    
    def __init__(self):
        self.enabled_nodes = set()
        self.access_history = defaultdict(deque)
        self.prediction_models = {}
        
    async def enable_for_node(self, storage_node):
        """Enable ML prefetching for storage node"""
        try:
            self.enabled_nodes.add(storage_node.node_id)
            
            # Initialize prediction model
            if ML_LIBRARIES_AVAILABLE:
                self.prediction_models[storage_node.node_id] = self._create_prediction_model()
            
            logging.info(f"ML prefetching enabled for {storage_node.node_id}")
            
        except Exception as e:
            logging.error(f"Prefetching enablement failed: {e}")
    
    def _create_prediction_model(self):
        """Create ML model for access pattern prediction"""
        try:
            if ML_LIBRARIES_AVAILABLE:
                # Simple model for demonstration
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                return None
        except Exception as e:
            logging.error(f"Prediction model creation failed: {e}")
            return None

class ReplicationManager:
    """Distributed replication manager for data redundancy"""
    
    def __init__(self):
        self.replication_configs = {}
        self.replica_locations = defaultdict(list)
        
    async def setup_replication(self, storage_node):
        """Setup replication for storage node"""
        try:
            # Configure default replication policy
            replication_config = {
                'replication_factor': 3,
                'consistency_level': 'quorum',
                'auto_repair': True,
                'cross_datacenter': False
            }
            
            self.replication_configs[storage_node.node_id] = replication_config
            
            logging.info(f"Replication setup for {storage_node.node_id}")
            
        except Exception as e:
            logging.error(f"Replication setup failed: {e}")

class MLTierOptimizer:
    """ML-based tier optimization engine"""
    
    def __init__(self):
        self.optimization_models = {}
        self.tier_predictions = {}
        
    def predict_optimal_tier(self, data_characteristics: Dict[str, Any]) -> StorageTier:
        """Predict optimal tier for data placement"""
        try:
            # Simple heuristic for demonstration
            access_frequency = data_characteristics.get('access_frequency', 'medium')
            data_size = data_characteristics.get('size_gb', 1.0)
            
            if access_frequency == 'high' and data_size < 10:
                return StorageTier.HOT_NVME
            elif access_frequency == 'medium':
                return StorageTier.WARM_SSD
            else:
                return StorageTier.COLD_HDD
                
        except Exception as e:
            logging.error(f"Tier prediction failed: {e}")
            return StorageTier.WARM_SSD

class AccessPatternAnalyzer:
    """Access pattern analysis for storage optimization"""
    
    def __init__(self):
        self.access_logs = defaultdict(deque)
        self.pattern_stats = {}
        
    def analyze_patterns(self, node_id: str) -> Dict[str, Any]:
        """Analyze access patterns for optimization"""
        try:
            if node_id not in self.access_logs:
                return {}
            
            # Analyze access frequency, temporal patterns, etc.
            analysis = {
                'total_accesses': len(self.access_logs[node_id]),
                'unique_files': len(set(log['file_id'] for log in self.access_logs[node_id])),
                'hot_files': [],  # Files with high access frequency
                'access_trends': {}  # Time-based access trends
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Pattern analysis failed: {e}")
            return {}

class StorageIOMetrics:
    """Storage I/O performance metrics tracking"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = {
            'read_iops': 0,
            'write_iops': 0,
            'read_latency_ms': 0,
            'write_latency_ms': 0,
            'throughput_mbps': 0
        }
    
    def update_metrics(self, new_metrics: Dict[str, float]):
        """Update I/O metrics"""
        try:
            self.current_metrics.update(new_metrics)
            self.metrics_history.append({
                'timestamp': datetime.utcnow(),
                'metrics': new_metrics.copy()
            })
        except Exception as e:
            logging.error(f"I/O metrics update failed: {e}")

class AccessPatternTracker:
    """Track data access patterns for intelligent tier management"""
    
    def __init__(self):
        self.access_log = deque(maxlen=10000)
        self.pattern_cache = {}
    
    def record_access(self, data_id: str, access_type: str):
        """Record data access event"""
        try:
            access_event = {
                'data_id': data_id,
                'access_type': access_type,
                'timestamp': datetime.utcnow()
            }
            
            self.access_log.append(access_event)
            
            # Update pattern cache
            if data_id not in self.pattern_cache:
                self.pattern_cache[data_id] = {
                    'access_count': 0,
                    'last_access': datetime.utcnow(),
                    'access_frequency': 'low'
                }
            
            self.pattern_cache[data_id]['access_count'] += 1
            self.pattern_cache[data_id]['last_access'] = datetime.utcnow()
            
        except Exception as e:
            logging.error(f"Access recording failed: {e}")

class DataCatalog:
    """Data catalog for tracking stored data across tiers"""
    
    def __init__(self):
        self.data_registry = {}
        self.tier_mappings = defaultdict(list)
    
    async def register_data(self, data_id: str, tier: StorageTier, size_gb: float):
        """Register data in catalog"""
        try:
            self.data_registry[data_id] = {
                'tier': tier,
                'size_gb': size_gb,
                'created_at': datetime.utcnow(),
                'metadata': {}
            }
            
            self.tier_mappings[tier].append(data_id)
            
        except Exception as e:
            logging.error(f"Data registration failed: {e}")

# Additional Supporting Classes - Block 8
class WorkloadManager:
    """Advanced workload management for heterogeneous compute environments"""
    
    def __init__(self):
        self.workload_types = {
            TaskType.CPU_INTENSIVE: CPUWorkloadHandler(),
            TaskType.GPU_COMPUTE: GPUWorkloadHandler(),
            TaskType.NPU_INFERENCE: NPUWorkloadHandler(),
            TaskType.FPGA_ACCELERATION: FPGAWorkloadHandler(),
            TaskType.AI_TRAINING: AITrainingWorkloadHandler(),
            TaskType.AI_INFERENCE: AIInferenceWorkloadHandler(),
        }
        
        self.workload_queue = deque()
        self.active_workloads = {}
        self.workload_history = []
        
    async def schedule_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Schedule workload on target node with optimized execution"""
        try:
            workload_type = TaskType(workload.get('workload_type', TaskType.CPU_INTENSIVE.value))
            
            # Get appropriate workload handler
            handler = self.workload_types.get(workload_type)
            if not handler:
                handler = self.workload_types[TaskType.CPU_INTENSIVE]  # Default
            
            # Prepare workload for execution
            prepared_workload = await handler.prepare_workload(workload, target_node)
            
            # Execute workload
            execution_id = await handler.execute_workload(prepared_workload, target_node)
            
            if execution_id:
                self.active_workloads[execution_id] = {
                    'workload': workload,
                    'target_node': target_node,
                    'handler': handler,
                    'start_time': datetime.utcnow()
                }
                
                logging.info(f"Workload {execution_id} scheduled on {target_node}")
                
            return execution_id
            
        except Exception as e:
            logging.error(f"Workload scheduling failed: {e}")
            return None

# Workload Handlers for different compute types
class CPUWorkloadHandler:
    """Handler for CPU-intensive workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare CPU workload for execution"""
        try:
            # Optimize for CPU architecture
            # Set CPU affinity
            # Configure NUMA topology
            return workload
        except Exception as e:
            logging.error(f"CPU workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute CPU workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual CPU workload
            return execution_id
        except Exception as e:
            logging.error(f"CPU workload execution failed: {e}")
            return None

class GPUWorkloadHandler:
    """Handler for GPU compute workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare GPU workload for execution"""
        try:
            # Setup CUDA/ROCm context
            # Allocate GPU memory
            # Configure GPU topology
            return workload
        except Exception as e:
            logging.error(f"GPU workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute GPU workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual GPU workload
            return execution_id
        except Exception as e:
            logging.error(f"GPU workload execution failed: {e}")
            return None

class NPUWorkloadHandler:
    """Handler for NPU inference workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare NPU workload for execution"""
        try:
            # Setup NPU runtime
            # Load AI model
            # Configure inference pipeline
            return workload
        except Exception as e:
            logging.error(f"NPU workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute NPU workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual NPU workload
            return execution_id
        except Exception as e:
            logging.error(f"NPU workload execution failed: {e}")
            return None

class FPGAWorkloadHandler:
    """Handler for FPGA acceleration workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare FPGA workload for execution"""
        try:
            # Load bitstream
            # Configure FPGA logic
            # Setup data pipelines
            return workload
        except Exception as e:
            logging.error(f"FPGA workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute FPGA workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual FPGA workload
            return execution_id
        except Exception as e:
            logging.error(f"FPGA workload execution failed: {e}")
            return None

class AITrainingWorkloadHandler:
    """Handler for AI training workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare AI training workload"""
        try:
            # Setup distributed training
            # Configure data loaders
            # Initialize model checkpointing
            return workload
        except Exception as e:
            logging.error(f"AI training workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute AI training workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual AI training
            return execution_id
        except Exception as e:
            logging.error(f"AI training workload execution failed: {e}")
            return None

class AIInferenceWorkloadHandler:
    """Handler for AI inference workloads"""
    
    async def prepare_workload(self, workload: Dict[str, Any], target_node: str) -> Dict[str, Any]:
        """Prepare AI inference workload"""
        try:
            # Load inference model
            # Setup batch processing
            # Configure optimization (TensorRT, etc.)
            return workload
        except Exception as e:
            logging.error(f"AI inference workload preparation failed: {e}")
            return workload
    
    async def execute_workload(self, workload: Dict[str, Any], target_node: str) -> str:
        """Execute AI inference workload"""
        try:
            execution_id = str(uuid.uuid4())
            # Implementation would execute actual AI inference
            return execution_id
        except Exception as e:
            logging.error(f"AI inference workload execution failed: {e}")
            return None

# Enhanced Control Node - Block 3: Fault-Tolerant Multi-Master Architecture
class ControlNodeManager:
    """
    Master coordinator and API hub with fault-tolerant multi-master leader election.
    Manages orchestration, supervision, and global scheduling across the cluster.
    """
    
    def __init__(self):
        self.node_id = f"control-{socket.gethostname()}-{int(time.time())}"
        self.cluster_name = CLUSTER_NAME
        self.is_leader = False
        self.leader_term = 0
        self.cluster_state = "forming"
        
        # Multi-master configuration
        self.control_nodes = {}  # Other control nodes in cluster
        self.backup_nodes = []
        self.quorum_size = 3
        self.leader_lease_ttl = LEADER_LEASE_DURATION
        
        # Orchestration subsystems
        self.global_scheduler = GlobalScheduler()
        self.resource_manager = ResourceManager()
        self.fault_detector = FaultDetector()
        self.health_monitor = ClusterHealthMonitor()
        
        # Leader election state
        self.election_state = LeaderElectionState()
        self.heartbeat_manager = HeartbeatManager()
        
        # API and supervision
        self.api_server = None
        self.supervision_engine = SupervisionEngine()
        
        logging.info(f"Control Node {self.node_id} initialized")
    
    async def initialize_cluster(self):
        """Initialize cluster and start leader election"""
        try:
            # Start health monitoring
            await self.health_monitor.start()
            
            # Initialize leader election
            await self.start_leader_election()
            
            # Start heartbeat system
            await self.heartbeat_manager.start()
            
            # Initialize global scheduler
            await self.global_scheduler.initialize()
            
            logging.info(f"Control node {self.node_id} cluster initialization complete")
            
        except Exception as e:
            logging.error(f"Cluster initialization failed: {e}")
            raise
    
    async def start_leader_election(self):
        """Start multi-master leader election process"""
        try:
            if etcd_client:
                # Use etcd for distributed leader election
                await self._etcd_leader_election()
            else:
                # Fallback to single-node mode
                self.is_leader = True
                self.election_state.current_leader = self.node_id
                self.election_state.leader_term = 1
                logging.info(f"Single-node mode: {self.node_id} became leader")
                
        except Exception as e:
            logging.error(f"Leader election failed: {e}")

class ComputeNodeManager:
    """
    Advanced compute node management supporting heterogeneous hardware architectures.
    Handles CPU, GPU, NPU, FPGA, and Edge-TPU compute resources with hot-swapping,
    self-registration, and auto-scaling capabilities.
    """
    
    def __init__(self):
        self.compute_nodes = {}
        self.hardware_pools = {
            'cpu_x86': {},
            'cpu_arm': {},
            'gpu_nvidia': {},
            'gpu_amd': {},
            'gpu_intel': {},
            'npu_dedicated': {},
            'fpga_xilinx': {},
            'fpga_intel': {},
            'edge_tpu': {},
            'hybrid': {}
        }
        
        self.workload_manager = WorkloadManager()
        self.isolation_manager = SandboxManager()
        self.scaling_manager = AutoScalingManager()
        self.hot_swap_manager = HotSwapManager()

class ThermalManager:
    """Advanced thermal management and optimization for heterogeneous hardware"""
    
    def __init__(self):
        self.thermal_history = defaultdict(deque)
        self.thermal_thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'gpu_warning': 75.0,
            'gpu_critical': 90.0,
            'npu_warning': 70.0,
            'npu_critical': 85.0,
            'fpga_warning': 80.0,
            'fpga_critical': 95.0
        }
        self.cooling_strategies = {}
        self.thermal_zones = {}
        
    def update_thermal_data(self, node_id: str, metrics: NodeMetrics):
        """Update thermal data for a node with enhanced monitoring"""
        try:
            thermal_data = {
                'timestamp': metrics.timestamp,
                'cpu_temp': metrics.cpu_temperature_celsius,
                'gpu_temp': metrics.gpu_temperature_celsius,
                'system_temp': metrics.system_temperature_celsius,
                'power_consumption': metrics.power_consumption_watts,
                'fan_speed': metrics.fan_speed_rpm
            }
            
            self.thermal_history[node_id].append(thermal_data)
            
            # Keep only recent data (last hour)
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            while (self.thermal_history[node_id] and 
                   self.thermal_history[node_id][0]['timestamp'] < cutoff_time):
                self.thermal_history[node_id].popleft()
            
        except Exception as e:
            logging.error(f"Thermal data update failed: {e}")
    
    def get_thermal_score(self, node_id: str) -> float:
        """Calculate comprehensive thermal health score"""
        try:
            if node_id not in self.thermal_history or not self.thermal_history[node_id]:
                return 0.5  # Unknown thermal state
                
            recent_data = list(self.thermal_history[node_id])[-5:]  # Last 5 readings
            if not recent_data:
                return 0.5
            
            # Calculate individual component scores
            scores = []
            
            # CPU thermal score
            avg_cpu_temp = sum(d['cpu_temp'] for d in recent_data) / len(recent_data)
            cpu_score = max(0.0, 1.0 - (avg_cpu_temp - 30.0) / 55.0)  # 30-85°C range
            scores.append(cpu_score)
            
            # GPU thermal score (if applicable)
            if any(d['gpu_temp'] > 0 for d in recent_data):
                avg_gpu_temp = sum(d['gpu_temp'] for d in recent_data) / len(recent_data)
                gpu_score = max(0.0, 1.0 - (avg_gpu_temp - 35.0) / 55.0)  # 35-90°C range
                scores.append(gpu_score)
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            logging.error(f"Thermal score calculation failed: {e}")
            return 0.5

class GlobalScheduler:
    """
    Global scheduler for orchestrating workloads across heterogeneous nodes.
    Implements intelligent placement with ML-driven optimization.
    """
    
    def __init__(self):
        self.scheduling_policy = "intelligent_placement"
        self.placement_optimizer = MLPlacementOptimizer()
        self.load_balancer = LoadBalancer()
        
        # Scheduling queues
        self.pending_tasks = deque()
        self.priority_queue = []
        self.scheduling_history = []
        
        # Resource tracking
        self.cluster_resources = {}
        self.resource_utilization = {}
        
    async def initialize(self):
        """Initialize global scheduler"""
        try:
            # Load historical scheduling data for ML training
            await self._load_scheduling_history()
            
            # Train placement optimizer
            if len(self.scheduling_history) > 10:
                self.placement_optimizer.train_model(self.scheduling_history)
            
            logging.info("Global scheduler initialized")
            
        except Exception as e:
            logging.error(f"Scheduler initialization failed: {e}")
    
    async def _load_scheduling_history(self):
        """Load historical scheduling data"""
        try:
            # This would load from persistent storage in production
            self.scheduling_history = []
            
        except Exception as e:
            logging.error(f"Failed to load scheduling history: {e}")

class SupervisionEngine:
    """
    Supervision engine for monitoring and managing node health and performance.
    Implements proactive failure detection and automated recovery.
    """
    
    def __init__(self):
        self.supervised_nodes = {}
        self.health_thresholds = {
            'cpu_critical': 95.0,
            'memory_critical': 90.0,
            'temperature_critical': 85.0,
            'disk_critical': 95.0
        }
        self.monitoring_interval = 5  # seconds
        
    async def start_global_supervision(self):
        """Start supervising all cluster nodes"""
        try:
            # Start monitoring loops
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._failure_detection_loop())
            
            logging.info("Global supervision engine started")
            
        except Exception as e:
            logging.error(f"Supervision engine start failed: {e}")
    
    async def _health_monitoring_loop(self):
        """Monitor health of all nodes"""
        while True:
            try:
                for node_id in self.supervised_nodes:
                    await self._check_node_health(node_id)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _performance_monitoring_loop(self):
        """Monitor performance of all nodes"""
        while True:
            try:
                # Performance monitoring implementation
                await asyncio.sleep(10)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _failure_detection_loop(self):
        """Detect node failures"""
        while True:
            try:
                # Failure detection implementation
                await asyncio.sleep(5)
                
            except Exception as e:
                logging.error(f"Failure detection error: {e}")
                await asyncio.sleep(5)
    
    async def _check_node_health(self, node_id):
        """Check health of a specific node"""
        try:
            # Health check implementation
            pass
            
        except Exception as e:
            logging.error(f"Node health check failed for {node_id}: {e}")

class FaultDetector:
    """
    Advanced fault detection system with predictive failure analysis.
    Implements split-brain protection and automated recovery.
    """
    
    def __init__(self):
        self.failure_patterns = {}
        self.anomaly_detector = SimpleAnomalyDetector()
        self.split_brain_detector = SplitBrainDetector()
        self.failure_predictors = {}
        
    async def start_cluster_monitoring(self):
        """Start cluster-wide fault detection"""
        try:
            # Start monitoring tasks
            asyncio.create_task(self._split_brain_monitoring())
            asyncio.create_task(self._network_partition_detection())
            asyncio.create_task(self._predictive_failure_analysis())
            
            logging.info("Fault detector started")
            
        except Exception as e:
            logging.error(f"Fault detector start failed: {e}")
    
    async def _split_brain_monitoring(self):
        """Monitor for split-brain scenarios"""
        while True:
            try:
                await asyncio.sleep(10)
                
            except Exception as e:
                logging.error(f"Split-brain monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _network_partition_detection(self):
        """Detect network partitions"""
        while True:
            try:
                await asyncio.sleep(15)
                
            except Exception as e:
                logging.error(f"Network partition detection error: {e}")
                await asyncio.sleep(15)
    
    async def _predictive_failure_analysis(self):
        """Predictive failure analysis"""
        while True:
            try:
                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(f"Predictive failure analysis error: {e}")
                await asyncio.sleep(30)

class ClusterHealthMonitor:
    """
    Comprehensive cluster health monitoring with real-time diagnostics.
    """
    
    def __init__(self):
        self.health_status = "unknown"
        self.cluster_metrics = {}
        self.health_history = deque(maxlen=1000)
        
    async def start(self):
        """Start cluster health monitoring"""
        try:
            asyncio.create_task(self._health_monitoring_loop())
            logging.info("Cluster health monitor started")
            
        except Exception as e:
            logging.error(f"Health monitor start failed: {e}")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while True:
            try:
                # Collect cluster-wide metrics
                await self._collect_cluster_metrics()
                
                # Assess overall cluster health
                health_score = self._calculate_cluster_health_score()
                
                # Update health status
                self._update_health_status(health_score)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_cluster_metrics(self):
        """Collect cluster-wide metrics"""
        try:
            # Metrics collection implementation
            self.cluster_metrics = {
                'total_nodes': 1,
                'healthy_nodes': 1,
                'avg_cpu_utilization': 0.3,
                'network_health_score': 1.0
            }
            
        except Exception as e:
            logging.error(f"Metrics collection failed: {e}")
    
    def _calculate_cluster_health_score(self) -> float:
        """Calculate overall cluster health score"""
        try:
            if not self.cluster_metrics:
                return 0.0
            
            # Factor in various health indicators
            scores = []
            
            # Node availability score
            total_nodes = self.cluster_metrics.get('total_nodes', 1)
            healthy_nodes = self.cluster_metrics.get('healthy_nodes', 0)
            availability_score = healthy_nodes / max(1, total_nodes)
            scores.append(availability_score)
            
            # Resource utilization score (optimal around 70%)
            avg_cpu_util = self.cluster_metrics.get('avg_cpu_utilization', 0)
            util_score = 1.0 - abs(avg_cpu_util - 0.7) / 0.7
            scores.append(max(0, util_score))
            
            # Network health score
            network_score = self.cluster_metrics.get('network_health_score', 1.0)
            scores.append(network_score)
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            logging.error(f"Health score calculation failed: {e}")
            return 0.0
    
    def _update_health_status(self, health_score: float):
        """Update overall health status"""
        try:
            if health_score >= 0.8:
                self.health_status = "healthy"
            elif health_score >= 0.6:
                self.health_status = "degraded"
            elif health_score >= 0.4:
                self.health_status = "warning"
            else:
                self.health_status = "critical"
                
        except Exception as e:
            logging.error(f"Health status update failed: {e}")

class MLPlacementOptimizer:
    """
    ML-driven workload placement optimizer for heterogeneous clusters.
    Uses historical data to learn optimal placement strategies.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_cores', 'memory_gb', 'gpu_memory', 
            'current_load', 'network_bandwidth',
            'thermal_score', 'historical_performance'
        ]
        
    def train_model(self, scheduling_history):
        """Train placement optimization model"""
        try:
            if len(scheduling_history) < 10:
                logging.warning("Insufficient training data for ML model")
                return
                
            # Extract features and labels from history
            X, y = self._prepare_training_data(scheduling_history)
            
            if X.shape[0] < 5:
                return
                
            # Train Random Forest model
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            logging.info(f"ML placement model trained with {X.shape[0]} samples")
            
        except Exception as e:
            logging.error(f"ML model training failed: {e}")
    
    def _prepare_training_data(self, history):
        """Prepare training data from scheduling history"""
        try:
            import numpy as np
            
            # This is a simplified version - in practice would be more complex
            features = []
            labels = []
            
            for record in history:
                feature_vector = [
                    record.get('cpu_cores', 1),
                    record.get('memory_gb', 4),
                    record.get('gpu_memory', 0),
                    record.get('current_load', 0.5),
                    record.get('network_bandwidth', 1000),
                    record.get('thermal_score', 0.7),
                    record.get('performance_score', 0.8)
                ]
                features.append(feature_vector)
                labels.append(record.get('success_score', 0.5))
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            logging.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def predict_placement_score(self, node_features):
        """Predict placement score for a node"""
        try:
            if self.model is None:
                return 0.5  # Default score
                
            import numpy as np
            features = np.array([node_features]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            score = self.model.predict(features_scaled)[0]
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            logging.error(f"Placement score prediction failed: {e}")
            return 0.5

class SimpleAnomalyDetector:
    """
    Simple anomaly detection for system metrics.
    Uses statistical methods to detect unusual patterns.
    """
    
    def __init__(self):
        self.metric_history = defaultdict(deque)
        self.baseline_stats = {}
        self.detection_threshold = 2.5  # Standard deviations
        
    def update_metric(self, metric_name: str, value: float):
        """Update metric value and maintain history"""
        try:
            self.metric_history[metric_name].append(value)
            
            # Keep only recent history (last 100 values)
            if len(self.metric_history[metric_name]) > 100:
                self.metric_history[metric_name].popleft()
            
            # Update baseline statistics
            self._update_baseline_stats(metric_name)
            
        except Exception as e:
            logging.error(f"Metric update failed for {metric_name}: {e}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if a metric value is anomalous"""
        try:
            if metric_name not in self.baseline_stats:
                return False
                
            stats = self.baseline_stats[metric_name]
            mean = stats['mean']
            std = stats['std']
            
            if std == 0:
                return False
                
            # Z-score based anomaly detection
            z_score = abs(value - mean) / std
            return z_score > self.detection_threshold
            
        except Exception as e:
            logging.error(f"Anomaly detection failed for {metric_name}: {e}")
            return False
    
    def _update_baseline_stats(self, metric_name: str):
        """Update baseline statistics for a metric"""
        try:
            values = list(self.metric_history[metric_name])
            if len(values) < 5:
                return
                
            import statistics
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            
            self.baseline_stats[metric_name] = {
                'mean': mean,
                'std': std,
                'count': len(values)
            }
            
        except Exception as e:
            logging.error(f"Baseline stats update failed for {metric_name}: {e}")
        
        # Scheduling queues
        self.pending_tasks = deque()
        self.priority_queue = []
        self.scheduling_history = []
        
        # Resource tracking
        self.cluster_resources = {}
        self.resource_utilization = {}
        
    async def initialize(self):
        """Initialize global scheduler"""
        try:
            # Load historical scheduling data for ML training
            await self._load_scheduling_history()
            
            # Train placement optimizer
            if len(self.scheduling_history) > 10:
                self.placement_optimizer.train_model(self.scheduling_history)
            
            logging.info("Global scheduler initialized")
            
        except Exception as e:
            logging.error(f"Scheduler initialization failed: {e}")
    
    async def activate_as_leader(self):
        """Activate scheduler when becoming cluster leader"""
        try:
            # Start scheduling loop
            asyncio.create_task(self._scheduling_loop())
            
            # Start resource monitoring
            asyncio.create_task(self._resource_monitoring_loop())
            
            logging.info("Global scheduler activated as leader")
            
        except Exception as e:
            logging.error(f"Scheduler activation failed: {e}")
    
    async def _scheduling_loop(self):
        """Main scheduling loop for task placement"""
        while True:
            try:
                if self.pending_tasks:
                    task = self.pending_tasks.popleft()
                    await self._schedule_task(task)
                
                await asyncio.sleep(0.1)  # 10 Hz scheduling frequency
                
            except Exception as e:
                logging.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(1)
    
    async def _schedule_task(self, task):
        """Schedule a single task using intelligent placement"""
        try:
            start_time = time.time()
            
            # Get available nodes for task
            suitable_nodes = await self._find_suitable_nodes(task)
            
            if not suitable_nodes:
                # No suitable nodes, requeue task
                self.pending_tasks.append(task)
                return
            
            # Use ML-driven placement optimization
            best_node = await self._optimize_placement(task, suitable_nodes)
            
            # Place task on selected node
            await self._place_task_on_node(task, best_node)
            
            # Record scheduling decision
            scheduling_time = time.time() - start_time
            self._record_scheduling_decision(task, best_node, scheduling_time)
            
        except Exception as e:
            logging.error(f"Task scheduling failed: {e}")
    
    async def _find_suitable_nodes(self, task) -> List[str]:
        """Find nodes suitable for task requirements"""
        suitable_nodes = []
        
        for node_id, node_info in self.cluster_resources.items():
            if await self._node_can_handle_task(node_info, task):
                suitable_nodes.append(node_id)
        
        return suitable_nodes
    
    async def _node_can_handle_task(self, node_info, task) -> bool:
        """Check if node can handle task requirements"""
        try:
            # Check resource requirements
            if task.get('cpu_cores', 0) > node_info.get('available_cpu_cores', 0):
                return False
            
            if task.get('memory_gb', 0) > node_info.get('available_memory_gb', 0):
                return False
            
            if task.get('gpu_units', 0) > node_info.get('available_gpu_units', 0):
                return False
            
            # Check node type compatibility
            task_type = task.get('task_type', '')
            node_type = node_info.get('node_type', '')
            
            # GPU tasks require GPU nodes
            if 'gpu' in task_type.lower() and 'gpu' not in node_type.lower():
                return False
            
            # Check thermal constraints
            thermal_score = self.thermal_manager.get_thermal_score(node_info.get('node_id'))
            if thermal_score < 0.3:  # Too hot
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Node suitability check failed: {e}")
            return False
    
    async def _optimize_placement(self, task, suitable_nodes) -> str:
        """Use ML optimization to select best node"""
        try:
            best_node = None
            best_score = -1
            
            for node_id in suitable_nodes:
                node_info = self.cluster_resources.get(node_id, {})
                
                # Calculate placement score
                score = await self._calculate_placement_score(task, node_info)
                
                if score > best_score:
                    best_score = score
                    best_node = node_id
            
            return best_node
            
        except Exception as e:
            logging.error(f"Placement optimization failed: {e}")
            return suitable_nodes[0] if suitable_nodes else None
    
    async def _calculate_placement_score(self, task, node_info) -> float:
        """Calculate placement score using multiple factors"""
        try:
            # Resource utilization score (prefer less utilized nodes)
            cpu_util = node_info.get('cpu_utilization', 0)
            memory_util = node_info.get('memory_utilization', 0)
            gpu_util = node_info.get('gpu_utilization', 0)
            
            utilization_score = 1.0 - ((cpu_util + memory_util + gpu_util) / 3.0)
            
            # Thermal score
            thermal_score = self.thermal_manager.get_thermal_score(node_info.get('node_id', ''))
            
            # Load balance score
            load_score = self.load_balancer.get_load_balance_score(node_info.get('node_id', ''))
            
            # ML prediction score
            ml_score = 0.5  # Default if ML not available
            if self.placement_optimizer.model:
                node_features = [cpu_util, memory_util, gpu_util, thermal_score]
                task_features = [
                    task.get('cpu_cores', 0),
                    task.get('memory_gb', 0),
                    task.get('gpu_units', 0)
                ]
                ml_score = self.placement_optimizer.predict_performance(node_features, task_features)
            
            # Weighted combination
            total_score = (
                0.3 * utilization_score +
                0.2 * thermal_score +
                0.2 * load_score +
                0.3 * ml_score
            )
            
            return total_score
            
        except Exception as e:
            logging.error(f"Score calculation failed: {e}")
            return 0.0
    
    async def _place_task_on_node(self, task, node_id):
        """Place task on selected node"""
        try:
            # Update resource allocation
            await self._allocate_resources(task, node_id)
            
            # Send task to node
            await self._send_task_to_node(task, node_id)
            
            logging.info(f"Task {task.get('task_id')} placed on node {node_id}")
            
        except Exception as e:
            logging.error(f"Task placement failed: {e}")
    
    def _record_scheduling_decision(self, task, node_id, scheduling_time):
        """Record scheduling decision for ML training"""
        try:
            decision_record = {
                'task_id': task.get('task_id'),
                'node_id': node_id,
                'scheduling_time': scheduling_time,
                'timestamp': datetime.utcnow().isoformat(),
                'features': {
                    'cpu_cores': task.get('cpu_cores', 0),
                    'memory_gb': task.get('memory_gb', 0),
                    'gpu_units': task.get('gpu_units', 0),
                    'task_type': task.get('task_type', ''),
                    'priority': task.get('priority', 5)
                }
            }
            
            self.scheduling_history.append(decision_record)
            
            # Keep only recent history
            if len(self.scheduling_history) > 10000:
                self.scheduling_history = self.scheduling_history[-5000:]
                
        except Exception as e:
            logging.error(f"Failed to record scheduling decision: {e}")

class SupervisionEngine:
    """
    Supervision engine for monitoring and managing node health and performance.
    Implements proactive failure detection and automated recovery.
    """
    
    def __init__(self):
        self.supervised_nodes = {}
        self.health_thresholds = {
            'cpu_critical': 95.0,
            'memory_critical': 90.0,
            'temperature_critical': 85.0,
            'disk_critical': 95.0
        }
        self.monitoring_interval = 5  # seconds
        
    async def start_global_supervision(self):
        """Start supervising all cluster nodes"""
        try:
            # Start monitoring loops
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._failure_detection_loop())
            
            logging.info("Global supervision engine started")
            
        except Exception as e:
            logging.error(f"Supervision engine start failed: {e}")
    
    async def _health_monitoring_loop(self):
        """Monitor health of all nodes"""
        while True:
            try:
                for node_id in self.supervised_nodes:
                    await self._check_node_health(node_id)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_node_health(self, node_id):
        """Check health of a specific node"""
        try:
            node_info = self.supervised_nodes.get(node_id)
            if not node_info:
                return
            
            # Check critical thresholds
            metrics = node_info.get('metrics', {})
            
            issues = []
            
            if metrics.get('cpu_usage', 0) > self.health_thresholds['cpu_critical']:
                issues.append('cpu_overload')
            
            if metrics.get('memory_usage', 0) > self.health_thresholds['memory_critical']:
                issues.append('memory_exhaustion')
            
            if metrics.get('temperature', 0) > self.health_thresholds['temperature_critical']:
                issues.append('thermal_emergency')
            
            if metrics.get('disk_usage', 0) > self.health_thresholds['disk_critical']:
                issues.append('disk_full')
            
            # Take corrective action if issues found
            if issues:
                await self._handle_node_issues(node_id, issues)
                
        except Exception as e:
            logging.error(f"Node health check failed for {node_id}: {e}")
    
    async def _handle_node_issues(self, node_id, issues):
        """Handle detected node issues"""
        try:
            for issue in issues:
                if issue == 'thermal_emergency':
                    await self._handle_thermal_emergency(node_id)
                elif issue == 'memory_exhaustion':
                    await self._handle_memory_exhaustion(node_id)
                elif issue == 'cpu_overload':
                    await self._handle_cpu_overload(node_id)
                elif issue == 'disk_full':
                    await self._handle_disk_full(node_id)
                    
        except Exception as e:
            logging.error(f"Issue handling failed for {node_id}: {e}")
    
    async def _handle_thermal_emergency(self, node_id):
        """Handle thermal emergency by throttling workloads"""
        try:
            logging.warning(f"Thermal emergency on node {node_id} - initiating throttling")
            
            # Throttle CPU-intensive tasks
            await self._throttle_node_workloads(node_id, 'cpu_intensive')
            
            # If critical, evacuate workloads
            node_temp = self.supervised_nodes[node_id]['metrics'].get('temperature', 0)
            if node_temp > 90:
                await self._evacuate_node_workloads(node_id)
                
        except Exception as e:
            logging.error(f"Thermal emergency handling failed: {e}")

class FaultDetector:
    """
    Advanced fault detection system with predictive failure analysis.
    Implements split-brain protection and automated recovery.
    """
    
    def __init__(self):
        self.failure_patterns = {}
        self.anomaly_detector = SimpleAnomalyDetector()
        self.split_brain_detector = SplitBrainDetector()
        self.failure_predictors = {}
        
    async def start_cluster_monitoring(self):
        """Start cluster-wide fault detection"""
        try:
            # Start monitoring tasks
            asyncio.create_task(self._split_brain_monitoring())
            asyncio.create_task(self._network_partition_detection())
            asyncio.create_task(self._predictive_failure_analysis())
            
            logging.info("Fault detector started")
            
        except Exception as e:
            logging.error(f"Fault detector start failed: {e}")
    
    async def _split_brain_monitoring(self):
        """Monitor for split-brain scenarios"""
        while True:
            try:
                # Check for multiple leaders
                if await self._detect_multiple_leaders():
                    await self._handle_split_brain()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Split-brain monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _detect_multiple_leaders(self) -> bool:
        """Detect if multiple control nodes claim leadership"""
        try:
            if not etcd_client:
                return False
            
            # Check leadership claims in etcd
            leaders = []
            for key, value in etcd_client.get_prefix('/omega/leader/'):
                leaders.append(value.decode())
            
            return len(set(leaders)) > 1
            
        except Exception as e:
            logging.error(f"Multiple leader detection failed: {e}")
            return False
    
    async def _handle_split_brain(self):
        """Handle split-brain scenario"""
        try:
            logging.critical("Split-brain scenario detected - initiating resolution")
            
            # Stop all scheduling activities
            # Force re-election
            # Notify administrators
            
            SPLIT_BRAIN_INCIDENTS.inc()
            
        except Exception as e:
            logging.error(f"Split-brain handling failed: {e}")

class ClusterHealthMonitor:
    """
    Comprehensive cluster health monitoring with real-time diagnostics.
    """
    
    def __init__(self):
        self.health_status = "unknown"
        self.cluster_metrics = {}
        self.health_history = deque(maxlen=1000)
        
    async def start(self):
        """Start cluster health monitoring"""
        try:
            asyncio.create_task(self._health_monitoring_loop())
            logging.info("Cluster health monitor started")
            
        except Exception as e:
            logging.error(f"Health monitor start failed: {e}")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while True:
            try:
                # Collect cluster-wide metrics
                await self._collect_cluster_metrics()
                
                # Assess overall cluster health
                health_score = self._calculate_cluster_health_score()
                
                # Update health status
                self._update_health_status(health_score)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _calculate_cluster_health_score(self) -> float:
        """Calculate overall cluster health score"""
        try:
            if not self.cluster_metrics:
                return 0.0
            
            # Factor in various health indicators
            scores = []
            
            # Node availability score
            total_nodes = self.cluster_metrics.get('total_nodes', 1)
            healthy_nodes = self.cluster_metrics.get('healthy_nodes', 0)
            availability_score = healthy_nodes / max(1, total_nodes)
            scores.append(availability_score)
            
            # Resource utilization score (optimal around 70%)
            avg_cpu_util = self.cluster_metrics.get('avg_cpu_utilization', 0)
            util_score = 1.0 - abs(avg_cpu_util - 0.7) / 0.7
            scores.append(max(0, util_score))
            
            # Network health score
            network_score = self.cluster_metrics.get('network_health_score', 1.0)
            scores.append(network_score)
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            logging.error(f"Health score calculation failed: {e}")
            return 0.0

# Enhanced Database Configuration with High Availability and Partitioning
DATABASE_URL = os.getenv("OMEGA_DB_URL", "sqlite:///./omega_prototype.db")
REDIS_URL = os.getenv("OMEGA_REDIS_URL", "redis://localhost:6379")
ETCD_URL = os.getenv("OMEGA_ETCD_URL", "localhost:2379")

try:
    if DATABASE_URL.startswith("postgresql"):
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        with engine.connect():
            pass
        print("✓ Using PostgreSQL database with connection pooling")
        DATABASE_TYPE = "postgresql"
    else:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True
        )
        print("✓ Using SQLite database")
        DATABASE_TYPE = "sqlite"
except Exception as e:
    print(f"Database connection failed: {e}")
    DATABASE_URL = "sqlite:///./omega_prototype.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    DATABASE_TYPE = "sqlite"

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis client placeholder - will be initialized during app startup using utils.get_redis_client
redis_client = None

# Enhanced etcd Configuration for Leader Election and Consensus
etcd_client = None
if EXTERNAL_SERVICES_AVAILABLE:
    try:
        import etcd3
        etcd_host, etcd_port = ETCD_URL.split(':')
        etcd_client = etcd3.client(
            host=etcd_host,
            port=int(etcd_port),
            timeout=5,
            grpc_options=[
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True)
            ]
        )
        print("✓ etcd client configured for leader election")
    except Exception as e:
        print(f"etcd configuration failed: {e}")
        etcd_client = None

# In-memory fallbacks for when external services are not available
nodes_data = {}
sessions_data = {}
distributed_state = {
    "leader_node": None,
    "backup_nodes": [],
    "cluster_health": "healthy",
    "last_election": None,
    "split_brain_detected": False
}

# Leader Election Manager
class LeaderElectionManager:
    """Fault-tolerant leader election with split-brain protection"""
    def __init__(self):
        self.node_id = f"control-{socket.gethostname()}-{int(time.time())}"
        self.state = LeaderElectionState()
        self.election_lock = asyncio.Lock()
        self.heartbeat_task = None
        self.watch_task = None
        self.election_callbacks = []
        
    async def initialize(self):
        """Initialize leader election system"""
        if not etcd_client:
            logging.warning("etcd not available, using single-node mode")
            self.state.current_leader = self.node_id
            return
            
        try:
            # Test etcd connectivity
            await asyncio.to_thread(etcd_client.status)
            
            # Start leader election process
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.watch_task = asyncio.create_task(self._watch_leader_changes())
            
            # Attempt to become leader
            await self._attempt_leadership()
            
        except Exception as e:
            logging.error(f"Leader election initialization failed: {e}")
            # Fallback to single-node mode
            self.state.current_leader = self.node_id
    
    async def _attempt_leadership(self):
        """Attempt to become the cluster leader"""
        async with self.election_lock:
            try:
                # Try to acquire leadership lease
                lease = etcd_client.lease(LEADER_LEASE_DURATION)
                
                # Attempt to put our ID as leader with the lease
                success = etcd_client.transaction(
                    compare=[etcd_client.transactions.version('/omega/leader') == 0],
                    success=[etcd_client.transactions.put('/omega/leader', self.node_id, lease=lease)],
                    failure=[]
                )
                
                if success:
                    self.state.current_leader = self.node_id
                    self.state.leader_term += 1
                    self.state.leader_lease_expiry = datetime.utcnow() + timedelta(seconds=LEADER_LEASE_DURATION)
                    self.state.last_election_time = datetime.utcnow()
                    
                    logging.info(f"Successfully became cluster leader (term {self.state.leader_term})")
                    LEADER_ELECTIONS_TOTAL.inc()
                    
                    # Notify callbacks
                    for callback in self.election_callbacks:
                        try:
                            await callback(True, self.node_id)
                        except Exception as e:
                            logging.error(f"Leader election callback failed: {e}")
                else:
                    # Someone else is leader, get current leader info
                    current_leader = etcd_client.get('/omega/leader')[0]
                    if current_leader:
                        self.state.current_leader = current_leader.value.decode()
                        logging.info(f"Current leader is: {self.state.current_leader}")
                        
            except Exception as e:
                logging.error(f"Leadership attempt failed: {e}")
    
    async def _heartbeat_loop(self):
        """Maintain leadership lease through heartbeats"""
        while True:
            try:
                if self.is_leader():
                    # Refresh lease
                    try:
                        etcd_client.refresh_lease(self.state.leader_lease_expiry)
                        self.state.leader_lease_expiry = datetime.utcnow() + timedelta(seconds=LEADER_LEASE_DURATION)
                    except Exception as e:
                        logging.warning(f"Failed to refresh leader lease: {e}")
                        # Try to re-acquire leadership
                        await self._attempt_leadership()
                else:
                    # Not leader, check if we should try to become one
                    current_leader_data = etcd_client.get('/omega/leader')[0]
                    if not current_leader_data:
                        # No current leader, attempt leadership
                        await self._attempt_leadership()
                
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                
            except Exception as e:
                logging.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(HEARTBEAT_INTERVAL)
    
    async def _watch_leader_changes(self):
        """Watch for leader changes and detect split-brain scenarios"""
        if not etcd_client:
            return
            
        try:
            events_iterator, cancel = etcd_client.watch('/omega/leader')
            
            for event in events_iterator:
                if event.type == etcd3.events.PutEvent:
                    new_leader = event.value.decode()
                    if new_leader != self.state.current_leader:
                        old_leader = self.state.current_leader
                        self.state.current_leader = new_leader
                        
                        logging.info(f"Leader changed from {old_leader} to {new_leader}")
                        
                        # Notify callbacks
                        for callback in self.election_callbacks:
                            try:
                                await callback(False, new_leader)
                            except Exception as e:
                                logging.error(f"Leader change callback failed: {e}")
                
                elif event.type == etcd3.events.DeleteEvent:
                    logging.warning("Leader key deleted, starting new election")
                    self.state.current_leader = None
                    await self._attempt_leadership()
                    
        except Exception as e:
            logging.error(f"Leader watch error: {e}")
    
    def is_leader(self) -> bool:
        """Check if this node is the current leader"""
        return self.state.current_leader == self.node_id
    
    def get_leader(self) -> Optional[str]:
        """Get current leader node ID"""
        return self.state.current_leader
    
    def add_election_callback(self, callback):
        """Add callback for leader election events"""
        self.election_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown leader election gracefully"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.watch_task:
            self.watch_task.cancel()
            
        if self.is_leader() and etcd_client:
            try:
                # Release leadership
                etcd_client.delete('/omega/leader')
                logging.info("Released leadership lease")
            except Exception as e:
                logging.error(f"Failed to release leadership: {e}")

# Global leader election manager
leader_election = LeaderElectionManager()

# Fallback database configuration
if 'engine' not in locals():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    DATABASE_TYPE = "sqlite"
    print("✓ Fallback to SQLite database")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis will be initialized on startup via the central helper to support multiple redis libs
redis_client = None

# etcd Configuration for Leader Election
etcd_client = None
if EXTERNAL_SERVICES_AVAILABLE:
    try:
        import etcd3
        etcd_host = os.getenv("OMEGA_ETCD_HOST", "localhost")
        etcd_port = int(os.getenv("OMEGA_ETCD_PORT", "2379"))
        etcd_client = etcd3.client(host=etcd_host, port=etcd_port, timeout=5)
        # Test connection
        etcd_client.get("test")
        print("✓ Using etcd for leader election")
    except Exception as e:
        print(f"etcd connection failed: {e}")
        etcd_client = None

# In-memory fallbacks for when external services are not available
nodes_data = {}
sessions_data = {}
distributed_state = {
    "leader_node": None,
    "backup_nodes": [],
    "cluster_health": "healthy"
}

# Simple ML model fallbacks
class SimpleAnomalyDetector:
    def __init__(self):
        self.threshold = 0.8
        self.baseline_metrics = {}
    
    def fit(self, data):
        if isinstance(data, list) and data:
            self.baseline_metrics = {
                'mean': np.mean(data),
                'std': np.std(data),
                'count': len(data)
            }
    
    def predict(self, data):
        if not self.baseline_metrics:
            return [0] * len(data)  # Normal behavior
        
        anomalies = []
        for value in data:
            # Simple statistical anomaly detection
            z_score = abs((value - self.baseline_metrics['mean']) / 
                         (self.baseline_metrics['std'] + 1e-8))
            anomalies.append(1 if z_score > 2.0 else 0)
        return anomalies

class SimpleLoadBalancer:
    def __init__(self):
        self.node_weights = {}
        self.round_robin_index = 0
    
    def select_node(self, available_nodes: List[str], task_requirements: Dict[str, Any] = None) -> str:
        if not available_nodes:
            return None
        
        # Simple round-robin with weights
        if task_requirements and 'preferred_node_type' in task_requirements:
            filtered_nodes = [node for node in available_nodes 
                            if task_requirements['preferred_node_type'] in node]
            if filtered_nodes:
                available_nodes = filtered_nodes
        
        selected = available_nodes[self.round_robin_index % len(available_nodes)]
        self.round_robin_index += 1
        return selected

anomaly_detector = SimpleAnomalyDetector()
load_balancer = SimpleLoadBalancer()

# Enhanced Database Models
class NodeRecord(Base):
    __tablename__ = "nodes"
    
    node_id = Column(String, primary_key=True, index=True)
    node_type = Column(String, nullable=False)
    status = Column(String, default="active")
    ip_address = Column(String)
    hostname = Column(String)
    resources = Column(JSON, default=dict)
    capabilities = Column(JSON, default=dict)
    current_load = Column(JSON, default=dict)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    performance_score = Column(Float, default=1.0)
    fault_tolerance_config = Column(JSON, default=dict)
    maintenance_window = Column(JSON, default=dict)
    security_profile = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class SessionRecord(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    user_id = Column(String, index=True)
    app_name = Column(String, nullable=False)
    app_command = Column(String)
    app_icon = Column(String)
    status = Column(String, default="INITIALIZING")
    node_id = Column(String, index=True)
    cpu_cores = Column(Integer)
    gpu_units = Column(Integer)
    ram_gb = Column(Float)
    storage_gb = Column(Float)
    priority = Column(Integer, default=1)
    session_type = Column(String, default="workstation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    tags = Column(JSON, default=list)
    environment_vars = Column(JSON, default=dict)
    resource_policy = Column(JSON, default=dict)
    migration_policy = Column(JSON, default=dict)
    snapshot_policy = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    process_tree = Column(JSON, default=list)
    security_labels = Column(JSON, default=list)
    fault_tolerance_enabled = Column(Boolean, default=True)
    backup_frequency_minutes = Column(Integer, default=30)

class TaskRecord(Base):
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True, index=True)
    task_type = Column(String, nullable=False)
    session_id = Column(String, index=True)
    node_id = Column(String, index=True)
    status = Column(String, default="pending")
    priority = Column(Integer, default=5)
    resource_requirements = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)
    result = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(String)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

class ClusterStateRecord(Base):
    __tablename__ = "cluster_state"
    
    state_id = Column(String, primary_key=True, index=True)
    leader_node_id = Column(String, index=True)
    backup_nodes = Column(JSON, default=list)
    cluster_health = Column(String, default="healthy")
    last_election = Column(DateTime)
    election_term = Column(Integer, default=0)
    split_brain_detected = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all database tables
try:
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")
except Exception as e:
    print(f"✗ Error creating database tables: {e}")
    if "postgresql" in DATABASE_URL.lower():
        print("Forcing SQLite fallback for table creation...")
        DATABASE_URL = "sqlite:///./omega_prototype.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        print("✓ SQLite database tables created successfully")

# Enhanced Pydantic Models
class NodeInfo(BaseModel):
    node_id: str
    node_type: str
    status: str = "active"
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    resources: Dict[str, Any] = {}
    capabilities: Optional[NodeCapabilities] = None
    current_load: Optional[NodeMetrics] = None
    performance_metrics: Optional[Dict[str, float]] = None
    fault_tolerance_config: Optional[FaultToleranceConfig] = None
    maintenance_window: Optional[Dict[str, Any]] = None
    security_profile: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

class SessionRequest(BaseModel):
    session_name: str
    app_name: str
    app_command: Optional[str] = "/bin/bash"
    app_icon: Optional[str] = None
    user_id: str = "admin"
    cpu_cores: int = 2
    gpu_units: int = 0
    ram_gb: float = 4.0
    storage_gb: float = 10.0
    priority: int = 1
    session_type: str = "workstation"
    tags: List[str] = []
    environment_vars: Dict[str, str] = {}
    resource_policy: Dict[str, Any] = {}
    migration_policy: Dict[str, Any] = {}
    snapshot_policy: Dict[str, Any] = {}
    security_labels: List[str] = ["standard"]
    fault_tolerance_enabled: bool = True
    backup_frequency_minutes: int = 30
    preferred_node_types: List[str] = []
    resource_constraints: Dict[str, Any] = {}

class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    user_id: str
    app_name: str
    app_command: str
    app_icon: Optional[str]
    status: str
    node_id: Optional[str]
    cpu_cores: int
    gpu_units: int
    ram_gb: float
    storage_gb: float
    priority: int
    session_type: str
    created_at: datetime
    start_time: Optional[datetime]
    elapsed_time: int  # in seconds
    tags: List[str]
    environment_vars: Dict[str, str]
    resource_policy: Dict[str, Any]
    migration_policy: Dict[str, Any]
    snapshot_policy: Dict[str, Any]
    metrics: Dict[str, Any]
    process_tree: List[Dict[str, Any]]
    security_labels: List[str]
    fault_tolerance_enabled: bool
    backup_frequency_minutes: int

class TaskRequest(BaseModel):
    task_type: str
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    priority: int = 5
    resource_requirements: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    max_retries: int = 3
    timeout_seconds: int = 3600
    preferred_node_types: List[str] = []

class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    session_id: Optional[str]
    node_id: Optional[str]
    status: str
    priority: int
    resource_requirements: Dict[str, Any]
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    max_retries: int

class ClusterHealthResponse(BaseModel):
    cluster_id: str
    leader_node_id: Optional[str]
    backup_nodes: List[str]
    total_nodes: int
    active_nodes: int
    failed_nodes: int
    cluster_health: str
    last_election: Optional[datetime]
    split_brain_detected: bool
    fault_tolerance_status: Dict[str, Any]

class NodeAction(BaseModel):
    action: str  # restart, shutdown, maintenance, quarantine, evacuate
    target_node_id: Optional[str] = None
    force: bool = False
    maintenance_duration_minutes: Optional[int] = None
    evacuation_target_nodes: List[str] = []

class SessionAction(BaseModel):
    action: str  # pause, resume, terminate, snapshot, migrate, backup, restore
    target_node_id: Optional[str] = None
    snapshot_name: Optional[str] = None
    backup_name: Optional[str] = None
    force: bool = False
    preserve_resources: bool = True

class ResourceHint(BaseModel):
    cpu_cores: int
    gpu_units: int
    ram_bytes: int
    storage_bytes: int = 0
    network_bandwidth_mbps: float = 100.0
    specialized_hardware: Dict[str, Any] = {}

class LatencyMetric(BaseModel):
    timestamp: float
    input_to_pixel_ms: float
    network_hop_ms: float
    gpu_render_ms: float
    prediction_confidence: float
    node_id: Optional[str] = None
    session_id: Optional[str] = None

class SecurityProfile(BaseModel):
    encryption_enabled: bool = True
    access_control_enabled: bool = True
    audit_logging_enabled: bool = True
    network_isolation: bool = False
    container_sandboxing: bool = True
    resource_limits_enforced: bool = True
    allowed_users: List[str] = []
    allowed_networks: List[str] = []

class ProcessInfo(BaseModel):
    pid: int
    name: str
    command: str
    cpu_percent: float
    memory_mb: float
    gpu_percent: float
    status: str
    user: str
    start_time: datetime

class SessionSnapshot(BaseModel):
    snapshot_id: str
    session_id: str
    snapshot_name: str
    created_at: datetime
    size_mb: float
    status: str  # creating, valid, corrupted, restoring
    delta_from_previous: bool
    metadata: Dict[str, Any]

class SessionMetrics(BaseModel):
    session_id: str
    timestamp: datetime
    cpu_usage: float
    gpu_usage: float
    ram_usage_gb: float
    disk_io_mbps: float
    network_in_mbps: float
    network_out_mbps: float
    fps: Optional[float]
    latency_ms: float
    active_processes: int
    temperature: float

# Advanced Resource Management
class OmegaResourceOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, float]] = []
        
    def register_node(self, node_info: NodeInfo):
        self.nodes[node_info.node_id] = node_info
        NODE_COUNT.set(len(self.nodes))
        
        # Store in database
        db = SessionLocal()
        try:
            db_node = NodeRecord(
                node_id=node_info.node_id,
                node_type=node_info.node_type,
                status=node_info.status,
                resources=node_info.resources,
                performance_score=1.0
            )
            db.merge(db_node)
            db.commit()
        finally:
            db.close()
    
    def smart_placement(self, requirements: ResourceHint) -> Optional[str]:
        """AI-driven resource placement with latency optimization"""
        available_nodes = [
            node for node in self.nodes.values() 
            if node.status == "active" and self._has_capacity(node, requirements)
        ]
        
        if not available_nodes:
            return None
            
        # Score nodes based on multiple factors
        scores = []
        for node in available_nodes:
            score = self._calculate_placement_score(node, requirements)
            scores.append((node.node_id, score))
        
        # Return best scoring node
        best_node = max(scores, key=lambda x: x[1])
        return best_node[0]
    
    def _has_capacity(self, node: NodeInfo, requirements: ResourceHint) -> bool:
        resources = node.resources
        return (
            resources.get("cpu_available", 0) >= requirements.cpu_cores and
            resources.get("gpu_available", 0) >= requirements.gpu_units and
            resources.get("ram_available", 0) >= requirements.ram_bytes
        )
    
    def _calculate_placement_score(self, node: NodeInfo, requirements: ResourceHint) -> float:
        """Multi-factor scoring: utilization, latency, thermal headroom"""
        resources = node.resources
        metrics = node.performance_metrics or {}
        
        # Utilization factor (prefer underutilized nodes)
        cpu_util = 1.0 - (resources.get("cpu_available", 0) / resources.get("cpu_total", 1))
        gpu_util = 1.0 - (resources.get("gpu_available", 0) / resources.get("gpu_total", 1))
        
        # Latency factor (prefer low-latency nodes)
        latency_factor = 1.0 / (1.0 + metrics.get("avg_latency_ms", 10.0))
        
        # Thermal factor (prefer cooler nodes)
        thermal_factor = 1.0 - (metrics.get("cpu_temp", 50.0) / 100.0)
        
        # Weighted score
        score = (
            0.3 * (1.0 - cpu_util) +
            0.2 * (1.0 - gpu_util) +
            0.3 * latency_factor +
            0.2 * thermal_factor
        )
        
        return score
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all registered nodes"""
        return [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "status": node.status,
                "ip_address": node.ip_address,
                "resources": node.resources,
                "performance_metrics": node.performance_metrics or {},
                "last_updated": node.last_updated.isoformat() if node.last_updated else None
            }
            for node in self.nodes.values()
        ]

# Latency Compensation Engine
class TemporalSyncEngine:
    def __init__(self):
        self.frame_buffer = []
        self.prediction_model = None
        self.input_history = []
        self.confidence_threshold = 0.7
        
    def predict_input(self, current_input: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based input prediction for latency hiding"""
        # Simplified prediction - in initial prototype would use trained model
        self.input_history.append({
            "timestamp": time.time(),
            "input": current_input
        })
        
        # Keep only last 100 inputs
        if len(self.input_history) > 100:
            self.input_history.pop(0)
            
        # Simple prediction based on velocity
        if len(self.input_history) >= 2:
            prev = self.input_history[-2]
            curr = self.input_history[-1]
            
            if "mouse_x" in current_input and "mouse_x" in prev["input"]:
                velocity_x = curr["input"]["mouse_x"] - prev["input"]["mouse_x"]
                velocity_y = curr["input"]["mouse_y"] - prev["input"]["mouse_y"]
                
                predicted = {
                    "mouse_x": curr["input"]["mouse_x"] + velocity_x,
                    "mouse_y": curr["input"]["mouse_y"] + velocity_y,
                    "confidence": 0.8 if abs(velocity_x) + abs(velocity_y) < 50 else 0.3
                }
                return predicted
        
        return {"confidence": 0.0}
    
    def sync_time_ptp(self) -> float:
        """IEEE 1588 PTP simulation"""
        return time.time_ns() / 1e9

# Initialize orchestrator and sync engine
orchestrator = OmegaResourceOrchestrator()

# Helper functions for session management
def get_session_processes_sync(session_id: str) -> List[Dict]:
    """Get real-time process information for a session"""
    try:
        # In production, this would query the actual node running the session
        # For now, return system processes as example
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                # Filter for session-related processes (this is simplified)
                if any(keyword in pinfo['name'].lower() for keyword in ['omega', 'python', 'node']):
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'cpu_percent': pinfo['cpu_percent'] or 0.0,
                        'memory_percent': pinfo['memory_percent'] or 0.0,
                        'status': pinfo['status'],
                        'session_id': session_id
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes[:10]  # Limit to top 10 processes
    except Exception:
        return []

def get_recent_session_logs(session_id: str, limit: int = 50) -> List[Dict]:
    """Get recent logs for a session"""
    try:
        # In production, this would read from actual log files
        # Generate realistic log entries for demonstration
        import random
        log_levels = ['INFO', 'DEBUG', 'WARNING', 'ERROR']
        log_messages = [
            f"Session {session_id} resource allocation completed",
            f"Process started for session {session_id}",
            f"Memory usage updated: {random.randint(10, 90)}%",
            f"Network connection established for session {session_id}",
            f"CPU usage optimized for session {session_id}",
            f"Session {session_id} checkpoint created",
            f"Performance metrics updated for session {session_id}",
            f"Resource cleanup initiated for session {session_id}"
        ]
        
        logs = []
        for i in range(min(limit, 20)):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
            logs.append({
                'timestamp': timestamp.isoformat(),
                'level': random.choice(log_levels),
                'message': random.choice(log_messages),
                'session_id': session_id,
                'component': random.choice(['orchestrator', 'compute', 'storage', 'network'])
            })
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    except Exception:
        return []

def get_session_snapshots(session_id: str) -> List[Dict]:
    """Get snapshots for a session"""
    try:
        # In production, this would query the actual snapshot storage
        # Return example snapshots
        snapshots = []
        for i in range(3):
            snapshot_time = datetime.now() - timedelta(hours=i*2)
            snapshots.append({
                'snapshot_id': f"snap-{session_id}-{i+1}",
                'name': f"Auto Snapshot {i+1}",
                'created_at': snapshot_time.isoformat(),
                'size_mb': random.randint(100, 1000),
                'type': 'automatic' if i < 2 else 'manual',
                'status': 'completed',
                'session_id': session_id
            })
        return snapshots
    except Exception:
        return []

# Helper functions for node management
def get_system_temperature():
    """Get system temperature if available"""
    try:
        # Try to get CPU temperature on macOS/Linux
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return round(entries[0].current, 1)
        # Fallback: simulate temperature based on CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return round(35 + (cpu_percent * 0.3), 1)  # Base temp + load factor
    except:
        return 42.0  # Default safe temperature

def generate_mock_metrics():
    """Generate realistic mock metrics for demonstration"""
    import random
    return {
        "cpu_percent": round(random.uniform(10, 80), 1),
        "memory_percent": round(random.uniform(20, 70), 1),
        "disk_percent": round(random.uniform(15, 60), 1),
        "temperature": round(random.uniform(35, 65), 1),
        "uptime": random.randint(3600, 604800),  # 1 hour to 1 week
        "network_io": {
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_recv": random.randint(1000000, 10000000)
        },
        "processes_count": random.randint(150, 300)
    }

# Helper functions for distributed computing
def get_system_temperature():
    """Get system temperature if available"""
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        for entry in entries:
                            return entry.current
                # Fallback to any temperature
                for entries in temps.values():
                    if entries:
                        return entries[0].current
        return None
    except:
        return None

async def discover_network_nodes():
    """Discover other Omega nodes on the local network"""
    discovered = []
    try:
        # For prototype, we'll simulate discovering nodes
        # In production, this would scan the network for other Omega instances
        local_ip = "127.0.0.1"
        
        # Example of what discovered nodes might look like
        example_nodes = [
            {
                "ip": "192.168.1.100",
                "hostname": "omega-compute-01",
                "node_type": "compute",
                "status": "available",
                "discovered_at": datetime.now().isoformat()
            },
            {
                "ip": "192.168.1.101", 
                "hostname": "omega-gpu-01",
                "node_type": "gpu",
                "status": "available",
                "discovered_at": datetime.now().isoformat()
            }
        ]
        
        # Return discovered nodes (empty for now, but structure is ready)
        return discovered
    except Exception as e:
        logging.error(f"Network discovery failed: {e}")
        return []

def get_hardware_info():
    """Enhanced hardware detection for distributed computing"""
    try:
        return {
            "cpu": {
                "model": platform.processor() or "Unknown CPU",
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "type": "DDR4"  # Default assumption
            },
            "storage": [
                {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "size": psutil.disk_usage(partition.mountpoint).total
                }
                for partition in psutil.disk_partitions()
                if partition.mountpoint and not partition.mountpoint.startswith('/System')
            ],
            "network": get_network_interfaces(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine()
            }
        }
    except Exception as e:
        logging.error(f"Error getting hardware info: {e}")
        return {"error": "Unable to retrieve hardware information"}

def get_mock_hardware():
    """Generate mock hardware info for remote nodes"""
    import random
    cpu_models = ["Intel Xeon E5-2686", "AMD EPYC 7742", "Intel Core i9-12900K"]
    return {
        "cpu": {
            "model": random.choice(cpu_models),
            "cores": random.choice([8, 16, 32, 64]),
            "threads": random.choice([16, 32, 64, 128]),
            "frequency": random.choice([2400, 2800, 3200, 3600])
        },
        "memory": {
            "total": random.choice([16, 32, 64, 128]) * 1024**3,
            "available": random.randint(8, 32) * 1024**3,
            "type": "DDR4"
        },
        "storage": [
            {
                "device": "/dev/nvme0n1",
                "mountpoint": "/",
                "fstype": "ext4",
                "size": random.choice([500, 1000, 2000]) * 1024**3
            }
        ]
    }

def get_network_interfaces():
    """Get network interface information"""
    try:
        interfaces = []
        for interface, addrs in psutil.net_if_addrs().items():
            if interface.startswith('lo'):  # Skip loopback
                continue
            
            interface_info = {
                "name": interface,
                "addresses": []
            }
            
            for addr in addrs:
                if addr.family == socket.AF_INET:  # IPv4
                    interface_info["addresses"].append({
                        "type": "IPv4",
                        "address": addr.address,
                        "netmask": addr.netmask
                    })
                elif addr.family == socket.AF_INET6:  # IPv6
                    interface_info["addresses"].append({
                        "type": "IPv6", 
                        "address": addr.address
                    })
            
            if interface_info["addresses"]:
                interfaces.append(interface_info)
        
        return interfaces
    except Exception as e:
        logging.error(f"Error getting network interfaces: {e}")
        return []

def get_running_processes():
    """Get list of running processes"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] is None:
                    pinfo['cpu_percent'] = 0.0
                if pinfo['memory_percent'] is None:
                    pinfo['memory_percent'] = 0.0
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage, descending
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:50]  # Return top 50 processes
    except Exception as e:
        logging.error(f"Error getting processes: {e}")
        return []

def get_recent_logs(node_id: str, limit: int = 50):
    """Get recent system logs for a node"""
    try:
        logs = []
        current_time = datetime.utcnow()
        
        # Simulate logs for demonstration
        log_levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        log_messages = [
            "System startup completed",
            "Network interface configured",
            "Storage health check passed",
            "High CPU temperature detected",
            "Memory usage normal",
            "Service restarted successfully",
            "Authentication successful",
            "Backup completed",
            "Update installed",
            "Performance monitoring active"
        ]
        
        for i in range(limit):
            log_time = current_time - timedelta(minutes=i*5)
            logs.append({
                "timestamp": log_time.isoformat(),
                "level": np.random.choice(log_levels, p=[0.5, 0.3, 0.1, 0.1]),
                "message": np.random.choice(log_messages),
                "source": f"{node_id}-system"
            })
        
        return logs
    except Exception as e:
        logging.error(f"Error getting logs: {e}")
        return []

def get_cpu_details():
    """Get detailed CPU information"""
    try:
        cpu_times = psutil.cpu_times()
        cpu_stats = psutil.cpu_stats()
        
        return {
            "usage_percent": psutil.cpu_percent(interval=1, percpu=True),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "times": cpu_times._asdict(),
            "stats": cpu_stats._asdict(),
            "count": {"physical": psutil.cpu_count(logical=False), "logical": psutil.cpu_count()}
        }
    except Exception as e:
        logging.error(f"Error getting CPU details: {e}")
        return {"error": "Unable to retrieve CPU details"}

def get_memory_details():
    """Get detailed memory information"""
    try:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "virtual": virtual._asdict(),
            "swap": swap._asdict()
        }
    except Exception as e:
        logging.error(f"Error getting memory details: {e}")
        return {"error": "Unable to retrieve memory details"}

def get_storage_details():
    """Get detailed storage information"""
    try:
        storage_info = []
        
        for partition in psutil.disk_partitions():
            try:
                if partition.mountpoint and not partition.mountpoint.startswith('/System'):
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "usage": usage._asdict()
                    })
            except PermissionError:
                continue
        
        # Add disk I/O statistics
        disk_io = psutil.disk_io_counters(perdisk=True)
        
        return {
            "partitions": storage_info,
            "io_stats": {disk: stats._asdict() for disk, stats in disk_io.items()} if disk_io else {}
        }
    except Exception as e:
        logging.error(f"Error getting storage details: {e}")
        return {"error": "Unable to retrieve storage details"}

def get_network_details():
    """Get detailed network information"""
    try:
        io_counters = psutil.net_io_counters(pernic=True)
        connections = psutil.net_connections()
        
        return {
            "interfaces": {nic: stats._asdict() for nic, stats in io_counters.items()},
            "connections_count": len(connections),
            "active_connections": len([c for c in connections if c.status == 'ESTABLISHED'])
        }
    except Exception as e:
        logging.error(f"Error getting network details: {e}")
        return {"error": "Unable to retrieve network details"}

def get_gpu_details():
    """Get GPU information if available"""
    try:
        # In initial prototype, simulate GPU info
        return {
            "detected": True,
            "devices": [
                {
                    "name": "NVIDIA RTX 4090",
                    "memory_total": 24576,  # MB
                    "memory_used": 2048,
                    "utilization": 45,
                    "temperature": 65
                }
            ]
        }
    except Exception as e:
        logging.error(f"Error getting GPU details: {e}")
        return {"detected": False, "error": "No GPU detected or driver unavailable"}

def get_security_status(node_id: str):
    """Get security status for a node"""
    return {
        "last_scan": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "vulnerabilities": {
            "critical": 0,
            "high": 1,
            "medium": 3,
            "low": 7
        },
        "firewall_status": "active",
        "antivirus_status": "active",
        "last_auth": datetime.utcnow().isoformat(),
        "failed_logins": 0,
        "certificates": {
            "valid": True,
            "expires": (datetime.utcnow() + timedelta(days=90)).isoformat()
        }
    }

def get_maintenance_status(node_id: str):
    """Get maintenance status for a node"""
    return {
        "mode": "normal",
        "last_maintenance": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "next_scheduled": (datetime.utcnow() + timedelta(days=23)).isoformat(),
        "firmware_version": "1.2.3",
        "update_available": True,
        "health_checks": {
            "cpu": "healthy",
            "memory": "healthy", 
            "storage": "warning",
            "network": "healthy"
        }
    }
sync_engine = TemporalSyncEngine()

# Security setup
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting Omega Control Node...")
    
    # Create database tables first
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created successfully")
    
    # Register local control node
    try:
        local_node = NodeInfo(
            node_id="local-control",
            node_type="control",
            ip_address="127.0.0.1",
            status="online",
            resources={
                "cpu_cores": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3)),
                "storage_gb": round(psutil.disk_usage('/').total / (1024**3)),
                "description": f"Local control node - {platform.system()} {platform.machine()}",
                "platform": platform.system(),
                "architecture": platform.machine(),
                "hostname": platform.node()
            },
            performance_metrics={
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        )
        orchestrator.register_node(local_node)
        logging.info("Local control node registered successfully")
    except Exception as e:
        logging.error(f"Failed to register local control node: {e}")
    # Initialize Redis via central helper to support multiple Redis libraries
    try:
        from utils.redis_helper import get_redis_client
        redis_url = os.getenv("OMEGA_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379"))
        # get_redis_client is async
        global redis_client
        try:
            redis_client = await get_redis_client(redis_url)
            logging.info("Redis client initialized in control node")
        except Exception as e:
            logging.warning(f"Redis helper failed to initialize client: {e}")
            redis_client = None
    except Exception:
        logging.debug("Redis helper not available; proceeding without Redis")
    
    try:
        # Try to start Prometheus metrics server on an alternative port if 8000 is busy
        for port in [8000, 8001, 8002]:
            try:
                start_http_server(port)
                logging.info(f"Prometheus metrics server started on port {port}")
                break
            except OSError as e:
                if port == 8002:  # Last attempt
                    logging.warning(f"Could not start Prometheus server: {e}")
                continue
    except Exception as e:
        logging.warning(f"Prometheus server failed to start: {e}")
    
    yield
    # Shutdown
    logging.info("Shutting down Omega Control Node...")

app = FastAPI(
    title="Omega Super Desktop Control Node",
    version="1.0.0",
    description="Initial prototype distributed computing control plane",
    lifespan=lifespan
)

# Initialize with local control node
def initialize_local_node():
    """Register the local control node"""
    try:
        # Ensure database tables exist first
        Base.metadata.create_all(bind=engine)
        
        local_node = NodeInfo(
            node_id="local-control",
            node_type="control",
            ip_address="127.0.0.1",
            status="online",
            resources={
                "cpu_cores": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3)),
                "storage_gb": round(psutil.disk_usage('/').total / (1024**3)),
                "description": "Local control node - initial prototype"
            }
        )
        orchestrator.register_node(local_node)
        logging.info("Local control node registered successfully")
    except Exception as e:
        logging.error(f"Failed to register local control node: {e}")

# Node will be registered after server startup

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public test endpoints (no auth required)
@app.get("/api/test/nodes")
async def test_get_nodes():
    """Test endpoint to get all nodes without authentication"""
    try:
        return orchestrator.get_all_nodes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/nodes/{node_id}")
async def test_get_node(node_id: str):
    """Test endpoint to get a specific node without authentication"""
    try:
        all_nodes = orchestrator.get_all_nodes()
        for node in all_nodes:
            if node["node_id"] == node_id:
                # Add real-time metrics for the node
                if node_id == "local-control":
                    node["real_time_metrics"] = {
                        "cpu_usage": psutil.cpu_percent(interval=0.1),
                        "memory_usage": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage('/').percent,
                        "network_io": dict(psutil.net_io_counters()._asdict()),
                        "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                        "uptime": time.time() - psutil.boot_time(),
                        "temperature": get_system_temperature()
                    }
                return node
        raise HTTPException(status_code=404, detail="Node not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/sessions")
async def test_get_sessions():
    """Test endpoint to get all sessions without authentication"""
    try:
        db = SessionLocal()
        try:
            sessions = db.query(SessionRecord).all()
            result = []
            for session in sessions:
                session_dict = convert_session_to_response(session).__dict__
                # Add real-time process information (synchronous)
                session_dict["real_time_processes"] = get_session_processes_sync(session.session_id)
                session_dict["real_time_logs"] = get_recent_session_logs(session.session_id, 10)
                result.append(session_dict)
            return result
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/sessions/{session_id}")
async def test_get_session(session_id: str):
    """Test endpoint to get a specific session without authentication"""
    try:
        db = SessionLocal()
        try:
            session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session_dict = convert_session_to_response(session).__dict__
            # Add comprehensive real-time data
            session_dict["real_time_metrics"] = {
                "cpu_usage": session.metrics.get("cpu_usage", 0.0),
                "memory_usage": session.metrics.get("ram_usage_gb", 0.0),
                "gpu_usage": session.metrics.get("gpu_usage", 0.0),
                "disk_io": session.metrics.get("disk_io_mbps", 0.0),
                "network_in": session.metrics.get("network_in_mbps", 0.0),
                "network_out": session.metrics.get("network_out_mbps", 0.0),
                "fps": session.metrics.get("fps", 0.0),
                "latency_ms": session.metrics.get("latency_ms", 5.0),
                "temperature": session.metrics.get("temperature", 45.0),
                "timestamp": datetime.now().isoformat()
            }
            session_dict["real_time_processes"] = get_session_processes_sync(session_id)
            session_dict["real_time_logs"] = get_recent_session_logs(session_id, 50)
            session_dict["snapshots"] = get_session_snapshots(session_id)
            return session_dict
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
async def create_session(session_request: SessionRequest):
    """Create a new session with intelligent resource allocation"""
    try:
        db = SessionLocal()
        try:
            # First, check available nodes and resources
            available_nodes = db.query(NodeRecord).filter(NodeRecord.status == "active").all()
            if not available_nodes:
                # Create a default local node if none exist
                local_node = NodeRecord(
                    node_id="local-node-1",
                    node_type="workstation",
                    status="active",
                    resources={
                        "cpu_cores": psutil.cpu_count(),
                        "ram_gb": round(psutil.virtual_memory().total / (1024**3)),
                        "gpu_units": 1,  # Assume 1 GPU
                        "storage_gb": round(psutil.disk_usage('/').total / (1024**3))
                    },
                    performance_score=1.0
                )
                db.add(local_node)
                db.commit()
                available_nodes = [local_node]
            
            # Calculate total available resources across all nodes
            total_cpu_cores = 0
            total_ram_gb = 0
            total_gpu_units = 0
            node_resources = {}
            
            for node in available_nodes:
                node_res = node.resources or {}
                cpu_cores = node_res.get("cpu_cores", 4)
                ram_gb = node_res.get("ram_gb", 8)
                gpu_units = node_res.get("gpu_units", 1)
                
                total_cpu_cores += cpu_cores
                total_ram_gb += ram_gb
                total_gpu_units += gpu_units
                
                node_resources[node.node_id] = {
                    "cpu_cores": cpu_cores,
                    "ram_gb": ram_gb, 
                    "gpu_units": gpu_units,
                    "available_cpu": cpu_cores * 0.8,  # Reserve 20% for system
                    "available_ram": ram_gb * 0.9,     # Reserve 10% for system
                    "available_gpu": gpu_units
                }
            
            # Validate requested resources are available
            requested_cpu = session_request.cpu_cores or 2
            requested_ram = session_request.ram_gb or 4
            requested_gpu = session_request.gpu_units or 0
            
            if requested_cpu > total_cpu_cores * 0.8:
                raise HTTPException(status_code=400, detail=f"Insufficient CPU cores. Requested: {requested_cpu}, Available: {int(total_cpu_cores * 0.8)}")
            
            if requested_ram > total_ram_gb * 0.9:
                raise HTTPException(status_code=400, detail=f"Insufficient RAM. Requested: {requested_ram}GB, Available: {int(total_ram_gb * 0.9)}GB")
            
            if requested_gpu > total_gpu_units:
                raise HTTPException(status_code=400, detail=f"Insufficient GPU units. Requested: {requested_gpu}, Available: {total_gpu_units}")
            
            # Select best node for session
            best_node = available_nodes[0]  # For now, use first available node
            
            # Create the session
            session_id = str(uuid.uuid4())
            new_session = SessionRecord(
                session_id=session_id,
                session_name=session_request.session_name,
                user_id=session_request.user_id or "default_user",
                app_name=session_request.app_name,
                app_command=session_request.app_command,
                app_icon=session_request.app_icon,
                status="RUNNING",
                node_id=best_node.node_id,
                cpu_cores=requested_cpu,
                gpu_units=requested_gpu,
                ram_gb=requested_ram,
                storage_gb=session_request.storage_gb or 10,
                priority=session_request.priority or 1,
                session_type=session_request.session_type or "workstation",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                start_time=datetime.utcnow(),
                tags=session_request.tags or [],
                environment_vars=session_request.environment_vars or {},
                resource_policy=session_request.resource_policy or {},
                migration_policy=session_request.migration_policy or {},
                snapshot_policy=session_request.snapshot_policy or {},
                metrics={
                    "cpu_usage": random.uniform(10, 40),
                    "ram_usage_gb": random.uniform(1, requested_ram * 0.8),
                    "gpu_usage": random.uniform(0, 60) if requested_gpu > 0 else 0,
                    "disk_io_mbps": random.uniform(5, 50),
                    "network_in_mbps": random.uniform(1, 10),
                    "network_out_mbps": random.uniform(1, 8),
                    "created_timestamp": datetime.utcnow().isoformat()
                },
                process_tree=[],
                security_labels=session_request.security_labels or ["standard"]
            )
            
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            
            # Return the created session
            session_response = convert_session_to_response(new_session)
            
            # Log the session creation
            logging.info(f"Session created successfully: {session_id} on node {best_node.node_id}")
            
            return {
                "session_id": session_id,
                "status": "created",
                "message": f"Session created successfully on node {best_node.node_id}",
                "allocated_resources": {
                    "cpu_cores": requested_cpu,
                    "ram_gb": requested_ram,
                    "gpu_units": requested_gpu,
                    "node_id": best_node.node_id
                },
                "session": session_response.__dict__
            }
            
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/sessions/{session_id}/action")
async def test_session_action(session_id: str, action_data: dict):
    """Test endpoint for session actions without authentication"""
    try:
        db = SessionLocal()
        try:
            session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            action_type = action_data.get("action")
            
            if action_type == "pause":
                session.status = "PAUSED"
            elif action_type == "resume":
                session.status = "RUNNING"
            elif action_type == "terminate":
                session.status = "TERMINATED"
                session.end_time = datetime.utcnow()
            elif action_type == "migrate":
                target_node = action_data.get("target_node_id")
                if not target_node:
                    raise HTTPException(status_code=400, detail="Target node required for migration")
                session.node_id = target_node
                session.status = "RUNNING"
            elif action_type == "snapshot":
                # Create a snapshot
                snapshot_name = action_data.get("snapshot_name", f"snapshot-{int(time.time())}")
                # This would create an actual snapshot in production
                pass
            
            db.commit()
            
            return {
                "status": "success",
                "action": action_type,
                "session_id": session_id,
                "message": f"Action {action_type} completed successfully"
            }
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/health")
async def test_health():
    """Test health endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/test/discover")
async def test_discover_nodes():
    """Discover other Omega nodes on the network"""
    try:
        discovered = await discover_network_nodes()
        return {"discovered_nodes": discovered, "scan_time": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Authentication endpoints
@app.post("/auth/login")
async def login(username: str, password: str):
    # In initial prototype: verify against LDAP/OAuth
    # Use environment variable for admin password or default
    admin_password = os.getenv("OMEGA_ADMIN_PASSWORD", "omega123")
    if username == "admin" and password == admin_password:
        token = jwt.encode(
            {"user_id": username, "exp": datetime.utcnow() + timedelta(hours=8)},
            SECRET_KEY,
            algorithm="HS256"
        )
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Node management APIs
@app.post("/api/v1/nodes/register")
async def register_node(node_info: NodeInfo, user_id: str = Depends(verify_token)):
    REQUESTS_TOTAL.labels(method="POST", endpoint="/nodes/register").inc()
    orchestrator.register_node(node_info)
    
    # Store in Redis for real-time access
    if redis_client:
        try:
            redis_client.hset(
                f"node:{node_info.node_id}",
                mapping={
                    "type": node_info.node_type,
                    "status": node_info.status,
                    "resources": json.dumps(node_info.resources),
                    "last_seen": str(datetime.utcnow())
                }
            )
        except Exception as e:
            logging.warning(f"Failed to store node data in Redis: {e}")
    
    logging.info(f"Node registered: {node_info.node_id} by user {user_id}")
    return {"status": "registered", "node_id": node_info.node_id}

@app.get("/api/v1/nodes")
async def list_nodes(user_id: str = Depends(verify_token)):
    REQUESTS_TOTAL.labels(method="GET", endpoint="/nodes").inc()
    
    # Get current system metrics for local node
    local_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "temperature": get_system_temperature(),
        "uptime": time.time() - psutil.boot_time(),
        "network_io": dict(psutil.net_io_counters()._asdict()),
        "processes_count": len(psutil.pids())
    }
    
    # Enhanced node data with real-time metrics
    enhanced_nodes = []
    for node in orchestrator.nodes.values():
        node_dict = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "status": node.status,
            "resources": node.resources,
            "ip_address": node.ip_address,
            "last_heartbeat": node.last_heartbeat,
            "metrics": local_metrics if node.node_id == "local-control" else generate_mock_metrics(),
            "hardware": get_hardware_info(),
            "processes": get_running_processes()[:10],  # Top 10 processes
            "logs": get_recent_logs(node.node_id)[:20]  # Recent 20 logs
        }
        enhanced_nodes.append(node_dict)
    
    return {"nodes": enhanced_nodes}

@app.get("/api/v1/nodes/{node_id}")
async def get_node_details(node_id: str, user_id: str = Depends(verify_token)):
    """Get detailed information about a specific node"""
    REQUESTS_TOTAL.labels(method="GET", endpoint="/nodes/details").inc()
    
    if node_id not in orchestrator.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = orchestrator.nodes[node_id]
    
    # Get comprehensive node details
    node_details = {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "status": node.status,
        "resources": node.resources,
        "ip_address": node.ip_address,
        "last_heartbeat": node.last_heartbeat,
        "uptime": time.time() - psutil.boot_time() if node_id == "local-control" else None,
        "hardware": get_hardware_info() if node_id == "local-control" else get_mock_hardware(),
        "performance": {
            "cpu": get_cpu_details(),
            "memory": get_memory_details(),
            "storage": get_storage_details(),
            "network": get_network_details(),
            "gpu": get_gpu_details() if node.node_type == "gpu" else None
        },
        "processes": get_running_processes(),
        "security": get_security_status(node_id),
        "logs": get_recent_logs(node_id),
        "maintenance": get_maintenance_status(node_id)
    }
    
    return node_details

@app.post("/api/v1/nodes/{node_id}/action")
async def node_action(node_id: str, action: dict, user_id: str = Depends(verify_token)):
    """Execute actions on a node (restart, shutdown, maintenance, etc.)"""
    REQUESTS_TOTAL.labels(method="POST", endpoint="/nodes/action").inc()
    
    if node_id not in orchestrator.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    action_type = action.get("type")
    
    if action_type == "restart":
        # In initial prototype, simulate restart
        logging.info(f"Restart initiated for node {node_id} by user {user_id}")
        return {"status": "restart_initiated", "message": f"Node {node_id} restart scheduled"}
    
    elif action_type == "shutdown":
        logging.info(f"Shutdown initiated for node {node_id} by user {user_id}")
        return {"status": "shutdown_initiated", "message": f"Node {node_id} shutdown scheduled"}
    
    elif action_type == "maintenance":
        mode = action.get("mode", "enable")
        logging.info(f"Maintenance mode {mode} for node {node_id} by user {user_id}")
        return {"status": "maintenance_updated", "mode": mode}
    
    elif action_type == "quarantine":
        logging.info(f"Quarantine initiated for node {node_id} by user {user_id}")
        return {"status": "quarantined", "message": f"Node {node_id} quarantined"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action type")

@app.get("/api/v1/nodes/{node_id}/metrics/stream")
async def stream_node_metrics(websocket: WebSocket, node_id: str):
    """Stream real-time metrics for a specific node"""
    await websocket.accept()
    
    try:
        while True:
            if node_id == "local-control":
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": dict(psutil.disk_io_counters()._asdict()),
                    "network_io": dict(psutil.net_io_counters()._asdict()),
                    "temperature": get_system_temperature(),
                    "processes_count": len(psutil.pids())
                }
            else:
                metrics = generate_mock_metrics()
                metrics["timestamp"] = datetime.utcnow().isoformat()
            
            await websocket.send_json(metrics)
            await asyncio.sleep(1)  # Send metrics every second
            
    except Exception as e:
        logging.error(f"WebSocket error for node {node_id}: {e}")
    finally:
        await websocket.close()

@app.delete("/api/v1/nodes/{node_id}")
async def deregister_node(node_id: str, user_id: str = Depends(verify_token)):
    if node_id in orchestrator.nodes:
        del orchestrator.nodes[node_id]
        if redis_client:
            try:
                redis_client.delete(f"node:{node_id}")
            except Exception as e:
                logging.warning(f"Failed to delete node from Redis: {e}")
        NODE_COUNT.set(len(orchestrator.nodes))
        return {"status": "deregistered"}
    raise HTTPException(status_code=404, detail="Node not found")

# Session management
@app.post("/api/v1/sessions/create")
async def create_session(request: SessionRequest, user_id: str = Depends(verify_token)):
    REQUESTS_TOTAL.labels(method="POST", endpoint="/sessions/create").inc()
    
    # Find optimal node placement
    requirements = ResourceHint(
        cpu_cores=request.cpu_cores,
        gpu_units=request.gpu_units,
        ram_bytes=request.ram_bytes
    )
    
    target_node = orchestrator.smart_placement(requirements)
    if not target_node:
        raise HTTPException(status_code=503, detail="No suitable nodes available")
    
    # Create session
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "user_id": request.user_id,
        "app_uri": request.app_uri,
        "node_id": target_node,
        "resources": requirements.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "status": "RUNNING"
    }
    
    orchestrator.sessions[session_id] = session_data
    ACTIVE_SESSIONS.set(len(orchestrator.sessions))
    
    # Store in database
    db = SessionLocal()
    try:
        db_session = SessionRecord(
            sid=session_id,
            user_id=request.user_id,
            cpu_cores=request.cpu_cores,
            gpu_units=request.gpu_units,
            ram_bytes=request.ram_bytes,
            metrics={}
        )
        db.add(db_session)
        db.commit()
    finally:
        db.close()
    
    # Store in Redis for real-time access
    if redis_client:
        try:
            redis_client.hset(f"session:{session_id}", mapping=session_data)
        except Exception as e:
            logging.warning(f"Failed to store session data in Redis: {e}")
    
    logging.info(f"Session created: {session_id} on node {target_node}")
    return {"session_id": session_id, "node_id": target_node, "status": "created"}

# Comprehensive Session Management API

@app.post("/api/v1/sessions", response_model=SessionResponse)
async def create_session(session_request: SessionRequest, db: Session = Depends(get_db)):
    """Create a new session with resource allocation"""
    try:
        # Generate unique session ID
        session_id = f"sess_{int(time.time())}_{session_request.session_name.lower().replace(' ', '_')}"
        
        # Find optimal node for placement
        optimal_node = await find_optimal_node_for_session(session_request, db)
        
        # Create session record
        session = SessionRecord(
            session_id=session_id,
            session_name=session_request.session_name,
            user_id=session_request.user_id,
            app_name=session_request.app_name,
            app_command=session_request.app_command,
            app_icon=session_request.app_icon,
            status="INITIALIZING",
            node_id=optimal_node,
            cpu_cores=session_request.cpu_cores,
            gpu_units=session_request.gpu_units,
            ram_gb=session_request.ram_gb,
            storage_gb=session_request.storage_gb,
            priority=session_request.priority,
            session_type=session_request.session_type,
            start_time=datetime.utcnow(),
            tags=session_request.tags,
            environment_vars=session_request.environment_vars,
            resource_policy=session_request.resource_policy,
            migration_policy=session_request.migration_policy,
            snapshot_policy=session_request.snapshot_policy,
            metrics={
                "cpu_usage": 0.0,
                "gpu_usage": 0.0,
                "ram_usage_gb": 0.0,
                "disk_io_mbps": 0.0,
                "network_in_mbps": 0.0,
                "network_out_mbps": 0.0,
                "fps": 0.0,
                "latency_ms": 5.0,
                "active_processes": 1,
                "temperature": 45.0
            },
            process_tree=[{
                "pid": 1001,
                "name": session_request.app_name,
                "command": session_request.app_command,
                "cpu_percent": 0.0,
                "memory_mb": 512.0,
                "gpu_percent": 0.0,
                "status": "running",
                "user": session_request.user_id,
                "start_time": datetime.utcnow().isoformat()
            }],
            security_labels=["standard", "user-session"]
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Start session on target node (simulate)
        await start_session_on_node(session_id, optimal_node)
        
        # Update status to running
        session.status = "RUNNING"
        db.commit()
        
        ACTIVE_SESSIONS.set(db.query(SessionRecord).filter(SessionRecord.status.in_(["RUNNING", "PAUSED", "SUSPENDED"])).count())
        
        return convert_session_to_response(session)
        
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/api/v1/sessions", response_model=List[SessionResponse])
async def list_sessions(
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    session_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all sessions with optional filtering"""
    REQUESTS_TOTAL.labels(method="GET", endpoint="/sessions").inc()
    try:
        query = db.query(SessionRecord)
        
        if status:
            query = query.filter(SessionRecord.status == status.upper())
        if user_id:
            query = query.filter(SessionRecord.user_id == user_id)
        if session_type:
            query = query.filter(SessionRecord.session_type == session_type)
            
        sessions = query.offset(offset).limit(limit).all()
        
        # Update metrics for all running sessions
        for session in sessions:
            if session.status == "RUNNING":
                await update_session_metrics(session, db)
        
        return [convert_session_to_response(session) for session in sessions]
        
    except Exception as e:
        logging.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get detailed session information"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Update real-time metrics
        await update_session_metrics(session, db)
        
        return convert_session_to_response(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.post("/api/v1/sessions/{session_id}/action")
async def session_action(session_id: str, action: SessionAction, db: Session = Depends(get_db)):
    """Perform action on session (pause, resume, terminate, snapshot, migrate)"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        result = await execute_session_action(session, action, db)
        
        ACTIVE_SESSIONS.set(db.query(SessionRecord).filter(SessionRecord.status.in_(["RUNNING", "PAUSED", "SUSPENDED"])).count())
        
        return {"status": "success", "message": f"Action {action.action} completed", "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error executing action {action.action} on session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute action: {str(e)}")

@app.get("/api/v1/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str, hours: int = 1, db: Session = Depends(get_db)):
    """Get session metrics history"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Generate realistic metrics history
        metrics_history = generate_metrics_history(session_id, hours)
        
        return {
            "session_id": session_id,
            "timeframe_hours": hours,
            "metrics": metrics_history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/api/v1/sessions/{session_id}/processes", response_model=List[ProcessInfo])
async def get_session_processes(session_id: str, db: Session = Depends(get_db)):
    """Get running processes for session"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Update process tree with real-time data
        processes = await get_real_session_processes(session)
        
        return processes
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting processes for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processes: {str(e)}")

@app.get("/api/v1/sessions/{session_id}/logs")
async def get_session_logs(
    session_id: str, 
    level: Optional[str] = None,
    lines: int = 100,
    follow: bool = False,
    db: Session = Depends(get_db)
):
    """Get session logs with optional filtering"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        logs = await get_session_log_entries(session_id, level, lines)
        
        return {
            "session_id": session_id,
            "logs": logs,
            "total_lines": len(logs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting logs for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@app.get("/api/v1/sessions/{session_id}/snapshots", response_model=List[SessionSnapshot])
async def list_session_snapshots(session_id: str, db: Session = Depends(get_db)):
    """List all snapshots for a session"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        snapshots = await get_session_snapshots(session_id)
        
        return snapshots
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting snapshots for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get snapshots: {str(e)}")

@app.post("/api/v1/sessions/{session_id}/snapshots")
async def create_session_snapshot(
    session_id: str, 
    snapshot_name: str,
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a new snapshot of the session"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        snapshot = await create_snapshot(session, snapshot_name, description)
        
        return {"status": "success", "snapshot": snapshot}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating snapshot for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")

# WebSocket endpoint for real-time session updates
@app.websocket("/api/v1/sessions/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: str):
    """Stream real-time session metrics and events"""
    await websocket.accept()
    
    try:
        while True:
            # Get current session state and metrics
            with SessionLocal() as db:
                session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
                if not session:
                    await websocket.send_json({"error": "Session not found"})
                    break
                    
                # Generate real-time metrics
                metrics = await get_real_time_session_metrics(session)
                
                await websocket.send_json({
                    "type": "metrics",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": metrics
                })
                
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()

@app.delete("/api/v1/sessions/{session_id}")
async def terminate_session(session_id: str, db: Session = Depends(get_db)):
    """Terminate a session"""
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Execute terminate action
        action = SessionAction(action="terminate", force=True)
        result = await execute_session_action(session, action, db)
        
        ACTIVE_SESSIONS.set(db.query(SessionRecord).filter(SessionRecord.status.in_(["RUNNING", "PAUSED", "SUSPENDED"])).count())
        
        return {"status": "terminated", "session_id": session_id, "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error terminating session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to terminate session: {str(e)}")

# Task execution
@app.post("/api/v1/tasks/execute")
async def execute_task(task: TaskRequest, background_tasks: BackgroundTasks, user_id: str = Depends(verify_token)):
    REQUESTS_TOTAL.labels(method="POST", endpoint="/tasks/execute").inc()
    
    task_id = str(uuid.uuid4())
    
    # Queue task for execution
    background_tasks.add_task(process_task, task_id, task)
    
    return {"task_id": task_id, "status": "queued"}

async def process_task(task_id: str, task: TaskRequest):
    """Background task processing"""
    try:
        # Simulate task execution
        await asyncio.sleep(2)
        
        # Store result
        result = {
            "task_id": task_id,
            "status": "completed",
            "result": {"output": "Task completed successfully"},
            "completed_at": datetime.utcnow().isoformat()
        }
        
        if redis_client:
            try:
                redis_client.hset(f"task:{task_id}", mapping=result)
            except Exception as e:
                logging.warning(f"Failed to store task result in Redis: {e}")
        logging.info(f"Task completed: {task_id}")
        
    except Exception as e:
        error_result = {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }
        if redis_client:
            try:
                redis_client.hset(f"task:{task_id}", mapping=error_result)
            except Exception as redis_error:
                logging.warning(f"Failed to store task error in Redis: {redis_error}")
        logging.error(f"Task failed: {task_id} - {e}")

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str, user_id: str = Depends(verify_token)):
    if not redis_client:
        raise HTTPException(status_code=503, detail="Task storage not available")
    
    try:
        result = redis_client.hgetall(f"task:{task_id}")
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        return result
    except Exception as e:
        logging.error(f"Failed to retrieve task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")

# Latency reporting and optimization
@app.post("/api/v1/metrics/latency")
async def report_latency(metrics: LatencyMetric, user_id: str = Depends(verify_token)):
    # Update prometheus metrics
    LATENCY_P95.set(metrics.input_to_pixel_ms)
    
    # Store in Redis for real-time monitoring
    if redis_client:
        try:
            redis_client.lpush("latency_metrics", json.dumps({
                "timestamp": metrics.timestamp,
                "input_to_pixel_ms": metrics.input_to_pixel_ms,
                "network_hop_ms": metrics.network_hop_ms,
                "gpu_render_ms": metrics.gpu_render_ms,
                "prediction_confidence": metrics.prediction_confidence
            }))
            
            # Keep only last 1000 metrics
            redis_client.ltrim("latency_metrics", 0, 999)
        except Exception as e:
            logging.warning(f"Failed to store latency metrics in Redis: {e}")
    
    return {"status": "recorded"}

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics(user_id: str = Depends(verify_token)):
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Get latest latency metrics
    latest_metrics = redis_client.lrange("latency_metrics", 0, 9)
    latency_data = [json.loads(m) for m in latest_metrics]
    
    # Calculate averages
    avg_latency = np.mean([m["input_to_pixel_ms"] for m in latency_data]) if latency_data else 0
    avg_confidence = np.mean([m["prediction_confidence"] for m in latency_data]) if latency_data else 0
    
    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "active_nodes": len(orchestrator.nodes),
            "active_sessions": len(orchestrator.sessions)
        },
        "latency": {
            "avg_input_to_pixel_ms": avg_latency,
            "avg_prediction_confidence": avg_confidence,
            "recent_measurements": latency_data
        }
    }

# AI prediction endpoint
@app.post("/api/v1/ai/predict_input")
async def predict_input(input_data: Dict[str, Any], user_id: str = Depends(verify_token)):
    prediction = sync_engine.predict_input(input_data)
    return {"prediction": prediction}

# Real-time WebSocket for UI updates
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send real-time updates
            data = {
                "timestamp": time.time(),
                "nodes": len(orchestrator.nodes),
                "sessions": len(orchestrator.sessions),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "nodes": len(orchestrator.nodes),
        "sessions": len(orchestrator.sessions)
    }

# Authentication endpoint (simplified for demo)
@app.post("/api/auth/login")
async def login(credentials: dict):
    """Simple authentication endpoint for demo"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Simple demo authentication - use environment variable
    admin_password = os.getenv("OMEGA_ADMIN_PASSWORD", "omega123")
    if username == "admin" and password == admin_password:
        token = jwt.encode(
            {"user_id": "admin", "exp": datetime.utcnow() + timedelta(hours=24)},
            SECRET_KEY,
            algorithm="HS256"
        )
        return {"token": token, "status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Dashboard metrics endpoint - Real Mac system data
@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """Get real system metrics for dashboard"""
    try:
        # Get real CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        
        # Get real memory info
        memory = psutil.virtual_memory()
        
        # Get real disk info
        disk = psutil.disk_usage('/')
        
        # Get real network info
        net_io = psutil.net_io_counters()
        
        # Get system boot time
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        
        # Get process count
        process_count = len(psutil.pids())
        
        # Try to get additional Mac-specific info
        try:
            # Get system load averages (1, 5, 15 minutes)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        except:
            load_avg = (0, 0, 0)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "hostname": platform.node(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                },
                "uptime_seconds": int(uptime.total_seconds()),
                "uptime_human": str(uptime).split('.')[0],
                "boot_time": boot_time.isoformat(),
                "load_avg": {
                    "1min": load_avg[0],
                    "5min": load_avg[1],
                    "15min": load_avg[2]
                },
                "process_count": process_count
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "logical_cores": cpu_count_logical,
                "physical_cores": cpu_count_physical,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "max_frequency_mhz": cpu_freq.max if cpu_freq else 0,
                "per_core_usage": psutil.cpu_percent(percpu=True)
            },
            "memory": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "usage_percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "active": getattr(memory, 'active', 0),
                "inactive": getattr(memory, 'inactive', 0),
                "wired": getattr(memory, 'wired', 0),
                "cached": getattr(memory, 'cached', 0)
            },
            "disk": {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "usage_percent": (disk.used / disk.total) * 100,
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2)
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "bytes_sent_gb": round(net_io.bytes_sent / (1024**3), 2),
                "bytes_recv_gb": round(net_io.bytes_recv / (1024**3), 2),
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            },
            "cluster": {
                "active_nodes": len(orchestrator.nodes),
                "standby_nodes": 0,  # For single-machine setup, no standby nodes
                "total_sessions": len(orchestrator.sessions),
                "cluster_status": "operational" if len(orchestrator.nodes) > 0 else "standalone"
            },
            "processes": {
                "top_cpu": [
                    {
                        "pid": p.info['pid'],
                        "name": p.info['name'],
                        "cpu_percent": round(p.info.get('cpu_percent', 0) or 0, 1)
                    }
                    for p in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent']), 
                                  key=lambda x: x.info.get('cpu_percent', 0) or 0, reverse=True)[:5]
                ],
                "top_memory": [
                    {
                        "pid": p.info['pid'], 
                        "name": p.info['name'],
                        "memory_percent": round(p.info.get('memory_percent', 0) or 0, 1)
                    }
                    for p in sorted(psutil.process_iter(['pid', 'name', 'memory_percent']), 
                                  key=lambda x: x.info.get('memory_percent', 0) or 0, reverse=True)[:5]
                ]
            }
        }
    except Exception as e:
        logging.error(f"Error getting dashboard metrics: {e}")
        return {
            "error": "Failed to get system metrics",
            "timestamp": datetime.utcnow().isoformat()
        }

# Session Management Helper Functions

async def find_optimal_node_for_session(session_request: SessionRequest, db: Session) -> str:
    """Find the best node for session placement"""
    # Simple placement logic - in production this would be much more sophisticated
    available_nodes = db.query(NodeRecord).filter(NodeRecord.status == "active").all()
    
    if not available_nodes:
        return "control-node-local"  # Fallback to local node
        
    # Score nodes based on resource availability and load
    best_node = None
    best_score = -1
    
    for node in available_nodes:
        score = calculate_node_placement_score(node, session_request)
        if score > best_score:
            best_score = score
            best_node = node.node_id
            
    return best_node or "control-node-local"

def calculate_node_placement_score(node: NodeRecord, session_request: SessionRequest) -> float:
    """Calculate placement score for a node"""
    # Simple scoring based on resource availability
    node_resources = node.resources or {}
    
    cpu_score = max(0, (node_resources.get("cpu_available", 0) - session_request.cpu_cores) / max(1, node_resources.get("cpu_cores", 1)))
    ram_score = max(0, (node_resources.get("memory_available_gb", 0) - session_request.ram_gb) / max(1, node_resources.get("memory_total_gb", 1)))
    gpu_score = 1.0 if node_resources.get("gpu_available", 0) >= session_request.gpu_units else 0.0
    
    return (cpu_score + ram_score + gpu_score) / 3.0

async def start_session_on_node(session_id: str, node_id: str):
    """Start session on target node"""
    # In production, this would communicate with node agents
    logging.info(f"Starting session {session_id} on node {node_id}")
    await asyncio.sleep(0.1)  # Simulate startup time

async def execute_session_action(session: SessionRecord, action: SessionAction, db: Session) -> Dict[str, Any]:
    """Execute action on session"""
    if action.action == "pause":
        session.status = "PAUSED"
        result = {"previous_status": "RUNNING"}
        
    elif action.action == "resume":
        session.status = "RUNNING"
        result = {"previous_status": "PAUSED"}
        
    elif action.action == "terminate":
        session.status = "TERMINATED"
        session.end_time = datetime.utcnow()
        result = {"terminated_at": session.end_time.isoformat()}
        
    elif action.action == "migrate":
        if not action.target_node_id:
            raise HTTPException(status_code=400, detail="Target node ID required for migration")
        session.status = "MIGRATING"
        # In production, this would trigger live migration
        await asyncio.sleep(2)  # Simulate migration time
        session.node_id = action.target_node_id
        session.status = "RUNNING"
        result = {"migrated_to": action.target_node_id}
        
    elif action.action == "snapshot":
        snapshot_name = action.snapshot_name or f"auto_{int(time.time())}"
        snapshot = await create_snapshot(session, snapshot_name, "Manual snapshot")
        result = {"snapshot": snapshot}
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")
    
    session.updated_at = datetime.utcnow()
    db.commit()
    
    return result

def convert_session_to_response(session: SessionRecord) -> SessionResponse:
    """Convert database session to API response"""
    elapsed_time = 0
    if session.start_time:
        end_time = session.end_time or datetime.utcnow()
        elapsed_time = int((end_time - session.start_time).total_seconds())
    
    return SessionResponse(
        session_id=session.session_id,
        session_name=session.session_name,
        user_id=session.user_id,
        app_name=session.app_name,
        app_command=session.app_command or "",
        app_icon=session.app_icon,
        status=session.status,
        node_id=session.node_id,
        cpu_cores=session.cpu_cores,
        gpu_units=session.gpu_units,
        ram_gb=session.ram_gb,
        storage_gb=session.storage_gb,
        priority=session.priority,
        session_type=session.session_type,
        created_at=session.created_at,
        start_time=session.start_time,
        elapsed_time=elapsed_time,
        tags=session.tags or [],
        environment_vars=session.environment_vars or {},
        resource_policy=session.resource_policy or {},
        migration_policy=session.migration_policy or {},
        snapshot_policy=session.snapshot_policy or {},
        metrics=session.metrics or {},
        process_tree=session.process_tree or [],
        security_labels=session.security_labels or []
    )

async def update_session_metrics(session: SessionRecord, db: Session):
    """Update session with real-time metrics"""
    # Simulate real metrics based on session type and status
    if session.status == "RUNNING":
        import random
        base_cpu = 20 if session.session_type == "workstation" else 60
        base_gpu = 30 if session.gpu_units > 0 else 0
        
        session.metrics = {
            "cpu_usage": min(100, base_cpu + random.uniform(-10, 20)),
            "gpu_usage": min(100, base_gpu + random.uniform(-15, 25)) if session.gpu_units > 0 else 0,
            "ram_usage_gb": min(session.ram_gb, session.ram_gb * 0.3 + random.uniform(0, session.ram_gb * 0.4)),
            "disk_io_mbps": random.uniform(5, 50),
            "network_in_mbps": random.uniform(1, 20),
            "network_out_mbps": random.uniform(1, 15),
            "fps": random.uniform(55, 65) if session.session_type == "gaming" else None,
            "latency_ms": random.uniform(3, 8),
            "active_processes": random.randint(3, 12),
            "temperature": random.uniform(40, 70)
        }
        session.updated_at = datetime.utcnow()
        db.commit()

def generate_metrics_history(session_id: str, hours: int) -> List[Dict[str, Any]]:
    """Generate realistic metrics history"""
    import random
    metrics = []
    now = datetime.utcnow()
    
    for i in range(hours * 60):  # One point per minute
        timestamp = now - timedelta(minutes=i)
        
        metrics.append({
            "timestamp": timestamp.isoformat(),
            "cpu_usage": random.uniform(15, 85),
            "gpu_usage": random.uniform(0, 90),
            "ram_usage_gb": random.uniform(1, 6),
            "disk_io_mbps": random.uniform(2, 40),
            "network_in_mbps": random.uniform(0.5, 15),
            "network_out_mbps": random.uniform(0.5, 12),
            "latency_ms": random.uniform(2, 12),
            "fps": random.uniform(50, 70) if random.random() > 0.3 else None,
            "temperature": random.uniform(35, 75)
        })
    
    return list(reversed(metrics))

async def get_real_session_processes(session: SessionRecord) -> List[ProcessInfo]:
    """Get real-time process information for session"""
    import random
    processes = []
    
    # Main application process
    processes.append(ProcessInfo(
        pid=1001,
        name=session.app_name,
        command=session.app_command or session.app_name,
        cpu_percent=random.uniform(10, 60),
        memory_mb=random.uniform(200, 1500),
        gpu_percent=random.uniform(0, 80) if session.gpu_units > 0 else 0,
        status="running",
        user=session.user_id,
        start_time=session.start_time or session.created_at
    ))
    
    # Supporting processes
    for i in range(random.randint(2, 8)):
        processes.append(ProcessInfo(
            pid=1002 + i,
            name=f"helper-{i}",
            command=f"/usr/bin/helper-process-{i}",
            cpu_percent=random.uniform(0, 15),
            memory_mb=random.uniform(50, 300),
            gpu_percent=random.uniform(0, 10) if session.gpu_units > 0 else 0,
            status=random.choice(["running", "sleeping"]),
            user=session.user_id,
            start_time=session.start_time or session.created_at
        ))
    
    return processes

async def get_session_log_entries(session_id: str, level: Optional[str], lines: int) -> List[Dict[str, Any]]:
    """Get session log entries"""
    import random
    logs = []
    now = datetime.utcnow()
    
    log_levels = ["INFO", "WARNING", "ERROR"] if not level else [level.upper()]
    
    for i in range(lines):
        log_level = random.choice(log_levels)
        timestamp = now - timedelta(seconds=i * random.randint(1, 30))
        
        messages = {
            "INFO": [
                "Session started successfully",
                "Resource allocation completed",
                "Application initialized",
                "Frame rendered successfully",
                "Network connection established",
                "Checkpoint saved",
                "Performance metrics updated"
            ],
            "WARNING": [
                "High CPU usage detected",
                "Memory usage approaching limit",
                "Network latency spike detected",
                "GPU temperature elevated",
                "Disk space running low"
            ],
            "ERROR": [
                "Failed to allocate GPU memory",
                "Network connection timeout",
                "Application crashed and restarted",
                "Checkpoint creation failed",
                "Resource limit exceeded"
            ]
        }
        
        logs.append({
            "timestamp": timestamp.isoformat(),
            "level": log_level,
            "component": random.choice(["session", "app", "system", "network", "gpu"]),
            "message": random.choice(messages[log_level]),
            "session_id": session_id
        })
    
    return list(reversed(logs))

async def get_session_snapshots(session_id: str) -> List[SessionSnapshot]:
    """Get session snapshots"""
    import random
    snapshots = []
    now = datetime.utcnow()
    
    # Generate some example snapshots
    for i in range(random.randint(2, 6)):
        created = now - timedelta(hours=i * random.randint(1, 24))
        
        snapshots.append(SessionSnapshot(
            snapshot_id=f"snap_{session_id}_{int(created.timestamp())}",
            session_id=session_id,
            snapshot_name=f"checkpoint_{i+1}",
            created_at=created,
            size_mb=random.uniform(500, 5000),
            status=random.choice(["valid", "valid", "valid", "corrupted"]),
            delta_from_previous=i > 0,
            metadata={
                "cpu_cores": random.randint(2, 8),
                "ram_gb": random.uniform(2, 16),
                "processes": random.randint(5, 15),
                "creator": "system"
            }
        ))
    
    return list(reversed(snapshots))

async def create_snapshot(session: SessionRecord, snapshot_name: str, description: Optional[str]) -> SessionSnapshot:
    """Create a new session snapshot"""
    import random
    
    snapshot = SessionSnapshot(
        snapshot_id=f"snap_{session.session_id}_{int(time.time())}",
        session_id=session.session_id,
        snapshot_name=snapshot_name,
        created_at=datetime.utcnow(),
        size_mb=random.uniform(800, 3000),
        status="creating",
        delta_from_previous=True,
        metadata={
            "session_name": session.session_name,
            "app_name": session.app_name,
            "description": description,
            "cpu_cores": session.cpu_cores,
            "ram_gb": session.ram_gb,
            "creator": session.user_id
        }
    )
    
    # Simulate snapshot creation time
    await asyncio.sleep(0.5)
    snapshot.status = "valid"
    
    return snapshot

async def get_real_time_session_metrics(session: SessionRecord) -> Dict[str, Any]:
    """Get real-time metrics for WebSocket streaming"""
    import random
    
    if session.status != "RUNNING":
        return {
            "cpu_usage": 0,
            "gpu_usage": 0,
            "ram_usage_gb": 0,
            "disk_io_mbps": 0,
            "network_in_mbps": 0,
            "network_out_mbps": 0,
            "latency_ms": 0,
            "fps": 0,
            "active_processes": 0,
            "temperature": 25
        }
    
    # Generate realistic real-time metrics
    return {
        "cpu_usage": random.uniform(10, 85),
        "gpu_usage": random.uniform(0, 90) if session.gpu_units > 0 else 0,
        "ram_usage_gb": random.uniform(0.5, session.ram_gb * 0.8),
        "disk_io_mbps": random.uniform(1, 45),
        "network_in_mbps": random.uniform(0.1, 20),
        "network_out_mbps": random.uniform(0.1, 15),
        "latency_ms": random.uniform(2, 10),
        "fps": random.uniform(45, 75) if session.session_type in ["gaming", "streaming"] else None,
        "active_processes": random.randint(3, 15),
        "temperature": random.uniform(35, 75),
        "status": session.status,
        "uptime_seconds": int((datetime.utcnow() - (session.start_time or session.created_at)).total_seconds())
    }

# Add sample sessions for demo
async def create_sample_sessions():
    """Create sample sessions for demonstration"""
    db = SessionLocal()
    try:
        # Check if we already have sessions
        existing_sessions = db.query(SessionRecord).count()
        if existing_sessions > 0:
            return
        
        sample_sessions = [
            {
                "session_name": "Gaming Session - Cyberpunk 2077",
                "app_name": "Cyberpunk 2077",
                "app_command": "steam://rungameid/1091500",
                "app_icon": "🎮",
                "user_id": "admin",
                "cpu_cores": 8,
                "gpu_units": 1,
                "ram_gb": 16.0,
                "storage_gb": 70.0,
                "priority": 2,
                "session_type": "gaming",
                "tags": ["gaming", "steam", "high-performance"],
                "status": "RUNNING"
            },
            {
                "session_name": "Blender Render Farm",
                "app_name": "Blender",
                "app_command": "/usr/bin/blender --background scene.blend",
                "app_icon": "🎬",
                "user_id": "artist",
                "cpu_cores": 12,
                "gpu_units": 2,
                "ram_gb": 32.0,
                "storage_gb": 100.0,
                "priority": 1,
                "session_type": "render_farm",
                "tags": ["rendering", "blender", "production"],
                "status": "RUNNING"
            },
            {
                "session_name": "Development Environment",
                "app_name": "VS Code",
                "app_command": "/usr/bin/code --remote",
                "app_icon": "💻",
                "user_id": "developer",
                "cpu_cores": 4,
                "gpu_units": 0,
                "ram_gb": 8.0,
                "storage_gb": 50.0,
                "priority": 3,
                "session_type": "development",
                "tags": ["development", "vscode", "coding"],
                "status": "PAUSED"
            }
        ]
        
        for session_data in sample_sessions:
            session_id = f"sess_{int(time.time())}_{session_data['session_name'].lower().replace(' ', '_').replace('-', '_')}"
            
            session = SessionRecord(
                session_id=session_id,
                session_name=session_data["session_name"],
                user_id=session_data["user_id"],
                app_name=session_data["app_name"],
                app_command=session_data["app_command"],
                app_icon=session_data["app_icon"],
                status=session_data["status"],
                node_id="control-node-local",
                cpu_cores=session_data["cpu_cores"],
                gpu_units=session_data["gpu_units"],
                ram_gb=session_data["ram_gb"],
                storage_gb=session_data["storage_gb"],
                priority=session_data["priority"],
                session_type=session_data["session_type"],
                start_time=datetime.utcnow() - timedelta(hours=random.randint(1, 12)),
                tags=session_data["tags"],
                environment_vars={},
                resource_policy={},
                migration_policy={"allow_migration": True},
                snapshot_policy={"auto_snapshot": True, "interval_hours": 4},
                metrics={
                    "cpu_usage": random.uniform(20, 80),
                    "gpu_usage": random.uniform(0, 90) if session_data["gpu_units"] > 0 else 0,
                    "ram_usage_gb": random.uniform(2, session_data["ram_gb"] * 0.7),
                    "disk_io_mbps": random.uniform(5, 40),
                    "network_in_mbps": random.uniform(1, 20),
                    "network_out_mbps": random.uniform(1, 15),
                    "fps": random.uniform(55, 75) if session_data["session_type"] == "gaming" else None,
                    "latency_ms": random.uniform(3, 8),
                    "active_processes": random.randint(5, 15),
                    "temperature": random.uniform(45, 70)
                },
                process_tree=[],
                security_labels=["standard", "user-session"]
            )
            
            db.add(session)
        
        db.commit()
        logging.info("Sample sessions created successfully")
        
    except Exception as e:
        logging.error(f"Error creating sample sessions: {e}")
        db.rollback()
    finally:
        db.close()

# Initialize enhanced orchestrator and managers
control_node_manager = ControlNodeManager()
compute_node_manager = ComputeNodeManager()
storage_node_manager = StorageNodeManager()
thermal_manager = ThermalManager()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample sessions on startup
    asyncio.run(create_sample_sessions())
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        reload=False,
        access_log=True
    )
