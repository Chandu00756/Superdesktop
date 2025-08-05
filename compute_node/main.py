"""
Omega Super Desktop Console v2.0 - Enhanced Compute Node
Heterogeneous compute node supporting CPU, GPU, NPU, FPGA, Edge-TPU
with hot-swapping, self-registration, and auto-scaling capabilities
"""

import asyncio
import logging
import socket
import ssl
import json
import os
import time
import uuid
import platform
import subprocess
import threading
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import psutil
import numpy as np

# Hardware detection and management
try:
    import pynvml  # NVIDIA GPU management
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Container and isolation
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from fastapi import FastAPI, BackgroundTasks
    import aiohttp
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

logger = logging.getLogger(__name__)

NODE_ID = "compute_node_1"
NODE_TYPE = "heterogeneous_compute_node"
CONTROL_NODE_HOST = "localhost"
CONTROL_NODE_PORT = 8443

class HardwareType(Enum):
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda" 
    GPU_OPENCL = "gpu_opencl"
    GPU_METAL = "gpu_metal"
    TPU = "tpu"
    NPU = "npu"
    FPGA = "fpga"
    EDGE_TPU = "edge_tpu"
    NEURAL_ENGINE = "neural_engine"

class ComputeCapability(Enum):
    TRAINING = "ml_training"
    INFERENCE = "ml_inference"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time_processing"
    PARALLEL_COMPUTE = "parallel_compute"
    VECTOR_PROCESSING = "vector_processing"
    MATRIX_OPERATIONS = "matrix_operations"
    SIGNAL_PROCESSING = "signal_processing"

class IsolationLevel(Enum):
    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"
    VM = "virtual_machine"
    HARDWARE = "hardware_isolation"

@dataclass
class HardwareDevice:
    """Represents a specific hardware device"""
    device_id: str
    device_type: HardwareType
    name: str
    memory_mb: int
    compute_units: int
    peak_performance_flops: float
    power_consumption_watts: float
    thermal_design_power: float
    
    # Status and utilization
    current_utilization: float = 0.0
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    available: bool = True
    
    # Capabilities
    supported_precisions: Set[str] = field(default_factory=lambda: {"fp32"})
    supported_frameworks: Set[str] = field(default_factory=set)
    hardware_features: Set[str] = field(default_factory=set)
    
    # Performance characteristics
    memory_bandwidth_gbps: float = 0.0
    cache_size_mb: float = 0.0
    interconnect_bandwidth_gbps: float = 0.0

@dataclass 
class WorkloadExecution:
    """Represents an executing workload with isolation"""
    execution_id: str
    task_spec: Dict[str, Any]
    assigned_devices: List[HardwareDevice]
    isolation_level: IsolationLevel
    container_id: Optional[str] = None
    process_id: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    
    # Resource allocation
    cpu_allocation: float = 1.0  # Number of CPU cores
    memory_allocation_mb: int = 1024
    gpu_memory_allocation_mb: int = 0
    
    # Performance tracking
    current_performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage_history: deque = field(default_factory=lambda: deque(maxlen=1000))
ssl_context = None
try:
    # Check if certificate files exist before trying to load them
    cert_file = "../security/certs/compute_node.crt"
    key_file = "../security/certs/compute_node.key"
    ca_file = "../security/certs/ca.crt"
    
    if os.path.exists(cert_file) and os.path.exists(key_file) and os.path.exists(ca_file):
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        ssl_context.load_verify_locations(cafile=ca_file)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        print("SSL certificates loaded successfully")
    else:
        print("SSL certificate files not found, will use HTTP connection")
        ssl_context = None
except Exception as e:
    print(f"SSL setup failed: {e}")
    print("Will attempt connection without SSL")
    ssl_context = None

# Security: TLS setup with error handling
resources = {
    "cpu": 16,
    "ram": 32,
    "network": "10Gbps"
}

async def register_with_control():
    """Register this compute node with the control center"""
    try:
        print(f"Connecting to control center at {CONTROL_NODE_HOST}:{CONTROL_NODE_PORT}")
        
        # Try connecting with or without SSL
        try:
            reader, writer = await asyncio.open_connection(
                CONTROL_NODE_HOST, CONTROL_NODE_PORT, ssl=ssl_context
            )
        except Exception as ssl_error:
            print(f"SSL connection failed: {ssl_error}")
            print("Trying HTTP connection on port 8443...")
            # Fallback to non-SSL connection
            reader, writer = await asyncio.open_connection(
                CONTROL_NODE_HOST, CONTROL_NODE_PORT
            )
        
        # Send registration data
        registration_data = {
            "node_id": NODE_ID,
            "node_type": NODE_TYPE,
            "resources": resources,
            "status": "active"
        }
        
        message = json.dumps(registration_data)
        writer.write(message.encode())
        await writer.drain()
        
        # Read response
        response = await reader.read(1024)
        print(f"Registration response: {response.decode()}")
        
        writer.close()
        await writer.wait_closed()
        
    except Exception as e:
        print(f"Registration failed: {e}")
        print("Will continue running without registration")

class HardwareManager:
    """Manages heterogeneous hardware resources"""
    
    def __init__(self):
        self.available_devices: Dict[str, HardwareDevice] = {}
        self.device_monitors: Dict[str, threading.Thread] = {}
        self.hardware_capabilities: Set[ComputeCapability] = set()
        
    async def initialize_hardware_detection(self):
        """Detect and initialize all available hardware"""
        try:
            logger.info("Initializing hardware detection...")
            
            # Detect CPU devices
            await self._detect_cpu_devices()
            
            # Detect GPU devices
            await self._detect_gpu_devices()
            
            # Detect specialized accelerators
            await self._detect_specialized_accelerators()
            
            # Start hardware monitoring
            await self._start_hardware_monitoring()
            
            # Determine overall compute capabilities
            self._determine_compute_capabilities()
            
            logger.info(f"Hardware detection complete. Found {len(self.available_devices)} devices")
            
        except Exception as e:
            logger.error(f"Error in hardware detection: {e}")
    
    async def _detect_cpu_devices(self):
        """Detect CPU capabilities and create device representations"""
        try:
            cpu_info = psutil.cpu_freq()
            cpu_count = psutil.cpu_count(logical=False)
            
            # Get CPU architecture and features
            cpu_arch = platform.machine()
            cpu_name = platform.processor()
            
            # Detect CPU features
            features = set()
            if cpu_arch in ['x86_64', 'AMD64']:
                features.update(['avx', 'avx2', 'sse4.1', 'sse4.2'])
            elif cpu_arch in ['arm64', 'aarch64']:
                features.update(['neon', 'fp16'])
            
            # Create CPU device representation
            cpu_device = HardwareDevice(
                device_id="cpu_0",
                device_type=HardwareType.CPU,
                name=cpu_name or f"{cpu_arch} CPU",
                memory_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
                compute_units=cpu_count,
                peak_performance_flops=cpu_count * (cpu_info.max if cpu_info else 3000) * 1e9,
                power_consumption_watts=65.0,
                thermal_design_power=95.0,
                supported_precisions={'fp32', 'fp64', 'int32', 'int64'},
                supported_frameworks={'numpy', 'scikit-learn', 'native'},
                hardware_features=features,
                memory_bandwidth_gbps=25.6,
                cache_size_mb=32.0
            )
            
            self.available_devices[cpu_device.device_id] = cpu_device
            logger.info(f"CPU detected: {cpu_device.name} ({cpu_count} cores)")
            
        except Exception as e:
            logger.error(f"Error detecting CPU devices: {e}")
    
    async def _detect_gpu_devices(self):
        """Detect GPU devices"""
        try:
            # NVIDIA CUDA GPUs
            if PYNVML_AVAILABLE:
                await self._detect_nvidia_gpus()
            
            # Apple Metal GPUs
            if platform.system() == "Darwin":
                await self._detect_metal_gpus()
            
        except Exception as e:
            logger.error(f"Error detecting GPU devices: {e}")
    
    async def _detect_nvidia_gpus(self):
        """Detect NVIDIA CUDA GPUs"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_device = HardwareDevice(
                    device_id=f"cuda_{i}",
                    device_type=HardwareType.GPU_CUDA,
                    name=name,
                    memory_mb=memory_info.total // (1024 * 1024),
                    compute_units=2048,  # Simplified
                    peak_performance_flops=10e12,  # 10 TFLOPS estimate
                    power_consumption_watts=250.0,
                    thermal_design_power=250.0,
                    supported_precisions={'fp32', 'fp16', 'int8', 'int32'},
                    supported_frameworks={'pytorch', 'tensorflow', 'cupy'},
                    hardware_features={'tensor_cores', 'cuda'},
                    memory_bandwidth_gbps=500.0
                )
                
                self.available_devices[gpu_device.device_id] = gpu_device
                logger.info(f"NVIDIA GPU detected: {name}")
                
        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPUs: {e}")
    
    async def _detect_metal_gpus(self):
        """Detect Apple Metal GPUs"""
        try:
            if platform.system() == "Darwin":
                metal_device = HardwareDevice(
                    device_id="metal_0",
                    device_type=HardwareType.GPU_METAL,
                    name="Apple GPU",
                    memory_mb=8192,
                    compute_units=1024,
                    peak_performance_flops=8e12,
                    power_consumption_watts=50.0,
                    thermal_design_power=75.0,
                    supported_precisions={'fp32', 'fp16'},
                    supported_frameworks={'metal', 'pytorch_mps'},
                    hardware_features={'metal', 'unified_memory'},
                    memory_bandwidth_gbps=200.0
                )
                
                self.available_devices[metal_device.device_id] = metal_device
                logger.info("Apple Metal GPU detected")
                
        except Exception as e:
            logger.error(f"Error detecting Metal GPUs: {e}")
    
    async def _detect_specialized_accelerators(self):
        """Detect TPU, NPU, FPGA devices"""
        # Simplified placeholder for specialized hardware detection
        logger.info("Specialized accelerator detection not fully implemented")
    
    async def _start_hardware_monitoring(self):
        """Start monitoring threads for all devices"""
        for device_id, device in self.available_devices.items():
            monitor_thread = threading.Thread(
                target=self._monitor_device,
                args=(device,),
                daemon=True
            )
            monitor_thread.start()
            self.device_monitors[device_id] = monitor_thread
    
    def _monitor_device(self, device: HardwareDevice):
        """Monitor individual device performance and health"""
        while device.available:
            try:
                if device.device_type == HardwareType.CPU:
                    device.current_utilization = psutil.cpu_percent(interval=1) / 100.0
                elif device.device_type == HardwareType.GPU_CUDA and PYNVML_AVAILABLE:
                    device_index = int(device.device_id.split('_')[1])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    device.current_utilization = utilization.gpu / 100.0
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring device {device.device_id}: {e}")
                time.sleep(10)
    
    def _determine_compute_capabilities(self):
        """Determine overall compute capabilities"""
        self.hardware_capabilities.clear()
        self.hardware_capabilities.add(ComputeCapability.BATCH_PROCESSING)
        
        gpu_devices = [d for d in self.available_devices.values() 
                      if d.device_type in [HardwareType.GPU_CUDA, HardwareType.GPU_METAL]]
        
        if gpu_devices:
            self.hardware_capabilities.add(ComputeCapability.TRAINING)
            self.hardware_capabilities.add(ComputeCapability.PARALLEL_COMPUTE)
            self.hardware_capabilities.add(ComputeCapability.MATRIX_OPERATIONS)

class EnhancedComputeNode:
    """Enhanced Compute Node with heterogeneous hardware support"""
    
    def __init__(self, node_id: str = None, config: Dict[str, Any] = None):
        self.node_id = node_id or NODE_ID
        self.config = config or {}
        
        # Hardware management
        self.hardware_manager = HardwareManager()
        
        # Enhanced capabilities
        self.hot_swap_manager = HotSwapManager(self.hardware_manager)
        self.auto_scaling_manager = AutoScalingManager(self.hardware_manager)
        self.sandbox_manager = AdvancedSandboxManager()
        self.self_registration_manager = SelfRegistrationManager(self)
        
        # Workload management
        self.active_executions: Dict[str, WorkloadExecution] = {}
        self.execution_queue = asyncio.Queue()
        
        # Self-registration
        self.control_node_url = f"{'https' if ssl_context else 'http'}://{CONTROL_NODE_HOST}:{CONTROL_NODE_PORT}"
        
        # Performance metrics
        self.performance_metrics = {
            'total_tasks_executed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'resource_utilization': 0.0
        }
        
        self.health_status = "initializing"
        
        logger.info(f"Enhanced Compute Node {self.node_id} initialized")
    
    async def initialize(self):
        """Initialize compute node with all capabilities"""
        try:
            self.health_status = "initializing"
            
            # Initialize hardware detection
            await self.hardware_manager.initialize_hardware_detection()
            
            # Start enhanced services
            await self.hot_swap_manager.start_monitoring()
            await self.auto_scaling_manager.start_scaling_monitor()
            
            # Start background services
            await self._start_background_services()
            
            # Enhanced self-registration
            await self.self_registration_manager.start_self_registration()
            
            self.health_status = "active"
            logger.info(f"Compute Node {self.node_id} fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing compute node: {e}")
            self.health_status = "failed"
            raise
    
    async def _start_background_services(self):
        """Start background monitoring services"""
        try:
            asyncio.create_task(self._resource_monitoring_loop())
            asyncio.create_task(self._health_reporting_loop())
            logger.info("Background services started")
            
        except Exception as e:
            logger.error(f"Error starting background services: {e}")
    
    async def _self_register_with_control_node(self):
        """Self-register with control node"""
        try:
            node_info = await self._prepare_node_info()
            
            if WEB_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.control_node_url}/api/nodes/register",
                        json=node_info,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Successfully registered: {result}")
                        else:
                            logger.error(f"Registration failed: {response.status}")
            else:
                logger.info("Web libraries not available, using basic registration")
                await self.register_with_control()
                
        except Exception as e:
            logger.error(f"Error self-registering: {e}")
    
    async def _prepare_node_info(self) -> Dict[str, Any]:
        """Prepare node information for registration"""
        try:
            hostname = platform.node()
            ip_address = await self._get_local_ip()
            
            total_cpu_cores = sum(d.compute_units for d in self.hardware_manager.available_devices.values() 
                                if d.device_type == HardwareType.CPU)
            total_memory_gb = sum(d.memory_mb for d in self.hardware_manager.available_devices.values() 
                                if d.device_type == HardwareType.CPU) / 1024.0
            total_gpu_count = len([d for d in self.hardware_manager.available_devices.values() 
                                 if d.device_type in [HardwareType.GPU_CUDA, HardwareType.GPU_METAL]])
            
            capabilities = [cap.value for cap in self.hardware_manager.hardware_capabilities]
            
            return {
                'node_id': self.node_id,
                'node_type': NODE_TYPE,
                'status': self.health_status,
                'hostname': hostname,
                'ip_address': ip_address,
                'port': 8001,
                'cpu_cores': total_cpu_cores,
                'memory_gb': total_memory_gb,
                'gpu_count': total_gpu_count,
                'capabilities': capabilities,
                'hardware_devices': [
                    {
                        'device_id': device.device_id,
                        'device_type': device.device_type.value,
                        'name': device.name,
                        'memory_mb': device.memory_mb,
                        'compute_units': device.compute_units
                    }
                    for device in self.hardware_manager.available_devices.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error preparing node info: {e}")
            return {}
    
    async def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except:
            return "127.0.0.1"
    
    async def _resource_monitoring_loop(self):
        """Monitor resource utilization"""
        while self.health_status == "active":
            try:
                total_utilization = 0.0
                device_count = 0
                
                for device in self.hardware_manager.available_devices.values():
                    total_utilization += device.current_utilization
                    device_count += 1
                
                if device_count > 0:
                    self.performance_metrics['resource_utilization'] = total_utilization / device_count
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _health_reporting_loop(self):
        """Report health status to control node"""
        while self.health_status == "active":
            try:
                await asyncio.sleep(30)
                logger.info(f"Node {self.node_id} health check - Status: {self.health_status}")
                
            except Exception as e:
                logger.error(f"Error in health reporting: {e}")
                await asyncio.sleep(30)
    
    async def execute_workload(self, workload_id: str, workload_spec: Dict[str, Any]) -> bool:
        """Execute workload with appropriate isolation"""
        try:
            # Determine isolation level
            isolation_level = IsolationLevel(workload_spec.get('isolation_level', 'process'))
            resource_limits = workload_spec.get('resource_limits', {})
            
            # Create sandbox
            sandbox_created = await self.sandbox_manager.create_sandbox(
                workload_id, isolation_level, resource_limits
            )
            
            if not sandbox_created:
                logger.error(f"Failed to create sandbox for {workload_id}")
                return False
            
            # Execute workload in sandbox
            execution = WorkloadExecution(
                workload_id=workload_id,
                node_id=self.node_id,
                isolation_level=isolation_level,
                status="running",
                resource_allocation=resource_limits,
                start_time=datetime.utcnow()
            )
            
            self.active_executions[workload_id] = execution
            
            # Update performance metrics
            self.performance_metrics['total_tasks_executed'] += 1
            
            logger.info(f"Workload {workload_id} started with {isolation_level.value} isolation")
            return True
            
        except Exception as e:
            logger.error(f"Workload execution failed: {e}")
            self.performance_metrics['failed_tasks'] += 1
            return False
    
    async def stop_workload(self, workload_id: str) -> bool:
        """Stop workload and cleanup sandbox"""
        try:
            if workload_id in self.active_executions:
                # Stop execution
                execution = self.active_executions[workload_id]
                execution.status = "completed"
                execution.end_time = datetime.utcnow()
                
                # Destroy sandbox
                await self.sandbox_manager.destroy_sandbox(workload_id)
                
                # Update metrics
                self.performance_metrics['successful_tasks'] += 1
                
                # Remove from active executions
                del self.active_executions[workload_id]
                
                logger.info(f"Workload {workload_id} completed successfully")
                return True
            else:
                logger.warning(f"Workload {workload_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Workload termination failed: {e}")
            return False

# Enhanced Compute Features - Hot-swapping, Auto-scaling, Advanced Sandboxing
class HotSwapManager:
    """Manages hot-swapping of hardware components"""
    
    def __init__(self, hardware_manager):
        self.hardware_manager = hardware_manager
        self.swap_events = deque(maxlen=100)
        self.monitoring_enabled = True
        
    async def start_monitoring(self):
        """Start monitoring for hot-swap events"""
        try:
            asyncio.create_task(self._monitor_hardware_changes())
            logger.info("Hot-swap monitoring started")
        except Exception as e:
            logger.error(f"Hot-swap monitoring failed: {e}")
    
    async def _monitor_hardware_changes(self):
        """Monitor for hardware changes"""
        previous_devices = set(self.hardware_manager.available_devices.keys())
        
        while self.monitoring_enabled:
            try:
                current_devices = set(self.hardware_manager.available_devices.keys())
                
                # Detect removed devices
                removed_devices = previous_devices - current_devices
                for device_id in removed_devices:
                    await self._handle_device_removal(device_id)
                
                # Detect new devices
                new_devices = current_devices - previous_devices
                for device_id in new_devices:
                    await self._handle_device_addition(device_id)
                
                previous_devices = current_devices
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Hardware monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_device_removal(self, device_id: str):
        """Handle device removal event"""
        try:
            event = {
                'type': 'device_removed',
                'device_id': device_id,
                'timestamp': datetime.utcnow(),
                'action': 'workload_migration_needed'
            }
            self.swap_events.append(event)
            logger.warning(f"Device {device_id} removed - initiating workload migration")
            
        except Exception as e:
            logger.error(f"Device removal handling failed: {e}")
    
    async def _handle_device_addition(self, device_id: str):
        """Handle device addition event"""
        try:
            device = self.hardware_manager.available_devices[device_id]
            event = {
                'type': 'device_added',
                'device_id': device_id,
                'device_type': device.device_type.value,
                'timestamp': datetime.utcnow(),
                'action': 'device_ready_for_workloads'
            }
            self.swap_events.append(event)
            logger.info(f"New device {device_id} added and ready for workloads")
            
        except Exception as e:
            logger.error(f"Device addition handling failed: {e}")

class AutoScalingManager:
    """Manages auto-scaling of compute resources"""
    
    def __init__(self, hardware_manager):
        self.hardware_manager = hardware_manager
        self.scaling_policies = {
            'cpu_threshold_scale_up': 0.8,
            'cpu_threshold_scale_down': 0.3,
            'memory_threshold_scale_up': 0.85,
            'memory_threshold_scale_down': 0.4,
            'scale_cooldown_seconds': 300
        }
        self.last_scaling_action = datetime.utcnow()
        self.active_scaling = True
        
    async def start_scaling_monitor(self):
        """Start auto-scaling monitoring"""
        try:
            asyncio.create_task(self._scaling_monitor_loop())
            logger.info("Auto-scaling monitoring started")
        except Exception as e:
            logger.error(f"Auto-scaling start failed: {e}")
    
    async def _scaling_monitor_loop(self):
        """Monitor resource utilization and trigger scaling"""
        while self.active_scaling:
            try:
                # Check if cooldown period has passed
                time_since_last_action = (datetime.utcnow() - self.last_scaling_action).seconds
                if time_since_last_action < self.scaling_policies['scale_cooldown_seconds']:
                    await asyncio.sleep(30)
                    continue
                
                # Analyze current resource utilization
                await self._analyze_and_scale()
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Scaling monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_and_scale(self):
        """Analyze utilization and make scaling decisions"""
        try:
            # Get average utilization across all devices
            total_utilization = 0.0
            device_count = 0
            
            for device in self.hardware_manager.available_devices.values():
                if device.available:
                    total_utilization += device.current_utilization
                    device_count += 1
            
            if device_count == 0:
                return
                
            avg_utilization = total_utilization / device_count
            
            # Make scaling decisions
            if avg_utilization > self.scaling_policies['cpu_threshold_scale_up']:
                await self._scale_up()
            elif avg_utilization < self.scaling_policies['cpu_threshold_scale_down']:
                await self._scale_down()
                
        except Exception as e:
            logger.error(f"Scaling analysis failed: {e}")
    
    async def _scale_up(self):
        """Scale up resources"""
        try:
            logger.info("High utilization detected - requesting scale up")
            # In a real implementation, this would request additional resources
            # from the control node or activate idle devices
            self.last_scaling_action = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
    
    async def _scale_down(self):
        """Scale down resources"""
        try:
            logger.info("Low utilization detected - considering scale down")
            # In a real implementation, this would gracefully reduce resources
            # or put devices into power-saving mode
            self.last_scaling_action = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")

class AdvancedSandboxManager:
    """Advanced sandboxing for workload isolation"""
    
    def __init__(self):
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.isolation_levels = {
            IsolationLevel.PROCESS: self._create_process_sandbox,
            IsolationLevel.CONTAINER: self._create_container_sandbox,
            IsolationLevel.VM: self._create_vm_sandbox,
            IsolationLevel.HARDWARE: self._create_hardware_sandbox
        }
        
    async def create_sandbox(self, workload_id: str, isolation_level: IsolationLevel, 
                           resource_limits: Dict[str, Any]) -> bool:
        """Create isolated sandbox for workload"""
        try:
            if isolation_level in self.isolation_levels:
                sandbox_info = await self.isolation_levels[isolation_level](
                    workload_id, resource_limits
                )
                self.active_sandboxes[workload_id] = sandbox_info
                logger.info(f"Sandbox created for {workload_id} with {isolation_level.value} isolation")
                return True
            else:
                logger.error(f"Unsupported isolation level: {isolation_level}")
                return False
                
        except Exception as e:
            logger.error(f"Sandbox creation failed: {e}")
            return False
    
    async def _create_process_sandbox(self, workload_id: str, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Create process-level isolation"""
        return {
            'type': 'process',
            'workload_id': workload_id,
            'cpu_limit': resource_limits.get('cpu_cores', 1),
            'memory_limit_mb': resource_limits.get('memory_mb', 1024),
            'network_isolation': False,
            'filesystem_isolation': False
        }
    
    async def _create_container_sandbox(self, workload_id: str, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Create container-level isolation"""
        sandbox_info = {
            'type': 'container',
            'workload_id': workload_id,
            'cpu_limit': resource_limits.get('cpu_cores', 2),
            'memory_limit_mb': resource_limits.get('memory_mb', 2048),
            'network_isolation': True,
            'filesystem_isolation': True
        }
        
        if DOCKER_AVAILABLE:
            # In a real implementation, would create Docker container
            sandbox_info['container_id'] = f"omega-{workload_id}"
            sandbox_info['image'] = 'omega-compute-runtime'
            
        return sandbox_info
    
    async def _create_vm_sandbox(self, workload_id: str, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Create VM-level isolation"""
        return {
            'type': 'vm',
            'workload_id': workload_id,
            'cpu_cores': resource_limits.get('cpu_cores', 2),
            'memory_mb': resource_limits.get('memory_mb', 4096),
            'disk_size_gb': resource_limits.get('disk_gb', 20),
            'network_isolation': True,
            'filesystem_isolation': True,
            'hypervisor': 'qemu-kvm'
        }
    
    async def _create_hardware_sandbox(self, workload_id: str, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Create hardware-level isolation"""
        return {
            'type': 'hardware',
            'workload_id': workload_id,
            'dedicated_cpu_cores': resource_limits.get('cpu_cores', 4),
            'dedicated_memory_mb': resource_limits.get('memory_mb', 8192),
            'dedicated_gpu': resource_limits.get('gpu_required', False),
            'numa_node': resource_limits.get('numa_node', 0),
            'sr_iov_enabled': True,
            'iommu_isolation': True
        }
    
    async def destroy_sandbox(self, workload_id: str) -> bool:
        """Destroy sandbox and cleanup resources"""
        try:
            if workload_id in self.active_sandboxes:
                sandbox_info = self.active_sandboxes[workload_id]
                
                # Cleanup based on sandbox type
                if sandbox_info['type'] == 'container' and DOCKER_AVAILABLE:
                    # In real implementation, would stop and remove container
                    pass
                elif sandbox_info['type'] == 'vm':
                    # In real implementation, would stop VM
                    pass
                
                del self.active_sandboxes[workload_id]
                logger.info(f"Sandbox destroyed for {workload_id}")
                return True
            else:
                logger.warning(f"Sandbox not found for {workload_id}")
                return False
                
        except Exception as e:
            logger.error(f"Sandbox destruction failed: {e}")
            return False

class SelfRegistrationManager:
    """Enhanced self-registration with automatic discovery"""
    
    def __init__(self, compute_node):
        self.compute_node = compute_node
        self.registration_retry_interval = 30
        self.heartbeat_interval = 15
        self.discovery_methods = [
            self._discover_via_dns,
            self._discover_via_mdns,
            self._discover_via_broadcast
        ]
        
    async def start_self_registration(self):
        """Start self-registration process"""
        try:
            # Try multiple discovery methods
            control_node_url = await self._discover_control_node()
            
            if control_node_url:
                self.compute_node.control_node_url = control_node_url
                await self._register_with_retries()
                asyncio.create_task(self._heartbeat_loop())
            else:
                logger.error("Could not discover control node")
                
        except Exception as e:
            logger.error(f"Self-registration failed: {e}")
    
    async def _discover_control_node(self) -> Optional[str]:
        """Discover control node using multiple methods"""
        for discovery_method in self.discovery_methods:
            try:
                url = await discovery_method()
                if url:
                    logger.info(f"Control node discovered: {url}")
                    return url
            except Exception as e:
                logger.warning(f"Discovery method failed: {e}")
        
        # Fallback to default
        return f"http://{CONTROL_NODE_HOST}:{CONTROL_NODE_PORT}"
    
    async def _discover_via_dns(self) -> Optional[str]:
        """Discover via DNS SRV records"""
        # In real implementation, would query DNS SRV records
        return None
    
    async def _discover_via_mdns(self) -> Optional[str]:
        """Discover via mDNS/Bonjour"""
        # In real implementation, would use mDNS
        return None
    
    async def _discover_via_broadcast(self) -> Optional[str]:
        """Discover via network broadcast"""
        # In real implementation, would send UDP broadcast
        return None
    
    async def _register_with_retries(self):
        """Register with control node with retry logic"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                success = await self.compute_node._self_register_with_control_node()
                if success:
                    logger.info("Registration successful")
                    return
                    
            except Exception as e:
                logger.warning(f"Registration attempt {retry_count + 1} failed: {e}")
            
            retry_count += 1
            await asyncio.sleep(self.registration_retry_interval)
        
        logger.error("Registration failed after all retries")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                # In real implementation, would send heartbeat to control node
                
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(self.heartbeat_interval)

# Legacy functions for compatibility
async def register_with_control():
    """Register this compute node with the control center"""
    try:
        print(f"Connecting to control center at {CONTROL_NODE_HOST}:{CONTROL_NODE_PORT}")
        
        # Simulated registration for basic compatibility
        node_info = {
            "node_id": NODE_ID,
            "node_type": NODE_TYPE,
            "resources": resources,
            "status": "active"
        }
        
        print(f"Node registered: {node_info}")
        return True
        
    except Exception as e:
        print(f"Registration failed: {e}")
        return False

async def heartbeat_loop():
    """Send periodic heartbeats to control center"""
    while True:
        try:
            print(f"Compute node {NODE_ID} is running...")
            await asyncio.sleep(30)
        except Exception as e:
            print(f"Heartbeat error: {e}")
            await asyncio.sleep(30)

async def main():
    """Main function - run enhanced compute node or fallback to legacy"""
    try:
        # Try to run enhanced compute node
        compute_node = EnhancedComputeNode()
        await compute_node.initialize()
        
        # Keep running
        while True:
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Enhanced mode failed: {e}")
        logger.info("Falling back to legacy mode")
        
        # Fallback to legacy registration
        await register_with_control()
        
        # Legacy heartbeat loop
        while True:
            await asyncio.sleep(5)
            logging.info("Compute node heartbeat (legacy mode)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Add simulated resources for legacy compatibility
    resources = {
        "cpu": 16,
        "ram": 32,
        "network": "10Gbps"
    }
    
    asyncio.run(main())
