"""
Omega Super Desktop Console - Advanced Compute Node
Initial prototype distributed processing agent with AI-driven optimization.
"""

import asyncio
import logging
import json
import time
import uuid
import psutil
import threading
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

# GPU libraries (simulation for initial prototype)
import subprocess
import platform

# Network and communication
import aiohttp
import websockets
from cryptography.fernet import Fernet

# Custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.models import NodeInfo, setup_logging

# Node configuration
NODE_ID = f"compute_node_{uuid.uuid4().hex[:8]}"
NODE_TYPE = "hybrid_node"  # CPU + GPU + Memory
CONTROL_NODE_URL = "http://localhost:8443"
AUTH_TOKEN = None

# Hardware monitoring
class HardwareMonitor:
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.gpu_available = self._detect_gpu()
        
    def _detect_gpu(self) -> int:
        """Detect available GPUs"""
        try:
            if platform.system() == "Linux":
                result = subprocess.run(['nvidia-smi', '-L'], 
                                      capture_output=True, text=True)
                return len([line for line in result.stdout.split('\n') if 'GPU' in line])
            elif platform.system() == "Darwin":  # macOS
                # Check for Metal-capable GPUs
                return 1 if self._has_metal_gpu() else 0
            return 0
        except:
            return 0
    
    def _has_metal_gpu(self) -> bool:
        """Check for Metal-capable GPU on macOS"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            return 'Metal' in result.stdout
        except:
            return False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get real-time hardware metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # CPU temperature (Linux only)
        cpu_temp = 45.0
        try:
            if platform.system() == "Linux":
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = temps['coretemp'][0].current
        except:
            pass
        
        # GPU metrics simulation
        gpu_util = np.random.uniform(20, 80) if self.gpu_available > 0 else 0
        gpu_temp = np.random.uniform(40, 75) if self.gpu_available > 0 else 0
        gpu_memory_used = np.random.uniform(2, 10) * 1024**3 if self.gpu_available > 0 else 0
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_temp": cpu_temp,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "gpu_count": self.gpu_available,
            "gpu_util": gpu_util,
            "gpu_temp": gpu_temp,
            "gpu_memory_used": gpu_memory_used,
            "timestamp": time.time()
        }

# Task execution engine
class TaskExecutor:
    def __init__(self):
        self.active_tasks = {}
        self.task_history = []
        self.max_concurrent_tasks = psutil.cpu_count()
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distributed task with performance monitoring"""
        task_id = task_data.get('task_id', str(uuid.uuid4()))
        task_type = task_data.get('task_type', 'generic')
        
        start_time = time.time()
        self.active_tasks[task_id] = {
            "start_time": start_time,
            "task_type": task_type,
            "status": "running"
        }
        
        try:
            # Route task based on type
            if task_type == "cpu_intensive":
                result = await self._execute_cpu_task(task_data)
            elif task_type == "gpu_compute":
                result = await self._execute_gpu_task(task_data)
            elif task_type == "memory_intensive":
                result = await self._execute_memory_task(task_data)
            else:
                result = await self._execute_generic_task(task_data)
            
            # Record completion
            duration = time.time() - start_time
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["duration"] = duration
            
            self.task_history.append({
                "task_id": task_id,
                "task_type": task_type,
                "duration": duration,
                "completed_at": datetime.utcnow().isoformat(),
                "result_size": len(str(result))
            })
            
            # Keep only last 100 tasks in history
            if len(self.task_history) > 100:
                self.task_history.pop(0)
                
            return {
                "task_id": task_id,
                "status": "completed",
                "duration": duration,
                "result": result
            }
            
        except Exception as e:
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_cpu_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """CPU-intensive computation simulation"""
        size = task_data.get('matrix_size', 1000)
        
        # Matrix multiplication simulation
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        start = time.time()
        result = np.dot(a, b)
        duration = time.time() - start
        
        return {
            "operation": "matrix_multiplication",
            "matrix_size": size,
            "computation_time": duration,
            "flops": (2 * size ** 3) / duration,
            "result_checksum": float(np.sum(result))
        }
    
    async def _execute_gpu_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """GPU computation simulation"""
        # Simulate GPU workload
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        
        return {
            "operation": "gpu_simulation",
            "gpu_cores_used": 2048,
            "memory_allocated": "4GB",
            "performance": "856 GFLOPS"
        }
    
    async def _execute_memory_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Memory-intensive task simulation"""
        size_mb = task_data.get('size_mb', 100)
        
        # Allocate and process memory
        data = np.random.bytes(size_mb * 1024 * 1024)
        
        # Simulate processing
        checksum = sum(data) % (2**32)
        
        return {
            "operation": "memory_processing",
            "size_mb": size_mb,
            "checksum": checksum,
            "bandwidth": f"{size_mb / 0.1:.2f} MB/s"
        }
    
    async def _execute_generic_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic task execution"""
        duration = task_data.get('duration', 1.0)
        await asyncio.sleep(duration)
        
        return {
            "operation": "generic_task",
            "duration": duration,
            "status": "completed"
        }

# Network latency optimizer
class LatencyOptimizer:
    def __init__(self):
        self.latency_history = []
        self.optimization_params = {
            "buffer_size": 64 * 1024,
            "compression_level": 6,
            "prediction_window": 50  # ms
        }
    
    def record_latency(self, latency_ms: float):
        """Record network latency measurement"""
        self.latency_history.append({
            "timestamp": time.time(),
            "latency_ms": latency_ms
        })
        
        # Keep only last 1000 measurements
        if len(self.latency_history) > 1000:
            self.latency_history.pop(0)
        
        # Auto-optimize parameters
        self._optimize_parameters()
    
    def _optimize_parameters(self):
        """Dynamically adjust parameters based on latency patterns"""
        if len(self.latency_history) < 10:
            return
        
        recent_latencies = [m["latency_ms"] for m in self.latency_history[-10:]]
        avg_latency = np.mean(recent_latencies)
        
        # Adjust buffer size based on latency
        if avg_latency > 20:
            self.optimization_params["buffer_size"] = min(128 * 1024, 
                                                         self.optimization_params["buffer_size"] * 1.2)
        elif avg_latency < 5:
            self.optimization_params["buffer_size"] = max(32 * 1024,
                                                         self.optimization_params["buffer_size"] * 0.8)
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get current optimization parameters"""
        return self.optimization_params.copy()

# Predictive caching system
class PredictiveCache:
    def __init__(self, max_size_mb: int = 1024):
        self.cache = {}
        self.access_patterns = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        
    def predict_access(self, key: str) -> float:
        """Predict probability of accessing a key"""
        if key not in self.access_patterns:
            return 0.1  # Low probability for new keys
        
        pattern = self.access_patterns[key]
        time_since_last = time.time() - pattern["last_access"]
        access_frequency = pattern["access_count"] / max(1, pattern["time_span"])
        
        # Simple prediction model
        probability = access_frequency * np.exp(-time_since_last / 3600)  # Decay over 1 hour
        return min(1.0, probability)
    
    def cache_data(self, key: str, data: bytes, priority: float = 1.0):
        """Cache data with priority and prediction"""
        data_size = len(data)
        
        # Evict if necessary
        while self.current_size + data_size > self.max_size_bytes and self.cache:
            self._evict_lru()
        
        self.cache[key] = {
            "data": data,
            "cached_at": time.time(),
            "access_count": 0,
            "priority": priority,
            "size": data_size
        }
        
        self.current_size += data_size
        
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item considering priority
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k]["cached_at"] / self.cache[k]["priority"])
        
        self.current_size -= self.cache[lru_key]["size"]
        del self.cache[lru_key]

# Main compute node class
class OmegaComputeNode:
    def __init__(self):
        self.logger = setup_logging(f"ComputeNode-{NODE_ID}")
        self.hardware_monitor = HardwareMonitor()
        self.task_executor = TaskExecutor()
        self.latency_optimizer = LatencyOptimizer()
        self.predictive_cache = PredictiveCache()
        self.running = False
        self.heartbeat_interval = 5.0
        
    async def start(self):
        """Start the compute node"""
        self.running = True
        self.logger.info(f"Starting Omega Compute Node {NODE_ID}")
        
        # Authenticate with control node
        await self._authenticate()
        
        # Register with control node
        await self._register_node()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._metrics_reporter()),
            asyncio.create_task(self._task_listener()),
            asyncio.create_task(self._latency_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        finally:
            await self._cleanup()
    
    async def _authenticate(self):
        """Authenticate with control node"""
        global AUTH_TOKEN
        
        auth_data = {
            "username": "compute_node",
            "password": "omega123"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONTROL_NODE_URL}/auth/login",
                    params=auth_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        AUTH_TOKEN = result["access_token"]
                        self.logger.info("Authentication successful")
                    else:
                        self.logger.error(f"Authentication failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
    
    async def _register_node(self):
        """Register this node with the control plane"""
        metrics = self.hardware_monitor.get_current_metrics()
        
        node_info = {
            "node_id": NODE_ID,
            "node_type": NODE_TYPE,
            "status": "active",
            "resources": {
                "cpu_cores": self.hardware_monitor.cpu_count,
                "cpu_available": self.hardware_monitor.cpu_count,
                "memory_total": self.hardware_monitor.memory_total,
                "memory_available": psutil.virtual_memory().available,
                "gpu_count": self.hardware_monitor.gpu_available,
                "gpu_available": self.hardware_monitor.gpu_available,
                "network": "10Gbps"
            },
            "performance_metrics": {
                "cpu_temp": metrics["cpu_temp"],
                "avg_latency_ms": 5.0,
                "task_completion_rate": 0.95
            }
        }
        
        try:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONTROL_NODE_URL}/api/v1/nodes/register",
                    json=node_info,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        self.logger.info("Node registration successful")
                    else:
                        self.logger.error(f"Registration failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to control node"""
        while self.running:
            try:
                metrics = self.hardware_monitor.get_current_metrics()
                
                heartbeat_data = {
                    "node_id": NODE_ID,
                    "timestamp": time.time(),
                    "status": "active",
                    "metrics": metrics,
                    "active_tasks": len(self.task_executor.active_tasks)
                }
                
                headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{CONTROL_NODE_URL}/api/v1/nodes/{NODE_ID}/heartbeat",
                        json=heartbeat_data,
                        headers=headers
                    ) as response:
                        if response.status != 200:
                            self.logger.warning(f"Heartbeat failed: {response.status}")
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _metrics_reporter(self):
        """Report detailed metrics to control node"""
        while self.running:
            try:
                # Collect comprehensive metrics
                metrics = {
                    "hardware": self.hardware_monitor.get_current_metrics(),
                    "tasks": {
                        "active": len(self.task_executor.active_tasks),
                        "completed": len(self.task_executor.task_history),
                        "success_rate": self._calculate_success_rate()
                    },
                    "cache": {
                        "size_mb": self.predictive_cache.current_size / (1024*1024),
                        "hit_rate": 0.85  # Simulated
                    },
                    "network": {
                        "latency_params": self.latency_optimizer.get_optimization_params(),
                        "avg_latency_ms": self._get_avg_latency()
                    }
                }
                
                headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{CONTROL_NODE_URL}/api/v1/metrics/node",
                        json={"node_id": NODE_ID, "metrics": metrics},
                        headers=headers
                    ) as response:
                        pass  # Metrics reporting is best-effort
                
            except Exception as e:
                self.logger.debug(f"Metrics reporting error: {e}")
            
            await asyncio.sleep(30)  # Report every 30 seconds
    
    async def _task_listener(self):
        """Listen for task assignments from control node"""
        # In initial prototype, this would use message queue (Redis, RabbitMQ)
        # For now, simulate task reception
        while self.running:
            try:
                # Simulate receiving tasks
                if np.random.random() < 0.1:  # 10% chance per iteration
                    task = {
                        "task_id": str(uuid.uuid4()),
                        "task_type": np.random.choice(["cpu_intensive", "gpu_compute", "memory_intensive"]),
                        "matrix_size": np.random.randint(500, 2000),
                        "priority": np.random.randint(1, 10)
                    }
                    
                    # Execute task asynchronously
                    asyncio.create_task(self._handle_task(task))
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Task listener error: {e}")
    
    async def _handle_task(self, task: Dict[str, Any]):
        """Handle incoming task execution"""
        self.logger.info(f"Executing task {task['task_id']} of type {task['task_type']}")
        
        result = await self.task_executor.execute_task(task)
        
        # Report task completion
        try:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONTROL_NODE_URL}/api/v1/tasks/{task['task_id']}/complete",
                    json=result,
                    headers=headers
                ) as response:
                    pass
        except Exception as e:
            self.logger.error(f"Task result reporting error: {e}")
    
    async def _latency_monitor(self):
        """Monitor and optimize network latency"""
        while self.running:
            try:
                # Measure latency to control node
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{CONTROL_NODE_URL}/health") as response:
                        latency_ms = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            self.latency_optimizer.record_latency(latency_ms)
                
            except Exception as e:
                self.logger.debug(f"Latency monitoring error: {e}")
            
            await asyncio.sleep(10)
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        if not self.task_executor.task_history:
            return 1.0
        
        successful = len([t for t in self.task_executor.task_history 
                         if "error" not in t])
        return successful / len(self.task_executor.task_history)
    
    def _get_avg_latency(self) -> float:
        """Get average network latency"""
        if not self.latency_optimizer.latency_history:
            return 5.0
        
        recent = self.latency_optimizer.latency_history[-10:]
        return np.mean([m["latency_ms"] for m in recent])
    
    async def _cleanup(self):
        """Cleanup on shutdown"""
        self.running = False
        self.logger.info("Compute node shutting down")

async def main():
    node = OmegaComputeNode()
    await node.start()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
