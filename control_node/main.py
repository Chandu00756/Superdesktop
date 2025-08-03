"""
Omega Super Desktop Console - Control Node
Initial prototype master coordinator and user interface hub with advanced features.
"""
import time
import json
import logging
import asyncio
import uvicorn
import psutil
import uuid
import numpy as np
import jwt
import platform
import subprocess
import socket
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from prometheus_client import start_http_server, Counter, Gauge, Histogram

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Simple in-memory metrics instead of optional dependencies
class SimpleMetrics:
    def __init__(self):
        self.requests_total = 0
        self.active_sessions = 0
        self.node_count = 0
        
    def inc_requests(self):
        self.requests_total += 1

# Prometheus metrics
REQUESTS_TOTAL = Counter('omega_requests_total', 'Total requests', ['method', 'endpoint'])
ACTIVE_SESSIONS = Gauge('omega_active_sessions', 'Number of active sessions')
NODE_COUNT = Gauge('omega_node_count', 'Number of registered nodes')
LATENCY_P95 = Gauge('omega_latency_p95_ms', 'P95 latency in milliseconds')

# Secret key for JWT tokens
SECRET_KEY = os.getenv("OMEGA_SECRET_KEY", "omega-super-desktop-secret-key-prototype-change-in-production")
        
metrics = SimpleMetrics()

# Global data storage (replace Redis if not available)
nodes_data = {}
sessions_data = {}
# Database setup with fallback to SQLite
import os
import logging
import sqlite3

try:
    # Try PostgreSQL first
    DATABASE_URL = os.getenv("OMEGA_DB_URL", "postgresql://omega:omega@localhost/omega_db")
    engine = create_engine(DATABASE_URL)
    # Test the connection
    with engine.connect():
        pass
    print("Using PostgreSQL database")
except Exception as e:
    # Fallback to SQLite
    print(f"PostgreSQL connection failed: {e}")
    print("Falling back to SQLite database")
    DATABASE_URL = "sqlite:///./omega_prototype.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
except Exception as e:
    print(f"Error creating database tables: {e}")
    # If PostgreSQL fails, force SQLite
    if "postgresql" in DATABASE_URL.lower():
        print("Forcing SQLite fallback...")
        DATABASE_URL = "sqlite:///./omega_prototype.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        print("SQLite database initialized successfully")

# Redis for real-time data (optional)
redis_client = None
try:
    import redis
    redis_url = os.getenv("OMEGA_REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    # Test Redis connection
    redis_client.ping()
    print("Using Redis for caching")
except Exception as e:
    print(f"Redis connection failed: {e}")
    print("Using in-memory cache instead")
    redis_client = None

# Simple ML model replacement
class SimpleAnomalyDetector:
    def __init__(self):
        self.threshold = 0.8
    
    def fit(self, data):
        pass
    
    def predict(self, data):
        return [1] * len(data)  # Normal behavior

anomaly_detector = SimpleAnomalyDetector()

class SessionRecord(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    user_id = Column(String, index=True)
    app_name = Column(String, nullable=False)
    app_command = Column(String)
    app_icon = Column(String)
    status = Column(String, default="INITIALIZING")  # INITIALIZING, RUNNING, PAUSED, SUSPENDED, MIGRATING, ERROR, TERMINATED
    node_id = Column(String, index=True)
    cpu_cores = Column(Integer)
    gpu_units = Column(Integer)
    ram_gb = Column(Float)
    storage_gb = Column(Float)
    priority = Column(Integer, default=1)
    session_type = Column(String, default="workstation")  # gaming, workstation, ai_compute, render_farm, development, streaming
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

class NodeRecord(Base):
    __tablename__ = "nodes"
    
    node_id = Column(String, primary_key=True, index=True)
    node_type = Column(String)
    status = Column(String, default="active")
    resources = Column(JSON)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    performance_score = Column(Float, default=1.0)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class NodeInfo(BaseModel):
    node_id: str
    node_type: str
    status: str = "active"
    ip_address: Optional[str] = None
    resources: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
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
    snapshot_policy: Dict[str, Any] = {}

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

class SessionAction(BaseModel):
    action: str  # pause, resume, terminate, snapshot, migrate
    target_node_id: Optional[str] = None
    snapshot_name: Optional[str] = None
    force: bool = False
    app_uri: str
    cpu_cores: int = 4
    gpu_units: int = 1
    ram_bytes: int = 8 * 1024 * 1024 * 1024  # 8GB
    low_latency: bool = False
    pin_gpu: Optional[str] = None

class ResourceHint(BaseModel):
    cpu_cores: int
    gpu_units: int
    ram_bytes: int

class TaskRequest(BaseModel):
    session_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 5  # 1-10 scale

class LatencyMetric(BaseModel):
    timestamp: float
    input_to_pixel_ms: float
    network_hop_ms: float
    gpu_render_ms: float
    prediction_confidence: float

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
