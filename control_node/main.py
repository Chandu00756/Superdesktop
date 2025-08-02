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
    
    sid = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    state = Column(String, default="RUNNING")
    cpu_cores = Column(Integer)
    gpu_units = Column(Integer)
    ram_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)

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
    user_id: str
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

@app.get("/api/v1/sessions")
async def list_sessions(user_id: str = Depends(verify_token)):
    REQUESTS_TOTAL.labels(method="GET", endpoint="/sessions").inc()
    return {"sessions": list(orchestrator.sessions.values())}

@app.delete("/api/v1/sessions/{session_id}")
async def terminate_session(session_id: str, user_id: str = Depends(verify_token)):
    if session_id in orchestrator.sessions:
        del orchestrator.sessions[session_id]
        if redis_client:
            try:
                redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logging.warning(f"Failed to delete session from Redis: {e}")
        ACTIVE_SESSIONS.set(len(orchestrator.sessions))
        
        # Update database
        db = SessionLocal()
        try:
            session = db.query(SessionRecord).filter(SessionRecord.sid == session_id).first()
            if session:
                session.state = "TERMINATED"
                session.updated_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logging.error(f"Failed to update session in database: {e}")
        finally:
            db.close()
        
        return {"status": "terminated"}
    raise HTTPException(status_code=404, detail="Session not found")

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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        reload=False,
        access_log=True
    )
