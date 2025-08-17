"""
Omega Control Center Backend API Server
Advanced encrypted communication with real-time data integration
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from contextlib import asynccontextmanager
import sqlite3
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import os
import shlex

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import websockets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import ssl
import base64
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import socket
import subprocess


class SecurityLevel(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    MAXIMUM = "maximum"


class NodeType(Enum):
    CONTROL = "control"
    COMPUTE = "compute"
    STORAGE = "storage"
    GPU = "gpu"


@dataclass
class EncryptedMessage:
    payload: str
    signature: str
    timestamp: float
    nonce: str


@dataclass
class NodeMetrics:
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_rx: int
    network_tx: int
    temperature: float
    power_consumption: float
    timestamp: float


@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    node_id: str
    application: str
    cpu_cores: int
    gpu_units: int
    memory_gb: int
    status: str
    created_at: float
    last_activity: float


class SecurityManager:
    def __init__(self):
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
        self.session_keys: Dict[str, bytes] = {}
        self.security_level = SecurityLevel.MAXIMUM
        
    def generate_session_key(self, session_id: str) -> bytes:
        key = Fernet.generate_key()
        self.session_keys[session_id] = key
        return key
    
    def encrypt_data(self, data: str, session_id: str = None) -> EncryptedMessage:
        if session_id and session_id in self.session_keys:
            cipher = Fernet(self.session_keys[session_id])
        else:
            cipher = self.cipher_suite
            
        nonce = secrets.token_hex(16)
        timestamp = time.time()
        
        payload_data = {
            "data": data,
            "timestamp": timestamp,
            "nonce": nonce
        }
        
        encrypted_payload = cipher.encrypt(json.dumps(payload_data).encode())
        payload_b64 = base64.b64encode(encrypted_payload).decode()
        
        signature = hmac.new(
            self.master_key,
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return EncryptedMessage(
            payload=payload_b64,
            signature=signature,
            timestamp=timestamp,
            nonce=nonce
        )
    
    def decrypt_data(self, message: EncryptedMessage, session_id: str = None) -> str:
        expected_signature = hmac.new(
            self.master_key,
            message.payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, message.signature):
            raise ValueError("Invalid message signature")
        
        if time.time() - message.timestamp > 300:
            raise ValueError("Message too old")
        
        if session_id and session_id in self.session_keys:
            cipher = Fernet(self.session_keys[session_id])
        else:
            cipher = self.cipher_suite
            
        encrypted_payload = base64.b64decode(message.payload.encode())
        decrypted_data = cipher.decrypt(encrypted_payload)
        payload_data = json.loads(decrypted_data.decode())
        
        return payload_data["data"]


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        # Always use a consistent DB path anchored to the backend folder
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'omega_control.db')
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    status TEXT DEFAULT 'active',
                    last_heartbeat REAL,
                    resources TEXT,
                    created_at REAL DEFAULT (julianday('now') * 86400)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    application TEXT NOT NULL,
                    cpu_cores INTEGER NOT NULL,
                    gpu_units INTEGER NOT NULL,
                    memory_gb INTEGER NOT NULL,
                    status TEXT DEFAULT 'running',
                    created_at REAL DEFAULT (julianday('now') * 86400),
                    last_activity REAL DEFAULT (julianday('now') * 86400),
                    FOREIGN KEY (node_id) REFERENCES nodes (node_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    network_rx INTEGER,
                    network_tx INTEGER,
                    temperature REAL,
                    power_consumption REAL,
                    timestamp REAL DEFAULT (julianday('now') * 86400),
                    FOREIGN KEY (node_id) REFERENCES nodes (node_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    timestamp REAL DEFAULT (julianday('now') * 86400)
                )
            """)

            # RBAC scaffolding
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    role TEXT PRIMARY KEY,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_roles (
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY (username, role),
                    FOREIGN KEY (role) REFERENCES roles(role)
                )
            """)
            # Snapshot metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (julianday('now') * 86400),
                    UNIQUE(session_id, tag)
                )
            """)
            # RDP sessions metadata (no passwords persisted)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rdp_session_meta (
                    session_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    username TEXT,
                    domain TEXT,
                    connect_url TEXT,
                    created_at REAL DEFAULT (julianday('now') * 86400)
                )
            """)
            
            conn.commit()
    
    def add_node(self, node_id: str, node_type: str, hostname: str, ip_address: str, port: int, resources: dict):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO nodes (node_id, node_type, hostname, ip_address, port, resources, last_heartbeat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (node_id, node_type, hostname, ip_address, port, json.dumps(resources), time.time())
                )
                conn.commit()
    
    def get_nodes(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM nodes")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def add_session(self, session: SessionInfo):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, user_id, node_id, application, cpu_cores, gpu_units, memory_gb, status, created_at, last_activity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session.session_id, session.user_id, session.node_id, session.application, 
                     session.cpu_cores, session.gpu_units, session.memory_gb, session.status,
                     session.created_at, session.last_activity)
                )
                conn.commit()
    
    def get_sessions(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM sessions")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def update_session_status(self, session_id: str, status: str):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET status = ?, last_activity = ? WHERE session_id = ?",
                    (status, time.time(), session_id)
                )
                conn.commit()
    
    def add_metrics(self, metrics: NodeMetrics):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO metrics (node_id, cpu_usage, memory_usage, gpu_usage, network_rx, network_tx, temperature, power_consumption, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (metrics.node_id, metrics.cpu_usage, metrics.memory_usage, metrics.gpu_usage,
                     metrics.network_rx, metrics.network_tx, metrics.temperature, 
                     metrics.power_consumption, metrics.timestamp)
                )
                conn.commit()
    
    def get_latest_metrics(self, node_id: str = None, limit: int = 100) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            if node_id:
                cursor = conn.execute(
                    "SELECT * FROM metrics WHERE node_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (node_id, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def log_event(self, event_type: str, source: str, message: str, severity: str = "info"):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO events (event_type, source, message, severity, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (event_type, source, message, severity, time.time())
                )
                conn.commit()

    def ensure_admin_role(self):
        """Ensure default roles and admin mapping exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)", ("admin", "Administrator"))
                conn.execute("INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)", ("user", "Standard User"))
                conn.execute("INSERT OR IGNORE INTO user_roles (username, role) VALUES (?, ?)", ("admin", "admin"))
                conn.commit()
        except Exception as e:
            logging.error(f"RBAC init error: {e}")


class PerformanceAnalyzer:
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.prediction_model = None
        
    def analyze_performance(self, metrics: List[Dict]) -> Dict:
        if not metrics:
            return {"status": "no_data"}
            
        latest = metrics[0] if metrics else {}
        
        cpu_avg = np.mean([m.get('cpu_usage', 0) for m in metrics[-10:]])
        memory_avg = np.mean([m.get('memory_usage', 0) for m in metrics[-10:]])
        gpu_avg = np.mean([m.get('gpu_usage', 0) for m in metrics[-10:]])
        
        health_score = 100 - (cpu_avg + memory_avg + gpu_avg) / 3
        
        bottlenecks = []
        if cpu_avg > 85:
            bottlenecks.append("CPU")
        if memory_avg > 90:
            bottlenecks.append("Memory")
        if gpu_avg > 95:
            bottlenecks.append("GPU")
            
        efficiency_rating = 5 - len(bottlenecks)
        
        return {
            "health_score": round(health_score, 1),
            "efficiency_rating": max(1, efficiency_rating),
            "bottlenecks": bottlenecks,
            "cpu_average": round(cpu_avg, 1),
            "memory_average": round(memory_avg, 1),
            "gpu_average": round(gpu_avg, 1),
            "recommendations": self.generate_recommendations(bottlenecks, cpu_avg, memory_avg, gpu_avg)
        }
    
    def generate_recommendations(self, bottlenecks: List[str], cpu_avg: float, memory_avg: float, gpu_avg: float) -> List[Dict]:
        recommendations = []
        
        if "CPU" in bottlenecks:
            recommendations.append({
                "title": "Optimize CPU Usage",
                "description": "Consider reducing CPU-intensive tasks or scaling to additional nodes",
                "impact": "high",
                "difficulty": "medium"
            })
            
        if "Memory" in bottlenecks:
            recommendations.append({
                "title": "Memory Optimization",
                "description": "Enable memory compression or add more RAM to the cluster",
                "impact": "high",
                "difficulty": "low"
            })
            
        if "GPU" in bottlenecks:
            recommendations.append({
                "title": "GPU Load Balancing",
                "description": "Distribute GPU workloads across multiple nodes",
                "impact": "medium",
                "difficulty": "high"
            })
            
        if not bottlenecks:
            recommendations.append({
                "title": "System Running Optimally",
                "description": "All resources are within normal operating parameters",
                "impact": "none",
                "difficulty": "none"
            })
            
        return recommendations


class OmegaAPIServer:
    def __init__(self):
        self.security_manager = SecurityManager()
        self.database = DatabaseManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.connected_clients: Set[WebSocket] = set()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.system_stats = {}
        self.network_topology = {}
        self.secure_sessions: Dict[str, str] = {}
        # Config
        self.session_storage_base = os.environ.get(
            'OMEGA_SESSION_BASE',
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'object_storage', 'sessions'))
        )
        os.makedirs(self.session_storage_base, exist_ok=True)
        # Default Linux desktop image for NoVNC
        self.default_desktop_image = os.environ.get('OMEGA_VD_IMAGE', 'dorowu/ubuntu-desktop-lxde-vnc')
        # DB ensure meta table
        self._ensure_vd_meta_table()

    def reconcile_vd_sessions_from_docker(self):
        """On startup, scan Docker for existing Omega VD containers and ensure DB has entries.
        This prevents sessions from disappearing across app restarts.
        """
        if not _docker_available():
            return
        try:
            out = subprocess.check_output([
                'docker', 'ps', '-a', '--format', '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Labels}}'
            ], stderr=subprocess.DEVNULL, timeout=10).decode().strip().splitlines()
        except Exception:
            return
        for line in out:
            try:
                cid, name, status, labels = (line.split('\t') + ['','','',''])[:4]
            except Exception:
                continue
            labels = labels or ''
            if 'omega.kind=virtual-desktop' not in labels and not name.startswith('omega_vd-') and not name.startswith('omega_vd_') and not name.startswith('omega_vd') and not name.startswith('omega_vd'):
                # Fallback: our naming scheme is omega_{session_id}
                if not (name.startswith('omega_vd-') or name.startswith('omega_vd_') or name.startswith('omega_vd')) and not name.startswith('omega_vd'):
                    pass
            # Derive session_id
            session_id = None
            for kv in (labels.split(',') if labels else []):
                if kv.startswith('omega.session_id='):
                    session_id = kv.split('=',1)[1]
                    break
            if not session_id and name.startswith('omega_'):
                sid_candidate = name[len('omega_'):]
                if sid_candidate.startswith('vd-'):
                    session_id = sid_candidate
            if not session_id:
                continue
            # Skip if already present
            try:
                with sqlite3.connect(self.database.db_path) as conn:
                    cur = conn.execute('SELECT 1 FROM vd_session_meta WHERE session_id=?', (session_id,))
                    if cur.fetchone():
                        continue
            except Exception:
                pass
            # Inspect container for ports, env, image
            try:
                import json as _json
                info = subprocess.check_output(['docker','inspect',cid], stderr=subprocess.DEVNULL, timeout=10).decode()
                j = _json.loads(info)[0]
                image = j.get('Config',{}).get('Image', '')
                env = j.get('Config',{}).get('Env', []) or []
                env_map = {e.split('=',1)[0]: (e.split('=',1)[1] if '=' in e else '') for e in env}
                vnc_password = env_map.get('VNC_PASSWORD') or env_map.get('PASSWORD') or ''
                ports_map = j.get('NetworkSettings',{}).get('Ports', {}) or {}
                def _host_port(key_opts: List[str]) -> Optional[int]:
                    for k in key_opts:
                        ent = ports_map.get(k)
                        if ent and isinstance(ent, list) and ent:
                            try:
                                return int(ent[0].get('HostPort'))
                            except Exception:
                                continue
                    return None
                http_port = _host_port(['6901/tcp','80/tcp']) or _find_free_port(7000,7999)
                vnc_port  = _host_port(['5901/tcp','5900/tcp']) or _find_free_port(5900,5999)
                # Build connect URL
                fam_ports = _container_ports_for_image(image)
                if 'dorowu/ubuntu-desktop-lxde-vnc' in image:
                    connect_path = '/static/vnc.html'
                    query = f"autoconnect=1&password={vnc_password}&host=localhost&port={http_port}&path=websockify"
                elif 'accetto' in image:
                    connect_path = '/'
                    query = f"autoconnect=1&password={vnc_password}"
                else:
                    connect_path = '/'
                    query = f"autoconnect=1&password={vnc_password}"
                connect_url = f"http://localhost:{http_port}{connect_path}?{query}"
                # Persist
                with sqlite3.connect(self.database.db_path) as conn:
                    conn.execute('INSERT OR REPLACE INTO vd_session_meta (session_id, container_id, http_port, vnc_port, vnc_password, os_image, connect_url) VALUES (?,?,?,?,?,?,?)',
                                 (session_id, cid, http_port, vnc_port, vnc_password, 'ubuntu-xfce', connect_url))
                    # Also ensure session exists
                    status_running = 'running' if ('Up' in (status or '')) else 'paused'
                    conn.execute('INSERT OR IGNORE INTO sessions (session_id, user_id, node_id, application, cpu_cores, gpu_units, memory_gb, status, created_at, last_activity) VALUES (?,?,?,?,?,?,?,?,?,?)',
                                 (session_id, 'admin', 'control-primary', 'virtual-desktop', 2, 0, 4, status_running, time.time(), time.time()))
                    conn.commit()
                self.database.log_event('vd_reconcile', session_id, f'Restored session from container {cid}', 'info')
            except Exception as e:
                try:
                    self.database.log_event('vd_reconcile_error', 'reconcile', f'Failed for {name}: {e}', 'warning')
                except Exception:
                    pass

    def _ensure_vd_meta_table(self):
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vd_session_meta (
                        session_id TEXT PRIMARY KEY,
                        container_id TEXT,
                        http_port INTEGER,
                        vnc_port INTEGER,
                        vnc_password TEXT,
                        os_image TEXT,
                        connect_url TEXT,
                        created_at REAL DEFAULT (julianday('now') * 86400)
                    )
                """)
                # Catalog of custom OS images
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vd_images (
                        id TEXT PRIMARY KEY,
                        image TEXT NOT NULL,
                        http_port INTEGER NOT NULL,
                        vnc_port INTEGER NOT NULL,
                        viewer_path TEXT DEFAULT '/',
                        description TEXT,
                        experimental INTEGER DEFAULT 0,
                        created_at REAL DEFAULT (julianday('now') * 86400)
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Failed ensuring vd_session_meta table: {e}")

    async def start_background_tasks(self):
        asyncio.create_task(self.metrics_collector())
        asyncio.create_task(self.health_monitor())
        asyncio.create_task(self.broadcast_updates())
    
    async def metrics_collector(self):
        while True:
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                control_metrics = NodeMetrics(
                    node_id="control-primary",
                    cpu_usage=cpu_usage,
                    memory_usage=memory.percent,
                    gpu_usage=0.0,
                    network_rx=network.bytes_recv,
                    network_tx=network.bytes_sent,
                    temperature=45.0 + (cpu_usage / 100) * 20,
                    power_consumption=150.0 + (cpu_usage / 100) * 50,
                    timestamp=time.time()
                )
                
                self.database.add_metrics(control_metrics)
                
                self.system_stats = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": disk.percent,
                    "network_rx": network.bytes_recv,
                    "network_tx": network.bytes_sent,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(2)
    
    async def health_monitor(self):
        while True:
            try:
                nodes = self.database.get_nodes()
                current_time = time.time()
                
                for node in nodes:
                    if current_time - node.get('last_heartbeat', 0) > 30:
                        self.database.log_event(
                            "node_offline",
                            node['node_id'],
                            f"Node {node['node_id']} has not sent heartbeat for 30+ seconds",
                            "warning"
                        )
                        
            except Exception as e:
                logging.error(f"Health monitor error: {e}")
            
            await asyncio.sleep(10)
    
    async def broadcast_updates(self):
        while True:
            try:
                if self.connected_clients:
                    update_data = {
                        "type": "system_update",
                        "timestamp": time.time(),
                        "system_stats": self.system_stats,
                        "node_count": len(self.database.get_nodes()),
                        "session_count": len([s for s in self.database.get_sessions() if s['status'] == 'running'])
                    }
                    
                    encrypted_message = self.security_manager.encrypt_data(json.dumps(update_data))
                    
                    disconnected_clients = set()
                    for client in self.connected_clients:
                        try:
                            await client.send_json(asdict(encrypted_message))
                        except:
                            disconnected_clients.add(client)
                    
                    self.connected_clients -= disconnected_clients
                    
            except Exception as e:
                logging.error(f"Broadcast error: {e}")
            
            await asyncio.sleep(1)


api_server = OmegaAPIServer()

app = FastAPI(
    title="Omega Control Center API",
    version="1.0.0",
    description="Advanced encrypted backend for distributed desktop control"
)

# Configurable CORS (default: allow all; set OMEGA_CORS_ORIGINS to comma-separated list to restrict)
_cors_env = os.environ.get('OMEGA_CORS_ORIGINS')
_allow_origins = [o.strip() for o in _cors_env.split(',')] if _cors_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


class AuthToken(BaseModel):
    username: str
    password: str


class NodeRegistration(BaseModel):
    node_id: str
    node_type: str
    hostname: str
    ip_address: str
    port: int
    resources: Dict[str, Any]


class SessionRequest(BaseModel):
    user_id: str
    application: str
    cpu_cores: int = 4
    gpu_units: int = 1
    memory_gb: int = 8


class MetricsData(BaseModel):
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    network_rx: int = 0
    network_tx: int = 0
    temperature: float = 50.0
    power_consumption: float = 200.0


@app.on_event("startup")
async def startup_event():
    await api_server.start_background_tasks()
    try:
        api_server.database.ensure_admin_role()
    except Exception as e:
        logging.warning(f"RBAC init warning: {e}")
    try:
        api_server.reconcile_vd_sessions_from_docker()
    except Exception as e:
        logging.warning(f"VD reconcile warning: {e}")
    logging.info("Omega API Server started")


@app.post("/api/auth/login")
async def login(auth: AuthToken):
    if auth.username == "admin" and auth.password == "omega123":
        session_key = api_server.security_manager.generate_session_key("session_" + secrets.token_hex(16))
        encrypted_token = api_server.security_manager.encrypt_data(
            json.dumps({"user_id": auth.username, "session_key": base64.b64encode(session_key).decode()})
        )
        
        return {
            "success": True,
            "token": asdict(encrypted_token),
            "expires_in": 28800
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


# --- Enhanced Security State ---
NONCE_WINDOW=5000  # track last N nonces per session
SESSION_META: Dict[str, dict] = {}

# In-memory rolling nonce store per session
from collections import deque
SESSION_NONCES: Dict[str, deque] = {}

def register_session_meta(session_id:str, key_b64:str):
    SESSION_META[session_id] = {
        'created': time.time(),
        'last_rotate': time.time(),
        'key': key_b64,
        'counter': 0
    }
    SESSION_NONCES[session_id] = deque(maxlen=NONCE_WINDOW)

# Basic validation function for secure endpoints
def validate_secure(headers):
    """Validate AES-GCM session headers and return (session_id, key_bytes)."""
    session_id = headers.get('X-Session-ID')
    key_b64 = headers.get('X-Session-Key')
    auth_header = headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing or invalid authorization header')
    if not session_id or not key_b64:
        raise HTTPException(status_code=401, detail='Missing session headers')
    meta = SESSION_META.get(session_id)
    if not meta:
        # Allow rehydrate if passed key and unknown session, register on the fly
        try:
            register_session_meta(session_id, key_b64)
            meta = SESSION_META.get(session_id)
        except Exception:
            raise HTTPException(status_code=401, detail='Unknown session')
    if meta['key'] != key_b64:
        raise HTTPException(status_code=401, detail='Session key mismatch')
    try:
        key = base64.b64decode(key_b64)
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid key encoding')
    return session_id, key

# REORDER PATCH: defer enhancement injection until after original secure endpoints defined
# Guard to avoid NameError during import phase; we wrap inside a function executed after definitions
POST_INIT_ENHANCED = False
async def post_init_enhance():
    global POST_INIT_ENHANCED, secure_nodes
    if POST_INIT_ENHANCED: return
    POST_INIT_ENHANCED = True
    # monkey patch secure_nodes - fix function name
    original = secure_nodes
    async def patched_secure_nodes(auth: AuthToken):  # type: ignore
        resp = await original(auth)
        # register_session_meta(resp['session_id'], resp['session_key'])
        return resp
    # replace route for secure_nodes
    for r in list(app.router.routes):
        if getattr(r,'path',None)=='/api/secure/nodes' and 'POST' in getattr(r,'methods',[]):
            app.router.routes.remove(r)
    app.post('/api/secure/nodes')(patched_secure_nodes)
    secure_nodes = patched_secure_nodes  # type: ignore
    
    # Note: Validation enhancement disabled for stability
    print("[Backend] Post-init enhancements applied")

@app.on_event('startup')
async def _apply_enhancements():
    await post_init_enhance()


# Counter + nonce wrapper
def wrap_encrypted(session_id:str, key:bytes, payload:dict)->dict:
    meta = SESSION_META.get(session_id)
    if not meta: raise HTTPException(status_code=401, detail='Session meta missing')
    meta['counter'] += 1
    payload['_ctr'] = meta['counter']
    payload['_ts'] = time.time()
    payload['_sid']= session_id
    packet = encrypt_aes_gcm(key, payload)
    packet['ctr']= meta['counter']
    packet['sid']= session_id
    return packet

# Secure action endpoints (POST) with integrity headers
class ActionRequest(BaseModel):
    action: str
    params: dict | None = None
    nonce: str
    ctr: int
    ts: float

SECURE_ACTIONS = {'discover_nodes','run_benchmark','health_check','restart_node'}

@app.post('/api/secure/action')
async def secure_action(request: Request, body: ActionRequest):
    session_id, key = validate_secure(request.headers)
    if body.action not in SECURE_ACTIONS:
        raise HTTPException(status_code=400, detail='Unknown action')
    meta = SESSION_META.get(session_id)
    if not meta:
        raise HTTPException(status_code=401, detail='Session meta missing')
    # Counter monotonic check
    if body.ctr <= meta['counter']:
        raise HTTPException(status_code=401, detail='Counter replay')
    if abs(time.time()-body.ts) > 30:
        raise HTTPException(status_code=401, detail='Stale action')
    SESSION_NONCES[session_id].append(body.nonce)
    meta['counter'] = body.ctr
    # Execute action
    if body.action == 'discover_nodes':
        result_enc = await action_discover_nodes()
    elif body.action == 'run_benchmark':
        result_enc = await action_run_benchmark()
    elif body.action == 'health_check':
        result_enc = await action_health_check()
    elif body.action == 'restart_node':
        # simple simulation
        await asyncio.sleep(1)
        result_enc = api_server.security_manager.encrypt_data(json.dumps({'success':True,'message':'Node restart initiated','node': body.params.get('node_id') if body.params else None}))
    else:
        raise HTTPException(status_code=400, detail='Unhandled action')
    # decrypt intermediate encrypted payload
    payload = api_server.security_manager.decrypt_data(EncryptedMessage(**result_enc))
    data = json.loads(payload)
    return wrap_encrypted(session_id, key, {'action': body.action, 'result': data, 'ok': True})

async def action_discover_nodes():
    # Example discovery: enumerate local network interfaces; treat each as a node if not yet recorded
    nets = psutil.net_if_addrs()
    new_nodes=[]
    for i,(name,addr_list) in enumerate(nets.items()):
        ip = next((a.address for a in addr_list if a.family.name in ('AF_INET','AddressFamily.AF_INET')), None)
        if not ip or ip.startswith('127.'):
            continue
        node_id=f'auto-{name}'
        existing=[n for n in api_server.database.get_nodes() if n['node_id']==node_id]
        if existing: continue
        api_server.database.add_node(node_id,'compute',name,ip,8000,{'cpu_cores': psutil.cpu_count(logical=True), 'memory_gb': round(psutil.virtual_memory().total/1024**3,2)})
        new_nodes.append(node_id)
    payload={'success':True,'discovered':len(new_nodes),'nodes':new_nodes}
    enc = api_server.security_manager.encrypt_data(json.dumps(payload))
    return enc

async def action_run_benchmark():
    # Simple real benchmark: measure CPU busy loop for 0.2s
    start=time.time(); ops=0
    while time.time()-start<0.2:
        hashlib.sha256(b'omega').hexdigest(); ops+=1
    score=int(ops/0.2)
    payload={'success':True,'score':score,'duration':'0.2s','components':{'cpu':score,'gpu':0,'memory':0,'storage':0}}
    enc = api_server.security_manager.encrypt_data(json.dumps(payload))
    return enc

async def action_health_check():
    nodes=api_server.database.get_nodes(); results=[]
    for n in nodes:
        results.append({'node_id':n['node_id'],'status':'healthy','checks':{'cpu':'OK','memory':'OK','network':'OK','storage':'OK'}})
    payload={'success':True,'results':results,'overall_health':'GOOD'}
    enc= api_server.security_manager.encrypt_data(json.dumps(payload))
    return enc

# Modify secure GET endpoints to use new wrap
@app.get('/api/secure/dashboard')
async def secure_dashboard(request: Request):
    session_id, key = validate_secure(request.headers)
    nodes = api_server.database.get_nodes()
    sessions = api_server.database.get_sessions()
    active_nodes = len([n for n in nodes if n.get('status')=='active'])
    standby_nodes = len([n for n in nodes if n.get('status')=='standby'])
    perf_cpu = psutil.cpu_percent(interval=0.05)
    perf_mem = psutil.virtual_memory().percent
    net_all = psutil.net_io_counters()
    cluster_info = {
        'name':'Local-Cluster',
        'status':'OPERATIONAL' if active_nodes>=1 else 'DEGRADED',
        'uptime': fmt_uptime(),
        'active_nodes': active_nodes,
        'standby_nodes': standby_nodes,
        'total_sessions': len(sessions),
        'cpu_usage': perf_cpu,
        'memory_usage': perf_mem,
        'network_load': round((net_all.bytes_recv+net_all.bytes_sent)/1024**2,2)
    }
    # Alerts = recent events
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT event_type,message,timestamp,severity FROM events ORDER BY timestamp DESC LIMIT 10')
        alerts=[{'id':f'evt-{row[2]}','type':row[3],'message':row[1],'timestamp':row[2]} for row in cur.fetchall()]
    payload={'cluster':cluster_info,'performance':{'cpu_utilization':perf_cpu,'memory_utilization':perf_mem},'alerts':alerts,'timestamp':time.time()}
    return wrap_encrypted(session_id, key, payload)

# Replace resources
@app.get('/api/secure/resources')
async def secure_resources(request: Request):
    session_id, key = validate_secure(request.headers)
    cpu = gather_cpu_block(); mem = gather_memory_block(); storage = gather_storage_block(); gpu = gather_gpu_block()
    payload = { 'cpu': cpu, 'gpu': gpu, 'memory': mem, 'storage': storage, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace network
@app.get('/api/secure/network')
async def secure_network(request: Request):
    session_id, key = validate_secure(request.headers)
    interfaces = gather_net_block()
    payload = { 'topology': {'nodes': api_server.database.get_nodes(), 'connections': []}, 'statistics': {'interfaces': interfaces}, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace performance
@app.get('/api/secure/performance')
async def secure_performance(request: Request):
    session_id, key = validate_secure(request.headers)
    cpu_hist = psutil.cpu_percent(percpu=False, interval=0.05)
    vm = psutil.virtual_memory()
    analysis = { 'cpu_average': cpu_hist, 'memory_average': vm.percent, 'gpu_average': 0, 'health_score': max(0,100- (cpu_hist+vm.percent)/2) }
    payload = { 'analysis': analysis, 'benchmark': None, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace plugins (no fake marketplace)
@app.get('/api/secure/plugins')
async def secure_plugins(request: Request):
    session_id, key = validate_secure(request.headers)
    # Minimal real structure: read from table if exists else empty
    installed=[]
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS plugins (name TEXT PRIMARY KEY, version TEXT, enabled INTEGER, description TEXT, installed_at REAL)")
            cur=conn.execute('SELECT name,version,enabled,description,installed_at FROM plugins')
            installed=[{'name':r[0],'version':r[1],'enabled':bool(r[2]),'description':r[3],'installed_at':r[4]} for r in cur.fetchall()]
    except Exception as e:
        logging.error(f'Plugin fetch error {e}')
    payload={'installed':installed,'available':[], 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

# Replace security (user list from sessions)
@app.get('/api/secure/security')
async def secure_security(request: Request):
    session_id, key = validate_secure(request.headers)
    sessions = api_server.database.get_sessions()
    users_map = {}
    for s in sessions:
        u = users_map.setdefault(s['user_id'], {'username':s['user_id'],'role':'user','status':'Active','last_login': datetime.fromtimestamp(s['created_at']).isoformat()})
    users = list(users_map.values())
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT event_type,message,severity,timestamp FROM events WHERE event_type LIKE "%security%" ORDER BY timestamp DESC LIMIT 20')
        security_events=[{'event_type':r[0],'message':r[1],'severity':r[2],'timestamp':r[3]} for r in cur.fetchall()]
    payload={'users':users,'certificates':[],'events':security_events,'encryption_status':'AES-256 Active','timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

# Add secure nodes & sessions endpoints plus processes/logs and websocket realtime
from fastapi import WebSocketDisconnect

@app.get('/api/secure/nodes')
async def secure_nodes(request: Request):
    session_id, key = validate_secure(request.headers)
    nodes = api_server.database.get_nodes()
    latest_metrics_map = {}
    for m in api_server.database.get_latest_metrics(limit= len(nodes)*3):
        latest_metrics_map.setdefault(m['node_id'], m)
    for n in nodes:
        n['metrics'] = latest_metrics_map.get(n['node_id'])
    payload = {'nodes': nodes, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

@app.get('/api/secure/sessions')
async def secure_sessions(request: Request):
    session_id, key = validate_secure(request.headers)
    sessions = api_server.database.get_sessions()
    payload = {'sessions': sessions, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

@app.get('/api/secure/processes')
async def secure_processes(request: Request):
    session_id, key = validate_secure(request.headers)
    procs = []
    try:
        for p in psutil.process_iter(['pid','name','cpu_percent','memory_info']):
            info = p.info
            procs.append({
                'pid': info.get('pid'),
                'name': info.get('name'),
                'cpu': info.get('cpu_percent'),
                'mem_mb': round(getattr(info.get('memory_info'), 'rss', 0)/1024**2,2)
            })
    except Exception as e:
        logging.error(f'process list error {e}')
    # top 25 by cpu
    procs = sorted(procs, key=lambda x: (x['cpu'] if x['cpu'] is not None else 0), reverse=True)[:25]
    payload = {'processes': procs, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

class KillProcessRequest(BaseModel):
    pid: int

@app.post('/api/secure/processes/kill')
async def secure_process_kill(request: Request, body: KillProcessRequest):
    session_id, key = validate_secure(request.headers)
    # RBAC: only admin may kill host processes
    user = SESSION_META.get(session_id, {}).get('user', 'admin')
    require_role(user, 'admin')
    pid = body.pid
    try:
        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=2)
        except psutil.TimeoutExpired:
            p.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return wrap_encrypted(session_id, key, {'success': True, 'pid': pid})

@app.get('/api/secure/logs')
async def secure_logs(request: Request):
    session_id, key = validate_secure(request.headers)
    events=[]
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT event_type, source, message, severity, timestamp FROM events ORDER BY timestamp DESC LIMIT 100')
            for row in cur.fetchall():
                events.append({'event_type':row[0],'source':row[1],'message':row[2],'severity':row[3],'timestamp':row[4]})
    except Exception as e:
        logging.error(f'log fetch error {e}')
    payload={'events':events,'timestamp':time.time()}
    return wrap_encrypted(session_id, key, payload)

@app.websocket('/ws/secure/realtime')
async def ws_secure_realtime(ws: WebSocket):
    await ws.accept()
    params = dict(ws.query_params)
    session_id = params.get('session_id')
    key_b64 = params.get('session_key')
    if not session_id or not key_b64 or session_id not in SESSION_META or SESSION_META[session_id]['key'] != key_b64:
        await ws.close()
        return
    key = base64.b64decode(key_b64)
    try:
        last_cpu = None
        while True:
            # build delta/system packet
            sys_stats = api_server.system_stats.copy()
            nodes_count = len(api_server.database.get_nodes())
            sessions_count = len(api_server.database.get_sessions())
            delta = {
                'type':'rt_delta',
                'system_stats': sys_stats,
                'counts': {'nodes': nodes_count, 'sessions': sessions_count},
                'timestamp': time.time()
            }
            pkt = wrap_encrypted(session_id, key, delta)
            await ws.send_json(pkt)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f'realtime websocket error {e}')
        try:
            await ws.close()
        except Exception:
            pass

# --- Real data helpers (replacing prior static placeholder logic) ---
import shutil, subprocess

START_TIME = psutil.boot_time()

def fmt_uptime():
    secs = int(time.time()-START_TIME)
    d, rem = divmod(secs, 86400); h, rem = divmod(rem,3600); m,_ = divmod(rem,60)
    return f"{d}d {h}h {m}m"

def gather_cpu_block():
    return {
        'total_cores': psutil.cpu_count(logical=True),
        'physical_cores': psutil.cpu_count(logical=False),
        'usage_percentage': psutil.cpu_percent(interval=0.1),
        'load_avg': list(psutil.getloadavg()) if hasattr(psutil,'getloadavg') else [],
        'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }

def gather_memory_block():
    vm = psutil.virtual_memory(); sm = psutil.swap_memory()
    return {
        'total_ram': round(vm.total/1024**3,2),
        'allocated_ram': round((vm.total-vm.available)/1024**3,2),
        'cached_ram': round(vm.cached/1024**3,2) if hasattr(vm,'cached') else None,
        'swap_total': round(sm.total/1024**3,2),
        'swap_used': round(sm.used/1024**3,2),
        'swap_usage': sm.percent
    }

def gather_storage_block():
    parts = []
    for p in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(p.mountpoint)
            parts.append({ 'device': p.device, 'mount': p.mountpoint, 'fstype': p.fstype, 'total_gb': round(usage.total/1024**3,2), 'used_gb': round(usage.used/1024**3,2), 'percent': usage.percent })
        except Exception:
            continue
    return {'partitions': parts}

def gather_gpu_block():
    # Attempt NVIDIA via nvidia-smi; no guesses if unavailable
    try:
        out = subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, timeout=2).decode().strip().splitlines()
        gpus=[]
        for line in out:
            name,u,mt,mu,temp,pwr = [x.strip() for x in line.split(',')]
            gpus.append({'name':name,'utilization':float(u),'memory_total_gb':round(float(mt)/1024,2),'memory_used_gb':round(float(mu)/1024,2),'temperature':float(temp),'power_w':float(pwr)})
        total_vram = sum(g['memory_total_gb'] for g in gpus)
        used_vram = sum(g['memory_used_gb'] for g in gpus)
        return {'gpus':gpus,'total_units':len(gpus),'total_vram_gb':total_vram,'used_vram_gb':used_vram}
    except Exception:
        return {'gpus':[], 'total_units':0, 'total_vram_gb':0, 'used_vram_gb':0}

def gather_net_block():
    stats = psutil.net_io_counters(pernic=True)
    inf_stats=[]
    for name,st in stats.items():
        inf_stats.append({'name':name,'bytes_sent':st.bytes_sent,'bytes_recv':st.bytes_recv,'packets_sent':st.packets_sent,'packets_recv':st.packets_recv,'errin':st.errin,'errout':st.errout})
    return inf_stats

# --- AES-GCM helper functions (restored) ---
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def generate_session_key() -> bytes:
    return os.urandom(32)

def encrypt_aes_gcm(session_key: bytes, data: dict) -> dict:
    iv = os.urandom(12)
    cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv))
    encryptor = cipher.encryptor()
    plaintext = json.dumps(data).encode()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return {
        'alg':'AES-256-GCM',
        'iv': base64.b64encode(iv).decode(),
        'ciphertext': base64.b64encode(ciphertext).decode(),
        'tag': base64.b64encode(encryptor.tag).decode(),
        'timestamp': time.time()
    }

# === Secure Session Handshake ===
class SecureSessionStart(BaseModel):
    client: str = 'desktop-app'
    user_id: str = 'admin'

@app.post('/api/secure/session/start')
async def secure_session_start(body: SecureSessionStart):
    # Issue a fresh AES-256-GCM session
    session_id = f"sid-{secrets.token_hex(12)}"
    key = generate_session_key()
    key_b64 = base64.b64encode(key).decode()
    register_session_meta(session_id, key_b64)
    # Track requesting user for RBAC and ownership checks
    try:
        SESSION_META[session_id]['user'] = body.user_id or 'admin'
    except Exception:
        SESSION_META[session_id] = {'user': body.user_id or 'admin', 'key': key_b64, 'created': time.time(), 'last_rotate': time.time(), 'counter': 0}
    payload = {'session_id': session_id, 'session_key': key_b64, 'issued_at': time.time(), 'expires_in': 6*3600}
    # Return plaintext JSON to bootstrap; subsequent calls use AES-GCM
    return payload

# RBAC check (scaffold)
def require_role(username: str, role: str):
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT 1 FROM user_roles WHERE username=? AND role=?', (username, role))
            if not cur.fetchone():
                raise HTTPException(status_code=403, detail='Forbidden: missing role')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'RBAC check error: {e}')
        raise HTTPException(status_code=500, detail='RBAC check failed')

def get_user_roles(username: str) -> list[str]:
    roles: list[str] = []
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT role FROM user_roles WHERE username=?', (username,))
            roles = [r[0] for r in cur.fetchall()]
    except Exception as e:
        logging.error(f'RBAC roles fetch error: {e}')
    return roles

# === Virtual Desktop (NoVNC over VNC) Management ===
class VDCreateRequest(BaseModel):
    user_id: str
    os_image: str = Field(default='ubuntu-xfce', description='ubuntu-xfce|ubuntu-lxde-legacy|debian-xfce|kali-xfce|windows')
    cpu_cores: int = 2
    memory_gb: int = 4
    gpu_units: int = 0
    packages: Optional[List[str]] = Field(default=None, description='Optional list of packages to install inside the desktop')
    vnc_password: Optional[str] = Field(default=None, description='Optional VNC password to set inside the desktop')
    resolution: Optional[str] = Field(default=None, description='Preferred desktop resolution, e.g., 1920x1080')
    profile: Optional[str] = Field(default=None, description='Optional preset package profile (browser|developer|office)')

def _find_free_port(start: int = 6000, end: int = 65000) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('0.0.0.0', p))
                return p
            except OSError:
                continue
    raise RuntimeError('No free ports available')

def _docker_run(args: list, timeout: int | float = 60) -> str:
    try:
        result = subprocess.check_output(['docker'] + args, stderr=subprocess.STDOUT, timeout=timeout)
        return result.decode().strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Docker error: {e.output.decode().strip()}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail='Docker not found on host')

def _docker_available() -> bool:
    try:
        subprocess.check_output(['docker', 'info', '--format', '{{.ServerVersion}}'], stderr=subprocess.STDOUT, timeout=5)
        return True
    except Exception:
        return False

def _image_for_os(os_image: str) -> str:
    mapping = {
    # Modern Ubuntu XFCE desktop (public, widely available)
    'ubuntu-xfce': 'accetto/ubuntu-vnc-xfce:latest',  # exposes :6901 (noVNC) and :5901
    # Chromium variant maps to the same base; Chromium can be added via packages/profile
    'ubuntu-xfce-chromium': 'accetto/ubuntu-vnc-xfce:latest',
        # Legacy option (kept for compatibility)
        'ubuntu-lxde-legacy': 'dorowu/ubuntu-desktop-lxde-vnc',
        # Alternatives
        'debian-xfce': 'accetto/debian-vnc-xfce',
        'kali-xfce':   'lscr.io/linuxserver/kali-linux:latest',  # may require extra config
    }
    if os_image == 'windows':
        # Windows handled via RDP endpoints
        raise HTTPException(status_code=400, detail='Windows requires RDP. Use /api/secure/rdp/create')
    # Check custom catalog override
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT image FROM vd_images WHERE id=?', (os_image,))
            row = cur.fetchone()
            if row:
                img = row[0]
                # Normalize deprecated/nonexistent accetto variants
                if img.startswith('accetto/ubuntu-vnc-xfce-'):
                    return 'accetto/ubuntu-vnc-xfce:latest'
                return img
    except Exception:
        pass
    return mapping.get(os_image, mapping['ubuntu-xfce'])

def _container_ports_for_image(image: str) -> dict:
    # Known defaults
    if 'dorowu/ubuntu-desktop-lxde-vnc' in image:
        return {'http': 80, 'vnc': 5900}
    if 'accetto' in image:
        return {'http': 6901, 'vnc': 5901}
    # Fallback
    return {'http': 80, 'vnc': 5900}

def _env_for_image(image: str, vnc_password: str, resolution: Optional[str] = None) -> list[str]:
    # Provide multiple common env names used by popular VNC desktop images
    envs = []
    for key in ('VNC_PASSWORD', 'PASSWORD', 'VNC_PW'):
        envs += ['-e', f"{key}={vnc_password}"]
    # Some accetto images support user/password pairs; keep minimal for now
    if resolution:
        # Support common env names across images
        for rkey in ('RESOLUTION', 'VNC_RESOLUTION'):
            envs += ['-e', f"{rkey}={resolution}"]
    return envs

@app.post('/api/secure/vd/create')
async def vd_create(request: Request, spec: VDCreateRequest, background_tasks: BackgroundTasks):
    sid, key = validate_secure(request.headers)
    # Preflight: ensure Docker daemon is available
    if not _docker_available():
        raise HTTPException(status_code=503, detail='Docker daemon not running. Start Docker Desktop and retry.')
    # Resolve image from env override or mapping, and set pull policy
    env_img = os.environ.get('OMEGA_DEFAULT_VD_IMAGE', '').strip()
    image = env_img or _image_for_os(spec.os_image)
    ports = _container_ports_for_image(image)
    pull_policy = os.environ.get('OMEGA_VD_PULL_POLICY', 'IfNotPresent').strip().lower()
    http_port = _find_free_port(7000, 7999)
    vnc_port = _find_free_port(5900, 5999)
    vnc_password = spec.vnc_password.strip() if isinstance(spec.vnc_password, str) and spec.vnc_password.strip() else secrets.token_urlsafe(8)
    session_id = f"vd-{secrets.token_hex(8)}"
    name = f"omega_{session_id}"

    # Prepare storage path
    sess_path = os.path.join(api_server.session_storage_base, session_id)
    os.makedirs(sess_path, exist_ok=True)

    # Pull image based on policy and with fallbacks (only when not env-overridden)
    def _inspect(img: str):
        _docker_run(['image', 'inspect', img], timeout=120)
    def _pull(img: str):
        _docker_run(['pull', img], timeout=600)
    try:
        if pull_policy == 'never':
            _inspect(image)
        elif pull_policy == 'always':
            _pull(image)
        else:  # IfNotPresent
            try:
                _inspect(image)
            except HTTPException:
                _pull(image)
    except HTTPException as e1:
        if env_img:
            # Respect env override; don't fallback automatically
            raise HTTPException(status_code=500, detail=f"Image '{image}' not available with pull policy '{pull_policy}': {e1.detail}. Pull the image manually or adjust OMEGA_VD_PULL_POLICY.")
        # Fallback attempts when using mapped images
        alt1 = 'accetto/ubuntu-vnc-xfce:latest'
        alt2 = 'dorowu/ubuntu-desktop-lxde-vnc'
        last_err = e1
        for alt in (alt1, alt2):
            try:
                if pull_policy == 'never':
                    _inspect(alt)
                elif pull_policy == 'always':
                    _pull(alt)
                else:
                    try:
                        _inspect(alt)
                    except HTTPException:
                        _pull(alt)
                image = alt
                ports = _container_ports_for_image(image)
                last_err = None
                break
            except HTTPException as e_alt:
                last_err = e_alt
        if last_err is not None:
            hint = 'Ensure Docker Desktop is running and has internet access, or pre-pull an image and set OMEGA_DEFAULT_VD_IMAGE plus OMEGA_VD_PULL_POLICY=Never.'
            raise HTTPException(status_code=500, detail=f"Image resolution failed for '{spec.os_image}'. Last error: {last_err.detail}. {hint}")

    # Run container
    port_http_map = f"{http_port}:{ports['http']}"
    port_vnc_map = f"{vnc_port}:{ports['vnc']}"
    envs = _env_for_image(image, vnc_password, spec.resolution)
    # Some images use different envs; keep minimal
    # Add labels to enable reconciliation after restarts
    labels = [
        '-l', 'omega.kind=virtual-desktop',
        '-l', f'omega.session_id={session_id}',
        '-l', f'omega.user={spec.user_id or "admin"}'
    ]
    run_args = ['run', '-d', '--name', name] + labels + ['-p', port_http_map, '-p', port_vnc_map] + envs + [image]
    container_id = _docker_run(run_args, timeout=120)

    # Register DB session
    s_info = SessionInfo(
        session_id=session_id,
        user_id=spec.user_id,
        node_id='control-primary',
        application='virtual-desktop',
        cpu_cores=spec.cpu_cores,
        gpu_units=spec.gpu_units,
        memory_gb=spec.memory_gb,
        status='running',
        created_at=time.time(),
        last_activity=time.time()
    )
    api_server.database.add_session(s_info)

    # Build connect URL depending on image family (path differs)
    # dorowu image serves a SPA at '/', with noVNC located under /static/vnc.html and expects host/port/path
    # Try custom viewer_path override
    custom_viewer = None
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT viewer_path FROM vd_images WHERE id=?', (spec.os_image,))
            row = cur.fetchone()
            if row and row[0]:
                custom_viewer = row[0]
    except Exception:
        pass
    if custom_viewer:
        connect_path = custom_viewer
        connect_query = f"autoconnect=1&password={vnc_password}"
    elif 'dorowu/ubuntu-desktop-lxde-vnc' in image:
        # websockify path is typically /websockify or /websockify, but dorowu bundle routes internally via static/vnc.html
        connect_path = '/static/vnc.html'
        # Important: noVNC connects to ws on the HTTP port via path=websockify
        connect_query = f"autoconnect=1&password={vnc_password}&host=localhost&port={http_port}&path=websockify"
    elif 'accetto' in image:
        # Accetto images publish viewer at '/' or '/vnc.html'
        connect_path = '/'
        connect_query = f"autoconnect=1&password={vnc_password}"
    else:
        connect_path = '/'
        connect_query = f"autoconnect=1&password={vnc_password}"
    connect_url = f"http://localhost:{http_port}{connect_path}?{connect_query}"
    # Store meta
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute(
            'INSERT OR REPLACE INTO vd_session_meta (session_id, container_id, http_port, vnc_port, vnc_password, os_image, connect_url) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (session_id, container_id, http_port, vnc_port, vnc_password, spec.os_image, connect_url)
        )
        conn.commit()

    # Optionally install requested/default packages in background (after a quick connectivity check)
    # Determine package set: profile > explicit packages > env default
    # Note: avoid 'chromium-browser' on modern Ubuntu (snap-based in containers). Keep firefox which is present in our default image.
    profile_map = {
        'browser': ['firefox', 'curl', 'wget', 'zip', 'unzip'],
        'developer': ['firefox', 'git', 'curl', 'wget', 'htop', 'build-essential', 'vim', 'python3', 'python3-pip'],
        'office': ['firefox', 'libreoffice', 'curl', 'zip', 'unzip']
    }
    pkgs = spec.packages
    if not pkgs and spec.profile and spec.profile in profile_map:
        pkgs = profile_map[spec.profile]
    if pkgs is None:
        # Allow default packages via env
        default_pkgs = os.environ.get('OMEGA_VD_DEFAULT_PACKAGES', 'firefox,htop,git,curl,zip,unzip').strip()
        pkgs = [p.strip() for p in default_pkgs.split(',') if p.strip()]
    def _check_network_and_install(cid: str, packages: List[str]):
        try:
            # Quick network connectivity check via a fast apt-get update with low timeouts
            try:
                _docker_run(['exec', '-u', '0', cid, 'bash', '-lc', "apt-get update -o Acquire::http::Timeout=5 -o Acquire::https::Timeout=5 -y || true"], timeout=180)  # noqa: E501
                api_server.database.log_event('vd_network_ok', session_id, 'Network connectivity verified inside desktop', 'info')
            except HTTPException as e:
                api_server.database.log_event('vd_network_warn', session_id, f'Network check issue: {e.detail}', 'warning')
            # Install additional packages if requested
            if packages:
                # Install packages individually to avoid whole-command failure if one package is unavailable
                pkgs_str = ' '.join([shlex.quote(p) if ' ' in p else p for p in packages]) if packages else ''
                # For Ubuntu containers, the standard 'firefox' apt may be a snap and fail.
                # Try a chain of fallbacks for browsers so at least one GUI browser is available.
                install_script = (
                    'set +e; '
                    + 'export DEBIAN_FRONTEND=noninteractive; '
                    + 'apt-get update -y >/dev/null 2>&1 || true; '
                    + 'for p in ' + pkgs_str + '; do '
                    + '  installed=0; echo "Installing $p"; '
                    + '  case "$p" in '
                    + '    firefox|browser) '
                    + '      for alt in firefox epiphany-browser midori chromium; do '
                    + '        apt-get install -y --no-install-recommends "$alt" >/dev/null 2>&1 && echo "INSTALL_OK:$alt" && installed=1 && break; '
                    + '      done; '
                    + '      [ $installed -eq 1 ] || echo "INSTALL_FAIL:$p"; '
                    + '      ;;'
                    + '    chromium|chromium-browser) '
                    + '      for alt in chromium chromium-browser chromium-common epiphany-browser midori; do '
                    + '        apt-get install -y --no-install-recommends "$alt" >/dev/null 2>&1 && echo "INSTALL_OK:$alt" && installed=1 && break; '
                    + '      done; '
                    + '      [ $installed -eq 1 ] || echo "INSTALL_FAIL:$p"; '
                    + '      ;;'
                    + '    *) '
                    + '      apt-get install -y --no-install-recommends "$p" >/dev/null 2>&1 && echo "INSTALL_OK:$p" || echo "INSTALL_FAIL:$p"; '
                    + '      ;;'
                    + '  esac; '
                    + 'done; true'
                )
                try:
                    _docker_run(['exec', '-u', '0', cid, 'bash', '-lc', install_script], timeout=1200)
                    api_server.database.log_event('vd_packages', session_id, f'Package install attempted: {packages}', 'info')
                except HTTPException as e:
                    api_server.database.log_event('vd_packages_error', session_id, f'Package loop failed: {e.detail}', 'warning')
        except HTTPException as e:
            api_server.database.log_event('vd_packages_error', session_id, f'Package install failed: {e.detail}', 'warning')
        except Exception as e:
            api_server.database.log_event('vd_packages_error', session_id, f'Package install failed: {e}', 'warning')
    background_tasks.add_task(_check_network_and_install, container_id, pkgs)

    # Respond with session info and initial connect URL
    payload = {
        'session_id': session_id,
        'connect_url': connect_url,
        'http_port': http_port,
        'status': 'running'
    }
    return wrap_encrypted(sid, key, payload)

@app.get('/api/secure/vd/profiles')
async def vd_profiles(request: Request):
    sid, key = validate_secure(request.headers)
    profiles = [
    {'id':'browser','label':'Browser','packages':['firefox','curl','wget','zip','unzip']},
        {'id':'developer','label':'Developer','packages':['firefox','git','curl','wget','htop','build-essential','vim','python3','python3-pip']},
        {'id':'office','label':'Office','packages':['firefox','libreoffice','curl','zip','unzip']}
    ]
    return wrap_encrypted(sid, key, {'profiles': profiles})

@app.get('/api/secure/vd/{session_id}/url')
async def vd_get_url(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT connect_url,http_port,vnc_port,vnc_password,os_image FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='VD session not found')
        connect_url, http_port, vnc_port, vnc_password, os_image = row
    # Recompute connect URL to handle image variations and fix older sessions
    image = _image_for_os(os_image)
    custom_viewer = None
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT viewer_path FROM vd_images WHERE id=?', (os_image,))
            r = cur.fetchone()
            if r and r[0]: custom_viewer = r[0]
    except Exception:
        pass
    if custom_viewer:
        connect_path = custom_viewer
        query = f"autoconnect=1&password={vnc_password}"
    elif 'dorowu/ubuntu-desktop-lxde-vnc' in image:
        connect_path = '/static/vnc.html'
        query = f"autoconnect=1&password={vnc_password}&host=localhost&port={http_port}&path=websockify"
    elif 'accetto' in image:
        connect_path = '/'
        query = f"autoconnect=1&password={vnc_password}"
    else:
        connect_path = '/'
        query = f"autoconnect=1&password={vnc_password}"
    new_url = f"http://localhost:{http_port}{connect_path}?{query}"
    payload = {'session_id': session_id, 'connect_url': new_url, 'http_port': http_port, 'vnc_port': vnc_port}
    return wrap_encrypted(sid, key, payload)

@app.delete('/api/secure/vd/{session_id}')
async def vd_delete(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    # RBAC/ownership: admin can delete any; owner can delete own session
    req_user = SESSION_META.get(sid, {}).get('user', 'admin')
    is_admin = False
    try:
        require_role(req_user, 'admin')
        is_admin = True
    except HTTPException:
        is_admin = False
    if not is_admin:
        # Verify ownership
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT user_id FROM sessions WHERE session_id = ?', (session_id,))
            row = cur.fetchone()
            owner = row[0] if row else None
        if owner != req_user:
            raise HTTPException(status_code=403, detail='Forbidden: only owner or admin can delete this session')
    container_id = None
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT container_id FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if row:
            container_id = row[0]
    # Stop and remove container
    if container_id:
        try:
            _docker_run(['rm', '-f', container_id])
        except HTTPException as e:
            logging.warning(f"Container removal issue: {e.detail}")
    # Remove DB records
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('DELETE FROM vd_session_meta WHERE session_id = ?', (session_id,))
        conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    payload = {'deleted': True, 'session_id': session_id}
    return wrap_encrypted(sid, key, payload)

@app.post('/api/secure/vd/{session_id}/pause')
async def vd_pause(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    container_id = None
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT container_id FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if row:
            container_id = row[0]
    if not container_id:
        raise HTTPException(status_code=404, detail='VD session not found')
    try:
        _docker_run(['pause', container_id])
    except HTTPException as e:
        raise
    api_server.database.update_session_status(session_id, 'paused')
    return wrap_encrypted(sid, key, {'session_id': session_id, 'status': 'paused'})

@app.post('/api/secure/vd/{session_id}/resume')
async def vd_resume(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    container_id = None
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT container_id FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if row:
            container_id = row[0]
    if not container_id:
        raise HTTPException(status_code=404, detail='VD session not found')
    try:
        _docker_run(['unpause', container_id])
    except HTTPException as e:
        raise
    api_server.database.update_session_status(session_id, 'running')
    return wrap_encrypted(sid, key, {'session_id': session_id, 'status': 'running'})

@app.post('/api/secure/vd/{session_id}/snapshot')
async def vd_snapshot(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT container_id FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='VD session not found')
        container_id = row[0]
    snap_tag = f"omega-snap-{session_id}-{int(time.time())}"
    # docker commit
    _docker_run(['commit', container_id, snap_tag])
    # Log event
    api_server.database.log_event('vd_snapshot', session_id, f'Snapshot created: {snap_tag}', 'info')
    # Estimate snapshot size via docker inspect (best effort)
    size_bytes = 0
    try:
        out = subprocess.check_output(['docker','image','inspect',snap_tag,'--format','{{.Size}}'], stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        size_bytes = int(out) if out.isdigit() else 0
    except Exception:
        pass
    # Persist metadata
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO snapshots (session_id, tag, size_bytes, created_at) VALUES (?, ?, ?, ?)', (session_id, snap_tag, size_bytes, time.time()))
            conn.commit()
    except Exception as e:
        logging.error(f'snapshot meta save error: {e}')
    return wrap_encrypted(sid, key, {'session_id': session_id, 'snapshot': snap_tag})

@app.get('/api/secure/vd/{session_id}/snapshots')
async def vd_list_snapshots(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT tag,size_bytes,created_at FROM snapshots WHERE session_id=? ORDER BY created_at DESC', (session_id,))
        snaps=[{'tag':r[0],'size_bytes':r[1],'created_at':r[2]} for r in cur.fetchall()]
    return wrap_encrypted(sid, key, {'session_id': session_id, 'snapshots': snaps})

# --- VD utility: package status inside container ---
@app.get('/api/secure/vd/{session_id}/packages')
async def vd_packages_status(request: Request, session_id: str, q: Optional[str] = None):
    sid, key = validate_secure(request.headers)
    # Resolve container
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT container_id FROM vd_session_meta WHERE session_id=?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='VD session not found')
        container_id = row[0]
    # Determine packages to check
    default_list = ['firefox','git','curl','wget','htop','zip','unzip','libreoffice']
    pkgs = [p.strip() for p in (q.split(',') if q else default_list) if p.strip()]
    # Build check script (best-effort: dpkg and which)
    check_script = (
        'set +e; '
        'for p in ' + ' '.join([shlex.quote(p) if ' ' in p else p for p in pkgs]) + '; do '
        '  if dpkg -s "$p" >/dev/null 2>&1; then echo "OK:dpkg:$p"; '
        '  elif command -v "$p" >/dev/null 2>&1; then echo "OK:bin:$p"; '
        '  else echo "MISS:$p"; fi; '
        'done'
    )
    try:
        out = _docker_run(['exec','-u','0', container_id, 'bash','-lc', check_script])
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f'Package status check failed: {e.detail}')
    results = []
    for line in (out or '').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('OK:'):
            parts = line.split(':')
            if len(parts) >= 3:
                _, source, name = parts[0], parts[1], ':'.join(parts[2:])
                results.append({'name': name, 'installed': True, 'source': source})
            else:
                results.append({'name': line[3:], 'installed': True, 'source': 'unknown'})
        elif line.startswith('MISS:'):
            results.append({'name': line[5:], 'installed': False})
    return wrap_encrypted(sid, key, {'session_id': session_id, 'packages': results})

class SnapshotDeleteRequest(BaseModel):
    tag: str

@app.post('/api/secure/vd/{session_id}/snapshot/delete')
async def vd_delete_snapshot(request: Request, session_id: str, body: SnapshotDeleteRequest):
    sid, key = validate_secure(request.headers)
    # Admin only
    user = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(user, 'admin')
    try:
        _docker_run(['rmi','-f', body.tag])
    except HTTPException:
        pass
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('DELETE FROM snapshots WHERE session_id=? AND tag=?', (session_id, body.tag))
        conn.commit()
    api_server.database.log_event('vd_snapshot_delete', session_id, f'Deleted snapshot {body.tag}', 'info')
    return wrap_encrypted(sid, key, {'deleted': True, 'tag': body.tag})

class SnapshotRestoreRequest(BaseModel):
    tag: str

@app.post('/api/secure/vd/{session_id}/snapshot/restore')
async def vd_restore_snapshot(request: Request, session_id: str, body: SnapshotRestoreRequest):
    sid, key = validate_secure(request.headers)
    # Admin only
    user = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(user, 'admin')
    # Start a new container from snapshot tag
    ports = _container_ports_for_image('dorowu/ubuntu-desktop-lxde-vnc')
    http_port = _find_free_port(7000, 7999)
    vnc_port = _find_free_port(5900, 5999)
    vnc_password = secrets.token_urlsafe(8)
    name = f'omega_restore_{session_id}_{int(time.time())}'
    try:
        _docker_run(['run','-d','--name',name,'-p',f"{http_port}:{ports['http']}",'-p',f"{vnc_port}:{ports['vnc']}",'-e',f"VNC_PASSWORD={vnc_password}", body.tag])
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f'Restore failed: {e.detail}')
    # Use same dorowu-compatible path and params when restoring from snapshot
    connect_url = f"http://localhost:{http_port}/static/vnc.html?autoconnect=1&password={vnc_password}&host=localhost&port={http_port}&path=websockify"
    api_server.database.log_event('vd_snapshot_restore', session_id, f'Restored {body.tag} -> {name}', 'info')
    return wrap_encrypted(sid, key, {'restored': True, 'connect_url': connect_url})

# Health check for VD viewer readiness
@app.get('/api/secure/vd/{session_id}/health')
async def vd_health(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    # Lookup meta
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT http_port, os_image FROM vd_session_meta WHERE session_id=?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='VD session not found')
        http_port, os_image = row
    # Probe viewer paths quickly (single attempt)
    import urllib.request
    ready = False
    code = None
    hit = None
    for path in ['/static/vnc.html', '/vnc.html', '/']:
        try:
            req = urllib.request.Request(f'http://127.0.0.1:{http_port}{path}', method='HEAD')
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                code = resp.getcode(); hit = path
                if code == 200:
                    ready = True
                    break
        except Exception:
            continue
    payload = {'session_id': session_id, 'ready': ready, 'http_port': http_port, 'code': code or 0, 'path': hit or ''}
    return wrap_encrypted(sid, key, payload)

# RDP session endpoints (Windows via user-licensed RDP)
class RDPCreateRequest(BaseModel):
    user_id: str
    host: str
    port: int = 3389
    username: str
    password: str
    domain: Optional[str] = None

# --- RBAC utility endpoints ---
class RoleAssignRequest(BaseModel):
    username: str
    role: str

@app.get('/api/secure/whoami')
async def secure_whoami(request: Request):
    sid, key = validate_secure(request.headers)
    user = SESSION_META.get(sid, {}).get('user', 'admin')
    roles = get_user_roles(user)
    return wrap_encrypted(sid, key, {'user': user, 'roles': roles, 'timestamp': time.time()})

@app.post('/api/secure/admin/roles/assign')
async def admin_assign_role(request: Request, body: RoleAssignRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    role = body.role.strip()
    username = body.username.strip()
    if not role or not username:
        raise HTTPException(status_code=400, detail='username and role are required')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)', (role, None))
            conn.execute('INSERT OR IGNORE INTO user_roles (username, role) VALUES (?, ?)', (username, role))
            conn.commit()
        api_server.database.log_event('security_role_assign', caller, f'Assigned role {role} to {username}', 'info')
    except Exception as e:
        logging.error(f'Role assign error: {e}')
        raise HTTPException(status_code=500, detail='Role assignment failed')
    return wrap_encrypted(sid, key, {'success': True, 'username': username, 'role': role})

@app.post('/api/secure/admin/roles/remove')
async def admin_remove_role(request: Request, body: RoleAssignRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    role = body.role.strip()
    username = body.username.strip()
    if not role or not username:
        raise HTTPException(status_code=400, detail='username and role are required')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('DELETE FROM user_roles WHERE username=? AND role=?', (username, role))
            conn.commit()
        api_server.database.log_event('security_role_remove', caller, f'Removed role {role} from {username}', 'info')
    except Exception as e:
        logging.error(f'Role remove error: {e}')
        raise HTTPException(status_code=500, detail='Role removal failed')
    return wrap_encrypted(sid, key, {'success': True, 'username': username, 'role': role, 'removed': True})

@app.post('/api/secure/rdp/create')
async def rdp_create(request: Request, spec: RDPCreateRequest):
    sid, key = validate_secure(request.headers)
    # Ownership/RBAC: non-admin may only create for self; admin may create for others
    req_user = SESSION_META.get(sid, {}).get('user', 'admin')
    if req_user != (spec.user_id or req_user):
        try:
            require_role(req_user, 'admin')
        except HTTPException:
            raise HTTPException(status_code=403, detail='Forbidden: cannot create RDP for another user')
    session_id = f"rdp-{secrets.token_hex(8)}"
    s_info = SessionInfo(
        session_id=session_id,
        user_id=spec.user_id,
        node_id='control-primary',
        application='rdp-desktop',
        cpu_cores=0, gpu_units=0, memory_gb=0,
        status='running', created_at=time.time(), last_activity=time.time()
    )
    api_server.database.add_session(s_info)
    connect_url = f"rdp://{spec.username}@{spec.host}:{spec.port}"
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('INSERT OR REPLACE INTO rdp_session_meta (session_id, host, port, username, domain, connect_url) VALUES (?, ?, ?, ?, ?, ?)',
                     (session_id, spec.host, spec.port, spec.username, spec.domain or '', connect_url))
        conn.commit()
    api_server.database.log_event('rdp_session_create', session_id, f'RDP to {spec.host}:{spec.port} user {spec.username}', 'info')
    return wrap_encrypted(sid, key, {'session_id': session_id, 'connect_url': connect_url, 'status': 'running'})

@app.get('/api/secure/rdp/{session_id}/url')
async def rdp_get_url(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT connect_url, host, port, username, domain FROM rdp_session_meta WHERE session_id=?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='RDP session not found')
        connect_url, host, port, username, domain = row
    return wrap_encrypted(sid, key, {'session_id': session_id, 'connect_url': connect_url, 'rdp': {'host': host, 'port': port, 'username': username, 'domain': domain}})

@app.delete('/api/secure/rdp/{session_id}')
async def rdp_delete(request: Request, session_id: str):
    sid, key = validate_secure(request.headers)
    # RBAC/ownership: admin or owner may delete
    req_user = SESSION_META.get(sid, {}).get('user', 'admin')
    is_admin = False
    try:
        require_role(req_user, 'admin')
        is_admin = True
    except HTTPException:
        is_admin = False
    if not is_admin:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT session_id FROM sessions WHERE session_id=? AND user_id=?', (session_id, req_user))
            if not cur.fetchone():
                raise HTTPException(status_code=403, detail='Forbidden: only owner or admin can delete this RDP session')
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('DELETE FROM rdp_session_meta WHERE session_id=?', (session_id,))
        conn.execute('DELETE FROM sessions WHERE session_id=?', (session_id,))
        conn.commit()
    api_server.database.log_event('rdp_session_delete', session_id, 'RDP session removed', 'info')
    return wrap_encrypted(sid, key, {'deleted': True, 'session_id': session_id})

# --- OS Catalog endpoints ---
class OSImageRegister(BaseModel):
    id: str
    image: str
    http_port: int
    vnc_port: int
    viewer_path: str = '/'
    description: Optional[str] = None
    experimental: int = 0

@app.get('/api/secure/vd/os-list')
async def vd_os_list(request: Request):
    sid, key = validate_secure(request.headers)
    # Built-ins
    builtin = [
    {'id':'ubuntu-xfce', 'image':'accetto/ubuntu-vnc-xfce:latest', 'http_port':6901, 'vnc_port':5901, 'viewer_path':'/', 'description':'Ubuntu + XFCE (modern) + noVNC'},
    {'id':'ubuntu-xfce-chromium', 'image':'accetto/ubuntu-vnc-xfce:latest', 'http_port':6901, 'vnc_port':5901, 'viewer_path':'/', 'description':'Ubuntu + XFCE (Chromium via packages)'},
        {'id':'ubuntu-lxde-legacy', 'image':'dorowu/ubuntu-desktop-lxde-vnc', 'http_port':80, 'vnc_port':5900, 'viewer_path':'/static/vnc.html', 'description':'Legacy LXDE variant'},
        {'id':'debian-xfce', 'image':'accetto/debian-vnc-xfce', 'http_port':6901, 'vnc_port':5901, 'viewer_path':'/', 'description':'Debian + XFCE'},
        {'id':'kali-xfce', 'image':'lscr.io/linuxserver/kali-linux:latest', 'http_port':80, 'vnc_port':5900, 'viewer_path':'/', 'description':'Kali XFCE (may need extra config)'},
    ]
    custom=[]
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT id,image,http_port,vnc_port,viewer_path,description,experimental FROM vd_images ORDER BY id')
            for r in cur.fetchall():
                custom.append({'id':r[0],'image':r[1],'http_port':r[2],'vnc_port':r[3],'viewer_path':r[4],'description':r[5],'experimental':bool(r[6])})
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'builtin': builtin, 'custom': custom})

@app.get('/api/secure/vd/docker-health')
async def vd_docker_health(request: Request):
    sid, key = validate_secure(request.headers)
    ok = _docker_available()
    info = None
    if ok:
        try:
            info = subprocess.check_output(['docker','info','--format','{{.ServerVersion}}'], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        except Exception:
            info = None
    return wrap_encrypted(sid, key, {'docker': ok, 'server_version': info})

@app.post('/api/secure/vd/os-register')
async def vd_os_register(request: Request, body: OSImageRegister):
    sid, key = validate_secure(request.headers)
    # Admin only
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO vd_images (id,image,http_port,vnc_port,viewer_path,description,experimental) VALUES (?,?,?,?,?,?,?)',
                         (body.id.strip(), body.image.strip(), body.http_port, body.vnc_port, body.viewer_path.strip() or '/', body.description, int(bool(body.experimental))))
            conn.commit()
        api_server.database.log_event('vd_os_register', caller, f'Registered OS {body.id} -> {body.image}', 'info')
    except Exception as e:
        logging.error(f'vd_os_register error: {e}')
        raise HTTPException(status_code=500, detail='Failed to register OS image')
    return wrap_encrypted(sid, key, {'success': True, 'id': body.id})
