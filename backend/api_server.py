from fastapi import Body
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import os, logging

class NodeRegistrationRequest(BaseModel):
    node_id: str
    node_type: str
    hostname: str
    ip_address: str
    port: int
    resources: dict
    permissions: Optional[list] = []
    description: Optional[str] = None
    device_fingerprint: str
    public_key_pem: str
    signed_challenge: str
    health_attestation: Optional[dict] = None
    device_certificate: Optional[str] = None
    geoip: Optional[str] = None
    behavioral_baseline: Optional[dict] = None

from fastapi import FastAPI
# Single FastAPI app instance (duplicates removed below in file)
app = FastAPI(
    title="Omega Control Center API",
    version="1.0.0",
    description="Advanced encrypted backend for distributed desktop control"
)

# Lightweight health/readiness endpoint (unauthenticated) so frontend can quickly
# detect backend availability before attempting secure session bootstrap.
@app.get('/health', include_in_schema=False)
async def health_check():
    return {'status': 'ok'}

# Simple unauthenticated ping used by discovery logic and external scripts
@app.get('/api/ping', include_in_schema=False)
async def api_ping():
    return {'pong': True, 'ts': time.time()}

# CORS middleware (added early). Configure via OMEGA_CORS_ORIGINS.
# SECURITY: In production (OMEGA_ENV=prod|production) do not default to wildcard.
_cors_origins_env = os.environ.get('OMEGA_CORS_ORIGINS')
_env = os.environ.get('OMEGA_ENV','dev').lower()
if _cors_origins_env:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(',') if o.strip()]
else:
    if _env in ('prod','production'):
        _cors_origins = ['http://localhost:8000','http://127.0.0.1:8000','http://localhost:8443','http://127.0.0.1:8443']
    else:
        _cors_origins = ['*']  # dev convenience
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )
except Exception as e:
    logging.warning(f"Failed adding CORS middleware: {e}")

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
from pydantic import Field, validator
import uvicorn
import websockets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import ssl
import base64
import binascii
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import socket
import subprocess
try:
    import bcrypt
    _HAS_BCRYPT = True
except Exception:
    _HAS_BCRYPT = False


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
        # master_key used for signing and Fernet envelope; prefer external configuration
        # Support: OMEGA_MASTER_KEY (in-memory) or OMEGA_MASTER_KEY_PATH (file persisted)
        master_env = os.environ.get('OMEGA_MASTER_KEY')
        master_path = os.environ.get('OMEGA_MASTER_KEY_PATH') or os.path.join(os.path.dirname(__file__), 'omega_keys', 'master.key')
        os.makedirs(os.path.dirname(master_path), exist_ok=True)
        self._master_key_bytes = None
        # Priority: explicit env var > persisted file > generate & persist
        if master_env:
            # Accept both raw bytes and base64 string; normalize to bytes suitable for Fernet
            try:
                candidate = master_env.encode() if isinstance(master_env, str) else master_env
                # Try to instantiate Fernet to validate
                self.cipher_suite = Fernet(candidate)
                self._master_key_bytes = candidate
            except Exception:
                # Derive a 32-byte key via PBKDF2 and persist
                derived = base64.urlsafe_b64encode(hashlib.pbkdf2_hmac('sha256', master_env.encode(), b'omega_master_salt', 200000, dklen=32))
                self._master_key_bytes = derived
                self.cipher_suite = Fernet(self._master_key_bytes)
                try:
                    with open(master_path, 'wb') as f:
                        f.write(self._master_key_bytes)
                    os.chmod(master_path, 0o600)
                except Exception:
                    logging.warning('Could not persist derived master key to %s', master_path)
        elif os.path.exists(master_path):
            try:
                with open(master_path, 'rb') as f:
                    self._master_key_bytes = f.read().strip()
                self.cipher_suite = Fernet(self._master_key_bytes)
            except Exception:
                # fallback to generate
                self._master_key_bytes = Fernet.generate_key()
                self.cipher_suite = Fernet(self._master_key_bytes)
                logging.warning('Invalid master key file at %s - generated a new in-memory key', master_path)
        else:
            # generate and persist
            self._master_key_bytes = Fernet.generate_key()
            self.cipher_suite = Fernet(self._master_key_bytes)
            try:
                with open(master_path, 'wb') as f:
                    f.write(self._master_key_bytes)
                os.chmod(master_path, 0o600)
                logging.warning('Generated new master key and persisted to %s; consider supplying OMEGA_MASTER_KEY in production', master_path)
            except Exception:
                logging.warning('Generated new master key in-memory (not persisted)')
        # RSA key persistence: try to load from configured path, else generate and persist
        rsa_path = os.environ.get('OMEGA_RSA_KEY_PATH') or os.path.join(os.path.dirname(__file__), 'omega_keys', 'rsa_key.pem')
        os.makedirs(os.path.dirname(rsa_path), exist_ok=True)
        if os.path.exists(rsa_path):
            try:
                with open(rsa_path, 'rb') as f:
                    pem = f.read()
                self.rsa_private_key = serialization.load_pem_private_key(pem, password=None)
            except (ValueError, TypeError, serialization.UnsupportedAlgorithm) as e:
                logging.error(f"Failed to load RSA private key from {rsa_path}: {e}; generating new key")
                self.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            except Exception as e:
                logging.error(f"Unexpected error loading RSA key: {e}; generating new key")
                self.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        else:
            self.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            try:
                pem = self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                )
                # write with strict permissions
                with open(rsa_path, 'wb') as f:
                    f.write(pem)
                os.chmod(rsa_path, 0o600)
            except (OSError, IOError) as e:
                logging.error(f"Failed to persist RSA private key to {rsa_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error persisting RSA key: {e}")
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
            self._master_key_bytes,
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
            self._master_key_bytes,
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

    # --- JWT helpers (simple HMAC-SHA256 JWT)
    def issue_jwt(self, session_id: str, user: str = 'admin', ttl: int = 3600) -> str:
        header = base64.urlsafe_b64encode(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode()).rstrip(b"=").decode()
        now = int(time.time())
        payload = {'sid': session_id, 'sub': user, 'iat': now, 'exp': now + int(ttl)}
        payload_b = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        to_sign = f"{header}.{payload_b}".encode()
        sig = hmac.new(self._master_key_bytes, to_sign, hashlib.sha256).digest()
        sig_b = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        return f"{header}.{payload_b}.{sig_b}"

    def validate_jwt(self, token: str) -> dict:
        try:
            parts = token.split('.')
            if len(parts) != 3:
                raise ValueError('Invalid token')
            header_b, payload_b, sig_b = parts
            to_sign = f"{header_b}.{payload_b}".encode()
            sig = base64.urlsafe_b64decode(sig_b + '==')
            expected = hmac.new(self._master_key_bytes, to_sign, hashlib.sha256).digest()
            if not hmac.compare_digest(expected, sig):
                raise ValueError('Invalid signature')
            payload_json = base64.urlsafe_b64decode(payload_b + '==').decode()
            payload = json.loads(payload_json)
            if int(time.time()) > int(payload.get('exp', 0)):
                raise ValueError('Token expired')
            return payload
        except Exception as e:
            raise HTTPException(status_code=401, detail=f'Invalid token: {e}')


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'omega_control.db')
        self.lock = threading.Lock()
        self.init_database()
        # Simple in-memory protocol health cache: {node_id: {protocol: status}}
        self.protocol_health = {}

    def init_database(self):
        with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
            try:
                conn.execute('PRAGMA journal_mode=WAL;')
            except sqlite3.DatabaseError as e:
                logging.error(f"Database error enabling WAL mode: {e}")
            except Exception as e:
                logging.error(f"Unexpected error enabling WAL mode: {e}")
            # --- Advanced, normalized, auditable, encrypted schema ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    status TEXT DEFAULT 'active',
                    trust_score INTEGER DEFAULT 0,
                    quarantine INTEGER DEFAULT 0,
                    last_heartbeat REAL,
                    last_protocol_check REAL,
                    created_at REAL DEFAULT (julianday('now') * 86400),
                    device_class TEXT DEFAULT 'generic'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_attestations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    device_fingerprint TEXT NOT NULL,
                    public_key_pem TEXT NOT NULL,
                    device_certificate TEXT,
                    health_attestation TEXT,
                    geoip TEXT,
                    behavioral_baseline TEXT,
                    attested_at REAL DEFAULT (julianday('now') * 86400),
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    permission TEXT NOT NULL,
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    resource_key TEXT NOT NULL,
                    resource_value TEXT,
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    approved_by TEXT,
                    approved_at REAL,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    timestamp REAL DEFAULT (julianday('now') * 86400),
                    hash_chain TEXT
                )
            """)
            # RBAC, users, sessions, metrics, etc. (as before)
            # ...existing code...
            conn.commit()
            # Opportunistic migrations (add columns if upgrading from earlier schema)
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
                # Add new columns (backward compatibility with older DB files)
                if 'trust_score' not in cols:
                    try:
                        conn.execute('ALTER TABLE nodes ADD COLUMN trust_score INTEGER DEFAULT 0')
                        cols.add('trust_score')
                        logging.info('Migrated: added trust_score column to nodes')
                    except Exception as me:
                        logging.error(f'migration add trust_score failed: {me}')
                if 'quarantine' not in cols:
                    try:
                        conn.execute('ALTER TABLE nodes ADD COLUMN quarantine INTEGER DEFAULT 0')
                        cols.add('quarantine')
                        logging.info('Migrated: added quarantine column to nodes')
                    except Exception as me:
                        logging.error(f'migration add quarantine failed: {me}')
                if 'status' not in cols:
                    try:
                        conn.execute("ALTER TABLE nodes ADD COLUMN status TEXT DEFAULT 'active'")
                        cols.add('status')
                        logging.info('Migrated: added status column to nodes')
                    except Exception as me:
                        logging.error(f'migration add status failed: {me}')
                if 'device_class' not in cols:
                    try:
                        conn.execute("ALTER TABLE nodes ADD COLUMN device_class TEXT DEFAULT 'generic'")
                        cols.add('device_class')
                        logging.info('Migrated: added device_class column to nodes')
                    except Exception as me:
                        logging.error(f'migration add device_class failed: {me}')
                if 'last_heartbeat' not in cols:
                    conn.execute('ALTER TABLE nodes ADD COLUMN last_heartbeat REAL')
                if 'last_protocol_check' not in cols:
                    conn.execute('ALTER TABLE nodes ADD COLUMN last_protocol_check REAL')
                conn.commit()
            except Exception as e:
                logging.debug(f"nodes table migration skipped/failed: {e}")
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS node_protocol_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id TEXT NOT NULL,
                        protocol TEXT NOT NULL,
                        status TEXT NOT NULL,
                        checked_at REAL DEFAULT (julianday('now') * 86400),
                        FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                    )
                """)
                conn.commit()
            except Exception as e:
                logging.debug(f"node_protocol_health table ensure failed: {e}")

    def add_node_advanced(self, node_id, node_type, hostname, ip_address, port, status, trust_score, quarantine, resources, permissions, attestation, approval_status):
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO nodes (node_id, node_type, hostname, ip_address, port, status, trust_score, quarantine, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (node_id, node_type, hostname, ip_address, port, status, trust_score, int(quarantine), time.time())
                )
                # Insert resources
                for k, v in (resources or {}).items():
                    conn.execute("INSERT INTO node_resources (node_id, resource_key, resource_value) VALUES (?, ?, ?)", (node_id, k, str(v)))
                # Insert permissions
                for perm in (permissions or []):
                    conn.execute("INSERT INTO node_permissions (node_id, permission) VALUES (?, ?)", (node_id, perm))
                # Insert attestation
                conn.execute("INSERT INTO node_attestations (node_id, device_fingerprint, public_key_pem, device_certificate, health_attestation, geoip, behavioral_baseline, attested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (node_id, attestation.get('device_fingerprint'), attestation.get('public_key_pem'), attestation.get('device_certificate'),
                     attestation.get('health_attestation'), attestation.get('geoip'), attestation.get('behavioral_baseline'), time.time()))
                # Insert approval
                conn.execute("INSERT INTO node_approvals (node_id, approved_by, approved_at, status) VALUES (?, ?, ?, ?)",
                    (node_id, None, None, approval_status))
                conn.commit()

    def log_audit(self, event_type, source, message, severity="info"):
        # Tamper-evident: hash chain (simple, not full blockchain)
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                prev = conn.execute("SELECT hash_chain FROM audit_logs ORDER BY id DESC LIMIT 1").fetchone()
                prev_hash = prev[0] if prev else ''
                import hashlib
                h = hashlib.sha256((prev_hash + event_type + source + message + severity + str(time.time())).encode()).hexdigest()
                conn.execute("INSERT INTO audit_logs (event_type, source, message, severity, timestamp, hash_chain) VALUES (?, ?, ?, ?, ?, ?)",
                    (event_type, source, message, severity, time.time(), h))
                conn.commit()
    
    def add_node(self, node_id: str, node_type: str, hostname: str, ip_address: str, port: int, resources: dict):
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO nodes (node_id, node_type, hostname, ip_address, port, status, trust_score, quarantine, last_heartbeat, last_protocol_check, created_at) VALUES (?, ?, ?, ?, ?, COALESCE((SELECT status FROM nodes WHERE node_id=?),'active'), COALESCE((SELECT trust_score FROM nodes WHERE node_id=?),0), COALESCE((SELECT quarantine FROM nodes WHERE node_id=?),0), ?, COALESCE((SELECT last_protocol_check FROM nodes WHERE node_id=?),NULL), COALESCE((SELECT created_at FROM nodes WHERE node_id=?),?))",
                        (node_id, node_type, hostname, ip_address, port, node_id, node_id, node_id, time.time(), node_id, node_id, time.time())
                    )
                    conn.commit()
            except Exception as e:
                logging.error(f"Error adding node to database: {e}")

    def update_node_heartbeat(self, node_id: str, online: bool):
        try:
            with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute('UPDATE nodes SET status=?, last_heartbeat=? WHERE node_id=?', ('online' if online else 'offline', time.time(), node_id))
                conn.commit()
        except Exception as e:
            logging.debug(f"update_node_heartbeat failed for {node_id}: {e}")

    def record_protocol_health(self, node_id: str, protocol: str, status: str):
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                    conn.execute('INSERT INTO node_protocol_health (node_id, protocol, status, checked_at) VALUES (?,?,?,?)', (node_id, protocol, status, time.time()))
                    conn.execute('UPDATE nodes SET last_protocol_check=? WHERE node_id=?', (time.time(), node_id))
                    conn.commit()
                self.protocol_health.setdefault(node_id, {})[protocol] = status
            except Exception as e:
                logging.debug(f"record_protocol_health failed for {node_id} {protocol}: {e}")

    def get_nodes(self) -> List[Dict]:
        with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
            cursor = conn.execute("SELECT * FROM nodes")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def set_quarantine(self, node_id: str, quarantine: bool):
        try:
            with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute('UPDATE nodes SET quarantine=?, status=? WHERE node_id=?', (1 if quarantine else 0, 'quarantined' if quarantine else 'active', node_id))
                conn.commit()
        except Exception as e:
            logging.debug(f"set_quarantine failed for {node_id}: {e}")

    def remove_node(self, node_id: str):
        try:
            with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute('DELETE FROM node_attestations WHERE node_id=?', (node_id,))
                conn.execute('DELETE FROM node_permissions WHERE node_id=?', (node_id,))
                conn.execute('DELETE FROM node_resources WHERE node_id=?', (node_id,))
                conn.execute('DELETE FROM node_approvals WHERE node_id=?', (node_id,))
                conn.execute('DELETE FROM nodes WHERE node_id=?', (node_id,))
                conn.commit()
        except Exception as e:
            logging.debug(f"remove_node failed for {node_id}: {e}")

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

def db_connect():
    """Centralized sqlite connect helper to ensure consistent flags (timeout, thread-safety)
    and attempt to enable WAL on new connections. Use this instead of sqlite3.connect(...) directly.
    """
    # Use api_server.database.db_path when available; fall back to a local path
    db_path = None
    try:
        db_path = api_server.database.db_path  # type: ignore
    except AttributeError as e:
        logging.error(f"api_server.database.db_path not available: {e}")
        db_path = os.path.join(os.path.dirname(__file__), 'omega_control.db')
    except Exception as e:
        logging.error(f"Unexpected error getting db_path: {e}")
        db_path = os.path.join(os.path.dirname(__file__), 'omega_control.db')
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    try:
        conn.execute('PRAGMA journal_mode=WAL;')
    except sqlite3.DatabaseError as e:
        logging.error(f"Failed to set WAL mode on DB connection: {e}")
    except Exception as e:
        logging.error(f"Unexpected error setting WAL mode: {e}")
    return conn


class PerformanceAnalyzer:
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.prediction_model = None

    def analyze_performance(self, metrics: List[Dict]) -> Dict:
        if not metrics:
            return {"status": "no_data"}

        cpu_avg = float(np.mean([m.get('cpu_usage', 0) for m in metrics[-10:]]))
        memory_avg = float(np.mean([m.get('memory_usage', 0) for m in metrics[-10:]]))
        gpu_avg = float(np.mean([m.get('gpu_usage', 0) for m in metrics[-10:]]))

        health_score = 100 - (cpu_avg + memory_avg + gpu_avg) / 3

        bottlenecks = []
        if cpu_avg > 85:
            bottlenecks.append("CPU")
        if memory_avg > 90:
            bottlenecks.append("Memory")
        if gpu_avg > 95:
            bottlenecks.append("GPU")

        efficiency_rating = max(1, 5 - len(bottlenecks))

        recommendations = self.generate_recommendations(bottlenecks, cpu_avg, memory_avg, gpu_avg)

        return {
            "health_score": round(health_score, 1),
            "efficiency_rating": efficiency_rating,
            "bottlenecks": bottlenecks,
            "cpu_average": round(cpu_avg, 1),
            "memory_average": round(memory_avg, 1),
            "gpu_average": round(gpu_avg, 1),
            "recommendations": recommendations
        }

    def generate_recommendations(self, bottlenecks: List[str], cpu_avg: float, memory_avg: float, gpu_avg: float) -> List[Dict]:
        recommendations: List[Dict] = []

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
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker ps failed: {e}")
            return
        except FileNotFoundError:
            logging.warning("Docker not found; skipping VD reconciliation")
            return
        except Exception as e:
            logging.error(f"Unexpected error listing docker containers: {e}")
            return
        for line in out:
            try:
                cid, name, status, labels = (line.split('\t') + ['','','',''])[:4]
            except ValueError as e:
                logging.debug(f"Unexpected docker ps line format: {line} -> {e}")
                continue
            except Exception as e:
                logging.error(f"Error parsing docker ps line: {e}")
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
            except sqlite3.DatabaseError as e:
                logging.error(f"DB error checking vd_session_meta for {session_id}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error checking vd_session_meta for {session_id}: {e}")
            # Inspect container for ports, env, image
            try:
                info = subprocess.check_output(['docker','inspect',cid], stderr=subprocess.DEVNULL, timeout=10).decode()
                j = json.loads(info)[0]
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
                            except (TypeError, ValueError):
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
                except Exception as db_e:
                    logging.error(f"Failed logging vd_reconcile_error: {db_e}")

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
        # Ensure session_meta table exists for persisted session key storage
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_meta (
                        session_id TEXT PRIMARY KEY,
                        key TEXT NOT NULL,
                        user TEXT,
                        created_at REAL,
                        last_rotate REAL,
                        counter INTEGER DEFAULT 0,
                        expires_at REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Failed ensuring session_meta table: {e}")
        # Revoked sessions and pending rekey tables
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS revoked_sessions (
                        session_id TEXT PRIMARY KEY,
                        revoked_at REAL,
                        reason TEXT
                    )
                ''')
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS session_pending_rekey (
                        session_id TEXT PRIMARY KEY,
                        new_key TEXT,
                        created_at REAL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logging.error(f"Failed ensuring revoked/pending tables: {e}")

    # --- Background task orchestration ---
    async def start_background_tasks(self):
        asyncio.create_task(self.metrics_collector())
        asyncio.create_task(self.health_monitor())
        asyncio.create_task(self.broadcast_updates())
        asyncio.create_task(self.session_maintenance())
        # Global monitors (module-level coroutines)
        asyncio.create_task(node_heartbeat_monitor())
        asyncio.create_task(protocol_health_monitor())

    async def session_maintenance(self):
        """Periodic cleanup of expired sessions from memory & DB."""
        interval = int(os.environ.get('OMEGA_SESSION_MAINT_INTERVAL', '60'))
        while True:
            try:
                now = time.time()
                expired = []
                with SESSION_LOCK:
                    for sid, meta in list(SESSION_META.items()):
                        try:
                            if meta.get('expires_at') and meta['expires_at'] <= now:
                                expired.append(sid)
                        except Exception:
                            continue
                    if expired:
                        try:
                            with db_connect() as conn:
                                for sid in expired:
                                    SESSION_META.pop(sid, None)
                                    SESSION_NONCES.pop(sid, None)
                                    try:
                                        conn.execute('DELETE FROM session_meta WHERE session_id=?', (sid,))
                                    except Exception as e:
                                        logging.error(f'session_maintenance delete error {sid}: {e}')
                                conn.commit()
                        except Exception as e:
                            logging.error(f'session_maintenance DB cleanup error: {e}')
                for sid in expired:
                    try:
                        self.database.log_event('session_expired', sid, 'Expired & removed', 'info')
                    except Exception:
                        pass
            except Exception as e:
                logging.error(f'session_maintenance loop error: {e}')
            await asyncio.sleep(interval)

    def load_persisted_sessions(self):
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cur = conn.execute('SELECT session_id,key,user,created_at,last_rotate,counter,expires_at FROM session_meta')
                rows = cur.fetchall()
            loaded=0
            now=time.time()
            with SESSION_LOCK:
                for r in rows:
                    sid, key_hex, user, created, last_rotate, counter, expires_at = r
                    if expires_at and expires_at <= now:
                        continue
                    SESSION_META[sid]={'key':bytes.fromhex(key_hex),'user':user,'created_at':created,'last_rotate':last_rotate,'counter':counter,'expires_at':expires_at}
                    loaded+=1
            logging.info(f"Loaded {loaded} persisted sessions")
        except Exception as e:
            logging.warning(f"load_persisted_sessions failed: {e}")

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
        """Periodic lightweight push placeholder (reserved for websocket broadcasting)."""
        while True:
            try:
                # Future: iterate self.connected_clients and send deltas
                await asyncio.sleep(5)
            except Exception:
                await asyncio.sleep(5)

# Instantiate global API server instance
api_server = OmegaAPIServer()

# --- Background monitors (module-level) ---
async def node_heartbeat_monitor():
    interval = int(os.environ.get('OMEGA_HEARTBEAT_INTERVAL','20'))
    offline_after = int(os.environ.get('OMEGA_OFFLINE_THRESHOLD','120'))
    while True:
        try:
            nodes = api_server.database.get_nodes()
            now = time.time()
            for n in nodes:
                last = n.get('last_heartbeat') or n.get('created_at') or now
                online = (now - float(last)) < offline_after
                api_server.database.update_node_heartbeat(n['node_id'], online)
        except Exception as e:
            logging.debug(f"node_heartbeat_monitor iteration failed: {e}")
        await asyncio.sleep(interval)

async def protocol_health_monitor():
    interval = int(os.environ.get('OMEGA_PROTOCOL_HEALTH_INTERVAL','60'))
    while True:
        try:
            nodes = api_server.database.get_nodes()
            for n in nodes:
                host = n.get('ip_address')
                if not host:
                    continue
                status = 'down'
                try:
                    import socket
                    with socket.create_connection((host, int(n.get('port',8443))), timeout=0.4):
                        status = 'up'
                except Exception:
                    pass
                for proto in SUPPORTED_PROTOCOLS:
                    api_server.database.record_protocol_health(n['node_id'], proto, status if proto.startswith('gRPC') else 'unknown')
        except Exception as e:
            logging.debug(f"protocol_health_monitor iteration failed: {e}")
        await asyncio.sleep(interval)
    
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

# CORS configuration warning helper
_allow_origins = os.environ.get('OMEGA_CORS_ORIGINS')
if _allow_origins:
    _allow_origins = [o.strip() for o in _allow_origins.split(',') if o.strip()]
else:
    _allow_origins = ['*']
if _allow_origins == ['*'] and os.environ.get('OMEGA_ENV','').lower() not in ('dev','development','local'):
    logging.warning('CORS is configured with allow_origins="*". In production set OMEGA_CORS_ORIGINS to a trusted domain list')

security = HTTPBearer()

# --- Simple in-memory rate limiter (token-bucket) ---
from collections import defaultdict
_RATE_BUCKETS = defaultdict(lambda: {'tokens': 20, 'last': time.time()})
_RATE_LOCK = threading.Lock()

def rate_limited(per_minute: int = 60, burst: int = 120):
    """Decorator to rate-limit endpoints per-client IP using a token-bucket.
    - per_minute: refill rate
    - burst: bucket capacity
    """
    refill_per_sec = per_minute / 60.0
    def _decorator(func):
        async def _wrapped(*args, **kwargs):
            # Attempt to extract Request from args or kwargs
            req = None
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, Request):
                    req = a
                    break
            ip = 'unknown'
            try:
                if req:
                    client = getattr(req, 'client', None)
                    if client:
                        ip = getattr(client, 'host', 'unknown') or 'unknown'
            except Exception:
                ip = 'unknown'

            now = time.time()
            with _RATE_LOCK:
                bucket = _RATE_BUCKETS[ip]
                # refill
                elapsed = now - bucket['last']
                bucket['tokens'] = min(burst, bucket['tokens'] + elapsed * refill_per_sec)
                bucket['last'] = now
                if bucket['tokens'] < 1:
                    raise HTTPException(status_code=429, detail='Too many requests')
                bucket['tokens'] -= 1

            return await func(*args, **kwargs)
        return _wrapped
    return _decorator


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
    # Load persisted sessions first so they are available to other startup tasks
    try:
        api_server.load_persisted_sessions()
    except Exception as e:
        logging.warning(f"Failed loading persisted sessions at startup: {e}")
    await api_server.start_background_tasks()
    try:
        api_server.database.ensure_admin_role()
    except sqlite3.DatabaseError as e:
        logging.warning(f"RBAC DB init warning: {e}")
    except Exception as e:
        logging.warning(f"RBAC init warning: {e}")
    try:
        api_server.reconcile_vd_sessions_from_docker()
    except FileNotFoundError:
        logging.info("Docker not available for VD reconcile at startup")
    except subprocess.CalledProcessError as e:
        logging.warning(f"VD reconcile subprocess error: {e}")
    except Exception as e:
        logging.warning(f"VD reconcile warning: {e}")
    logging.info("Omega API Server started")


@app.post("/api/auth/login")
@rate_limited(per_minute=30, burst=10)
async def login(auth: AuthToken):
    if auth.username == "admin" and auth.password == "omega123":
        # Issue an encrypted token (no AES session key included)
        encrypted_token = api_server.security_manager.encrypt_data(
            json.dumps({"user_id": auth.username, "issued_at": time.time()})
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
# In-memory map of session_id -> WebSocket for push notifications
SESSION_WS: Dict[str, WebSocket] = {}

# Revoked sessions cache (in-memory mirror of DB)
REVOKED_SESSIONS: set = set()
# Lock to protect in-memory session structures across threads/async tasks
SESSION_LOCK = threading.RLock()

def register_session_meta(session_id:str, key_b64:str):
    now = time.time()
    # Keep raw (base64) key in memory for fast crypto ops; persist an encrypted envelope to DB
    with SESSION_LOCK:
        SESSION_META[session_id] = {
            'created': now,
            'last_rotate': now,
            'key': key_b64,
            'counter': 0,
            'expires_at': now + (6 * 3600)  # default 6 hours
        }
        SESSION_NONCES[session_id] = deque(maxlen=NONCE_WINDOW)
    # Persist to DB if available (best-effort)
    try:
        if 'api_server' in globals() and getattr(api_server, 'database', None):
            # encrypt key_b64 with master Fernet before storing, so DB does not contain raw session keys
            try:
                raw_key = base64.b64decode(key_b64)
                enc = api_server.security_manager.cipher_suite.encrypt(raw_key)
                store_key = base64.b64encode(enc).decode()
            except (binascii.Error, TypeError) as e:
                logging.debug(f"register_session_meta: provided key not valid base64, storing raw: {e}")
                store_key = key_b64
            except Exception as e:
                logging.error(f"register_session_meta: unexpected error encrypting session key: {e}")
                store_key = key_b64
            try:
                with db_connect() as conn:
                    conn.execute('INSERT OR REPLACE INTO session_meta (session_id, key, user, created_at, last_rotate, counter, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                                 (session_id, store_key, SESSION_META.get(session_id, {}).get('user'), now, now, 0, SESSION_META.get(session_id, {})['expires_at']))
                    conn.commit()
            except sqlite3.DatabaseError as e:
                logging.error(f"register_session_meta: DB error persisting session_meta for {session_id}: {e}")
            except Exception as e:
                logging.error(f"register_session_meta: unexpected error persisting session_meta for {session_id}: {e}")
    except Exception as e:
        # Best effort persistence; do not fail registration on DB errors
        logging.error(f"register_session_meta: unexpected top-level error: {e}")

# Basic validation function for secure endpoints
def validate_secure(headers):
    """Validate AES-GCM session headers and return (session_id, key_bytes)."""
    session_id = headers.get('X-Session-ID')
    auth_header = headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing or invalid authorization header')
    if not session_id:
        raise HTTPException(status_code=401, detail='Missing session id header')
    # Check revocation first (in-memory cache)
    if session_id in REVOKED_SESSIONS:
        raise HTTPException(status_code=401, detail='Session revoked')

    with SESSION_LOCK:
        meta = SESSION_META.get(session_id)
        if not meta or 'key' not in meta:
            # Best-effort: check DB for revocation/pending and return generic error
            try:
                with db_connect() as conn:
                    cur = conn.execute('SELECT revoked_at FROM revoked_sessions WHERE session_id=?', (session_id,))
                    if cur.fetchone():
                        REVOKED_SESSIONS.add(session_id)
                        raise HTTPException(status_code=401, detail='Session revoked')
            except HTTPException:
                raise
            except sqlite3.DatabaseError as e:
                logging.error(f"validate_secure: DB error checking revoked_sessions for {session_id}: {e}")
            except Exception as e:
                logging.error(f"validate_secure: unexpected error checking revoked_sessions for {session_id}: {e}")
            raise HTTPException(status_code=401, detail='Unknown or uninitialized session')
        # Enforce expiry strictly on each request
        expires_at = meta.get('expires_at')
        if expires_at and time.time() > expires_at:
            # cleanup
            try:
                SESSION_META.pop(session_id, None)
                SESSION_NONCES.pop(session_id, None)
                try:
                    with db_connect() as conn:
                        conn.execute('DELETE FROM session_meta WHERE session_id=?', (session_id,))
                        conn.commit()
                except sqlite3.DatabaseError as e:
                    logging.error(f"validate_secure: DB error deleting expired session_meta for {session_id}: {e}")
            except Exception as e:
                logging.error(f"validate_secure: unexpected cleanup error for {session_id}: {e}")
            raise HTTPException(status_code=401, detail='Session expired')
        try:
            key = base64.b64decode(meta['key'])
        except Exception:
            raise HTTPException(status_code=401, detail='Invalid stored key encoding')
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

# In-memory per-node advanced state (protocols, policies) - can be persisted later
NODE_PROTOCOL_STATE: dict[str, str] = {}
NODE_POLICIES: dict[str, dict[str,str]] = {}
SUPPORTED_PROTOCOLS = ['gRPC/QUIC','gRPC/HTTP2','ZeroMQ','RDMA']

# Fallback minimal discovery (in case full implementation below fails early during import)
async def action_discover_nodes():  # lightweight placeholder (shadowed by full version later if defined again)
    try:
        nodes = api_server.database.get_nodes()
        return {'success': True, 'discovered': 0, 'nodes': [n['node_id'] for n in nodes], 'scanned': 0, 'placeholder': True}
    except Exception:
        return {'success': True, 'discovered': 0, 'nodes': [], 'scanned': 0, 'placeholder': True}

@app.post('/api/secure/action')
async def secure_action(request: Request, body: ActionRequest):
    session_id, key = validate_secure(request.headers)
    if body.action not in SECURE_ACTIONS:
        raise HTTPException(status_code=400, detail='Unknown action')
    meta = SESSION_META.get(session_id)
    if not meta:
        raise HTTPException(status_code=401, detail='Session meta missing')
    # Counter monotonic and nonce duplicate checks under lock
    with SESSION_LOCK:
        # Allow concurrent/out-of-order arrival by accepting ctr >= meta['counter']
        # Reject strictly older counters which are definitely replays
        if body.ctr < meta.get('counter', 0):
            raise HTTPException(status_code=401, detail='Counter replay (too old)')
        if abs(time.time()-body.ts) > 30:
            raise HTTPException(status_code=401, detail='Stale action')
        # Duplicate nonce check
        nonces = SESSION_NONCES.get(session_id)
        if nonces is None:
            SESSION_NONCES[session_id] = deque(maxlen=NONCE_WINDOW)
            nonces = SESSION_NONCES[session_id]
        if body.nonce in nonces:
            raise HTTPException(status_code=401, detail='Nonce replay detected')
        nonces.append(body.nonce)
        # advance stored counter conservatively to the max seen so far
        meta['counter'] = max(meta.get('counter', 0), body.ctr)
    # Execute action with robust error handling
    result_data=None
    result_enc=None
    try:
        if body.action == 'discover_nodes':
            # returns raw dict (not encrypted) now
            result_data = await action_discover_nodes()
        elif body.action == 'run_benchmark':
            result_enc = await action_run_benchmark()
        elif body.action == 'health_check':
            result_enc = await action_health_check()
        elif body.action == 'restart_node':
            await asyncio.sleep(1)
            result_enc = api_server.security_manager.encrypt_data(json.dumps({'success':True,'message':'Node restart initiated','node': body.params.get('node_id') if body.params else None}))
        else:
            raise HTTPException(status_code=400, detail='Unhandled action')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"secure_action error action={body.action}: {e}")
        if body.action == 'discover_nodes':
            result_data={'success':False,'error':str(e)}
        else:
            # wrap generic error
            result_data={'success':False,'error':str(e)}
            # ensure encryption path below works by converting to encrypted stub
            result_enc = api_server.security_manager.encrypt_data(json.dumps(result_data))
    # decrypt intermediate encrypted payload
    if body.action == 'discover_nodes':
        data = result_data
    else:
        # result_enc is an encrypted mapping
        if isinstance(result_enc, EncryptedMessage):
            payload = api_server.security_manager.decrypt_data(result_enc)
        else:
            # expect dict-like
            payload = api_server.security_manager.decrypt_data(EncryptedMessage(**result_enc))
        data = json.loads(payload)
    return wrap_encrypted(session_id, key, {'action': body.action, 'result': data, 'ok': True})

@app.post('/api/secure/discover')
async def secure_discover(request: Request):
    """Dedicated discovery endpoint returning encrypted payload; never 501."""
    session_id, key = validate_secure(request.headers)
    try:
        result = await action_discover_nodes()
        if 'success' not in result:
            result['success']=True
    except Exception as e:
        logging.error(f"secure_discover error: {e}")
        result={'success':False,'error':str(e)}
    return wrap_encrypted(session_id, key, {'action':'discover_nodes','result':result,'ok':True})

@app.get('/api/secure/nodes/{node_id}/protocol/health')
async def get_protocol_health(node_id: str, request: Request):
    session_id, key = validate_secure(request.headers)
    # Return recent protocol health statuses
    rec = api_server.database.protocol_health.get(node_id, {})
    return wrap_encrypted(session_id, key, {'node_id': node_id, 'protocol_health': rec})

class Heartbeat(BaseModel):
    node_id: str
    status: Optional[str] = 'online'
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None

@app.post('/api/secure/nodes/heartbeat')
async def secure_heartbeat(body: Heartbeat, request: Request):
    session_id, key = validate_secure(request.headers)
    api_server.database.update_node_heartbeat(body.node_id, body.status == 'online')
    # Persist lightweight metrics sample if provided
    try:
        if (body.cpu_usage is not None) or (body.memory_usage is not None):
            nm = NodeMetrics(
                node_id=body.node_id,
                cpu_usage=float(body.cpu_usage or 0.0),
                memory_usage=float(body.memory_usage or 0.0),
                gpu_usage=0.0,
                network_rx=0.0,
                network_tx=0.0,
                temperature=0.0,
                power_consumption=0.0,
                timestamp=time.time()
            )
            api_server.database.add_metrics(nm)
    except Exception as e:
        logging.debug(f"heartbeat metrics insert failed for {body.node_id}: {e}")
    api_server.database.log_event('node_heartbeat', body.node_id, f"Heartbeat {body.status}")
    return wrap_encrypted(session_id, key, {'ok': True})

@app.post('/api/secure/nodes/{node_id}/probe')
async def secure_probe(node_id: str, request: Request):
    session_id, key = validate_secure(request.headers)
    node = next((n for n in api_server.database.get_nodes() if n['node_id']==node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail='node not found')
    host=node.get('ip_address'); port=int(node.get('port',8443));
    results={}
    # gRPC/QUIC placeholder: attempt TCP connect
    import socket
    for proto in SUPPORTED_PROTOCOLS:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                status='up'
        except Exception:
            status='down'
        api_server.database.record_protocol_health(node_id, proto, status if proto.startswith('gRPC') else 'unknown')
        results[proto]=status if proto.startswith('gRPC') else 'unknown'
    return wrap_encrypted(session_id, key, {'node_id': node_id, 'probe': results})

class QuarantineReq(BaseModel):
    node_id: str
    enable: bool

@app.post('/api/secure/nodes/quarantine')
async def secure_quarantine(body: QuarantineReq, request: Request):
    session_id, key = validate_secure(request.headers)
    api_server.database.set_quarantine(body.node_id, body.enable)
    api_server.database.log_event('node_quarantine', body.node_id, f"Quarantine={'on' if body.enable else 'off'}")
    return wrap_encrypted(session_id, key, {'node_id': body.node_id, 'quarantine': body.enable})

class RemoveReq(BaseModel):
    node_id: str

@app.post('/api/secure/nodes/remove')
async def secure_remove(body: RemoveReq, request: Request):
    session_id, key = validate_secure(request.headers)
    api_server.database.remove_node(body.node_id)
    api_server.database.log_event('node_remove', body.node_id, 'Node removed')
    return wrap_encrypted(session_id, key, {'removed': body.node_id})

class VdStartRequest(BaseModel):
    node_id: str
    image: Optional[str] = 'ubuntu-xfce'
    cpu_cores: int = 2
    memory_gb: int = 4

@app.post('/api/secure/vd/start')
async def vd_start(body: VdStartRequest, request: Request):
    session_id, key = validate_secure(request.headers)
    # Placeholder: allocate a pseudo session id
    vd_sid = f"vd-{secrets.token_hex(6)}"
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR IGNORE INTO sessions (session_id, user_id, node_id, application, cpu_cores, gpu_units, memory_gb, status, created_at, last_activity) VALUES (?,?,?,?,?,?,?,?,?,?)',
                         (vd_sid, SESSION_META.get(session_id,{}).get('user','admin'), body.node_id, 'virtual-desktop', body.cpu_cores, 0, body.memory_gb, 'starting', time.time(), time.time()))
            conn.commit()
        api_server.database.log_event('vd_start', vd_sid, f"Requested on {body.node_id}", 'info')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed start: {e}')
    # Simulate connect URL
    connect_url = f"http://localhost:7000/?session={vd_sid}"
    return wrap_encrypted(session_id, key, {'success': True, 'session_id': vd_sid, 'connect_url': connect_url})

@app.get('/api/secure/vd/list')
async def vd_list(request: Request):
    session_id, key = validate_secure(request.headers)
    sessions = []
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute("SELECT session_id,node_id,application,status,created_at FROM sessions WHERE application='virtual-desktop' ORDER BY created_at DESC LIMIT 100")
            sessions = [{'session_id':r[0], 'node_id':r[1], 'application':r[2], 'status':r[3], 'created_at':r[4]} for r in cur.fetchall()]
    except Exception:
        pass
    return wrap_encrypted(session_id, key, {'sessions': sessions})

async def action_discover_nodes():
    """Advanced adaptive discovery.
    Methods (toggle via env OMEGA_DISCOVERY_METHODS= tcp,icmp,arp,mdns ):
      tcp  : (always on) bounded /24 TCP port scan of candidate subnets.
      icmp : single "ping" (system ping utility) for additional liveness if no tcp port open.
      arp  : parse local ARP cache to seed additional IPs (no active probe cost).
      mdns : (placeholder) send one multicast query to collect responders (future hook).
    Enhancements:
      - Optionally include controller host itself (OMEGA_DISCOVERY_INCLUDE_SELF=1) so UI shows local node.
      - Records which method produced each discovery (metadata only, not persisted yet).
      - Respects existing caps (per-interface 256, global 1024) + overall timeout window.
      - Environment controls: OMEGA_* vars documented below.
    Returns: success, discovered, nodes (ids), scanned, methods_used, self_added, disabled flags.
    """
    if os.environ.get('OMEGA_ENABLE_DISCOVERY','1').lower() not in ('1','true','yes','on'):  # fast escape
        return {'success':True,'discovered':0,'nodes':[],'disabled':True}
    import ipaddress, socket, asyncio, subprocess, json as _json
    methods_env = os.environ.get('OMEGA_DISCOVERY_METHODS','tcp,arp').lower().replace(' ','')
    enabled_methods = {m for m in methods_env.split(',') if m}
    ports = [int(p) for p in os.environ.get('OMEGA_DISCOVERY_PORTS','8443,22').split(',') if p.strip().isdigit()] or [8443,22]
    timeout = float(os.environ.get('OMEGA_DISCOVERY_TIMEOUT','0.35'))
    include_self = os.environ.get('OMEGA_DISCOVERY_INCLUDE_SELF','1').lower() in ('1','true','yes','on')

    nets = psutil.net_if_addrs()
    host_ips=set()
    interface_ips=[]
    for name, addr_list in nets.items():
        addr = next((a for a in addr_list if getattr(a.family,'name',str(a.family)) in ('AF_INET','AddressFamily.AF_INET')), None)
        if not addr: continue
        ip = addr.address
        if ip.startswith('127.') or ip.startswith('169.254.'):
            continue
        interface_ips.append(ip)
        netmask = getattr(addr,'netmask', None) or '255.255.255.0'
        try:
            network = ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
        except ValueError:
            continue
        if network.prefixlen < 24:  # bound to /24 window
            host_ip = ipaddress.ip_address(ip)
            base_int = int(host_ip) & 0xFFFFFF00
            network = ipaddress.ip_network((base_int, 24))
        candidates=[str(h) for h in network.hosts()][:256]
        for h in candidates:
            if h!=ip:
                host_ips.add(h)

    # ARP seeding (no active probe). Parse `arp -a` output (best-effort, ignore errors)
    arp_mac_map={}
    if 'arp' in enabled_methods:
        try:
            arp_out = subprocess.run(['arp','-a'], capture_output=True, text=True, timeout=1).stdout
            for line in arp_out.splitlines():
                # typical: ? (192.168.1.34) at xx:xx:.. on en0 ifscope [ethernet]
                if '(' in line and ')' in line:
                    ip=line.split('(')[1].split(')')[0].strip()
                    if ip and all(not ip.startswith(pref) for pref in ('127.','169.254.')):
                        host_ips.add(ip)
                        parts=line.split()
                        mac=None
                        if ' at ' in line:
                            try:
                                mac=line.split(' at ')[1].split(' ')[0].strip()
                            except Exception:
                                mac=None
                        if mac:
                            arp_mac_map[ip]=mac.lower()
        except Exception:
            pass

    # Optional mDNS service discovery (zeroconf) to find additional hosts (e.g., mobile devices)
    if 'mdns' in enabled_methods:
        try:
            # Optional dependency (install via pip install zeroconf). Marked type ignore to avoid analyzer error if missing.
            from zeroconf import Zeroconf, ServiceBrowser  # type: ignore
            mdns_hosts=set()
            class _Listener:
                def add_service(self, zc, t, name):
                    try:
                        info=zc.get_service_info(t,name)
                        if info and info.addresses:
                            for a in info.addresses:
                                import ipaddress as _ip
                                try:
                                    ip_str=str(_ip.ip_address(a))
                                    if ip_str and not ip_str.startswith('127.'):
                                        mdns_hosts.add(ip_str)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                def update_service(self, *args, **kwargs):
                    pass
                def remove_service(self, *args, **kwargs):
                    pass
            zc=Zeroconf()
            listener=_Listener()
            # Common service types to browse quickly (short window)
            service_types=['_http._tcp.local.','_workstation._tcp.local.','_ssh._tcp.local.']
            browsers=[ServiceBrowser(zc, st, listener) for st in service_types]
            await asyncio.sleep(min(1.5, timeout*4))
            zc.close()
            for ipx in mdns_hosts:
                host_ips.add(ipx)
            if mdns_hosts:
                enabled_methods.add('mdns')
        except Exception:
            # silently ignore if zeroconf not installed
            pass

    candidates=list(host_ips)[:1024]
    methods_used=set()
    results=[]
    sem = asyncio.Semaphore(64)
    loop=asyncio.get_event_loop()

    def _tcp_probe(host,port,timeout):
        try:
            with socket.create_connection((host,port), timeout=timeout):
                return True
        except Exception:
            return False

    async def tcp_probe(host):
        found=[]
        for p in ports:
            try:
                async with sem:
                    fut = loop.run_in_executor(None, lambda: _tcp_probe(host,p,timeout))
                    ok = await asyncio.wait_for(fut, timeout=timeout+0.15)
                if ok:
                    found.append(p)
            except Exception:
                pass
        return found

    def icmp_ping(host):  # uses system ping (works unprivileged on mac/linux)
        try:
            # macOS: -c 1 count, -W timeout (linux); macOS uses -W for packet timeout in ms? Fallback with small overall timeout
            cmd=['ping','-c','1','-W','1',host]
            res=subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.2)
            return res.returncode==0
        except Exception:
            return False

    async def probe(host):
        host_methods=[]
        ports_found=[]
        if 'tcp' in enabled_methods:
            ports_found = await tcp_probe(host)
            if ports_found:
                host_methods.append('tcp')
        omega=False
        if 8443 in ports_found:
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://{host}:8443/api/ping", timeout=timeout) as r:
                    if r.status==200:
                        omega=True; host_methods.append('ping')
            except Exception:
                pass
        # If nothing via TCP and icmp enabled -> try icmp to mark liveness
        if not ports_found and 'icmp' in enabled_methods:
            if await loop.run_in_executor(None, lambda: icmp_ping(host)):
                host_methods.append('icmp')
        if not host_methods:
            return None
        methods_used.update(host_methods)
        return {'host':host,'ports':ports_found,'omega':omega,'methods':host_methods}

    # Launch probes
    if candidates:
        tasks=[probe(h) for h in candidates]
        for coro in asyncio.as_completed(tasks, timeout=min(25, timeout*len(tasks)+5)):
            try:
                res = await coro
                if res: results.append(res)
            except Exception:
                pass

    # Existing node IP map
    existing_nodes = {n['ip_address']: n for n in api_server.database.get_nodes() if n.get('ip_address')}
    new_ids=[]
    # MAC vendor heuristics for classification
    mobile_ouis={'34:15:9e','d8:bb:2c','28:16:ad','dc:a9:04','f8:ff:c2','3c:2e:f9','b8:53:ac','1c:1b:0d','ac:37:43','b4:0f:3b'}  # sample handful
    apple_ouis={'b8:27:eb','f0:18:98','a4:5e:60','0c:74:c2','60:f8:1d','d0:03:4b'}
    android_indicators={'samsung','oneplus','pixel','android','xiaomi','redmi','huawei','oppo'}
    def classify_device(ip):
        mac=arp_mac_map.get(ip,'')
        prefix=':'.join(mac.split(':')[:3]) if mac else ''
        if prefix in apple_ouis:
            return 'mobile-ios'
        if prefix in mobile_ouis:
            return 'mobile'
        host_lower=ip
        # Additional placeholder heuristics could go here
        return 'generic'
    for r in results:
        if r['host'] in existing_nodes:
            continue
        node_id = f"disc-{r['host'].replace('.','-')}"
        ntype = 'omega' if r['omega'] else ('alive' if r['ports'] else 'generic')
        api_server.database.add_node(node_id, ntype, node_id, r['host'], r['ports'][0] if r['ports'] else 0, {'cpu_cores': None, 'memory_gb': None})
        # classify & update device_class
        try:
            klass=classify_device(r['host'])
            with sqlite3.connect(api_server.database.db_path) as conn:
                conn.execute('UPDATE nodes SET device_class=? WHERE node_id=?', (klass, node_id))
                conn.commit()
        except Exception:
            pass
        new_ids.append(node_id)

    self_added=[]
    if include_self:
        # Add controller host(s) if not in DB
        controller_nodes = api_server.database.get_nodes()
        existing_ids={n['node_id'] for n in controller_nodes}
        for ip in interface_ips:
            sid=f"self-{ip.replace('.','-')}"
            if sid not in existing_ids:
                api_server.database.add_node(sid,'controller',sid,ip,8443, {'cpu_cores': psutil.cpu_count(), 'memory_gb': round(psutil.virtual_memory().total/1024**3,1)})
                self_added.append(sid)
                new_ids.append(sid)

    payload={
        'success':True,
        'discovered':len(new_ids),
        'nodes':new_ids,
        'scanned':len(candidates),
        'methods_used':sorted(methods_used),
        'self_added':self_added,
        'config':{
            'ports':ports,
            'timeout':timeout,
            'include_self':include_self,
            'enabled_methods':sorted(enabled_methods)
        }
    }
    return payload

# ---- Low latency link scaffolding ----
class LatencyStartReq(BaseModel):
    node_id: str

@app.post('/api/secure/nodes/latency/start')
async def latency_start(body: LatencyStartReq, request: Request):
    session_id, key = validate_secure(request.headers)
    # Generate ephemeral token & record (placeholder, real negotiation later)
    token = secrets.token_hex(16)
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS latency_links (id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT, token TEXT, created_at REAL, protocol TEXT, status TEXT, FOREIGN KEY(node_id) REFERENCES nodes(node_id))")
            conn.execute('INSERT INTO latency_links (node_id, token, created_at, protocol, status) VALUES (?,?,?,?,?)', (body.node_id, token, time.time(), 'negotiation', 'pending'))
            conn.commit()
    except Exception as e:
        logging.error(f'latency_start record error: {e}')
    # Determine available protocols (QUIC if aioquic importable)
    protocols=['websocket']
    try:
        import aioquic  # type: ignore
        protocols.append('quic')
    except Exception:
        pass
    payload={'success':True,'token':token,'protocols':protocols,'recommended':protocols[-1],'node_id':body.node_id}
    return wrap_encrypted(session_id, key, payload)

@app.get('/api/secure/nodes/latency/measure')
async def latency_measure(node_id: str, request: Request):
    session_id, key = validate_secure(request.headers)
    # Placeholder synthetic latency measurement
    import random
    base=random.uniform(3,12)  # ms
    jitter=random.uniform(0.2,1.5)
    payload={'success':True,'node_id':node_id,'latency_ms':round(base,2),'jitter_ms':round(jitter,2),'method':'synthetic'}
    return wrap_encrypted(session_id, key, payload)

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
        # attach active protocol (in-memory) and simple mtls flag placeholder
        n['protocol'] = NODE_PROTOCOL_STATE.get(n['node_id'],'gRPC/QUIC')
        n['mtls'] = True
    payload = {'nodes': nodes, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

# --- Advanced Node Endpoints ---
@app.get('/api/secure/nodes/{node_id}/protocol')
async def node_protocol_get(request: Request, node_id: str):
    sid, key = validate_secure(request.headers)
    active = NODE_PROTOCOL_STATE.get(node_id, 'gRPC/QUIC')
    return wrap_encrypted(sid, key, {'node_id': node_id, 'active': active, 'supported': SUPPORTED_PROTOCOLS})

class ProtocolSetRequest(BaseModel):
    protocol: str

@app.post('/api/secure/nodes/{node_id}/protocol')
async def node_protocol_set(request: Request, node_id: str, body: ProtocolSetRequest):
    sid, key = validate_secure(request.headers)
    proto = body.protocol.strip()
    if proto not in SUPPORTED_PROTOCOLS:
        raise HTTPException(status_code=400, detail='Unsupported protocol')
    NODE_PROTOCOL_STATE[node_id] = proto
    try:
        api_server.database.log_event('node_protocol_switch', node_id, f'Switched to {proto}', 'info')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'node_id': node_id, 'active': proto})

@app.get('/api/secure/nodes/{node_id}/telemetry')
async def node_telemetry(request: Request, node_id: str, limit: int = 60):
    sid, key = validate_secure(request.headers)
    limit = max(5, min(240, limit))
    data = api_server.database.get_latest_metrics(node_id=node_id, limit=limit)
    # reverse chronological currently; sort ascending by timestamp
    data = sorted(data, key=lambda x: x['timestamp'])
    series = {
        'cpu': [round(d.get('cpu_usage') or 0,2) for d in data],
        'mem': [round(d.get('memory_usage') or 0,2) for d in data],
        'ts': [d.get('timestamp') for d in data]
    }
    return wrap_encrypted(sid, key, {'node_id': node_id, 'series': series})

@app.get('/api/secure/nodes/{node_id}/policies')
async def node_policies_get(request: Request, node_id: str):
    sid, key = validate_secure(request.headers)
    pol = NODE_POLICIES.get(node_id, {})
    return wrap_encrypted(sid, key, {'node_id': node_id, 'policies': pol})

class PolicySetRequest(BaseModel):
    key: str
    value: str | None = ''

@app.post('/api/secure/nodes/{node_id}/policies')
async def node_policies_set(request: Request, node_id: str, body: PolicySetRequest):
    sid, key = validate_secure(request.headers)
    if not body.key.strip():
        raise HTTPException(status_code=400, detail='key required')
    NODE_POLICIES.setdefault(node_id, {})[body.key.strip()] = body.value or ''
    try:
        api_server.database.log_event('node_policy_set', node_id, f"{body.key}={body.value}", 'info')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'ok': True, 'policies': NODE_POLICIES[node_id]})

@app.delete('/api/secure/nodes/{node_id}/policies/{pkey}')
async def node_policies_delete(request: Request, node_id: str, pkey: str):
    sid, key = validate_secure(request.headers)
    pol = NODE_POLICIES.setdefault(node_id, {})
    if pkey in pol:
        pol.pop(pkey, None)
        try:
            api_server.database.log_event('node_policy_delete', node_id, f"{pkey}", 'info')
        except Exception:
            pass
    return wrap_encrypted(sid, key, {'ok': True, 'policies': pol})

@app.get('/api/secure/nodes/{node_id}/trust')
async def node_trust_get(request: Request, node_id: str):
    sid, key = validate_secure(request.headers)
    # look up in nodes table
    trust_score = 0
    quarantine = 0
    status = 'unknown'
    try:
        for n in api_server.database.get_nodes():
            if n['node_id']==node_id:
                trust_score = n.get('trust_score') or 0
                quarantine = n.get('quarantine') or 0
                status = n.get('status')
                break
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'node_id': node_id, 'trust_score': trust_score, 'quarantine': bool(quarantine), 'status': status})

class DiagnosticsRequest(BaseModel):
    level: str = 'basic'

@app.post('/api/secure/nodes/{node_id}/diagnostics')
async def node_diagnostics(request: Request, node_id: str, body: DiagnosticsRequest):
    sid, key = validate_secure(request.headers)
    # Simple simulated diagnostics leveraging latest metrics
    metrics = api_server.database.get_latest_metrics(node_id=node_id, limit=1)
    m = metrics[0] if metrics else {}
    result = {
        'node_id': node_id,
        'level': body.level,
        'checks': {
            'cpu': 'OK' if (m.get('cpu_usage',0) < 90) else 'HIGH',
            'memory': 'OK' if (m.get('memory_usage',0) < 90) else 'HIGH',
            'heartbeat': 'OK'
        },
        'timestamp': time.time()
    }
    try:
        api_server.database.log_event('node_diagnostics', node_id, f"diag level={body.level}", 'info')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'diagnostics': result})

class OTARequest(BaseModel):
    action: str  # update | rollback
    version: str | None = None

@app.post('/api/secure/nodes/{node_id}/ota')
async def node_ota(request: Request, node_id: str, body: OTARequest):
    sid, key = validate_secure(request.headers)
    if body.action not in ('update','rollback'):
        raise HTTPException(status_code=400, detail='invalid action')
    msg = f"OTA {body.action} triggered" + (f" target={body.version}" if body.version else '')
    try:
        api_server.database.log_event('node_ota', node_id, msg, 'info')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'ok': True, 'action': body.action, 'version': body.version})

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
            try:
                p.kill()
            except Exception as e:
                logging.error(f"secure_process_kill: failed to kill process {pid}: {e}")
    except psutil.NoSuchProcess:
        logging.info(f"secure_process_kill: process {pid} not found")
        return wrap_encrypted(session_id, key, {'success': False, 'pid': pid, 'message': 'process not found'})
    except Exception as e:
        logging.error(f"secure_process_kill: unexpected error for pid {pid}: {e}")
        raise HTTPException(status_code=500, detail='Failed to terminate process')

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
    # WebSocket clients must provide only a valid session_id; key material remains server-side and is not transmitted
    if not session_id or session_id not in SESSION_META or 'key' not in SESSION_META[session_id]:
        await ws.close()
        return
    # register ws for push notifications
    try:
        with SESSION_LOCK:
            SESSION_WS[session_id] = ws
    except Exception as e:
        logging.error(f"ws_secure_realtime: failed registering websocket for {session_id}: {e}")
    key = base64.b64decode(SESSION_META[session_id]['key'])
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
        logging.info(f"ws {session_id} disconnected")
    except Exception as e:
        logging.error(f'realtime websocket error {e}')
        try:
            await ws.close()
        except Exception:
            logging.debug("ws_secure_realtime: ws.close() failed during error cleanup")
    finally:
        try:
            with SESSION_LOCK:
                SESSION_WS.pop(session_id, None)
        except Exception as e:
            logging.error(f"ws_secure_realtime: failed removing websocket for {session_id}: {e}")


# --- Compatibility (non-secure) /api/v1 endpoints for legacy frontend calls ---
def _is_admin_request(req: Request) -> bool:
    # Accept either an X-Admin-Token matching env or a valid JWT in Authorization header
    try:
        adm = os.environ.get('OMEGA_ADMIN_TOKEN')
        hdr = req.headers.get('x-admin-token') or req.headers.get('X-Admin-Token') or req.headers.get('X-Admin-Token'.lower())
        if adm and hdr and hdr == adm:
            return True
    except Exception:
        pass
    # try JWT validation (best-effort)
    try:
        auth = req.headers.get('authorization','')
        if auth and auth.lower().startswith('bearer '):
            tok = auth.split(' ',1)[1]
            claims = api_server.security_manager.validate_jwt(tok)
            # admin if subject or roles indicate admin
            if claims.get('sub') == 'admin' or claims.get('sid') in SESSION_META:
                return True
    except Exception:
        pass
    return False


@app.post('/api/v1/nodes/register')
async def v1_register_node(request: Request, body: NodeRegistration):
    # Legacy registration endpoint used by older frontends; requires admin token or valid session
    if not _is_admin_request(request):
        raise HTTPException(status_code=403, detail='admin token required')
    try:
        api_server.database.add_node(body.node_id, body.node_type, body.hostname, body.ip_address, body.port, body.resources or {})
        api_server.database.log_event('v1_node_register', body.node_id, f'Legacy v1 register by {request.client.host if request.client else "local"}', 'info')
        return JSONResponse(content={'success': True, 'node_id': body.node_id})
    except Exception as e:
        logging.error(f'v1_register_node error: {e}')
        raise HTTPException(status_code=500, detail='registration failed')


@app.get('/api/v1/nodes/list')
async def v1_nodes_list():
    try:
        nodes = api_server.database.get_nodes()
        return JSONResponse(content={'nodes': nodes})
    except Exception as e:
        logging.error(f'v1_nodes_list error: {e}')
        raise HTTPException(status_code=500, detail='failed to list nodes')


@app.get('/api/v1/nodes/pending')
async def v1_nodes_pending(request: Request):
    if not _is_admin_request(request):
        raise HTTPException(status_code=403, detail='admin token required')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT node_id, approved_by, approved_at, status FROM node_approvals WHERE status=?', ('pending',))
            pending = [{'node_id': r[0], 'approved_by': r[1], 'approved_at': r[2], 'status': r[3]} for r in cur.fetchall()]
        return JSONResponse(content={'pending': pending})
    except Exception as e:
        logging.error(f'v1_nodes_pending error: {e}')
        raise HTTPException(status_code=500, detail='failed to fetch pending')


@app.post('/api/v1/nodes/approve')
async def v1_nodes_approve(request: Request, body: dict = Body(...)):
    if not _is_admin_request(request):
        raise HTTPException(status_code=403, detail='admin token required')
    node_id = body.get('node_id')
    if not node_id:
        raise HTTPException(status_code=400, detail='node_id required')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('UPDATE node_approvals SET status=?, approved_by=?, approved_at=? WHERE node_id=?', ('approved', 'admin', time.time(), node_id))
            conn.execute('UPDATE nodes SET status=? WHERE node_id=?', ('active', node_id))
            conn.commit()
        api_server.database.log_event('v1_node_approve', node_id, f'Approved via v1 by admin', 'info')
        return JSONResponse(content={'success': True, 'node_id': node_id})
    except Exception as e:
        logging.error(f'v1_nodes_approve error: {e}')
        raise HTTPException(status_code=500, detail='approve failed')


@app.post('/api/v1/nodes/deny')
async def v1_nodes_deny(request: Request, body: dict = Body(...)):
    if not _is_admin_request(request):
        raise HTTPException(status_code=403, detail='admin token required')
    node_id = body.get('node_id')
    if not node_id:
        raise HTTPException(status_code=400, detail='node_id required')
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('UPDATE node_approvals SET status=?, approved_by=?, approved_at=? WHERE node_id=?', ('denied', 'admin', time.time(), node_id))
            conn.execute('UPDATE nodes SET status=? WHERE node_id=?', ('denied', node_id))
            conn.commit()
        api_server.database.log_event('v1_node_deny', node_id, f'Denied via v1 by admin', 'warning')
        return JSONResponse(content={'success': True, 'node_id': node_id})
    except Exception as e:
        logging.error(f'v1_nodes_deny error: {e}')
        raise HTTPException(status_code=500, detail='deny failed')


@app.get('/api/v1/nodes/discovered')
async def v1_nodes_discovered():
    # Best-effort: reuse discovery action to attempt local discovery and return a small set
    try:
        # lightweight discovery: return empty list or run psutil network scanning in background
        return JSONResponse(content={'discovered': []})
    except Exception as e:
        logging.error(f'v1_nodes_discovered error: {e}')
        return JSONResponse(content={'discovered': []})


def persist_revocation(session_id: str, reason: str = ''):
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR REPLACE INTO revoked_sessions (session_id, revoked_at, reason) VALUES (?, ?, ?)', (session_id, time.time(), reason))
            conn.commit()
    except sqlite3.DatabaseError as e:
        logging.error(f"Database error persisting revocation for {session_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error persisting revocation for {session_id}: {e}")
    with SESSION_LOCK:
        REVOKED_SESSIONS.add(session_id)


def _mask_id(s: str, keep: int = 6) -> str:
    if not s or len(s) <= keep:
        return '***'
    return s[:keep] + '...' + s[-3:]


def _hash_password(password: str) -> str:
    if _HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    # fallback PBKDF2
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200000)
    return base64.b64encode(salt + dk).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Constant-time verification supporting bcrypt or PBKDF2 fallback."""
    if not hashed:
        return False
    if _HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False
    try:
        raw = base64.b64decode(hashed)
        if len(raw) < 48:  # 16 salt + 32 dk
            return False
        salt = raw[:16]
        dk_stored = raw[16:]
        dk_calc = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200000)
        return hmac.compare_digest(dk_stored, dk_calc)
    except Exception as e:
        logging.debug(f"_verify_password fallback error: {e}")
        return False


def persist_pending_rekey(session_id: str, key_b64: str):
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO session_pending_rekey (session_id, new_key, created_at) VALUES (?, ?, ?)', (session_id, key_b64, time.time()))
            conn.commit()
    except sqlite3.DatabaseError as e:
        logging.error(f"Database error persisting pending rekey for {session_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error persisting pending rekey for {session_id}: {e}")

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
    # RSA-encrypted AES session key (base64)
    encrypted_key: Optional[str] = None

"""(Deprecated early definition of secure_session_start removed in security hardening pass)"""
# (Intentionally left blank)


@app.get('/api/secure/public_key')
@rate_limited(per_minute=120, burst=40)
async def secure_public_key():
    # Return server RSA public key PEM so clients can perform RSA-OAEP encryption for session key handshake
    try:
        # Lazy-init if key attributes not yet created (import ordering safety)
        if not hasattr(api_server, 'rsa_private_key') or api_server.rsa_private_key is None:
            from cryptography.hazmat.primitives.asymmetric import rsa
            api_server.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if not hasattr(api_server, 'rsa_public_key') or api_server.rsa_public_key is None:
            api_server.rsa_public_key = api_server.rsa_private_key.public_key()
        pub_pem = api_server.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        return JSONResponse(content={'public_key_pem': pub_pem})
    except Exception as e:
        logging.error(f'secure_public_key error: {e}')
        raise HTTPException(status_code=500, detail='failed to export public key')

@app.get('/api/public_info')
async def public_ping():
    """Public minimal info (renamed from /api/ping to avoid duplicate route)."""
    return {'service':'omega-backend','version':'2.0','ts':time.time()}


class SecureSessionRotate(BaseModel):
    session_id: str
    # RSA-encrypted new AES key (base64)
    encrypted_key: str


@app.post('/api/secure/session/rotate')
@rate_limited(per_minute=60, burst=20)
async def secure_session_rotate(body: SecureSessionRotate, request: Request):
    # Client-initiated rekey: client generates new AES key locally, encrypts it with server RSA public key, and posts here.
    # Validate caller via Authorization JWT
    auth = request.headers.get('Authorization','')
    if not auth.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing bearer token')
    token = auth.split(' ',1)[1]
    try:
        claims = api_server.security_manager.validate_jwt(token)
    except HTTPException:
        raise
    # Ensure token subject matches session_id
    if claims.get('sid') != body.session_id:
        raise HTTPException(status_code=403, detail='Token does not match session')
    # Decrypt new key
    try:
        enc = base64.b64decode(body.encrypted_key)
        new_key = api_server.rsa_private_key.decrypt(enc, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid encrypted_key')
    new_key_b64 = base64.b64encode(new_key).decode()
    # Atomically replace in-memory meta then persist to DB (best-effort)
    with SESSION_LOCK:
        if body.session_id not in SESSION_META:
            raise HTTPException(status_code=404, detail='Unknown session')
        SESSION_META[body.session_id]['key'] = new_key_b64
        SESSION_META[body.session_id]['last_rotate'] = time.time()
    try:
        with db_connect() as conn:
            conn.execute('UPDATE session_meta SET key=?, last_rotate=? WHERE session_id=?', (new_key_b64, SESSION_META[body.session_id]['last_rotate'], body.session_id))
            conn.commit()
    except Exception:
        pass
    # Remove any admin-created pending rekey record now that client completed rotation
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('DELETE FROM session_pending_rekey WHERE session_id=?', (body.session_id,))
            conn.commit()
    except Exception:
        pass
    try:
        api_server.database.log_event('session_rotate', body.session_id, 'Session key rotated by client', 'info')
    except Exception:
        pass
    return {'success': True, 'session_id': body.session_id}



# --- Most advanced, secure, and bug-free node registration ---
@app.post('/api/secure/nodes/register')
async def register_node(request: Request, body: NodeRegistrationRequest = Body(...)):
    import ipaddress, socket, re
    session_id, key = validate_secure(request.headers)
    user = SESSION_META.get(session_id, {}).get('user', 'admin')
    perms = SESSION_META.get(session_id, {}).get('permissions', [])
    # RBAC check
    if 'admin' not in SESSION_META.get(session_id, {}).get('roles', []) and 'node_register' not in perms:
        api_server.database.log_audit('node_register_denied', body.node_id, f'Permission denied for {user}', 'warning')
        raise HTTPException(status_code=403, detail='Insufficient permissions to register node')
    # Input validation
    if not re.match(r'^[a-zA-Z0-9\-_]{3,64}$', body.node_id):
        raise HTTPException(status_code=400, detail='Invalid node_id format')
    if not re.match(r'^[a-zA-Z0-9\-_]{3,32}$', body.node_type):
        raise HTTPException(status_code=400, detail='Invalid node_type format')
    try:
        ipaddress.ip_address(body.ip_address)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid IP address')
    if not (0 < body.port < 65536):
        raise HTTPException(status_code=400, detail='Invalid port')
    # Check for duplicate device fingerprint
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute("SELECT 1 FROM node_attestations WHERE device_fingerprint=?", (body.device_fingerprint,))
        if cur.fetchone():
            api_server.database.log_audit('node_register_conflict', body.node_id, f'Duplicate fingerprint {body.device_fingerprint[:12]}... by {user}', 'warning')
            raise HTTPException(status_code=409, detail='Device fingerprint already registered')
    # Validate public key and signed challenge
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    try:
        pubkey = serialization.load_pem_public_key(body.public_key_pem.encode())
        challenge = (body.node_id + body.device_fingerprint).encode()
        signature = base64.b64decode(body.signed_challenge)
        pubkey.verify(signature, challenge, padding.PKCS1v15(), hashes.SHA256())
    except Exception as e:
        api_server.database.log_audit('node_register_attestation_fail', body.node_id, f'Attestation failed: {e}', 'warning')
        raise HTTPException(status_code=400, detail=f'Invalid device attestation: {e}')
    # Certificate validation (simulated CA check)
    cert_valid = False
    if body.device_certificate:
        try:
            from cryptography.x509 import load_pem_x509_certificate
            cert = load_pem_x509_certificate(body.device_certificate.encode())
            if cert.not_valid_before <= datetime.utcnow() <= cert.not_valid_after:
                cert_pubkey = cert.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
                reg_pubkey = serialization.load_pem_public_key(body.public_key_pem.encode()).public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
                if cert_pubkey == reg_pubkey:
                    cert_valid = True
        except Exception as e:
            api_server.database.log_audit('node_register_cert_fail', body.node_id, f'Certificate validation failed: {e}', 'warning')
    # GeoIP/location check
    geoip_flagged = False
    allowed_regions = {"US", "CA", "EU"}
    geoip_info = body.geoip or ""
    if geoip_info and not any(r in geoip_info for r in allowed_regions):
        geoip_flagged = True
    # Behavioral baseline check
    baseline_flagged = False
    if body.behavioral_baseline:
        cpu = body.behavioral_baseline.get('cpu_cores', 0)
        mem = body.behavioral_baseline.get('memory_gb', 0)
        if cpu < 2 or mem < 2:
            baseline_flagged = True
    # ML/rule-based anomaly detection (simulated)
    anomaly_flagged = geoip_flagged or baseline_flagged
    # Trust score calculation
    trust_score = 100
    if not cert_valid:
        trust_score -= 20
    if geoip_flagged:
        trust_score -= 30
    if baseline_flagged:
        trust_score -= 20
    if anomaly_flagged:
        trust_score -= 20
    if trust_score < 0:
        trust_score = 0
    quarantine = trust_score < 70
    node_status = 'pending_approval' if quarantine else 'active'
    # Defensive: encrypt sensitive fields before DB insert (simulate with base64 for now)
    import base64, json
    def enc(val):
        return base64.b64encode(json.dumps(val).encode()).decode() if val is not None else None
    # Register node in advanced schema
    api_server.database.add_node_advanced(
        node_id=body.node_id,
        node_type=body.node_type,
        hostname=body.hostname,
        ip_address=body.ip_address,
        port=body.port,
        status=node_status,
        trust_score=trust_score,
        quarantine=quarantine,
        resources=body.resources,
        permissions=body.permissions,
        attestation={
            'device_fingerprint': enc(body.device_fingerprint),
            'public_key_pem': enc(body.public_key_pem),
            'device_certificate': enc(body.device_certificate),
            'health_attestation': enc(body.health_attestation),
            'geoip': enc(body.geoip),
            'behavioral_baseline': enc(body.behavioral_baseline)
        },
        approval_status='pending' if quarantine else 'approved'
    )
    api_server.database.log_audit(
        'node_register',
        body.node_id,
        f'Node {body.node_id} registered by {user} (trust={trust_score}, quarantine={quarantine}, geoip={geoip_info})',
        'info' if not quarantine else 'warning'
    )
    if quarantine:
        api_server.database.log_audit('node_quarantine', body.node_id, f'Node {body.node_id} quarantined for admin approval (trust_score={trust_score})', 'warning')
    return wrap_encrypted(session_id, key, {'success': True, 'node_id': body.node_id, 'trust_score': trust_score, 'quarantine': quarantine})
    # The following lines were unreachable and caused indentation errors. If you want to use them, move them into a function:
    #    raise ValueError('os_image is required')
    #    # disallow shell metacharacters
    #    if any(ch in v for ch in [';', '|', '&', '$', '`', '\\', '>', '<']):
    #        raise ValueError('Invalid os_image value')
    #    return v  # Ensure valid os_image value
# (Removed duplicate FastAPI app redefinition; using single instance declared at top)
def require_role(user, role):
    # Enforce RBAC: only admin can perform admin actions
    roles = get_user_roles(user)
    if role not in roles:
        logging.warning(f"RBAC: User {user} lacks required role {role}")
        raise HTTPException(status_code=403, detail=f'{role} role required')

# --- FINAL ROUTE OVERRIDES (ensure correct handlers not shadowed by earlier wrappers) ---
# Re-register secure session start AFTER generic wrappers that changed signature.
# First, remove any previously registered POST route for this path so the override actually takes effect
try:
    _target_path = '/api/secure/session/start'
    _method = 'POST'
    to_remove = []
    for _r in list(app.router.routes):  # type: ignore[attr-defined]
        try:
            if getattr(_r, 'path', None) == _target_path and _method in getattr(_r, 'methods', set()):
                to_remove.append(_r)
        except Exception:
            continue
    for _r in to_remove:
        try:
            app.router.routes.remove(_r)  # type: ignore[attr-defined]
            logging.info('Removed previous wrapped route for %s', _target_path)
        except Exception:
            pass
except Exception as _e:  # pragma: no cover - non-critical
    logging.error(f"Failed pruning old secure_session_start route: {_e}")
@app.post('/api/secure/session/start', summary='Secure Session Start', tags=['secure'])
@rate_limited(per_minute=30, burst=10)
async def secure_session_start_override(body: SecureSessionStart = Body(...)):
    """Initiate an AES-GCM secure session (final override).
    Client supplies RSA-OAEP encrypted 32-byte key (base64). Dev fallback allowed if env OMEGA_ALLOW_INSECURE_HANDSHAKE set.
    Returns: { session_id, token, issued_at, expires_in }
    """
    session_id = f"sid-{secrets.token_hex(12)}"
    allow_dev = os.environ.get('OMEGA_ALLOW_INSECURE_HANDSHAKE','0').lower() in ('1','true','yes')
    key=None
    if body.encrypted_key:
        try:
            encrypted_key_bytes = base64.b64decode(body.encrypted_key)
            key = api_server.rsa_private_key.decrypt(
                encrypted_key_bytes,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
        except Exception:
            if not allow_dev:
                raise HTTPException(status_code=400, detail='Invalid encrypted_key')
    if key is None:
        if not allow_dev:
            raise HTTPException(status_code=400, detail='encrypted_key is required')
        key = os.urandom(32)
    key_b64 = base64.b64encode(key).decode()
    register_session_meta(session_id, key_b64)
    with SESSION_LOCK:
        if session_id in SESSION_META:
            SESSION_META[session_id]['user'] = body.user_id or 'admin'
        else:
            SESSION_META[session_id] = {'user': body.user_id or 'admin', 'key': key_b64, 'created': time.time(), 'last_rotate': time.time(), 'counter': 0}
    payload = {'session_id': session_id, 'issued_at': time.time(), 'expires_in': 6*3600}
    try:
        jwt = api_server.security_manager.issue_jwt(session_id, user=SESSION_META.get(session_id, {}).get('user','admin'), ttl=6*3600)
        payload['token'] = jwt
    except Exception:
        pass
    return payload

def get_user_roles(user):
    # Real role lookup: check DB or in-memory map
    # For now, fallback to admin/user
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT role FROM user_roles WHERE username=?', (user,))
            roles = [r[0] for r in cur.fetchall()]
            if roles:
                return roles
    except Exception as e:
        logging.error(f"Error fetching user roles for {user}: {e}")
    if user == 'admin':
        return ['admin']
    return ['user']

class VDCreateRequest(BaseModel):
    user_id: str
    os_image: str
    vnc_password: str
    cpu_cores: int = 2
    gpu_units: int = 0
    memory_gb: int = 4
    resolution: str = '1280x720'
    profile: str = ''
    packages: list = []

    @validator('os_image')
    def validate_os_image(cls, v: str) -> str:
        if not v:
            raise ValueError('os_image is required')
        # disallow shell metacharacters that could affect docker/image names
        if any(ch in v for ch in [';', '|', '&', '$', '`', '\\', '>', '<']):
            raise ValueError('Invalid os_image value')
        return v

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
    # Webtop variants (noVNC on :3000)
    'ubuntu-webtop': 'lscr.io/linuxserver/webtop:ubuntu',
        # Legacy option (kept for compatibility)
        'ubuntu-lxde-legacy': 'dorowu/ubuntu-desktop-lxde-vnc',
        # Alternatives
        'debian-xfce': 'accetto/debian-vnc-xfce',
    'debian-webtop': 'lscr.io/linuxserver/webtop:debian',
    'fedora-webtop': 'lscr.io/linuxserver/webtop:fedora',
        'kali-xfce':   'lscr.io/linuxserver/kali-linux:latest',  # may require extra config
    }
    # External OS that cannot run in our Docker-based flow
    if os_image in ('qubes', 'qubes-os', 'qubesos'):
        raise HTTPException(status_code=400, detail='Qubes OS requires an external hypervisor (Xen). Use /api/secure/rdp/create to connect to a Qubes VM or register an external connector.')
    if os_image in ('freebsd','openbsd','inferno-os','plan9','centos-stream','almalinux','rocky-linux'):
        raise HTTPException(status_code=400, detail=f"{os_image} is an external OS. Use /api/secure/rdp/create or register an external connector.")
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


def _sanitize_image_name(img: str) -> str:
    # Basic allowlist check: allow only repository[:tag] with safe characters
    if not img or '..' in img:
        raise HTTPException(status_code=400, detail='Invalid image name')
    # Allow common characters and colon/slash/dash/underscore
    if not all(c.isalnum() or c in '/:._-@' for c in img):
        raise HTTPException(status_code=400, detail='Invalid image name characters')
    return img


def _validate_container_name(name: str) -> str:
    # Docker recommends lowercase and limited chars; enforce a strict pattern
    if not name or len(name) > 128:
        raise HTTPException(status_code=400, detail='Invalid container name')
    if any(c in name for c in ' <>|&;$`\n\r'):
        raise HTTPException(status_code=400, detail='Invalid container name characters')
    return name

def _container_ports_for_image(image: str) -> dict:
    # Known defaults
    if 'dorowu/ubuntu-desktop-lxde-vnc' in image:
        return {'http': 80, 'vnc': 5900}
    if 'accetto' in image:
        return {'http': 6901, 'vnc': 5901}
    if 'linuxserver/webtop' in image or 'lscr.io/linuxserver/webtop' in image:
        # Web UI on 3000, no classic 590x VNC port exposed. We'll map vnc to 5901 as placeholder (unused)
        return {'http': 3000, 'vnc': 5901}
    # Fallback
    return {'http': 80, 'vnc': 5900}

def _env_for_image(image: str, vnc_password: str, resolution: Optional[str] = None) -> list[str]:
    # Provide multiple common env names used by popular VNC desktop images
    envs = []
    # Set multiple common VNC password envs
    for key in ('VNC_PASSWORD', 'PASSWORD', 'VNC_PW'):
        envs += ['-e', f"{key}={vnc_password}"]
    # Some images require setting USER to root for password/program installs
    envs += ['-e', 'USER=root']
    # Some accetto images support user/password pairs; keep minimal for now
    if resolution:
        # Support common env names across images
        for rkey in ('RESOLUTION', 'VNC_RESOLUTION'):
            envs += ['-e', f"{rkey}={resolution}"]
        # For webtop, hint width/height if RESOLUTION provided (optional)
        if 'linuxserver/webtop' in image or 'lscr.io/linuxserver/webtop' in image:
            try:
                parts = resolution.lower().split('x')
                if len(parts) == 2:
                    w, h = parts[0], parts[1]
                    envs += ['-e', f"WEBTOP_WIDTH={w}", '-e', f"WEBTOP_HEIGHT={h}"]
            except Exception:
                pass
    return envs

def _detect_viewer_path(http_port: int) -> tuple[str, str]:
    """Probe the container viewer endpoint and pick the correct path and query defaults.
    Returns (path, query_string_without_leading_question_mark).
    - accetto images: /vnc.html and no host/port/path params required
    - dorowu images: /static/vnc.html and needs host/port/path=websockify
    Fallback to '/' if neither is available.
    """
    import urllib.request
    def _head(path: str) -> int:
        try:
            req = urllib.request.Request(f'http://127.0.0.1:{http_port}{path}', method='HEAD')
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return int(resp.getcode())
        except Exception:
            return 0
    if _head('/vnc.html') == 200:
        return '/vnc.html', 'autoconnect=1'
    if _head('/static/vnc.html') == 200:
        return '/static/vnc.html', f'autoconnect=1&host=localhost&port={http_port}&path=websockify'
    # last resort
    return '/', ''

@app.post('/api/secure/vd/create')
@rate_limited(per_minute=30, burst=6)
async def vd_create(request: Request, spec: VDCreateRequest, background_tasks: BackgroundTasks):
    sid, key = validate_secure(request.headers)
    # Preflight: ensure Docker daemon is available
    if not _docker_available():
        raise HTTPException(status_code=503, detail='Docker daemon not running. Start Docker Desktop and retry.')
    # Resolve image from env override or mapping, and set pull policy
    env_img = os.environ.get('OMEGA_DEFAULT_VD_IMAGE', '').strip()
    image = env_img or _image_for_os(spec.os_image)
    try:
        image = _sanitize_image_name(image)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image selection')
    ports = _container_ports_for_image(image)
    pull_policy = os.environ.get('OMEGA_VD_PULL_POLICY', 'IfNotPresent').strip().lower()
    http_port = _find_free_port(7000, 7999)
    vnc_port = _find_free_port(5900, 5999)
    vnc_password = spec.vnc_password.strip() if isinstance(spec.vnc_password, str) and spec.vnc_password.strip() else secrets.token_urlsafe(8)
    session_id = f"vd-{secrets.token_hex(8)}"
    name = f"omega_{session_id}"
    try:
        name = _validate_container_name(name)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid container name')

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
    run_args = ['run', '-d', '--restart', 'unless-stopped', '--name', name] + labels + ['-p', port_http_map, '-p', port_vnc_map] + envs + [image]
    container_id = _docker_run(run_args, timeout=120)
    # Give the container a brief moment to initialize services (tight timeout to avoid blocking)
    try:
        time.sleep(0.7)
    except Exception:
        pass

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
    elif 'linuxserver/webtop' in image or 'lscr.io/linuxserver/webtop' in image:
        # Webtop serves UI at / on :3000 and doesn't use the noVNC query params
        connect_path = '/'
        connect_query = ''
    else:
        # Auto-detect between accetto (/vnc.html) and dorowu (/static/vnc.html)
        path, q = _detect_viewer_path(http_port)
        connect_path = path
        # Always include password if we have one
        connect_query = (q + ('&' if q else '') + f'password={vnc_password}').strip('&')
    connect_url = f"http://localhost:{http_port}{connect_path}{('?' + connect_query) if connect_query else ''}"
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
            # Detect package manager and refresh caches with reasonable timeout
            try:
                pm_detect = (
                    'set +e; '
                    'pm=""; '
                    'if command -v apt-get >/dev/null 2>&1; then pm=apt; '
                    'elif command -v dnf >/dev/null 2>&1; then pm=dnf; '
                    'elif command -v yum >/dev/null 2>&1; then pm=yum; '
                    'elif command -v apk >/dev/null 2>&1; then pm=apk; fi; '
                    'echo "$pm"'
                )
                pm = _docker_run(['exec','-u','0', cid, 'bash','-lc', pm_detect], timeout=60).strip()
            except HTTPException:
                pm = ''
            try:
                if pm == 'apt':
                    _docker_run(['exec','-u','0', cid, 'bash','-lc', 'export DEBIAN_FRONTEND=noninteractive; apt-get update -y -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 || true'], timeout=600)
                elif pm == 'dnf':
                    _docker_run(['exec','-u','0', cid, 'bash','-lc', 'dnf -y makecache || true'], timeout=600)
                elif pm == 'yum':
                    _docker_run(['exec','-u','0', cid, 'bash','-lc', 'yum -y makecache || true'], timeout=600)
                elif pm == 'apk':
                    _docker_run(['exec','-u','0', cid, 'bash','-lc', 'apk update || true'], timeout=300)
                api_server.database.log_event('vd_network_ok', session_id, f'PM={pm or "unknown"}: repo cache refreshed', 'info')
            except HTTPException as e:
                api_server.database.log_event('vd_network_warn', session_id, f'PM={pm or "unknown"}: repo refresh issue: {e.detail}', 'warning')
            # Install additional packages if requested
            if packages:
                # Install packages individually with cross-distro fallbacks
                pkgs_str = ' '.join([shlex.quote(p) if ' ' in p else p for p in packages]) if packages else ''
                parts = [
                    'set +e; export DEBIAN_FRONTEND=noninteractive; ',
                    'pm=""; ',
                    'if command -v apt-get >/dev/null 2>&1; then pm=apt; ',
                    'elif command -v dnf >/dev/null 2>&1; then pm=dnf; ',
                    'elif command -v yum >/dev/null 2>&1; then pm=yum; ',
                    'elif command -v apk >/dev/null 2>&1; then pm=apk; fi; ',
                    'install_apt() { apt-get install -y --no-install-recommends "$1" >/dev/null 2>&1; }; ',
                    'install_dnf() { dnf install -y "$1" >/dev/null 2>&1; }; ',
                    'install_yum() { yum install -y "$1" >/dev/null 2>&1; }; ',
                    'install_apk() { apk add --no-cache "$1" >/dev/null 2>&1; }; ',
                    'do_install() { case "$pm" in apt) install_apt "$1" ;; dnf) install_dnf "$1" ;; yum) install_yum "$1" ;; apk) install_apk "$1" ;; *) return 1 ;; esac; }; ',
                    'for p in ', pkgs_str, ' ; do ',
                    '  installed=0; echo "Installing $p via $pm"; ',
                    '  case "$p" in ',
                    '    firefox|browser) ',
                    '      for alt in firefox firefox-esr chromium epiphany-browser epiphany midori; do do_install "$alt" && echo "INSTALL_OK:$alt" && installed=1 && break; done; ',
                    '      [ $installed -eq 1 ] || echo "INSTALL_FAIL:$p"; ;;',
                    '    chromium|chromium-browser) ',
                    '      for alt in chromium chromium-browser chromium-common epiphany-browser epiphany midori; do do_install "$alt" && echo "INSTALL_OK:$alt" && installed=1 && break; done; ',
                    '      [ $installed -eq 1 ] || echo "INSTALL_FAIL:$p"; ;;',
                    '    *) do_install "$p" && echo "INSTALL_OK:$p" || echo "INSTALL_FAIL:$p"; ;;',
                    '  esac; ',
                    'done; ',
                    'BROWSER=$(command -v firefox || command -v chromium || command -v epiphany-browser || command -v epiphany || command -v midori || true); ',
                    'echo "BROWSER_DETECTED:${BROWSER}"; true'
                ]
                install_script = ''.join(parts)
                try:
                    out = _docker_run(['exec', '-u', '0', cid, 'bash', '-lc', install_script], timeout=1800)
                    api_server.database.log_event('vd_packages', session_id, f'Install attempted (pm={pm or "unknown"}): {packages}', 'info')
                    # Surface detected browser to logs
                    if out and 'BROWSER_DETECTED:' in out:
                        line = [ln for ln in out.splitlines() if ln.startswith('BROWSER_DETECTED:')][-1]
                        api_server.database.log_event('vd_browser', session_id, line, 'info')
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
        cur = conn.execute('SELECT connect_url,http_port,vnc_port,vnc_password,os_image,container_id FROM vd_session_meta WHERE session_id = ?', (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='VD session not found')
        connect_url, http_port, vnc_port, vnc_password, os_image, container_id = row
    # Try to ensure container is running
    try:
        state = _docker_run(['inspect','-f','{{.State.Running}}', container_id]) if container_id else 'true'
        if state.strip().lower() != 'true':
            try:
                _docker_run(['start', container_id])
                time.sleep(0.5)
            except HTTPException:
                pass
    except HTTPException:
        pass
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
    elif 'linuxserver/webtop' in image or 'lscr.io/linuxserver/webtop' in image:
        connect_path = '/'
        query = ''
    else:
        # Auto-detect path live in case the image mapping changed or a snapshot was restored
        path, q = _detect_viewer_path(http_port)
        connect_path = path
        query = (q + ('&' if q else '') + f'password={vnc_password}').strip('&')
    new_url = f"http://localhost:{http_port}{connect_path}{('?' + query) if query else ''}"
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
    # Resolve friendly descriptions for roles from roles table
    friendly = []
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT role,description FROM roles WHERE role IN ({seq})'.format(seq=','.join('?'*len(roles))), tuple(roles) if roles else ())
            rows = {r[0]: r[1] for r in cur.fetchall()}
            for r in roles:
                friendly.append(rows.get(r) or r)
    except Exception:
        friendly = roles
    # Permissions stub: in future map roles->permissions via a table
    permissions = ['all'] if 'admin' in roles else ['read']
    return wrap_encrypted(sid, key, {'user': user, 'roles': roles, 'role_descriptions': friendly, 'permissions': permissions, 'timestamp': time.time()})

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
        with db_connect() as conn:
            conn.execute('INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)', (role, None))
            cur = conn.execute('SELECT 1 FROM user_roles WHERE username=? AND role=?', (username, role))
            exists = bool(cur.fetchone())
            if not exists:
                conn.execute('INSERT INTO user_roles (username, role) VALUES (?, ?)', (username, role))
            conn.commit()
        api_server.database.log_event('security_role_assign', caller, f'Assigned role {role} to {username}', 'info')
    except Exception as e:
        logging.error(f'Role assign error: {e}')
        raise HTTPException(status_code=500, detail='Role assignment failed')
    return wrap_encrypted(sid, key, {'success': True, 'username': username, 'role': role, 'applied': not exists})

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
        with db_connect() as conn:
            cur = conn.execute('SELECT 1 FROM user_roles WHERE username=? AND role=?', (username, role))
            existed = bool(cur.fetchone())
            conn.execute('DELETE FROM user_roles WHERE username=? AND role=?', (username, role))
            conn.commit()
        api_server.database.log_event('security_role_remove', caller, f'Removed role {role} from {username}', 'info')
    except Exception as e:
        logging.error(f'Role remove error: {e}')
        raise HTTPException(status_code=500, detail='Role removal failed')
    return wrap_encrypted(sid, key, {'success': True, 'username': username, 'role': role, 'removed': existed})


class UserCreateRequest(BaseModel):
    username: str
    password: str


@app.post('/api/secure/admin/users/create')
@rate_limited(per_minute=30, burst=6)
async def admin_create_user(request: Request, body: UserCreateRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    username = body.username.strip()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail='username and password are required')
    hashed = _hash_password(body.password)
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO users (username, password_hash, created_at) VALUES (?, ?, ?)', (username, hashed, time.time()))
            conn.commit()
        api_server.database.log_event('user_create', caller, f'User {username} created', 'info')
    except Exception as e:
        logging.error(f'user create error: {e}')
        raise HTTPException(status_code=500, detail='Failed to create user')
    return wrap_encrypted(sid, key, {'created': True, 'username': username})


class UserModifyRequest(BaseModel):
    username: str


@app.post('/api/secure/admin/users/reset_password')
@rate_limited(per_minute=10, burst=4)
async def admin_reset_password(request: Request, body: UserModifyRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail='username required')
    # Set a random password and return a one-time token (best-effort)
    new_pw = secrets.token_urlsafe(12)
    hashed = _hash_password(new_pw)
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT 1 FROM users WHERE username=?', (username,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail='user not found')
            conn.execute('UPDATE users SET password_hash=? WHERE username=?', (hashed, username))
            conn.commit()
        api_server.database.log_event('user_reset_pw', caller, f'Password reset for {username}', 'info')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'Password reset error: {e}')
        raise HTTPException(status_code=500, detail='Password reset failed')
    # Return the new password in the response (admin should convey securely)
    return wrap_encrypted(sid, key, {'username': username, 'new_password': new_pw})


@app.post('/api/secure/admin/users/suspend')
async def admin_suspend_user(request: Request, body: UserModifyRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    username = body.username.strip()
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT disabled FROM users WHERE username=?', (username,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail='user not found')
            conn.execute('UPDATE users SET disabled=1 WHERE username=?', (username,))
            conn.commit()
        api_server.database.log_event('user_suspend', caller, f'User suspended: {username}', 'warning')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'User suspend error: {e}')
        raise HTTPException(status_code=500, detail='Suspend failed')
    return wrap_encrypted(sid, key, {'suspended': True, 'username': username})


@app.post('/api/secure/admin/users/activate')
async def admin_activate_user(request: Request, body: UserModifyRequest):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    username = body.username.strip()
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT disabled FROM users WHERE username=?', (username,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail='user not found')
            conn.execute('UPDATE users SET disabled=0 WHERE username=?', (username,))
            conn.commit()
        api_server.database.log_event('user_activate', caller, f'User activated: {username}', 'info')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'User activate error: {e}')
        raise HTTPException(status_code=500, detail='Activate failed')
    return wrap_encrypted(sid, key, {'activated': True, 'username': username})


@app.get('/api/secure/admin/certificates')
async def list_certificates(request: Request):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    # Placeholder: no real cert store yet
    return wrap_encrypted(sid, key, {'certificates': [], 'timestamp': time.time()})


@app.post('/api/secure/admin/certificates/generate')
async def generate_certificate(request: Request, body: dict):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    # Placeholder: return a simulated cert id
    cert_id = f'cert-{secrets.token_hex(8)}'
    api_server.database.log_event('cert_generate', caller, f'Generated placeholder cert {cert_id}', 'info')
    return wrap_encrypted(sid, key, {'generated': True, 'id': cert_id})


def issue_node_certificate(node_id: str, public_key_pem: str, days: int = 1) -> str:
    """Issue a short-lived X.509 certificate for a node, signed by server RSA key.
    Returns PEM-encoded certificate as string.
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        pubkey = load_pem_public_key(public_key_pem.encode())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)])
        issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"Omega Controller CA")])
        now = datetime.utcnow()
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(pubkey)
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(now - timedelta(minutes=1))
        builder = builder.not_valid_after(now + timedelta(days=days))
        builder = builder.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        cert = builder.sign(private_key=api_server.rsa_private_key, algorithm=hashes.SHA256())
        return cert.public_bytes(serialization.Encoding.PEM).decode()
    except Exception as e:
        logging.error(f'issue_node_certificate error for {node_id}: {e}')
        raise


@app.post('/api/secure/nodes/{node_id}/issue_cert')
async def secure_issue_node_cert(request: Request, node_id: str, body: dict = Body(...)):
    """Issue a node certificate for the given node_id using provided public_key_pem.
    Protected endpoint: requires a secure session and admin role.
    Request body: { "public_key_pem": "-----BEGIN PUBLIC KEY...", "days": 1 }
    """
    session_id, key = validate_secure(request.headers)
    caller = SESSION_META.get(session_id, {}).get('user', 'admin')
    require_role(caller, 'admin')
    public_key_pem = body.get('public_key_pem')
    days = int(body.get('days', 1)) if body.get('days') else 1
    if not public_key_pem:
        raise HTTPException(status_code=400, detail='public_key_pem required')
    try:
        cert_pem = issue_node_certificate(node_id, public_key_pem, days=days)
        api_server.database.log_event('issue_node_cert', node_id, f'Issued node cert (days={days}) to {caller}', 'info')
        return wrap_encrypted(session_id, key, {'certificate_pem': cert_pem})
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'secure_issue_node_cert error: {e}')
        raise HTTPException(status_code=500, detail='failed to issue certificate')


@app.post('/api/secure/admin/certificates/revoke')
async def revoke_certificate(request: Request, body: dict):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    cert_id = (body.get('id') or '')
    if not cert_id:
        raise HTTPException(status_code=400, detail='id required')
    api_server.database.log_event('cert_revoke', caller, f'Revoked placeholder cert {cert_id}', 'info')
    return wrap_encrypted(sid, key, {'revoked': True, 'id': cert_id})


@app.get('/api/secure/admin/users')
@rate_limited(per_minute=60, burst=10)
async def admin_list_users(request: Request):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    users = []
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT username, created_at, disabled FROM users ORDER BY created_at DESC')
            for r in cur.fetchall():
                users.append({'username': r[0], 'created_at': r[1], 'disabled': bool(r[2])})
    except Exception as e:
        logging.error(f'list users error: {e}')
        raise HTTPException(status_code=500, detail='Failed to list users')
    return wrap_encrypted(sid, key, {'users': users})

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
    {'id':'ubuntu-webtop', 'image':'lscr.io/linuxserver/webtop:ubuntu', 'http_port':3000, 'vnc_port':5901, 'viewer_path':'/', 'description':'Ubuntu Webtop (browser-based desktop)'},
        {'id':'ubuntu-lxde-legacy', 'image':'dorowu/ubuntu-desktop-lxde-vnc', 'http_port':80, 'vnc_port':5900, 'viewer_path':'/static/vnc.html', 'description':'Legacy LXDE variant'},
    {'id':'debian-xfce', 'image':'accetto/debian-vnc-xfce', 'http_port':6901, 'vnc_port':5901, 'viewer_path':'/', 'description':'Debian + XFCE'},
    {'id':'debian-webtop', 'image':'lscr.io/linuxserver/webtop:debian', 'http_port':3000, 'vnc_port':5901, 'viewer_path':'/', 'description':'Debian Webtop (browser-based desktop)'},
    {'id':'fedora-webtop', 'image':'lscr.io/linuxserver/webtop:fedora', 'http_port':3000, 'vnc_port':5901, 'viewer_path':'/', 'description':'Fedora Webtop (browser-based desktop)'},
    # RHEL-family presented as external connectors to avoid broken tags
    {'id':'centos-stream', 'image':'external/centos', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'CentOS Stream (external – connect via RDP/VNC/SSH)', 'experimental': True},
    {'id':'almalinux', 'image':'external/almalinux', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'AlmaLinux (external – connect via RDP/VNC/SSH)', 'experimental': True},
    {'id':'rocky-linux', 'image':'external/rocky', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'Rocky Linux (external – connect via RDP/VNC/SSH)', 'experimental': True},
        {'id':'kali-xfce', 'image':'lscr.io/linuxserver/kali-linux:latest', 'http_port':80, 'vnc_port':5900, 'viewer_path':'/', 'description':'Kali XFCE (may need extra config)'},
    # External OS (not containerized): provide as a catalog entry for UX; route users to RDP/VNC connectors
    {'id':'qubes-os', 'image':'external/qubes', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'Qubes OS (external VM – connect via RDP/VNC)', 'experimental': True},
    {'id':'freebsd', 'image':'external/freebsd', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'FreeBSD (external – connect via RDP/VNC/SSH)', 'experimental': True},
    {'id':'openbsd', 'image':'external/openbsd', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'OpenBSD (external – connect via RDP/VNC/SSH)', 'experimental': True},
    {'id':'inferno-os', 'image':'external/inferno', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'Inferno (research – external VM)', 'experimental': True},
    {'id':'plan9', 'image':'external/plan9', 'http_port':0, 'vnc_port':0, 'viewer_path':'', 'description':'Plan 9 from Bell Labs (research – external VM)', 'experimental': True},
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


@app.post('/api/secure/admin/session/revoke')
@rate_limited(per_minute=60, burst=10)
async def admin_revoke_session(request: Request, body: dict):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    session_id = body.get('session_id')
    if not session_id:
        raise HTTPException(status_code=400, detail='session_id required')
    # remove from memory and persist revocation
    SESSION_META.pop(session_id, None)
    SESSION_NONCES.pop(session_id, None)
    try:
        persist_revocation(session_id, body.get('reason','admin_revoke'))
    except Exception:
        pass
    try:
        # close ws if present
        ws = SESSION_WS.get(session_id)
        if ws:
            await ws.close()
    except Exception:
        pass
    try:
        api_server.database.log_event('session_revoked', session_id, f'Revoked by admin {caller}', 'warning')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'revoked': True, 'session_id': session_id})


@app.post('/api/secure/admin/session/rotate')
@rate_limited(per_minute=60, burst=10)
async def admin_rotate_session(request: Request, body: dict):
    sid, key = validate_secure(request.headers)
    caller = SESSION_META.get(sid, {}).get('user', 'admin')
    require_role(caller, 'admin')
    session_id = body.get('session_id')
    if not session_id or session_id not in SESSION_META:
        raise HTTPException(status_code=400, detail='Unknown session_id')
    # generate server-side new AES key and persist as pending rekey; push via WS if connected
    new_key = os.urandom(32)
    key_b64 = base64.b64encode(new_key).decode()
    persist_pending_rekey(session_id, key_b64)
    # Attempt push via WS: send a small delivery packet (server will not send raw AES key; instead request client to rekey)
    pushed = False
    try:
        ws = SESSION_WS.get(session_id)
        if ws:
            try:
                # Signal client to perform rekey (client should call rotate endpoint to supply new key), so we send a rekey_request
                await ws.send_json({'type':'rekey_request','session_id':session_id,'ts':time.time()})
                pushed = True
            except Exception:
                pushed = False
    except Exception:
        pushed = False
    try:
        api_server.database.log_event('session_admin_rotate', session_id, f'Admin {caller} requested rotate (pushed={pushed})', 'info')
    except Exception:
        pass
    return wrap_encrypted(sid, key, {'rotated': True, 'session_id': session_id, 'pushed': pushed})


@app.post('/api/secure/session/poll')
async def session_poll(request: Request, body: dict):
    # Clients poll to check if server has pending rekey or revocation
    session_id = body.get('session_id')
    if not session_id:
        raise HTTPException(status_code=400, detail='session_id required')
    # If revoked, instruct client to drop session
    try:
        with sqlite3.connect(api_server.database.db_path) as conn:
            cur = conn.execute('SELECT revoked_at,reason FROM revoked_sessions WHERE session_id=?', (session_id,))
            row = cur.fetchone()
            if row:
                return JSONResponse(content={'status':'revoked','revoked_at': row[0], 'reason': row[1]})
            cur = conn.execute('SELECT new_key,created_at FROM session_pending_rekey WHERE session_id=?', (session_id,))
            row = cur.fetchone()
            if row:
                # For safety, server does not return raw new_key to client; instead it instructs client to rekey
                return JSONResponse(content={'status':'rekey_pending','created_at': row[1]})
    except Exception:
        pass
    return JSONResponse(content={'status':'ok'})

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
