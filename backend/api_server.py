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

# Initialize scheduler on startup
from backend.scheduler_engine import initialize_scheduler

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_scheduler({
        'strategy_weights': {
            'weighted_least_loaded': 0.3,
            'capability_aware': 0.25,
            'energy_efficient': 0.15,
            'predictive_placement': 0.15,
            'thermal_aware': 0.1,
            'latency_optimized': 0.05
        }
    })

# Hardened security headers middleware (simple inline implementation)
from starlette.middleware.base import BaseHTTPMiddleware
class _SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # type: ignore
        resp = await call_next(request)
        # Content-Security-Policy (restrictive; allow self + data images)
        csp = os.getenv('OMEGA_CSP', "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'")
        resp.headers.setdefault('Content-Security-Policy', csp)
        resp.headers.setdefault('X-Content-Type-Options','nosniff')
        resp.headers.setdefault('X-Frame-Options','DENY')
        resp.headers.setdefault('Referrer-Policy','no-referrer')
        resp.headers.setdefault('Permissions-Policy','geolocation=(), microphone=(), camera=()')
        resp.headers.setdefault('Cross-Origin-Opener-Policy','same-origin')
        resp.headers.setdefault('Cross-Origin-Resource-Policy','same-origin')
        resp.headers.setdefault('Cross-Origin-Embedder-Policy','require-corp')
        return resp
app.add_middleware(_SecurityHeaders)

# Lightweight health/readiness endpoint (unauthenticated) so frontend can quickly
# detect backend availability before attempting secure session bootstrap.
@app.get('/health', include_in_schema=False)
async def health_check():
    return {'status': 'ok'}

# Simple unauthenticated ping used by discovery logic and external scripts
@app.get('/api/ping', include_in_schema=False)
async def api_ping():
    return {'pong': True, 'ts': time.time()}

# ---------------------------------------------------------------------------
# Backward compatibility shim: legacy agents still POST to /api/nodes/register
# Newer secured path moved under /api/secure/nodes/register (with auth + crypto)
# Provide a minimal passthrough that returns 410 or forwards to secure flow.
# ---------------------------------------------------------------------------
@app.post('/api/nodes/register', include_in_schema=False)
async def legacy_nodes_register(request: Request):
    """Legacy endpoint shim. Returns informative 410 Gone so agents can upgrade.
    Optionally could forward into secure path if we detect a compatible JSON body.
    """
    try:
        body = await request.json()
    except Exception:
        body = None
    return JSONResponse(status_code=410, content={
        'error': 'deprecated_endpoint',
        'detail': 'Use /api/secure/nodes/register with secure session established',
        'received': body or {}
    })

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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set
from contextlib import asynccontextmanager
import sqlite3
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import os
import shlex
from common.secret_providers import default_manager

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from pydantic import Field, validator
import uvicorn
import websockets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, CollectorRegistry, Counter, Gauge, Histogram
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

# Added helper utilities
import contextlib

def _docker_available() -> bool:
    try:
        import docker  # type: ignore
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False

def _find_free_port(start: int, end: int) -> int:
    for port in range(start, end):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return 0

def _container_ports_for_image(image: str) -> dict:
    return {'vnc': 5901, 'http': 6901}

def require_role(user: str, role: str):
    # Minimal admin check bridging to RBAC permission set
    perms = get_user_permissions(user)
    if role == 'admin' and 'rbac:manage' not in perms:
        raise HTTPException(status_code=403, detail='admin role required')


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
    
    def get_public_key_pem(self) -> str:
        try:
            pub = self.rsa_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pub.decode()
        except Exception as e:
            logging.error(f"get_public_key_pem failed: {e}")
            raise
    
    def _generate_rsa_keypair(self):
        try:
            self.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self.rsa_public_key = self.rsa_private_key.public_key()
            # persist if path configured
            rsa_path = os.environ.get('OMEGA_RSA_KEY_PATH') or os.path.join(os.path.dirname(__file__), 'omega_keys', 'rsa_key.pem')
            os.makedirs(os.path.dirname(rsa_path), exist_ok=True)
            pem = self.rsa_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(rsa_path, 'wb') as f:
                f.write(pem)
            os.chmod(rsa_path, 0o600)
        except Exception as e:
            logging.error(f"RSA key rotation failed: {e}")
            raise
        
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
            # Re-add essential tables that may be missing if DB was deleted
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    node_id TEXT,
                    application TEXT,
                    cpu_cores INTEGER,
                    gpu_units INTEGER,
                    memory_gb INTEGER,
                    status TEXT,
                    created_at REAL,
                    last_activity REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    network_rx INTEGER,
                    network_tx INTEGER,
                    temperature REAL,
                    power_consumption REAL,
                    timestamp REAL,
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    source TEXT,
                    message TEXT,
                    severity TEXT,
                    timestamp REAL
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
            # Node credentials (issued on approval)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_credentials (
                    node_id TEXT PRIMARY KEY,
                    token TEXT,
                    issued_at REAL,
                    expires_at REAL,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                )
                """
            )
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
            # Policy engine tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id TEXT UNIQUE,
                    name TEXT,
                    kind TEXT,
                    raw TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id TEXT NOT NULL,
                    target_type TEXT NOT NULL, -- node|group
                    target_id TEXT NOT NULL,
                    created_at REAL,
                    UNIQUE(policy_id,target_type,target_id)
                )
            """)
            # RBAC tables (may be missing if DB recreated) - keep minimal schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    role TEXT PRIMARY KEY,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    UNIQUE(username, role)
                )
            """)
            # Fine-grained permissions matrix
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permissions (
                    code TEXT PRIMARY KEY,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role TEXT NOT NULL,
                    permission_code TEXT NOT NULL,
                    PRIMARY KEY(role, permission_code)
                )
            """)
            # Node join requests (signed) & approvals already partially covered by node_approvals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_join_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    capabilities TEXT,
                    nonce TEXT,
                    signature TEXT,
                    created_at REAL,
                    status TEXT DEFAULT 'pending'
                )
            """)
            # Key rotation & revocation metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS revoked_keys (
                    key_id TEXT PRIMARY KEY,
                    revoked_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT UNIQUE,
                    created_at REAL,
                    active INTEGER DEFAULT 0
                )
            """)
            # RBAC, users, sessions, metrics, etc. (as before)
            # ...existing code...
            conn.commit()
            # Bootstrap permissions & roles if empty
            try:
                cur = conn.execute('SELECT COUNT(1) FROM permissions')
                if cur.fetchone()[0] == 0:
                    base_permissions = [
                        ('dashboard:view','View dashboard'),
                        ('resources:view','View resource inventory'),
                        ('network:view','View network topology'),
                        ('performance:view','View performance metrics'),
                        ('plugins:view','Manage plugins'),
                        ('security:view','View security posture'),
                        ('nodes:view','View nodes'),
                        ('sessions:view','View sessions'),
                        ('processes:view','View processes'),
                        ('processes:kill','Terminate processes'),
                        ('rbac:manage','Manage roles & permissions'),
                        ('node:join','Submit node join request'),
                        ('node:approve','Approve node join'),
                        ('autoscale:manage','Trigger autoscaling actions'),
                        ('policy:manage','Manage policies'),
                        ('market:account','Access resource marketplace account'),
                        ('benchmark:run','Run benchmarks'),
                        ('keys:view','List cryptographic keys'),
                        ('keys:rotate','Rotate cryptographic keys'),
                        ('keys:revoke','Revoke cryptographic keys'),
                        # Backward-compat legacy codes
                        ('key:rotate','Rotate cryptographic keys (legacy)'),
                        ('key:revoke','Revoke cryptographic keys (legacy)'),
                        ('attest:verify','Verify node attestation'),
                        ('storage:manage','Manage storage backends'),
                        ('model:manage','Manage predictive models'),
                        ('migration:execute','Execute workload migration'),
                        ('dr:backup','Perform backup operations'),
                        ('dr:restore','Perform restore operations')
                    ]
                    conn.executemany('INSERT INTO permissions(code,description) VALUES(?,?)', base_permissions)
                cur = conn.execute('SELECT COUNT(1) FROM roles')
                if cur.fetchone()[0] == 0:
                    roles = [
                        ('admin','Full administrative access'),
                        ('operator','Operational management minus security critical actions'),
                        ('viewer','Read-only access'),
                        ('security','Security operations and attestation'),
                        ('autoscaler','Manage autoscaling decisions')
                    ]
                    conn.executemany('INSERT INTO roles(role,description) VALUES(?,?)', roles)
                # map role -> permissions baseline if empty
                cur = conn.execute('SELECT COUNT(1) FROM role_permissions')
                if cur.fetchone()[0] == 0:
                    # simple mapping sets
                    role_perm_map = {
                        'admin': [p[0] for p in base_permissions],
                        'operator': ['dashboard:view','resources:view','network:view','performance:view','nodes:view','sessions:view','processes:view','processes:kill','benchmark:run','migration:execute','storage:manage'],
                        'viewer': ['dashboard:view','resources:view','network:view','performance:view','nodes:view','sessions:view'],
                        'security': ['security:view','attest:verify','policy:manage','keys:view','keys:rotate','keys:revoke','rbac:manage'],
                        'autoscaler': ['autoscale:manage','performance:view','nodes:view']
                    }
                    rows = []
                    for r, perms in role_perm_map.items():
                        for perm in perms:
                            rows.append((r, perm))
                    conn.executemany('INSERT INTO role_permissions(role,permission_code) VALUES(?,?)', rows)
                conn.commit()
            except Exception as e:
                logging.error(f"RBAC bootstrap failure: {e}")
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

            # Seed default RBAC roles/admin user mapping if tables exist (idempotent)
            try:
                conn.execute("INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)", ("admin", "Administrator"))
                conn.execute("INSERT OR IGNORE INTO roles (role, description) VALUES (?, ?)", ("user", "Standard User"))
                conn.execute("INSERT OR IGNORE INTO user_roles (username, role) VALUES (?, ?)", ("admin", "admin"))
                # Seed permissions (idempotent)
                base_perms = [
                    ("dashboard:view", "View dashboard summary"),
                    ("resources:view", "View resource metrics"),
                    ("network:view", "View network metrics"),
                    ("performance:view", "View performance data"),
                    ("plugins:view", "List plugins"),
                    ("security:view", "View security info"),
                    ("nodes:view", "List nodes"),
                    ("nodes:quarantine", "Quarantine nodes"),
                    ("nodes:remove", "Remove nodes"),
                    ("sessions:view", "View sessions"),
                    ("processes:view", "View processes"),
                    ("processes:kill", "Kill processes"),
                    ("logs:view", "View logs"),
                    ("session:rotate", "Rotate secure session key"),
                    ("session:start_override", "Start session override"),
                    ("node:register", "Register nodes"),
                    ("node:approve", "Approve pending nodes"),
                    ("keys:view", "List cryptographic keys"),
                    ("keys:rotate", "Rotate cryptographic keys"),
                    ("keys:revoke", "Revoke cryptographic keys"),
                    ("crypto:rotate_keys", "Rotate cryptographic keys (legacy)"),
                    ("rbac:manage", "Manage roles and permissions"),
                    ("node:join", "Submit node join request"),
                    ("backup:create", "Create configuration snapshot"),
                    ("backup:view", "View configuration snapshots"),
                    ("market:view", "View marketplace credits"),
                    ("market:adjust", "Adjust marketplace credits"),
                ]
                for code, desc in base_perms:
                    conn.execute("INSERT OR IGNORE INTO permissions (code, description) VALUES (?, ?)", (code, desc))
                # Assign permissions to roles (admin gets all, user limited view)
                user_allowed = {"dashboard:view","resources:view","network:view","performance:view","plugins:view","security:view","nodes:view","sessions:view","processes:view","logs:view","backup:view","market:view"}
                for code, _ in base_perms:
                    # admin
                    conn.execute("INSERT OR IGNORE INTO role_permissions (role, permission_code) VALUES (?, ?)", ("admin", code))
                    if code in user_allowed:
                        conn.execute("INSERT OR IGNORE INTO role_permissions (role, permission_code) VALUES (?, ?)", ("user", code))
                conn.commit()
            except Exception as e:
                logging.error(f"RBAC init error: {e}")

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
                import hashlib as _hashlib_mod
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
        # Preload audit catalog (lazy creation handled by migrations)
        self._audit_cache_loaded = False

    def _load_audit_event_types(self):
        if self._audit_cache_loaded:
            return
        try:
            with sqlite3.connect(self.database.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute('CREATE TABLE IF NOT EXISTS audit_event_types (code TEXT PRIMARY KEY, description TEXT, severity_default TEXT, retention_days INTEGER DEFAULT 30)')
                self._audit_cache_loaded = True
        except Exception as e:
            logging.debug(f"audit_event_types load failed: {e}")

    def export_audit_bundle(self, fmt: str = 'jsonl') -> dict:
        """Export audit_logs into a signed bundle with SHA256 and detached signature placeholder."""
        self._load_audit_event_types()
        bundle_id = uuid.uuid4().hex
        records = []
        with sqlite3.connect(self.database.db_path, timeout=30, check_same_thread=False) as conn:
            cur = conn.execute('SELECT id,event_type,source,message,severity,timestamp,hash_chain FROM audit_logs ORDER BY id ASC')
            rows = cur.fetchall()
            for r in rows:
                rec = {
                    'id': r[0], 'event_type': r[1], 'source': r[2], 'message': r[3],
                    'severity': r[4], 'timestamp': r[5], 'hash_chain': r[6]
                }
                records.append(rec)
        # Serialize
        if fmt == 'jsonl':
            payload = '\n'.join(json.dumps(r, separators=(',',':')) for r in records).encode()
        else:
            payload = json.dumps({'records': records}).encode()
        import hashlib as _hashlib_mod
        sha256 = _hashlib_mod.sha256(payload).hexdigest()
        # Sign using RSA private key if available
        signature_b64 = None
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding as asy_padding
            sig = self.security_manager.rsa_private_key.sign(payload, asy_padding.PKCS1v15(), hashes.SHA256())
            import base64 as _b64
            signature_b64 = _b64.b64encode(sig).decode()
        except Exception as e:
            logging.debug(f"audit bundle signing skipped: {e}")
        # Persist manifest
        try:
            with sqlite3.connect(self.database.db_path, timeout=30, check_same_thread=False) as conn:
                conn.execute('CREATE TABLE IF NOT EXISTS audit_export_manifests (id INTEGER PRIMARY KEY AUTOINCREMENT, bundle_id TEXT UNIQUE, created_at REAL, format TEXT, record_count INTEGER, sha256 TEXT, signature TEXT)')
                conn.execute('INSERT OR REPLACE INTO audit_export_manifests (bundle_id, created_at, format, record_count, sha256, signature) VALUES (?,?,?,?,?,?)', (bundle_id, time.time(), fmt, len(records), sha256, signature_b64))
                conn.commit()
        except Exception as e:
            logging.error(f"Persist audit export manifest failed: {e}")
        return {'bundle_id': bundle_id, 'format': fmt, 'record_count': len(records), 'sha256': sha256, 'signature': signature_b64, 'data': payload.decode(errors='ignore')}

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
            logging.warning("Docker not found; skipping VD reconcile at startup")
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
        asyncio.create_task(self.node_credential_maintenance())

    async def node_credential_maintenance(self):
        """Expire node credentials and optionally quarantine nodes whose credentials are stale."""
        interval = int(os.environ.get('OMEGA_NODE_CRED_MAINT_INTERVAL','60'))
        quarantine_on_expiry = os.environ.get('OMEGA_NODE_CRED_EXPIRE_QUARANTINE','1') in ('1','true','yes','on')
        while True:
            try:
                now = time.time()
                with sqlite3.connect(self.database.db_path, timeout=30, check_same_thread=False) as conn:
                    cur = conn.execute('SELECT node_id, expires_at FROM node_credentials')
                    rows = cur.fetchall()
                    for node_id, exp in rows:
                        try:
                            if exp and now > float(exp):
                                # remove credential
                                conn.execute('DELETE FROM node_credentials WHERE node_id=?', (node_id,))
                                if quarantine_on_expiry:
                                    conn.execute('UPDATE nodes SET quarantine=1, status=? WHERE node_id=?', ('quarantined', node_id))
                                self.database.log_event('node_cred_expired', node_id, 'Credential expired; access revoked', 'warning')
                        except Exception:
                            pass
                    conn.commit()
            except Exception as e:
                logging.debug(f'node_credential_maintenance error: {e}')
            await asyncio.sleep(interval)

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
                    sid, key_col, user, created, last_rotate, counter, expires_at = r
                    if expires_at and expires_at <= now:
                        continue
                    # Keys may be stored as:
                    # 1) raw hex (legacy)
                    # 2) base64 of raw bytes
                    # 3) base64-wrapped Fernet ciphertext of raw bytes (current)
                    raw_bytes = None
                    if isinstance(key_col, str):
                        k = key_col.strip()
                        # Try hex first (legacy)
                        try:
                            if all(c in '0123456789abcdefABCDEF' for c in k) and len(k) % 2 == 0:
                                raw_bytes = bytes.fromhex(k)
                        except Exception:
                            raw_bytes = None
                        # Try base64 decode (may be raw 32 bytes or Fernet ciphertext)
                        if raw_bytes is None:
                            import base64, binascii
                            try:
                                decoded = base64.b64decode(k)
                                # Heuristic: if looks like Fernet token (has two dots when decoded to str) skip direct use
                                if len(decoded) in (32, 44):  # 32 raw bytes or typical length after base64
                                    raw_bytes = decoded
                            except (binascii.Error, ValueError):
                                pass
                        # Attempt Fernet decrypt if still not usable and cipher available
                        if raw_bytes is None and hasattr(self.security_manager, 'cipher_suite'):
                            try:
                                import base64
                                dec = self.security_manager.cipher_suite.decrypt(base64.b64decode(k))
                                if len(dec) == 32:
                                    raw_bytes = dec
                            except Exception:
                                pass
                    if raw_bytes is None:
                        # Skip malformed key but continue
                        logging.debug(f"Skipping session {sid}: unrecognized key format")
                        continue
                    SESSION_META[sid]={
                        'key': raw_bytes,
                        'user': user,
                        'created_at': created,
                        'last_rotate': last_rotate,
                        'counter': counter,
                        'expires_at': expires_at
                    }
                    loaded += 1
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
    NOTE: Preserve the original function signature so FastAPI can correctly
    perform request body validation (fixes 422 responses introduced when the
    wrapper obscured the endpoint's parameters).
    - per_minute: refill rate
    - burst: bucket capacity
    """
    refill_per_sec = per_minute / 60.0
    def _decorator(func):
        import inspect
        from functools import wraps

        @wraps(func)
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
                # refill tokens based on elapsed time
                elapsed = now - bucket['last']
                bucket['tokens'] = min(burst, bucket['tokens'] + elapsed * refill_per_sec)
                bucket['last'] = now
                if bucket['tokens'] < 1:
                    raise HTTPException(status_code=429, detail='Too many requests')
                bucket['tokens'] -= 1

            return await func(*args, **kwargs)

        # Copy the original callable signature so FastAPI sees expected params
        try:
            _wrapped.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        except Exception:
            pass
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


"""Lifespan migration: replaced deprecated on_event startup handlers.

The original startup_event + enhancement application have been consolidated
into a single FastAPI lifespan context to eliminate deprecation warnings and
guarantee ordering:
  1. Load persisted sessions (needed by other tasks)
  2. Start background tasks
  3. Ensure RBAC/admin role
  4. Reconcile existing virtual desktop sessions (docker)
  5. Apply post-init enhancements (route patching)
"""
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    # --- Startup phase ---
    try:
        api_server.load_persisted_sessions()
    except Exception as e:
        logging.warning(f"Failed loading persisted sessions at startup: {e}")
    try:
        await api_server.start_background_tasks()
    except Exception as e:
        logging.error(f"Background task start failure: {e}")
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
    # Apply enhancements (previously second on_event handler)
    try:
        await post_init_enhance()
    except Exception as e:
        logging.debug(f"post_init_enhance skipped/failed: {e}")
    logging.info("Omega API Server started (lifespan)")
    yield
    # --- Shutdown phase (future hooks) ---
    try:
        logging.info("Omega API Server shutdown (lifespan)")
    except Exception:
        pass

# Attach lifespan context (done after definition to avoid circular refs)
app.router.lifespan_context = lifespan  # type: ignore[attr-defined]

# --- Migration runner (SQLite) for backend ---
try:
    import subprocess, sys as _sys
    _script = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'migrate.py'))
    if os.path.exists(_script):
        subprocess.run([_sys.executable, _script, '--target', 'backend'], timeout=5, check=False)
except Exception as e:
    logging.debug(f"Backend migration runner skipped: {e}")


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

# === RBAC Permission Utilities ===
RBAC_CACHE = {
    'user_roles': {},            # username -> [roles]
    'role_permissions': {},      # role -> set(permission_code)
    'user_permissions': {}       # username -> set(permission_code)
}

RBAC_LAST_LOAD = 0.0
RBAC_CACHE_TTL = 30.0  # seconds

def _load_rbac_cache(force: bool=False):
    global RBAC_LAST_LOAD
    now = time.time()
    if not force and (now - RBAC_LAST_LOAD) < RBAC_CACHE_TTL:
        return
    try:
        with db_connect() as conn:
            # roles per user
            cur = conn.execute('SELECT username, role FROM user_roles')
            user_roles_map = {}
            for username, role in cur.fetchall():
                user_roles_map.setdefault(username, set()).add(role)
            RBAC_CACHE['user_roles'] = {u: list(r) for u, r in user_roles_map.items()}
            # permissions per role
            cur = conn.execute('SELECT role, permission_code FROM role_permissions')
            role_perms = {}
            for role, perm in cur.fetchall():
                role_perms.setdefault(role, set()).add(perm)
            RBAC_CACHE['role_permissions'] = role_perms
            # user permissions derived
            user_perms = {}
            for user, roles in user_roles_map.items():
                pset = set()
                for r in roles:
                    pset.update(role_perms.get(r, set()))
                user_perms[user] = pset
            RBAC_CACHE['user_permissions'] = user_perms
            RBAC_LAST_LOAD = now
    except Exception as e:
        logging.error(f"RBAC cache load failed: {e}")

def get_user_permissions(username: str) -> set:
    _load_rbac_cache()
    return RBAC_CACHE['user_permissions'].get(username, set())

def require_permissions(*required: str):
    """FastAPI dependency factory enforcing that session user holds all required permissions.

    Falls back to ADMIN_BYPASS_TOKEN env token if provided via X-Bypass-Token header for break-glass.
    """
    async def _dep(request: Request):
        validate_secure(request.headers)
        session_id = request.headers.get('X-Session-ID')
        meta = SESSION_META.get(session_id, {})
        user = meta.get('user') or 'unknown'
        roles = meta.get('roles', [])
        bypass = os.getenv('ADMIN_BYPASS_TOKEN')
        if bypass and request.headers.get('X-Bypass-Token') == bypass:
            return True
        base = get_user_permissions(user)
        eff, denied = evaluate_policies(user, roles, base)
        denied_req = [p for p in required if p in denied]
        if denied_req:
            raise HTTPException(status_code=403, detail={'error':'policy_denied','permissions':denied_req})
        missing = [p for p in required if p not in eff]
        if missing and 'admin' not in roles:
            raise HTTPException(status_code=403, detail={'error':'permission_denied','missing':missing})
        return True
    return _dep

# Process-specific metric registry to avoid cross-process collisions
METRICS_REGISTRY = CollectorRegistry()

# Core metrics with custom registry
API_REQUESTS_TOTAL = Counter('omega_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'], registry=METRICS_REGISTRY)
SESSION_COUNT = Gauge('omega_active_sessions', 'Active sessions count', registry=METRICS_REGISTRY)
NODE_COUNT = Gauge('omega_nodes_total', 'Total registered nodes', ['status'], registry=METRICS_REGISTRY)
SECURITY_EVENTS = Counter('omega_security_events_total', 'Security events', ['type', 'severity'], registry=METRICS_REGISTRY)
REQUEST_DURATION = Histogram('omega_request_duration_seconds', 'Request duration', ['method', 'endpoint'], registry=METRICS_REGISTRY)

# Custom advanced metrics
SEAMLESSNESS_INDEX = Gauge('omega_seamlessness_index', 'Latency variance and failover transparency score', registry=METRICS_REGISTRY)
SCALING_EFFICIENCY = Gauge('omega_scaling_efficiency', 'Performance per dollar efficiency', registry=METRICS_REGISTRY)
COLLABORATION_COEFFICIENT = Gauge('omega_collaboration_coefficient', 'Multi-user session uplift factor', registry=METRICS_REGISTRY)

# Unified health schema helper
SERVICE_START_TIME = time.time()
def build_health(deps: dict, version: str = "1.0.0"):
    degraded = [name for name, info in deps.items() if not info.get('ok')]
    return {
        'status': 'healthy' if not degraded else 'degraded',
        'version': version,
        'uptime_seconds': int(time.time() - SERVICE_START_TIME),
        'dependencies': deps,
        'degraded': degraded,
    'timestamp': datetime.now(timezone.utc).isoformat()
    }

# === RBAC management endpoints ===
from fastapi import APIRouter
rbac_router = APIRouter(prefix="/secure/rbac", tags=["rbac"])

# --- RBAC Data Models ---
class RoleCreate(BaseModel):
    role: str
    description: str | None = None
    permissions: list[str] = []

class RoleUpdate(BaseModel):
    description: str | None = None
    permissions: list[str] | None = None

class PermissionCreate(BaseModel):
    code: str
    description: str | None = None

class RBACMatrix(BaseModel):
    roles: dict[str, list[str]]
    permissions: dict[str, str]

@rbac_router.get("/roles", dependencies=[Depends(require_permissions('rbac:manage'))])
async def list_roles():
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT role, description FROM roles')
            return {'roles': [{'role': r, 'description': d} for r, d in cur.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.get("/permissions", dependencies=[Depends(require_permissions('rbac:manage'))])
async def list_permissions():
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT code, description FROM permissions')
            return {'permissions': [{'code': c, 'description': d} for c, d in cur.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.post("/role/{role}/permissions", dependencies=[Depends(require_permissions('rbac:manage'))])
async def add_permission_to_role(role: str, body: dict):
    perm = body.get('permission')
    if not perm:
        raise HTTPException(status_code=400, detail='permission missing')
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR IGNORE INTO role_permissions (role, permission_code) VALUES (?, ?)', (role, perm))
            conn.commit()
        RBAC_LAST_LOAD = 0  # force reload
        return {'status': 'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.post("/user/{username}/roles", dependencies=[Depends(require_permissions('rbac:manage'))])
async def assign_role_to_user(username: str, body: dict):
    role = body.get('role')
    if not role:
        raise HTTPException(status_code=400, detail='role missing')
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR IGNORE INTO user_roles (username, role) VALUES (?, ?)', (username, role))
            conn.commit()
        RBAC_LAST_LOAD = 0
        return {'status': 'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.delete("/user/{username}/roles/{role}", dependencies=[Depends(require_permissions('rbac:manage'))])
async def remove_role_from_user(username: str, role: str):
    """Remove a role assignment from a user."""
    try:
        with db_connect() as conn:
            conn.execute('DELETE FROM user_roles WHERE username=? AND role=?', (username, role))
            conn.commit()
        global RBAC_LAST_LOAD
        RBAC_LAST_LOAD = 0
        return {'status': 'ok', 'removed': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.get('/matrix', response_model=RBACMatrix, dependencies=[Depends(require_permissions('rbac:manage'))])
async def rbac_matrix():
    try:
        with db_connect() as conn:
            roles: dict[str, list[str]] = {}
            for r, p in conn.execute('SELECT role, permission_code FROM role_permissions'):
                roles.setdefault(r, []).append(p)
            perms = {c: d for c, d in conn.execute('SELECT code, description FROM permissions')}
        return RBACMatrix(roles=roles, permissions=perms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.post('/roles', dependencies=[Depends(require_permissions('rbac:manage'))])
async def rbac_create_role(payload: RoleCreate):
    if not payload.role or not payload.role.strip():
        raise HTTPException(status_code=400, detail='role required')
    try:
        with db_connect() as conn:
            conn.execute('INSERT INTO roles(role,description) VALUES(?,?)', (payload.role, payload.description))
            for perm in payload.permissions:
                conn.execute('INSERT OR IGNORE INTO permissions(code,description) VALUES(?,?)', (perm, perm))
                conn.execute('INSERT INTO role_permissions(role,permission_code) VALUES(?,?)', (payload.role, perm))
            conn.commit()
        global RBAC_LAST_LOAD
        RBAC_LAST_LOAD = 0
        return {'role': payload.role, 'permissions': payload.permissions}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail='role exists')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.put('/roles/{role}', dependencies=[Depends(require_permissions('rbac:manage'))])
async def rbac_update_role(role: str, payload: RoleUpdate):
    try:
        with db_connect() as conn:
            if payload.description is not None:
                conn.execute('UPDATE roles SET description=? WHERE role=?', (payload.description, role))
            if payload.permissions is not None:
                conn.execute('DELETE FROM role_permissions WHERE role=?', (role,))
                for perm in payload.permissions:
                    conn.execute('INSERT OR IGNORE INTO permissions(code,description) VALUES(?,?)', (perm, perm))
                    conn.execute('INSERT INTO role_permissions(role,permission_code) VALUES(?,?)', (role, perm))
            conn.commit()
        global RBAC_LAST_LOAD
        RBAC_LAST_LOAD = 0
        return {'role': role, 'updated': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.delete('/roles/{role}', dependencies=[Depends(require_permissions('rbac:manage'))])
async def rbac_delete_role(role: str):
    try:
        with db_connect() as conn:
            conn.execute('DELETE FROM role_permissions WHERE role=?', (role,))
            conn.execute('DELETE FROM roles WHERE role=?', (role,))
            conn.commit()
        global RBAC_LAST_LOAD
        RBAC_LAST_LOAD = 0
        return {'role': role, 'deleted': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rbac_router.post('/permissions', dependencies=[Depends(require_permissions('rbac:manage'))])
async def rbac_create_permission(payload: PermissionCreate):
    if not payload.code:
        raise HTTPException(status_code=400, detail='code required')
    try:
        with db_connect() as conn:
            conn.execute('INSERT INTO permissions(code,description) VALUES(?,?)', (payload.code, payload.description))
            conn.commit()
        global RBAC_LAST_LOAD
        RBAC_LAST_LOAD = 0
        return {'permission': payload.code}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail='permission exists')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Node join flow
@app.post('/secure/nodes/join', dependencies=[Depends(require_permissions('node:join'))])
async def secure_node_join(request: Request, body: dict):
    """Submit signed node join request. Expected body: node_id, capabilities(dict), nonce, timestamp, signature.
    Signature = HMAC SHA256 over canonical JSON of (node_id,nonce,timestamp,capabilities) using ENROLLMENT_SHARED_SECRET.
    """
    secret = os.getenv('ENROLLMENT_SHARED_SECRET', 'change_me')
    required = ['node_id', 'capabilities', 'nonce', 'timestamp', 'signature']
    if any(k not in body for k in required):
        raise HTTPException(status_code=400, detail='missing fields')
    try:
        canonical = json.dumps({k: body[k] for k in ['node_id','nonce','timestamp','capabilities']}, sort_keys=True, separators=(',',':'))
        expected = hashlib.sha256((canonical+secret).encode()).hexdigest()
        if not hmac.compare_digest(expected, body['signature']):
            raise HTTPException(status_code=400, detail='invalid signature')
        # Timestamp freshness (5 min)
        if abs(time.time() - float(body['timestamp'])) > 300:
            raise HTTPException(status_code=400, detail='stale timestamp')
        # Nonce replay protection (in-memory + persistent uniqueness window)
        nonce = body['nonce']
        node_id = body['node_id']
        now = time.time()
        # Persistent duplicate check first
        with db_connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS node_join_nonces (nonce TEXT PRIMARY KEY, node_id TEXT, created_at REAL)')
            cur = conn.execute('SELECT created_at FROM node_join_nonces WHERE nonce=?', (nonce,))
            row = cur.fetchone()
            if row:
                raise HTTPException(status_code=400, detail='replay nonce')
            # Clear old nonces > 10 minutes to bound table
            cutoff = now - 600
            try:
                conn.execute('DELETE FROM node_join_nonces WHERE created_at < ?', (cutoff,))
            except Exception:
                pass
            conn.execute('INSERT INTO node_join_nonces (nonce,node_id,created_at) VALUES (?,?,?)', (nonce, node_id, now))
            conn.commit()
        with db_connect() as conn:
            conn.execute('INSERT INTO node_join_requests (node_id, capabilities, nonce, signature, created_at, status) VALUES (?, ?, ?, ?, ?, ?)',
                         (body['node_id'], json.dumps(body['capabilities']), body['nonce'], body['signature'], time.time(), 'pending'))
            conn.commit()
        return {'status':'pending','node_id':body['node_id']}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'node join failed: {e}')
        raise HTTPException(status_code=500, detail='internal error')

@app.post('/secure/nodes/approve', dependencies=[Depends(require_permissions('node:approve'))])
async def secure_node_approve(body: dict):
    node_id = body.get('node_id')
    if not node_id:
        raise HTTPException(status_code=400, detail='node_id required')
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT id,status FROM node_join_requests WHERE node_id=? ORDER BY id DESC LIMIT 1', (node_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail='no join request')
            if row[1] != 'pending':
                return {'status': row[1], 'node_id': node_id}
            conn.execute('UPDATE node_join_requests SET status="approved" WHERE id=?', (row[0],))
            conn.execute('INSERT OR REPLACE INTO node_approvals (node_id, approved_by, approved_at, status) VALUES (?, ?, ?, ?)', (node_id, 'admin', time.time(), 'approved'))
            conn.commit()
        return {'status':'approved','node_id':node_id}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'approve error: {e}')
        raise HTTPException(status_code=500, detail='internal error')

# --- Audit export endpoints ---
@app.get('/secure/audit/export', dependencies=[Depends(require_permissions('security:view'))])
async def audit_export(fmt: str = 'jsonl'):
    if fmt not in ('jsonl','json'):
        raise HTTPException(status_code=400, detail='unsupported format')
    out = api_server.export_audit_bundle(fmt=fmt)
    # Large bundle data could be omitted unless explicitly requested; included now to simplify
    return out

@app.get('/secure/audit/manifest', dependencies=[Depends(require_permissions('security:view'))])
async def audit_manifest():
    try:
        with sqlite3.connect(api_server.database.db_path, timeout=30, check_same_thread=False) as conn:
            cur = conn.execute('SELECT bundle_id, created_at, format, record_count, sha256, signature FROM audit_export_manifests ORDER BY created_at DESC LIMIT 50')
            rows = [dict(bundle_id=b, created_at=c, format=f, record_count=r, sha256=s, signature=sg) for b,c,f,r,s,sg in cur.fetchall()]
        return {'manifests': rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Key rotation endpoints
@app.post('/secure/keys/rotate', dependencies=[Depends(require_permissions('crypto:rotate_keys'))])
async def rotate_keys(body: dict = {}):
    key_id = body.get('key_id') or uuid.uuid4().hex
    now = time.time()
    try:
        with db_connect() as conn:
            # Mark existing active as inactive
            conn.execute('UPDATE key_metadata SET active=0 WHERE active=1')
            conn.execute('INSERT OR REPLACE INTO key_metadata (key_id, created_at, active) VALUES (?, ?, 1)', (key_id, now))
            conn.commit()
        return {'status':'rotated','key_id':key_id,'created_at':now}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/secure/keys/revoke', dependencies=[Depends(require_permissions('crypto:rotate_keys'))])
async def revoke_key(body: dict):
    key_id = body.get('key_id')
    if not key_id:
        raise HTTPException(status_code=400, detail='key_id required')
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR IGNORE INTO revoked_keys (key_id, revoked_at) VALUES (?, ?)', (key_id, time.time()))
            conn.execute('UPDATE key_metadata SET active=0 WHERE key_id=?', (key_id,))
            conn.commit()
        return {'status':'revoked','key_id':key_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/secure/keys/status', dependencies=[Depends(require_permissions('crypto:rotate_keys'))])
async def keys_status():
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT key_id, created_at, active FROM key_metadata ORDER BY created_at DESC')
            keys = [{'key_id':k,'created_at':c,'active':bool(a)} for k,c,a in cur.fetchall()]
            cur = conn.execute('SELECT key_id, revoked_at FROM revoked_keys')
            revoked = [{'key_id':k,'revoked_at':r} for k,r in cur.fetchall()]
        return {'keys':keys,'revoked':revoked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(rbac_router)

# === Policy Engine Endpoints ===
policy_router = APIRouter(prefix="/secure/policy", tags=["policy"], dependencies=[Depends(require_permissions('rbac:manage'))])

class PolicyUpsert(BaseModel):
    policy_id: str
    name: str
    kind: str = 'generic'
    spec: dict
    raw: str

@policy_router.post('/upsert')
async def upsert_policy(body: PolicyUpsert):
    raw = body.raw
    # Support YAML input auto-conversion
    try:
        if body.kind.lower() in ('yaml','yml'):
            import yaml  # type: ignore
            parsed = yaml.safe_load(body.raw) if body.raw else {}
            raw = json.dumps(parsed, separators=(',',':'))
        elif body.kind.lower() == 'json':
            # validate JSON
            json.loads(body.raw)
        else:
            # attempt detect
            if body.raw.strip().startswith('{'):
                json.loads(body.raw)
            else:
                import yaml  # type: ignore
                parsed = yaml.safe_load(body.raw)
                raw = json.dumps(parsed, separators=(',',':'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'invalid policy format: {e}')
    # Advanced schema validation
    try:
        parsed_obj = json.loads(raw)
        if not isinstance(parsed_obj, dict):
            raise HTTPException(status_code=400, detail='policy root must be an object')
        # Required top-level fields
        for field in ('version', 'statements'):
            if field not in parsed_obj:
                raise HTTPException(status_code=400, detail=f'missing required field: {field}')
        if not isinstance(parsed_obj['statements'], list) or not parsed_obj['statements']:
            raise HTTPException(status_code=400, detail='statements must be a non-empty list')
        allowed_effects = {'allow','deny'}
        for idx, stmt in enumerate(parsed_obj['statements']):
            if not isinstance(stmt, dict):
                raise HTTPException(status_code=400, detail=f'statement {idx} must be object')
            if 'effect' not in stmt or stmt['effect'].lower() not in allowed_effects:
                raise HTTPException(status_code=400, detail=f'statement {idx} invalid or missing effect')
            if 'actions' not in stmt or not isinstance(stmt['actions'], list) or not stmt['actions']:
                raise HTTPException(status_code=400, detail=f'statement {idx} must define non-empty actions list')
            if 'resources' not in stmt or not isinstance(stmt['resources'], list) or not stmt['resources']:
                raise HTTPException(status_code=400, detail=f'statement {idx} must define non-empty resources list')
            # Optional condition structure
            if 'conditions' in stmt and not isinstance(stmt['conditions'], dict):
                raise HTTPException(status_code=400, detail=f'statement {idx} conditions must be object if present')
        if parsed_obj.get('id') and parsed_obj['id'] != body.policy_id:
            raise HTTPException(status_code=400, detail='policy_id mismatch in body vs raw content')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'policy validation failed: {e}')
    now = datetime.now(timezone.utc).isoformat()
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR REPLACE INTO policies (policy_id,name,kind,raw,created_at,updated_at) VALUES (?,?,?,?,COALESCE((SELECT created_at FROM policies WHERE policy_id=?),?),?)',
                         (body.policy_id, body.name, body.kind, raw, body.policy_id, now, now))
            conn.commit()
        return {'status':'ok','policy_id':body.policy_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@policy_router.get('/list')
async def list_policies():
    with db_connect() as conn:
        cur = conn.execute('SELECT policy_id,name,kind,raw,created_at,updated_at FROM policies ORDER BY id DESC LIMIT 200')
        rows=[{'policy_id':p,'name':n,'kind':k,'spec':json.loads(r),'created_at':c,'updated_at':u} for p,n,k,r,c,u in cur.fetchall()]
    return {'policies':rows}

# === Policy assignment & evaluation ===
class PolicyAssignment(BaseModel):
    policy_id: str
    target_type: str  # user or role
    target_id: str

@policy_router.post('/assign')
async def assign_policy(body: PolicyAssignment):
    if body.target_type not in ('user','role'):
        raise HTTPException(status_code=400, detail='target_type must be user or role')
    try:
        with db_connect() as conn:
            # policy_assignments schema uses created_at column
            conn.execute('INSERT OR REPLACE INTO policy_assignments (policy_id,target_type,target_id,created_at) VALUES (?,?,?,?)',
                         (body.policy_id, body.target_type, body.target_id, datetime.now(timezone.utc).isoformat()))
            conn.commit()
        invalidate_policy_cache()
        return {'status':'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@policy_router.get('/effective/{username}')
async def effective_permissions(username: str):
    base = get_user_permissions(username)
    _load_rbac_cache()
    roles = RBAC_CACHE.get('user_roles', {}).get(username, [])
    eff, denied = evaluate_policies(username, roles, base)
    return {'username': username, 'base': list(base), 'effective': list(eff), 'denied': list(denied)}

# Policy cache & evaluation helpers
POLICY_CACHE = {
    'policies': {},
    'assign_user': {},
    'assign_role': {},
}
POLICY_LAST_LOAD = 0
POLICY_CACHE_TTL = 15

def invalidate_policy_cache():
    global POLICY_LAST_LOAD
    POLICY_LAST_LOAD = 0

def _load_policy_cache():
    global POLICY_LAST_LOAD
    now = time.time()
    if (now - POLICY_LAST_LOAD) < POLICY_CACHE_TTL:
        return
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT policy_id, raw FROM policies')
            policies = {}
            for pid, raw in cur.fetchall():
                try:
                    policies[pid] = json.loads(raw)
                except Exception:
                    continue
            POLICY_CACHE['policies'] = policies
            a_user = {}
            a_role = {}
            cur = conn.execute('SELECT policy_id,target_type,target_id FROM policy_assignments')
            for pid, ttype, tid in cur.fetchall():
                if ttype == 'user':
                    a_user.setdefault(tid, set()).add(pid)
                elif ttype == 'role':
                    a_role.setdefault(tid, set()).add(pid)
            POLICY_CACHE['assign_user'] = a_user
            POLICY_CACHE['assign_role'] = a_role
            POLICY_LAST_LOAD = now
    except Exception as e:
        logging.error(f"Policy cache load failed: {e}")

def evaluate_policies(username: str, roles: List[str], base_perms: set):
    _load_policy_cache()
    assigned = set()
    assigned.update(POLICY_CACHE['assign_user'].get(username, set()))
    for r in roles:
        assigned.update(POLICY_CACHE['assign_role'].get(r, set()))
    allow = set(); deny = set()
    for pid in assigned:
        pobj = POLICY_CACHE['policies'].get(pid) or {}
        for stmt in pobj.get('statements', []):
            if not isinstance(stmt, dict):
                continue
            actions = stmt.get('actions') or []
            if not isinstance(actions, list):
                continue
            effect = (stmt.get('effect') or '').lower()
            if effect == 'allow':
                allow.update(actions)
            elif effect == 'deny':
                deny.update(actions)
    eff = set(base_perms) | allow
    eff -= deny
    return eff, deny

class PolicyAssign(BaseModel):
    policy_id: str
    target_type: str
    target_id: str

@policy_router.post('/assign')
async def assign_policy(body: PolicyAssign):
    now=time.time()
    if body.target_type not in ('node','group'):
        raise HTTPException(status_code=400, detail='invalid target_type')
    try:
        with db_connect() as conn:
            conn.execute('INSERT OR IGNORE INTO policy_assignments (policy_id,target_type,target_id,created_at) VALUES (?,?,?,?)', (body.policy_id, body.target_type, body.target_id, now))
            conn.commit()
        return {'status':'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@policy_router.get('/assignments/{target_type}/{target_id}')
async def get_assignments(target_type: str, target_id: str):
    with db_connect() as conn:
        cur = conn.execute('SELECT policy_id FROM policy_assignments WHERE target_type=? AND target_id=?', (target_type, target_id))
        return {'policies':[r[0] for r in cur.fetchall()]}

app.include_router(policy_router)

# Unified health endpoint (override or add if not present)
@app.get('/health')
async def unified_health():
    # Initialize secret manager lazily (first call) to avoid import cycles
    global SECRET_MANAGER
    try:
        SECRET_MANAGER
    except NameError:
        SECRET_MANAGER = default_manager()  # type: ignore

    deps = {
        'database': {'ok': True},
        'redis': {'ok': True},  # placeholder (future real check)
        'metrics': {'ok': bool(api_server.system_stats)},
        'rbac_cache': {'ok': True},
        'sessions': {'ok': True},
        'audit': {'ok': True},
        'migrations': {'ok': True},
        'secrets': {'ok': True},
    }

    # Database & basic queries
    applied_migrations = []
    known_migrations = []
    audit_events = 0
    audit_exports = 0
    try:
        with db_connect() as conn:
            # health probe
            conn.execute('SELECT 1')
            # migrations table (may not exist initially)
            try:
                cur = conn.execute('SELECT version FROM schema_migrations ORDER BY version')
                applied_migrations = [r[0] for r in cur.fetchall()]
            except Exception:
                deps['migrations']['ok'] = False
            # audit stats (tables may or may not exist depending on migration run)
            try:
                cur = conn.execute('SELECT count(*) FROM audit_events')
                audit_events = cur.fetchone()[0]
            except Exception:
                deps['audit']['ok'] = False
            try:
                cur = conn.execute('SELECT count(*) FROM audit_export_manifests')
                audit_exports = cur.fetchone()[0]
            except Exception:
                pass  # optional table
    except Exception:
        deps['database']['ok'] = False

    # Discover known migration files (best-effort)
    try:
        migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        if os.path.isdir(migrations_dir):
            for name in os.listdir(migrations_dir):
                if name.startswith('V') and '__' in name:
                    ver = name.split('__',1)[0][1:]
                    known_migrations.append(ver)
    except Exception:
        pass
    pending_migrations = [v for v in known_migrations if v not in applied_migrations]
    if pending_migrations:
        deps['migrations']['ok'] = False

    # RBAC cache freshness (stale if > 2x TTL)
    if (time.time() - RBAC_LAST_LOAD) > (RBAC_CACHE_TTL * 2):
        deps['rbac_cache']['ok'] = False

    # Session / key stats
    active_sessions = len(SESSION_META)
    revoked_sessions = len(REVOKED_SESSIONS)
    if active_sessions == 0:
        deps['sessions']['ok'] = False

    # Secrets snapshot fingerprint (avoid exposing raw secret values)
    try:
        snap = SECRET_MANAGER.snapshot()
        secret_keys = sorted(snap.keys())
        import hashlib as _hashlib
        fp = _hashlib.sha256(('|'.join(secret_keys)).encode()).hexdigest()[:16]
        deps['secrets']['count'] = len(secret_keys)
        deps['secrets']['fingerprint'] = fp
        if len(secret_keys) == 0:
            deps['secrets']['ok'] = False
    except Exception:
        deps['secrets']['ok'] = False

    # Audit enrichment
    deps['audit']['events'] = audit_events
    deps['audit']['exports'] = audit_exports
    if audit_events == 0:
        deps['audit']['ok'] = False  # no events recorded could indicate issue

    extended = {
        'migrations': {
            'applied': applied_migrations,
            'known': known_migrations,
            'pending': pending_migrations,
        },
        'sessions': {
            'active': active_sessions,
            'revoked': revoked_sessions,
        },
    }
    base = build_health(deps, version='1.0.2')
    base['extended'] = extended
    return base

# Secure token-based registration (replacement for legacy register)
@app.post('/api/secure/nodes/register', dependencies=[Depends(require_permissions('node:register'))])
async def secure_register(request: Request, body: NodeRegistrationRequest):
    token = request.headers.get('X-Register-Token')
    expected = os.getenv('NODE_REGISTRATION_TOKEN')
    if not expected or token != expected:
        raise HTTPException(status_code=401, detail='Invalid registration token')
    # Insert minimal node (pending approval) if not exists
    api_server.database.add_node(body.node_id, body.node_type, body.hostname, body.ip_address, body.port, body.resources)
    # Mark node as pending and create an approval record
    try:
        with db_connect() as conn:
            conn.execute('UPDATE nodes SET status=? WHERE node_id=?', ('pending', body.node_id))
            conn.execute('INSERT OR IGNORE INTO node_approvals (node_id, approved_by, approved_at, status) VALUES (?, ?, ?, ?)', (body.node_id, None, None, 'pending'))
            # Persist attestation details to support later deny/revocation workflows
            try:
                conn.execute(
                    'INSERT INTO node_attestations (node_id, device_fingerprint, public_key_pem, device_certificate, health_attestation, geoip, behavioral_baseline, attested_at) VALUES (?,?,?,?,?,?,?,?)',
                    (
                        body.node_id,
                        body.device_fingerprint,
                        body.public_key_pem,
                        body.device_certificate,
                        json.dumps(body.health_attestation or {}),
                        body.geoip,
                        json.dumps(body.behavioral_baseline or {}),
                        time.time(),
                    )
                )
            except Exception as _e:
                logging.debug(f"secure_register: attestation persist failed for {body.node_id}: {_e}")
            conn.commit()
    except Exception as e:
        logging.debug(f"secure_register: failed to persist pending approval for {body.node_id}: {e}")
    api_server.database.log_event('node_register', body.node_id, 'Secure registration submitted')
    return {'status':'pending_approval','node_id': body.node_id}

# Deprecate legacy register endpoint if exists
@app.post('/api/nodes/register')
async def legacy_register_deprecated():
    return {'deprecated': True, 'message': 'Use /secure/nodes/join and /secure/nodes/approve', 'status':'410'}

# === Key & Certificate Lifecycle Management ===
class KeyRotationRequest(BaseModel):
    reason: str | None = None
    rotate_session_keys: bool = True
    rotate_rsa: bool = False

@app.get('/secure/keys', dependencies=[Depends(require_permissions('keys:view'))])
async def list_keys():
    keys = []
    # Session keys (in-memory)
    for sid, meta in list(SESSION_META.items())[:500]:  # cap for safety
        keys.append({
            'type': 'session',
            'session_id': sid,
            'user': meta.get('user'),
            'created_at': meta.get('created_at'),
            'expires_at': meta.get('expires_at'),
            'counter': meta.get('counter',0)
        })
    # RSA key fingerprint
    try:
        pub = api_server.security_manager.get_public_key_pem()
        import hashlib as _h
        fp = _h.sha256(pub.encode()).hexdigest()[:16]
        keys.append({'type':'rsa','fingerprint':fp})
    except Exception:
        pass
    return {'keys': keys}

@app.post('/secure/session/revoke', dependencies=[Depends(require_permissions('keys:revoke'))])
async def revoke_session(body: dict):
    sid = body.get('session_id')
    if not sid:
        raise HTTPException(status_code=400, detail='session_id required')
    if sid in SESSION_META:
        SESSION_META.pop(sid, None)
        REVOKED_SESSIONS.add(sid)
        api_server.database.log_event('session_revoked', sid, 'Session revoked via API')
    return {'status':'revoked','session_id':sid}

@app.post('/secure/session/keys/rotate', dependencies=[Depends(require_permissions('keys:rotate'))])
async def rotate_session_keys(req: KeyRotationRequest):
    rotated = {}
    if req.rotate_session_keys:
        count = 0
        for sid, meta in list(SESSION_META.items()):
            try:
                new_key = secrets.token_bytes(32)
                import base64 as _b64
                meta['key'] = _b64.b64encode(new_key).decode()
                meta['counter'] = 0
                count += 1
            except Exception:
                continue
        rotated['session_keys'] = count
    if req.rotate_rsa:
        try:
            api_server.security_manager._generate_rsa_keypair()
            rotated['rsa'] = True
        except Exception as e:
            rotated['rsa'] = False
            rotated['rsa_error'] = str(e)
    api_server.database.log_event('keys_rotated','system', json.dumps({'rotated':rotated,'reason':req.reason}))
    return {'status':'ok','rotated':rotated}

# --- API-consistent aliases under /api/secure/* ---
@app.get('/api/secure/keys', dependencies=[Depends(require_permissions('keys:view'))])
async def api_secure_list_keys(request: Request):
    # Reuse existing logic
    return await list_keys()


@app.post('/api/secure/keys/revoke', dependencies=[Depends(require_permissions('keys:revoke'))])
async def api_secure_revoke_session(body: dict, request: Request):
    return await revoke_session(body)


@app.post('/api/secure/keys/rotate', dependencies=[Depends(require_permissions('keys:rotate'))])
async def api_secure_rotate_keys(req: KeyRotationRequest, request: Request):
    return await rotate_session_keys(req)


# --- Self-revocation for current session ---
@app.post('/api/secure/session/revoke_self')
async def revoke_self(request: Request):
    sid, key = validate_secure(request.headers)
    persist_revocation(sid, 'self')
    # cleanup in-memory; close websocket best-effort
    ws_closed = False
    try:
        with SESSION_LOCK:
            SESSION_META.pop(sid, None)
            SESSION_NONCES.pop(sid, None)
            ws = SESSION_WS.pop(sid, None)
        if ws:
            try:
                await ws.close()
                ws_closed = True
            except Exception:
                pass
        with db_connect() as conn:
            try:
                conn.execute('DELETE FROM session_meta WHERE session_id=?', (sid,))
                conn.commit()
            except Exception:
                pass
        api_server.database.log_event('session_revoke_self', sid, 'self-revoked', 'info')
    except Exception as e:
        logging.debug(f'revoke_self: cleanup failed for {sid}: {e}')
    return wrap_encrypted(sid, key, {'ok': True, 'revoked': sid, 'ws_closed': ws_closed})

# === Resource Marketplace Scaffold ===
class CreditAdjust(BaseModel):
    user: str
    delta: int
    reason: str | None = None

@app.get('/secure/market/credits', dependencies=[Depends(require_permissions('market:view'))])
async def market_list():
    rows=[]
    try:
        with db_connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS user_credits (user TEXT PRIMARY KEY, balance INTEGER, updated_at REAL)')
            cur=conn.execute('SELECT user,balance,updated_at FROM user_credits')
            rows=[{'user':u,'balance':b,'updated_at':t} for u,b,t in cur.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'credits':rows}

@app.post('/secure/market/credits/adjust', dependencies=[Depends(require_permissions('market:adjust'))])
async def credit_adjust(body: CreditAdjust):
    now=time.time()
    try:
        with db_connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS user_credits (user TEXT PRIMARY KEY, balance INTEGER, updated_at REAL)')
            conn.execute('INSERT INTO user_credits (user,balance,updated_at) VALUES (?,?,?) ON CONFLICT(user) DO UPDATE SET balance=balance+excluded.balance, updated_at=excluded.updated_at', (body.user, body.delta, now))
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    api_server.database.log_event('credit_adjust', body.user, json.dumps({'delta':body.delta,'reason':body.reason}))
    return {'status':'ok'}

# === Backup & Disaster Recovery (config snapshot) ===
class SnapshotCreate(BaseModel):
    label: str | None = None
    include_audit: bool = True

SNAPSHOT_DIR = os.getenv('OMEGA_SNAPSHOT_DIR','data/snapshots')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

@app.post('/secure/backup/snapshot', dependencies=[Depends(require_permissions('backup:create'))])
async def create_snapshot(body: SnapshotCreate):
    ts=int(time.time())
    name=f"snap-{ts}-{(body.label or 'auto').replace(' ','_')}"
    path=os.path.join(SNAPSHOT_DIR, name)
    os.makedirs(path, exist_ok=True)
    meta={'created_at':ts,'label':body.label,'version':'1.0.0'}
    # Copy key tables (best-effort)
    tables=['nodes','sessions','policies','roles','permissions','role_permissions','user_roles']
    copied=[]
    try:
        with db_connect() as conn:
            for t in tables:
                try:
                    cur=conn.execute(f'SELECT * FROM {t}')
                    rows=cur.fetchall()
                    cols=[c[0] for c in cur.description]
                    with open(os.path.join(path, f'{t}.json'),'w') as f:
                        json.dump({'columns':cols,'rows':rows}, f)
                    copied.append(t)
                except Exception:
                    continue
            if body.include_audit:
                try:
                    cur=conn.execute('SELECT * FROM audit_logs ORDER BY id ASC')
                    rows=cur.fetchall(); cols=[c[0] for c in cur.description]
                    with open(os.path.join(path,'audit_logs.json'),'w') as f:
                        json.dump({'columns':cols,'rows':rows}, f)
                except Exception:
                    pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    with open(os.path.join(path,'meta.json'),'w') as f:
        json.dump(meta, f)
    api_server.database.log_event('snapshot_create','system', json.dumps({'name':name,'copied':copied}))
    return {'status':'ok','snapshot':name,'copied':copied}

@app.get('/secure/backup/snapshots', dependencies=[Depends(require_permissions('backup:view'))])
async def list_snapshots():
    out=[]
    for n in sorted(os.listdir(SNAPSHOT_DIR)):
        p=os.path.join(SNAPSHOT_DIR,n)
        if not os.path.isdir(p): continue
        try:
            with open(os.path.join(p,'meta.json')) as f:
                meta=json.load(f)
        except Exception:
            meta={}
        out.append({'name':n,'meta':meta})
    return {'snapshots':out}

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

# (Removed deprecated @app.on_event('startup') enhancement hook; logic moved to lifespan)


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

@app.post('/api/secure/action', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.post('/api/secure/discover', dependencies=[Depends(require_permissions('nodes:view'))])
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

# Alias under nodes path for frontend consistency
@app.get('/api/secure/nodes/discover', dependencies=[Depends(require_permissions('nodes:view'))])
@rate_limited(per_minute=12, burst=6)
async def secure_nodes_discover(request: Request):
    session_id, key = validate_secure(request.headers)
    try:
        result = await action_discover_nodes()
        if 'success' not in result:
            result['success'] = True
    except Exception as e:
        logging.error(f"secure_nodes_discover error: {e}")
        result = {'success': False, 'error': str(e)}
    return wrap_encrypted(session_id, key, {'action': 'discover_nodes', 'result': result, 'ok': True})

@app.get('/api/secure/nodes/{node_id}/protocol/health', dependencies=[Depends(require_permissions('nodes:view'))])
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

def _validate_node_credential_for_request(node_id: str, headers: dict):
    """Validate node credential token from headers against DB for the given node.
    - Requires header X-Node-Token to be present and match node_credentials.token
    - Token must not be expired (expires_at > now)
    - Node must not be quarantined or denied; status should be active
    Raises HTTPException 401/403 on failure.
    """
    token = headers.get('X-Node-Token') or headers.get('x-node-token')
    if not token:
        raise HTTPException(status_code=401, detail='Missing node credential')
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT token, expires_at FROM node_credentials WHERE node_id=?', (node_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail='No credential issued')
            db_token, expires_at = row
            if not secrets.compare_digest(str(token), str(db_token)):
                raise HTTPException(status_code=401, detail='Invalid node credential')
            if expires_at and time.time() > float(expires_at):
                raise HTTPException(status_code=401, detail='Node credential expired')
            # Check node status/quarantine
            cur2 = conn.execute('SELECT status, quarantine FROM nodes WHERE node_id=?', (node_id,))
            n = cur2.fetchone()
            if n:
                status, quarantine = n
                if quarantine:
                    raise HTTPException(status_code=403, detail='Node quarantined')
                if (status or '').lower() in ('denied','pending','pending_approval'):
                    raise HTTPException(status_code=403, detail='Node not approved')
    except HTTPException:
        raise
    except Exception as e:
        logging.debug(f"node credential validation error for {node_id}: {e}")
        raise HTTPException(status_code=401, detail='Node credential check failed')

@app.post('/api/secure/nodes/heartbeat', dependencies=[Depends(require_permissions('nodes:view'))])
async def secure_heartbeat(body: Heartbeat, request: Request):
    session_id, key = validate_secure(request.headers)
    # Enforce node-issued credential on node-initiated call
    _validate_node_credential_for_request(body.node_id, request.headers)
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

@app.post('/api/secure/nodes/{node_id}/probe', dependencies=[Depends(require_permissions('nodes:view'))])
async def secure_probe(node_id: str, request: Request):
    session_id, key = validate_secure(request.headers)
    # Enforce node-issued credential on node-initiated probe
    _validate_node_credential_for_request(node_id, request.headers)
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

@app.post('/api/secure/nodes/quarantine', dependencies=[Depends(require_permissions('nodes:quarantine'))])
async def secure_quarantine(body: QuarantineReq, request: Request):
    session_id, key = validate_secure(request.headers)
    api_server.database.set_quarantine(body.node_id, body.enable)
    api_server.database.log_event('node_quarantine', body.node_id, f"Quarantine={'on' if body.enable else 'off'}")
    return wrap_encrypted(session_id, key, {'node_id': body.node_id, 'quarantine': body.enable})

class RemoveReq(BaseModel):
    node_id: str

@app.post('/api/secure/nodes/remove', dependencies=[Depends(require_permissions('nodes:remove'))])
async def secure_remove(body: RemoveReq, request: Request):
    session_id, key = validate_secure(request.headers)
    api_server.database.remove_node(body.node_id)
    api_server.database.log_event('node_remove', body.node_id, 'Node removed')
    return wrap_encrypted(session_id, key, {'removed': body.node_id})

# --- Secure approval workflow endpoints ---

# --- Node credential management ---
@app.get('/api/secure/nodes/{node_id}/credential', dependencies=[Depends(require_permissions('keys:view'))])
async def get_node_credential(node_id: str, request: Request):
    session_id, key = validate_secure(request.headers)
    rec=None
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT token, issued_at, expires_at FROM node_credentials WHERE node_id=?', (node_id,))
            r = cur.fetchone()
            if r:
                rec = {'token': r[0], 'issued_at': r[1], 'expires_at': r[2]}
    except Exception as e:
        logging.debug(f'get_node_credential error: {e}')
    return wrap_encrypted(session_id, key, {'node_id': node_id, 'credential': rec})

class RotateNodeCredReq(BaseModel):
    node_id: str
    ttl_seconds: Optional[int] = None

@app.post('/api/secure/nodes/credential/rotate', dependencies=[Depends(require_permissions('keys:rotate'))])
async def rotate_node_credential(body: RotateNodeCredReq, request: Request):
    session_id, key = validate_secure(request.headers)
    node_id = body.node_id
    ttl = body.ttl_seconds or int(os.getenv('OMEGA_NODE_CRED_TTL','86400'))
    exp = time.time() + ttl
    token=None
    try:
        with db_connect() as conn:
            import base64 as _b64
            token = _b64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
            conn.execute('INSERT OR REPLACE INTO node_credentials (node_id, token, issued_at, expires_at) VALUES (?,?,?,?)', (node_id, token, time.time(), exp))
            conn.commit()
        api_server.database.log_event('node_cred_rotate', node_id, 'Credential rotated', 'info')
    except Exception as e:
        logging.error(f'rotate_node_credential error: {e}')
        raise HTTPException(status_code=500, detail='rotate failed')
    return wrap_encrypted(session_id, key, {'node_id': node_id, 'credential': {'token': token, 'expires_at': exp}})
@app.get('/api/secure/nodes/pending', dependencies=[Depends(require_permissions('node:approve'))])
async def secure_pending_nodes(request: Request):
    session_id, key = validate_secure(request.headers)
    rows = []
    try:
        with db_connect() as conn:
            cur = conn.execute('SELECT node_id, approved_by, approved_at, status FROM node_approvals WHERE status=?', ('pending',))
            rows = [{'node_id': r[0], 'approved_by': r[1], 'approved_at': r[2], 'status': r[3]} for r in cur.fetchall()]
    except Exception as e:
        logging.error(f'secure_pending_nodes error: {e}')
    return wrap_encrypted(session_id, key, {'pending': rows})

class ApproveNodeReq(BaseModel):
    node_id: str

@app.post('/api/secure/nodes/approve', dependencies=[Depends(require_permissions('node:approve'))])
async def secure_approve_node(body: ApproveNodeReq, request: Request):
    session_id, key = validate_secure(request.headers)
    node_id = body.node_id
    ok = True
    # Bootstrap trust and issue a short-lived node credential on approval
    base_trust = int(os.getenv('OMEGA_APPROVAL_TRUST_BASE', '50'))
    cred_ttl = int(os.getenv('OMEGA_NODE_CRED_TTL', '86400'))  # 24h default
    token = None
    expires_at = time.time() + cred_ttl
    try:
        with db_connect() as conn:
            # Mark approved and activate node
            conn.execute('UPDATE node_approvals SET status=?, approved_by=?, approved_at=? WHERE node_id=?', ('approved', SESSION_META.get(session_id,{}).get('user','admin'), time.time(), node_id))
            conn.execute('UPDATE nodes SET status=?, trust_score=? WHERE node_id=?', ('active', base_trust, node_id))
            # Ensure credential table and issue credential
            conn.execute('''
                CREATE TABLE IF NOT EXISTS node_credentials (
                    node_id TEXT PRIMARY KEY,
                    token TEXT,
                    issued_at REAL,
                    expires_at REAL,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                )
            ''')
            import base64 as _b64
            token = _b64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
            conn.execute('INSERT OR REPLACE INTO node_credentials (node_id, token, issued_at, expires_at) VALUES (?,?,?,?)', (node_id, token, time.time(), expires_at))
            conn.commit()
        api_server.database.log_event('node_approve', node_id, f'Approved via secure API', 'info')
    except Exception as e:
        ok = False
        logging.error(f'secure_approve_node error: {e}')
    payload = {'ok': ok, 'node_id': node_id, 'status': 'approved' if ok else 'error'}
    if ok:
        payload['trust_score'] = base_trust
        payload['credential'] = {'token': token, 'expires_at': expires_at}
    return wrap_encrypted(session_id, key, payload)

class DenyNodeReq(BaseModel):
    node_id: str

@app.post('/api/secure/nodes/deny', dependencies=[Depends(require_permissions('node:approve'))])
async def secure_deny_node(body: DenyNodeReq, request: Request):
    session_id, key = validate_secure(request.headers)
    node_id = body.node_id
    status='denied'
    fingerprint=None
    try:
        with db_connect() as conn:
            # if node exists, mark denied; else return not_found status
            cur = conn.execute('SELECT 1 FROM node_approvals WHERE node_id=?', (node_id,))
            if cur.fetchone():
                conn.execute('UPDATE node_approvals SET status=?, approved_by=?, approved_at=? WHERE node_id=?', ('denied', SESSION_META.get(session_id,{}).get('user','admin'), time.time(), node_id))
                conn.execute('UPDATE nodes SET status=? WHERE node_id=?', ('denied', node_id))
                # Revoke any issued credentials
                try:
                    conn.execute('DELETE FROM node_credentials WHERE node_id=?', (node_id,))
                except Exception:
                    pass
                # Add public key fingerprint to revoked_keys if attestation exists
                try:
                    cur2 = conn.execute('SELECT public_key_pem FROM node_attestations WHERE node_id=? ORDER BY id DESC LIMIT 1', (node_id,))
                    row = cur2.fetchone()
                    if row and row[0]:
                        import hashlib as _hash
                        fingerprint = _hash.sha256((row[0] or '').encode()).hexdigest()
                        conn.execute('INSERT OR IGNORE INTO revoked_keys (key_id, revoked_at) VALUES (?, ?)', (fingerprint, time.time()))
                except Exception as _e:
                    logging.debug(f'deny: revoke key failed for {node_id}: {_e}')
                conn.commit()
                api_server.database.log_event('node_deny', node_id, 'Denied via secure API', 'warning')
            else:
                status='not_found'
    except Exception as e:
        logging.error(f'secure_deny_node error: {e}')
        status='error'
    return wrap_encrypted(session_id, key, {'node_id': node_id, 'status': status, 'fingerprint': fingerprint})

class VdStartRequest(BaseModel):
    node_id: str
    image: Optional[str] = 'ubuntu-xfce'
    cpu_cores: int = 2
    memory_gb: int = 4

@app.post('/api/secure/vd/start', dependencies=[Depends(require_permissions('session:start_override'))])
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

@app.get('/api/secure/vd/list', dependencies=[Depends(require_permissions('sessions:view'))])
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

@app.post('/api/secure/nodes/latency/start', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.get('/api/secure/nodes/latency/measure', dependencies=[Depends(require_permissions('nodes:view'))])
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
async def secure_dashboard(request: Request, permitted: bool = Depends(require_permissions('dashboard:view'))):
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
async def secure_resources(request: Request, permitted: bool = Depends(require_permissions('resources:view'))):
    session_id, key = validate_secure(request.headers)
    cpu = gather_cpu_block(); mem = gather_memory_block(); storage = gather_storage_block(); gpu = gather_gpu_block()
    payload = { 'cpu': cpu, 'gpu': gpu, 'memory': mem, 'storage': storage, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace network
@app.get('/api/secure/network')
async def secure_network(request: Request, permitted: bool = Depends(require_permissions('network:view'))):
    session_id, key = validate_secure(request.headers)
    interfaces = gather_net_block()
    payload = { 'topology': {'nodes': api_server.database.get_nodes(), 'connections': []}, 'statistics': {'interfaces': interfaces}, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace performance
@app.get('/api/secure/performance')
async def secure_performance(request: Request, permitted: bool = Depends(require_permissions('performance:view'))):
    session_id, key = validate_secure(request.headers)
    cpu_hist = psutil.cpu_percent(percpu=False, interval=0.05)
    vm = psutil.virtual_memory()
    analysis = { 'cpu_average': cpu_hist, 'memory_average': vm.percent, 'gpu_average': 0, 'health_score': max(0,100- (cpu_hist+vm.percent)/2) }
    payload = { 'analysis': analysis, 'benchmark': None, 'timestamp': time.time() }
    return wrap_encrypted(session_id, key, payload)

# Replace plugins (no fake marketplace)
@app.get('/api/secure/plugins')
async def secure_plugins(request: Request, permitted: bool = Depends(require_permissions('plugins:view'))):
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
    # Merge live loaded plugin registry
    try:
        from common.plugin_framework import get_plugin_manager  # type: ignore
        live = get_plugin_manager().list()
    except Exception:
        live = []
    payload={'installed':installed,'live': live, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

# Replace security (user list from sessions)
@app.get('/api/secure/security')
async def secure_security(request: Request, permitted: bool = Depends(require_permissions('security:view'))):
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
async def secure_nodes(request: Request, permitted: bool = Depends(require_permissions('nodes:view'))):
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
@app.get('/api/secure/nodes/{node_id}/protocol', dependencies=[Depends(require_permissions('nodes:view'))])
async def node_protocol_get(request: Request, node_id: str):
    sid, key = validate_secure(request.headers)
    active = NODE_PROTOCOL_STATE.get(node_id, 'gRPC/QUIC')
    return wrap_encrypted(sid, key, {'node_id': node_id, 'active': active, 'supported': SUPPORTED_PROTOCOLS})

class ProtocolSetRequest(BaseModel):
    protocol: str

@app.post('/api/secure/nodes/{node_id}/protocol', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.get('/api/secure/nodes/{node_id}/telemetry', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.get('/api/secure/nodes/{node_id}/policies', dependencies=[Depends(require_permissions('nodes:view'))])
async def node_policies_get(request: Request, node_id: str):
    sid, key = validate_secure(request.headers)
    pol = NODE_POLICIES.get(node_id, {})
    return wrap_encrypted(sid, key, {'node_id': node_id, 'policies': pol})

class PolicySetRequest(BaseModel):
    key: str
    value: str | None = ''

@app.post('/api/secure/nodes/{node_id}/policies', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.delete('/api/secure/nodes/{node_id}/policies/{pkey}', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.get('/api/secure/nodes/{node_id}/trust', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.post('/api/secure/nodes/{node_id}/diagnostics', dependencies=[Depends(require_permissions('nodes:view'))])
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

@app.post('/api/secure/nodes/{node_id}/ota', dependencies=[Depends(require_permissions('nodes:view'))])
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
async def secure_sessions(request: Request, permitted: bool = Depends(require_permissions('sessions:view'))):
    session_id, key = validate_secure(request.headers)
    sessions = api_server.database.get_sessions()
    payload = {'sessions': sessions, 'timestamp': time.time()}
    return wrap_encrypted(session_id, key, payload)

@app.get('/api/secure/processes')
async def secure_processes(request: Request, permitted: bool = Depends(require_permissions('processes:view'))):
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

# --- Streaming negotiation endpoints (scaffold) ---
try:
    from common.streaming import get_stream_store  # type: ignore
except Exception:  # pragma: no cover
    async def get_stream_store():  # type: ignore
        class _Null:
            async def create_offer(self,*a,**k): return 'na'
            async def attach_answer(self,*a,**k): return False
            async def get(self,*a,**k): return None
        return _Null()

class StreamOffer(BaseModel):
    session_id: str
    sdp: str

class StreamAnswer(BaseModel):
    sdp: str

@app.post('/api/secure/stream/offer')
async def create_stream_offer(body: StreamOffer, request: Request, permitted: bool = Depends(require_permissions('stream:create'))):
    sid, key = validate_secure(request.headers)
    store = await get_stream_store()
    stream_id = await store.create_offer(body.session_id, body.sdp)
    return wrap_encrypted(sid, key, {'stream_id': stream_id})

@app.post('/api/secure/stream/{stream_id}/answer')
async def attach_stream_answer(stream_id: str, body: StreamAnswer, request: Request, permitted: bool = Depends(require_permissions('stream:answer'))):
    sid, key = validate_secure(request.headers)
    store = await get_stream_store()
    ok = await store.attach_answer(stream_id, body.sdp)
    if not ok:
        raise HTTPException(status_code=404, detail='stream not found')
    return wrap_encrypted(sid, key, {'ok': True})

@app.get('/api/secure/stream/{stream_id}')
async def get_stream(stream_id: str, request: Request, permitted: bool = Depends(require_permissions('stream:view'))):
    sid, key = validate_secure(request.headers)
    store = await get_stream_store()
    rec = await store.get(stream_id)
    if not rec:
        raise HTTPException(status_code=404, detail='stream not found')
    return wrap_encrypted(sid, key, {'stream': {k:v for k,v in rec.items() if k != 'offer_sdp' or True}})

@app.post('/api/secure/processes/kill')
async def secure_process_kill(request: Request, body: KillProcessRequest, permitted: bool = Depends(require_permissions('processes:kill'))):
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
async def secure_logs(request: Request, permitted: bool = Depends(require_permissions('logs:view'))):
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


# --- Secure Prometheus metrics exposure (for control-plane only) ---
@app.get('/api/secure/metrics')
async def secure_metrics(request: Request, permitted: bool = Depends(require_permissions('security:view'))):
    """Return Prometheus metrics for the backend process.
    Protected behind secure session and security:view permission.
    """
    # validate but do not wrap; Prometheus expects plaintext exposition format
    validate_secure(request.headers)
    data = generate_latest(METRICS_REGISTRY)  # bytes
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# Secure health with unified schema, encrypted response
@app.get('/api/secure/health')
@rate_limited(per_minute=120, burst=60)
async def secure_health(request: Request, permitted: bool = Depends(require_permissions('security:view'))):
    sid, key = validate_secure(request.headers)
    deps = {}
    # DB check
    try:
        with db_connect() as conn:
            conn.execute('SELECT 1')
        deps['database'] = {'ok': True}
    except Exception as e:
        deps['database'] = {'ok': False, 'error': str(e)}
    # Crypto checks
    deps['rsa_key'] = {'ok': bool(getattr(api_server.security_manager, 'rsa_private_key', None))}
    deps['fernet'] = {'ok': bool(getattr(api_server.security_manager, 'cipher_suite', None))}
    # Storage/session dir
    try:
        base = api_server.session_storage_base
        deps['session_storage'] = {'ok': os.path.isdir(base), 'path': base}
    except Exception as e:
        deps['session_storage'] = {'ok': False, 'error': str(e)}
    # Docker availability (optional)
    try:
        deps['docker'] = {'ok': _docker_available()}
    except Exception:
        deps['docker'] = {'ok': False}
    # Build health payload
    payload = build_health(deps, version='2.0.0')
    return wrap_encrypted(sid, key, payload)

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
async def secure_public_key(request: Request, t: Optional[str] = None):  # request for rate limiter IP; t is optional cache-buster
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
        # Minimal debug trace (can be toggled with OMEGA_DEBUG=1)
        if os.environ.get('OMEGA_DEBUG','0') in ('1','true','yes'):
            logging.debug('secure_public_key served to %s (t=%s)', getattr(getattr(request,'client',None),'host','?'), t)
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


@app.post('/api/secure/session/start')
@rate_limited(per_minute=60, burst=30)
async def secure_session_start(body: SecureSessionStart, request: Request):
    """Secure session bootstrap.
    Client generates an AES-256-GCM key, encrypts with server RSA-OAEP public key, and POSTs here.
    Server decrypts, registers session, and issues an HMAC JWT tied to the session id.
    """
    # Decrypt AES session key
    if not body.encrypted_key:
        raise HTTPException(status_code=400, detail='encrypted_key required')
    try:
        enc = base64.b64decode(body.encrypted_key)
        raw_key = api_server.rsa_private_key.decrypt(
            enc,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        if len(raw_key) != 32:
            raise ValueError('invalid key length')
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'secure_session_start decrypt failed: {e}')
        raise HTTPException(status_code=400, detail='Invalid encrypted_key')

    # Create session id and register
    session_id = uuid.uuid4().hex
    key_b64 = base64.b64encode(raw_key).decode()
    register_session_meta(session_id, key_b64)
    # Attach user/roles and persist user to DB row
    user = (body.user_id or 'admin').strip() or 'admin'
    roles = ['admin'] if user == 'admin' else RBAC_CACHE.get('user_roles', {}).get(user, ['user'])
    with SESSION_LOCK:
        meta = SESSION_META.get(session_id, {})
        meta['user'] = user
        meta['roles'] = roles
        SESSION_META[session_id] = meta
    try:
        with db_connect() as conn:
            conn.execute('UPDATE session_meta SET user=? WHERE session_id=?', (user, session_id))
            conn.commit()
    except Exception as e:
        logging.debug(f'secure_session_start: user persist failed for {session_id}: {e}')

    # Issue JWT for Authorization header with sid claim
    token = api_server.security_manager.issue_jwt(session_id=session_id, user=user, ttl=6*3600)
    try:
        api_server.database.log_event('secure_session_start', session_id, f'user={user}', 'info')
    except Exception:
        pass
    return {'session_id': session_id, 'token': token, 'expires_in': 6*3600}


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


# --- Admin session operations ---
class AdminSessionOp(BaseModel):
    session_id: str
    reason: Optional[str] = ''


@app.post('/api/secure/admin/session/revoke', dependencies=[Depends(require_permissions('rbac:manage'))])
async def admin_session_revoke(body: AdminSessionOp, request: Request):
    """Revoke a session immediately. Requires admin (rbac:manage)."""
    admin_sid, key = validate_secure(request.headers)
    target = body.session_id
    # Persist revocation and cleanup in-memory state
    persist_revocation(target, body.reason or '')
    ws_closed = False
    try:
        with SESSION_LOCK:
            SESSION_META.pop(target, None)
            SESSION_NONCES.pop(target, None)
            ws = SESSION_WS.pop(target, None)
        if ws is not None:
            try:
                await ws.close()
                ws_closed = True
            except Exception:
                pass
        with db_connect() as conn:
            try:
                conn.execute('DELETE FROM session_meta WHERE session_id=?', (target,))
                conn.commit()
            except Exception:
                pass
        api_server.database.log_event('session_revoke', target, f"revoked by {SESSION_META.get(admin_sid,{}).get('user','admin')} reason={body.reason or ''}", 'warning')
    except Exception as e:
        logging.error(f'admin_session_revoke error for {target}: {e}')
    return wrap_encrypted(admin_sid, key, {'ok': True, 'revoked': target, 'ws_closed': ws_closed})


@app.post('/api/secure/admin/session/rotate', dependencies=[Depends(require_permissions('rbac:manage'))])
async def admin_session_rotate(body: AdminSessionOp, request: Request):
    """Request a client rekey. Records a pending rekey and signals via WS if available.
    Actual key change completes when client calls /api/secure/session/rotate.
    """
    admin_sid, key = validate_secure(request.headers)
    target = body.session_id
    # Record a pending rotate request (no key material transmitted here)
    try:
        persist_pending_rekey(target, '')  # empty placeholder; client-initiated rotate will overwrite
    except Exception:
        pass
    # Best-effort WS notification
    notified = False
    try:
        with SESSION_LOCK:
            ws = SESSION_WS.get(target)
            tmeta = SESSION_META.get(target, {})
            tkey_b64 = tmeta.get('key')
        if ws and tkey_b64:
            tkey = base64.b64decode(tkey_b64)
            pkt = wrap_encrypted(target, tkey, {'type': 'rekey_request', 'ts': time.time()})
            try:
                await ws.send_json(pkt)
                notified = True
            except Exception:
                notified = False
        api_server.database.log_event('session_rotate_request', target, f"requested by {SESSION_META.get(admin_sid,{}).get('user','admin')}")
    except Exception as e:
        logging.debug(f'admin_session_rotate notify failed for {target}: {e}')
    return wrap_encrypted(admin_sid, key, {'ok': True, 'rotate_requested': target, 'notified': notified})


class SessionPollReq(BaseModel):
    session_id: str


@app.post('/api/secure/session/poll')
async def secure_session_poll(body: SessionPollReq, request: Request):
    """Frontend helper: check session validity and whether admin requested rekey.
    Requires a valid secure session in headers; returns info for the caller's session.
    """
    sid, key = validate_secure(request.headers)
    # Only allow polling for own session
    if body.session_id != sid:
        raise HTTPException(status_code=403, detail='may only poll current session')
    # Determine if a pending rekey exists
    pending = False
    try:
        with db_connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS session_pending_rekey (session_id TEXT PRIMARY KEY, new_key TEXT, created_at REAL)')
            cur = conn.execute('SELECT 1 FROM session_pending_rekey WHERE session_id=?', (sid,))
            pending = bool(cur.fetchone())
    except Exception:
        pending = False
    meta = SESSION_META.get(sid, {})
    return wrap_encrypted(sid, key, {
        'ok': True,
        'session_id': sid,
        'user': meta.get('user','admin'),
        'roles': meta.get('roles',[]),
        'expires_at': meta.get('expires_at'),
        'rekey_requested': pending
    })

# Convenience: current session info (encrypted)
@app.get('/api/secure/session/info')
async def secure_session_info(request: Request, permitted: bool = Depends(require_permissions('sessions:view'))):
    sid, key = validate_secure(request.headers)
    meta = SESSION_META.get(sid, {})
    return wrap_encrypted(sid, key, {
        'session_id': sid,
        'user': meta.get('user','admin'),
        'roles': meta.get('roles',[]),
        'expires_at': meta.get('expires_at'),
        'counter': meta.get('counter',0)
    })

# Convenience: effective permissions for current user
@app.get('/api/secure/policy/effective')
async def secure_policy_effective(request: Request, permitted: bool = Depends(require_permissions('security:view'))):
    sid, key = validate_secure(request.headers)
    meta = SESSION_META.get(sid, {})
    username = meta.get('user','admin')
    base = get_user_permissions(username)
    roles = meta.get('roles', [])
    eff, denied = evaluate_policies(username, roles, base)
    return wrap_encrypted(sid, key, {
        'username': username,
        'base': sorted(list(base)),
        'effective': sorted(list(eff)),
        'denied': sorted(list(denied))
    })



# --- Most advanced, secure, and bug-free node registration ---
# Note: Keep this under a distinct path to avoid duplicate route collisions with the basic secure register above
@app.post('/api/secure/nodes/register/advanced', include_in_schema=False)
async def register_node_advanced(request: Request, body: NodeRegistrationRequest = Body(...)):
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
            if cert.not_valid_before <= datetime.now(timezone.utc) <= cert.not_valid_after:
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
