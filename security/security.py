"""
Omega Super Desktop Console - Security Module
Initial prototype authentication, encryption, and access control.
"""

import ssl
import jwt
import os
from typing import Dict, Any

# Certificate-based mutual TLS

def create_ssl_context(certfile: str, keyfile: str, cafile: str) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    context.load_verify_locations(cafile=cafile)
    context.verify_mode = ssl.CERT_REQUIRED
    return context

# JWT authentication
SECRET_KEY = os.environ.get("OMEGA_JWT_SECRET", "supersecretkey")

def generate_jwt(payload: Dict[str, Any]) -> str:
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_jwt(token: str) -> Dict[str, Any]:
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

# Role-based access control
class RBAC:
    def __init__(self):
        self.roles = {"admin": ["all"], "user": ["read", "execute"]}

    def check_access(self, role: str, action: str) -> bool:
        return action in self.roles.get(role, []) or "all" in self.roles.get(role, [])

# Encryption utilities
from cryptography.fernet import Fernet

def generate_key() -> bytes:
    return Fernet.generate_key()

def encrypt_data(data: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_data(token: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.decrypt(token)
