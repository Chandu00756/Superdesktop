"""
Frontend Integration Module for Omega Control Center
Handles encrypted communication between frontend and backend
"""

import asyncio
import json
import logging
import time
import base64
import hmac
import hashlib
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import aiohttp
import websockets
from cryptography.fernet import Fernet


@dataclass
class EncryptedMessage:
    payload: str
    signature: str
    timestamp: float
    nonce: str


class FrontendCrypto:
    def __init__(self, master_key: bytes = None):
        self.master_key = master_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.session_keys: Dict[str, bytes] = {}
    
    def decrypt_message(self, encrypted_msg: Dict[str, Any]) -> str:
        message = EncryptedMessage(**encrypted_msg)
        
        expected_signature = hmac.new(
            self.master_key,
            message.payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, message.signature):
            raise ValueError("Invalid message signature")
        
        if time.time() - message.timestamp > 300:
            raise ValueError("Message too old")
        
        encrypted_payload = base64.b64decode(message.payload.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_payload)
        payload_data = json.loads(decrypted_data.decode())
        
        return payload_data["data"]
    
    def encrypt_message(self, data: str) -> Dict[str, Any]:
        nonce = base64.b64encode(bytes(range(16))).decode()
        timestamp = time.time()
        
        payload_data = {
            "data": data,
            "timestamp": timestamp,
            "nonce": nonce
        }
        
        encrypted_payload = self.cipher_suite.encrypt(json.dumps(payload_data).encode())
        payload_b64 = base64.b64encode(encrypted_payload).decode()
        
        signature = hmac.new(
            self.master_key,
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "payload": payload_b64,
            "signature": signature,
            "timestamp": timestamp,
            "nonce": nonce
        }


class BackendConnector:
    def __init__(self, base_url: str = "http://127.0.0.1:8443"):
        self.base_url = base_url
        self.crypto = FrontendCrypto()
        self.session = None
        self.auth_token = None
        self.websocket = None
        self.update_callbacks: Dict[str, Callable] = {}
        
    async def start(self):
        self.session = aiohttp.ClientSession()
        logging.info("Backend connector started")
    
    async def stop(self):
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
        logging.info("Backend connector stopped")
    
    async def authenticate(self, username: str, password: str) -> bool:
        try:
            async with self.session.post(f"{self.base_url}/api/auth/login", json={
                "username": username,
                "password": password
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data["token"]
                    return True
                return False
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return False
    
    async def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {json.dumps(self.auth_token)}"
            
            async with self.session.request(
                method,
                f"{self.base_url}{endpoint}",
                json=data,
                headers=headers
            ) as response:
                if response.status == 200:
                    encrypted_response = await response.json()
                    decrypted_data = self.crypto.decrypt_message(encrypted_response)
                    return json.loads(decrypted_data)
                else:
                    logging.error(f"Request failed: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Request error: {e}")
            return None
    
    async def connect_websocket(self):
        try:
            self.websocket = await websockets.connect("ws://127.0.0.1:8443/ws/realtime")
            asyncio.create_task(self.websocket_handler())
            logging.info("WebSocket connected")
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
    
    async def websocket_handler(self):
        try:
            async for message in self.websocket:
                try:
                    encrypted_msg = json.loads(message)
                    decrypted_data = self.crypto.decrypt_message(encrypted_msg)
                    update_data = json.loads(decrypted_data)
                    
                    update_type = update_data.get("type", "unknown")
                    if update_type in self.update_callbacks:
                        self.update_callbacks[update_type](update_data)
                        
                except Exception as e:
                    logging.error(f"WebSocket message processing error: {e}")
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
    
    def register_update_callback(self, update_type: str, callback: Callable):
        self.update_callbacks[update_type] = callback
    
    async def get_dashboard_data(self) -> Optional[Dict]:
        return await self.make_request("/api/dashboard/metrics")
    
    async def get_nodes_data(self) -> Optional[Dict]:
        return await self.make_request("/api/nodes")
    
    async def get_sessions_data(self) -> Optional[Dict]:
        return await self.make_request("/api/sessions")
    
    async def get_resources_data(self) -> Optional[Dict]:
        return await self.make_request("/api/resources")
    
    async def get_network_data(self) -> Optional[Dict]:
        return await self.make_request("/api/network")
    
    async def get_performance_data(self) -> Optional[Dict]:
        return await self.make_request("/api/performance")
    
    async def get_security_data(self) -> Optional[Dict]:
        return await self.make_request("/api/security")
    
    async def get_plugins_data(self) -> Optional[Dict]:
        return await self.make_request("/api/plugins")
    
    async def create_session(self, user_id: str, application: str, cpu_cores: int = 4, gpu_units: int = 1, memory_gb: int = 8) -> Optional[Dict]:
        return await self.make_request("/api/sessions/create", "POST", {
            "user_id": user_id,
            "application": application,
            "cpu_cores": cpu_cores,
            "gpu_units": gpu_units,
            "memory_gb": memory_gb
        })
    
    async def terminate_session(self, session_id: str) -> Optional[Dict]:
        return await self.make_request(f"/api/sessions/{session_id}", "DELETE")
    
    async def discover_nodes(self) -> Optional[Dict]:
        return await self.make_request("/api/actions/discover_nodes", "POST")
    
    async def run_benchmark(self) -> Optional[Dict]:
        return await self.make_request("/api/actions/run_benchmark", "POST")
    
    async def health_check(self) -> Optional[Dict]:
        return await self.make_request("/api/actions/health_check", "POST")


backend_connector = BackendConnector()


async def init_backend_connection():
    await backend_connector.start()
    auth_success = await backend_connector.authenticate("admin", "omega123")
    if auth_success:
        await backend_connector.connect_websocket()
        logging.info("Backend connection initialized successfully")
        return True
    else:
        logging.error("Backend authentication failed")
        return False


if __name__ == "__main__":
    async def test_connection():
        await init_backend_connection()
        
        dashboard_data = await backend_connector.get_dashboard_data()
        print("Dashboard data:", dashboard_data)
        
        await backend_connector.stop()
    
    asyncio.run(test_connection())
