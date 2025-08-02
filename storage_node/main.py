"""
Omega Super Desktop Console - Storage Node
Initial prototype distributed storage and data management agent.
"""

import asyncio
import logging
import os
import socket
import ssl
import json
from typing import Dict, Any

NODE_ID = "storage_node_1"
NODE_TYPE = "storage_node"
STORAGE_CAPACITY = "1TB"  # Storage capacity
CONTROL_NODE_HOST = "localhost"
CONTROL_NODE_PORT = 8443

# Security: TLS setup
ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
ssl_context.load_cert_chain(certfile="../security/certs/storage_node.crt", keyfile="../security/certs/storage_node.key")
ssl_context.load_verify_locations(cafile="../security/certs/ca.crt")
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Simulated resources
resources = {
    "primary_storage": "1TB NVMe SSD",
    "secondary_storage": "4TB HDD",
    "network_storage": "NAS compatible",
    "replication": "3x"
}

async def register_with_control():
    """Register this storage node with the control center"""
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
            "capacity": STORAGE_CAPACITY,
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

async def heartbeat_loop():
    """Send periodic heartbeats to control center"""
    while True:
        try:
            print(f"Storage node {NODE_ID} is running...")
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
        except Exception as e:
            print(f"Heartbeat error: {e}")
            await asyncio.sleep(30)

async def main():
    await register_with_control()
    # Initial prototype: listen for storage requests, manage data
    while True:
        await asyncio.sleep(5)
        logging.info("Storage node heartbeat.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
