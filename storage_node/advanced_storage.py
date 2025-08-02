"""
Omega Super Desktop Console - Advanced Storage Node
Initial prototype distributed storage with intelligent tiering and replication.
"""

import asyncio
import logging
import json
import time
import uuid
import os
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import aiohttp
from aiohttp import web
import numpy as np

# Custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.models import setup_logging

# Configuration
NODE_ID = f"storage_node_{uuid.uuid4().hex[:8]}"
NODE_TYPE = "storage_node"
CONTROL_NODE_URL = "http://localhost:8443"
AUTH_TOKEN = None

# Storage configuration
STORAGE_ROOT = Path("/tmp/omega_storage")
PRIMARY_STORAGE = STORAGE_ROOT / "primary"
SECONDARY_STORAGE = STORAGE_ROOT / "secondary"
CACHE_STORAGE = STORAGE_ROOT / "cache"
REPLICATION_FACTOR = 3

class StorageEngine:
    def __init__(self):
        self.logger = setup_logging(f"StorageEngine-{NODE_ID}")
        self.blocks = {}  # Block metadata
        self.replication_map = {}  # Block -> [node_ids]
        self.access_patterns = {}  # Access frequency tracking
        self.cache_policy = "LRU"  # LRU, LFU, or AI
        self.compression_enabled = True
        self.encryption_key = self._generate_encryption_key()
        
        # Initialize storage directories
        self._init_storage_dirs()
        
    def _init_storage_dirs(self):
        """Initialize storage directory structure"""
        for path in [PRIMARY_STORAGE, SECONDARY_STORAGE, CACHE_STORAGE]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories for different storage tiers
        (PRIMARY_STORAGE / "hot").mkdir(exist_ok=True)
        (PRIMARY_STORAGE / "warm").mkdir(exist_ok=True)
        (SECONDARY_STORAGE / "cold").mkdir(exist_ok=True)
        (CACHE_STORAGE / "l1").mkdir(exist_ok=True)
        (CACHE_STORAGE / "l2").mkdir(exist_ok=True)
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for data at rest"""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()
    
    async def store_block(self, block_id: str, data: bytes, tier: str = "hot") -> Dict[str, Any]:
        """Store a data block with automatic tiering"""
        start_time = time.time()
        
        # Compress data if enabled
        if self.compression_enabled:
            import gzip
            compressed_data = gzip.compress(data)
            compression_ratio = len(data) / len(compressed_data)
        else:
            compressed_data = data
            compression_ratio = 1.0
        
        # Encrypt data
        from cryptography.fernet import Fernet
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(compressed_data)
        
        # Determine storage path based on tier
        if tier == "hot":
            storage_path = PRIMARY_STORAGE / "hot" / f"{block_id}.dat"
        elif tier == "warm":
            storage_path = PRIMARY_STORAGE / "warm" / f"{block_id}.dat"
        else:  # cold
            storage_path = SECONDARY_STORAGE / "cold" / f"{block_id}.dat"
        
        # Write data
        async with aiofiles.open(storage_path, 'wb') as f:
            await f.write(encrypted_data)
        
        # Calculate checksum
        checksum = hashlib.sha256(data).hexdigest()
        
        # Store metadata
        metadata = {
            "block_id": block_id,
            "size": len(data),
            "compressed_size": len(compressed_data),
            "encrypted_size": len(encrypted_data),
            "compression_ratio": compression_ratio,
            "checksum": checksum,
            "tier": tier,
            "created_at": datetime.utcnow().isoformat(),
            "access_count": 0,
            "last_access": datetime.utcnow().isoformat(),
            "storage_path": str(storage_path)
        }
        
        self.blocks[block_id] = metadata
        self.access_patterns[block_id] = {
            "access_times": [time.time()],
            "access_frequency": 1,
            "last_prediction": time.time()
        }
        
        duration = time.time() - start_time
        self.logger.info(f"Stored block {block_id} in {duration:.3f}s (tier: {tier})")
        
        return {
            "block_id": block_id,
            "size": len(data),
            "tier": tier,
            "compression_ratio": compression_ratio,
            "storage_time": duration,
            "checksum": checksum
        }
    
    async def retrieve_block(self, block_id: str) -> Optional[bytes]:
        """Retrieve a data block with caching"""
        if block_id not in self.blocks:
            return None
        
        metadata = self.blocks[block_id]
        storage_path = Path(metadata["storage_path"])
        
        # Update access patterns
        self._update_access_pattern(block_id)
        
        # Check cache first
        cached_data = await self._check_cache(block_id)
        if cached_data:
            self.logger.debug(f"Cache hit for block {block_id}")
            return cached_data
        
        # Read from storage
        start_time = time.time()
        try:
            async with aiofiles.open(storage_path, 'rb') as f:
                encrypted_data = await f.read()
            
            # Decrypt data
            from cryptography.fernet import Fernet
            fernet = Fernet(self.encryption_key)
            compressed_data = fernet.decrypt(encrypted_data)
            
            # Decompress if needed
            if self.compression_enabled:
                import gzip
                data = gzip.decompress(compressed_data)
            else:
                data = compressed_data
            
            # Verify checksum
            checksum = hashlib.sha256(data).hexdigest()
            if checksum != metadata["checksum"]:
                self.logger.error(f"Checksum mismatch for block {block_id}")
                return None
            
            # Cache for future access
            await self._cache_block(block_id, data)
            
            duration = time.time() - start_time
            self.logger.debug(f"Retrieved block {block_id} in {duration:.3f}s")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving block {block_id}: {e}")
            return None
    
    async def delete_block(self, block_id: str) -> bool:
        """Delete a data block"""
        if block_id not in self.blocks:
            return False
        
        metadata = self.blocks[block_id]
        storage_path = Path(metadata["storage_path"])
        
        try:
            # Remove file
            if storage_path.exists():
                storage_path.unlink()
            
            # Remove from cache
            await self._evict_from_cache(block_id)
            
            # Remove metadata
            del self.blocks[block_id]
            if block_id in self.access_patterns:
                del self.access_patterns[block_id]
            
            self.logger.info(f"Deleted block {block_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting block {block_id}: {e}")
            return False
    
    def _update_access_pattern(self, block_id: str):
        """Update access patterns for intelligent caching"""
        current_time = time.time()
        
        if block_id in self.access_patterns:
            pattern = self.access_patterns[block_id]
            pattern["access_times"].append(current_time)
            pattern["access_frequency"] += 1
            
            # Keep only last 100 access times
            if len(pattern["access_times"]) > 100:
                pattern["access_times"] = pattern["access_times"][-100:]
        
        # Update block metadata
        if block_id in self.blocks:
            self.blocks[block_id]["access_count"] += 1
            self.blocks[block_id]["last_access"] = datetime.utcnow().isoformat()
    
    async def _check_cache(self, block_id: str) -> Optional[bytes]:
        """Check if block is in cache"""
        cache_paths = [
            CACHE_STORAGE / "l1" / f"{block_id}.cache",
            CACHE_STORAGE / "l2" / f"{block_id}.cache"
        ]
        
        for cache_path in cache_paths:
            if cache_path.exists():
                try:
                    async with aiofiles.open(cache_path, 'rb') as f:
                        return await f.read()
                except:
                    # Remove corrupted cache file
                    cache_path.unlink(missing_ok=True)
        
        return None
    
    async def _cache_block(self, block_id: str, data: bytes):
        """Cache block data for faster access"""
        cache_path = CACHE_STORAGE / "l1" / f"{block_id}.cache"
        
        try:
            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(data)
        except Exception as e:
            self.logger.debug(f"Cache write error for {block_id}: {e}")
    
    async def _evict_from_cache(self, block_id: str):
        """Remove block from cache"""
        cache_paths = [
            CACHE_STORAGE / "l1" / f"{block_id}.cache",
            CACHE_STORAGE / "l2" / f"{block_id}.cache"
        ]
        
        for cache_path in cache_paths:
            cache_path.unlink(missing_ok=True)
    
    async def optimize_storage(self):
        """Optimize storage based on access patterns"""
        self.logger.info("Starting storage optimization...")
        
        current_time = time.time()
        hot_threshold = current_time - 3600  # 1 hour
        warm_threshold = current_time - 86400  # 24 hours
        
        for block_id, metadata in self.blocks.items():
            if block_id not in self.access_patterns:
                continue
            
            pattern = self.access_patterns[block_id]
            last_access = max(pattern["access_times"]) if pattern["access_times"] else 0
            
            # Determine optimal tier
            if last_access > hot_threshold and pattern["access_frequency"] > 10:
                target_tier = "hot"
            elif last_access > warm_threshold:
                target_tier = "warm"
            else:
                target_tier = "cold"
            
            # Move if necessary
            if metadata["tier"] != target_tier:
                await self._move_to_tier(block_id, target_tier)
        
        self.logger.info("Storage optimization complete")
    
    async def _move_to_tier(self, block_id: str, target_tier: str):
        """Move block to different storage tier"""
        if block_id not in self.blocks:
            return
        
        metadata = self.blocks[block_id]
        current_path = Path(metadata["storage_path"])
        
        # Determine new path
        if target_tier == "hot":
            new_path = PRIMARY_STORAGE / "hot" / f"{block_id}.dat"
        elif target_tier == "warm":
            new_path = PRIMARY_STORAGE / "warm" / f"{block_id}.dat"
        else:  # cold
            new_path = SECONDARY_STORAGE / "cold" / f"{block_id}.dat"
        
        try:
            # Move file
            shutil.move(str(current_path), str(new_path))
            
            # Update metadata
            metadata["tier"] = target_tier
            metadata["storage_path"] = str(new_path)
            
            self.logger.debug(f"Moved block {block_id} to {target_tier} tier")
            
        except Exception as e:
            self.logger.error(f"Error moving block {block_id} to {target_tier}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        total_blocks = len(self.blocks)
        total_size = sum(meta["size"] for meta in self.blocks.values())
        compressed_size = sum(meta["compressed_size"] for meta in self.blocks.values())
        
        # Tier distribution
        tier_stats = {"hot": 0, "warm": 0, "cold": 0}
        for meta in self.blocks.values():
            tier_stats[meta["tier"]] += 1
        
        # Cache stats
        cache_l1_size = sum(f.stat().st_size for f in (CACHE_STORAGE / "l1").glob("*.cache"))
        cache_l2_size = sum(f.stat().st_size for f in (CACHE_STORAGE / "l2").glob("*.cache"))
        
        return {
            "total_blocks": total_blocks,
            "total_size_bytes": total_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": total_size / compressed_size if compressed_size > 0 else 1.0,
            "tier_distribution": tier_stats,
            "cache_l1_size": cache_l1_size,
            "cache_l2_size": cache_l2_size,
            "storage_efficiency": (total_size - compressed_size) / total_size if total_size > 0 else 0.0
        }

class ReplicationManager:
    def __init__(self, storage_engine: StorageEngine):
        self.storage_engine = storage_engine
        self.logger = setup_logging(f"ReplicationManager-{NODE_ID}")
        self.peer_nodes = []
        self.replication_factor = REPLICATION_FACTOR
        
    def add_peer_node(self, node_id: str, endpoint: str):
        """Add peer storage node for replication"""
        self.peer_nodes.append({
            "node_id": node_id,
            "endpoint": endpoint,
            "status": "active",
            "last_sync": time.time()
        })
        
    async def replicate_block(self, block_id: str, data: bytes) -> List[str]:
        """Replicate block to peer nodes"""
        successful_replicas = []
        
        # Select peer nodes for replication
        target_peers = self.peer_nodes[:self.replication_factor-1]  # -1 for local copy
        
        for peer in target_peers:
            try:
                success = await self._send_to_peer(peer, block_id, data)
                if success:
                    successful_replicas.append(peer["node_id"])
            except Exception as e:
                self.logger.error(f"Replication to {peer['node_id']} failed: {e}")
        
        return successful_replicas
    
    async def _send_to_peer(self, peer: Dict[str, Any], block_id: str, data: bytes) -> bool:
        """Send block data to peer node"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{peer['endpoint']}/api/v1/storage/replicate",
                    json={
                        "block_id": block_id,
                        "data": data.hex(),  # Hex encode for JSON
                        "source_node": NODE_ID
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Error sending to peer {peer['node_id']}: {e}")
            return False
    
    async def verify_replicas(self, block_id: str) -> Dict[str, bool]:
        """Verify block replicas across peer nodes"""
        verification_results = {}
        
        for peer in self.peer_nodes:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{peer['endpoint']}/api/v1/storage/verify/{block_id}",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        verification_results[peer["node_id"]] = response.status == 200
            except Exception:
                verification_results[peer["node_id"]] = False
        
        return verification_results

class OmegaStorageNode:
    def __init__(self):
        self.logger = setup_logging(f"StorageNode-{NODE_ID}")
        self.storage_engine = StorageEngine()
        self.replication_manager = ReplicationManager(self.storage_engine)
        self.running = False
        
    async def start(self):
        """Start the storage node"""
        self.running = True
        self.logger.info(f"Starting Omega Storage Node {NODE_ID}")
        
        # Authenticate with control node
        await self._authenticate()
        
        # Register with control node
        await self._register_node()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._start_api_server())
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
            "username": "storage_node",
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
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
    
    async def _register_node(self):
        """Register with control node"""
        stats = self.storage_engine.get_storage_stats()
        
        node_info = {
            "node_id": NODE_ID,
            "node_type": NODE_TYPE,
            "status": "active",
            "resources": {
                "primary_storage": "1TB NVMe SSD",
                "secondary_storage": "4TB HDD",
                "network_storage": "NAS compatible",
                "replication": f"{REPLICATION_FACTOR}x",
                "total_blocks": stats["total_blocks"],
                "total_size_gb": stats["total_size_bytes"] / (1024**3),
                "compression_ratio": stats["compression_ratio"]
            },
            "performance_metrics": {
                "avg_read_latency_ms": 2.5,
                "avg_write_latency_ms": 5.0,
                "storage_efficiency": stats["storage_efficiency"]
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
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                stats = self.storage_engine.get_storage_stats()
                
                heartbeat_data = {
                    "node_id": NODE_ID,
                    "timestamp": time.time(),
                    "status": "active",
                    "storage_stats": stats
                }
                
                headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{CONTROL_NODE_URL}/api/v1/nodes/{NODE_ID}/heartbeat",
                        json=heartbeat_data,
                        headers=headers
                    ) as response:
                        pass
                
            except Exception as e:
                self.logger.debug(f"Heartbeat error: {e}")
            
            await asyncio.sleep(30)
    
    async def _optimization_loop(self):
        """Periodic storage optimization"""
        while self.running:
            try:
                await self.storage_engine.optimize_storage()
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
            
            await asyncio.sleep(3600)  # Every hour
    
    async def _health_monitor(self):
        """Monitor storage health"""
        while self.running:
            try:
                # Check disk space
                for path in [PRIMARY_STORAGE, SECONDARY_STORAGE, CACHE_STORAGE]:
                    usage = shutil.disk_usage(path)
                    usage_percent = (usage.used / usage.total) * 100
                    
                    if usage_percent > 90:
                        self.logger.warning(f"High disk usage on {path}: {usage_percent:.1f}%")
                    elif usage_percent > 95:
                        self.logger.error(f"Critical disk usage on {path}: {usage_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _start_api_server(self):
        """Start internal API server for storage operations"""
        
        app = web.Application()
        
        # Storage API routes
        app.router.add_post('/api/v1/storage/store', self._handle_store)
        app.router.add_get('/api/v1/storage/retrieve/{block_id}', self._handle_retrieve)
        app.router.add_delete('/api/v1/storage/delete/{block_id}', self._handle_delete)
        app.router.add_post('/api/v1/storage/replicate', self._handle_replicate)
        app.router.add_get('/api/v1/storage/verify/{block_id}', self._handle_verify)
        app.router.add_get('/api/v1/storage/stats', self._handle_stats)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Start on different port than control node
        site = web.TCPSite(runner, '0.0.0.0', 8444)
        await site.start()
        
        self.logger.info("Storage API server started on port 8444")
        
        # Keep server running
        while self.running:
            await asyncio.sleep(1)
        
        await runner.cleanup()
    
    async def _handle_store(self, request):
        """Handle storage request"""
        data = await request.json()
        block_id = data.get("block_id")
        block_data = bytes.fromhex(data.get("data", ""))
        tier = data.get("tier", "hot")
        
        result = await self.storage_engine.store_block(block_id, block_data, tier)
        
        # Replicate to peers
        if data.get("replicate", True):
            replicas = await self.replication_manager.replicate_block(block_id, block_data)
            result["replicas"] = replicas
        
        return web.json_response(result)
    
    async def _handle_retrieve(self, request):
        """Handle retrieval request"""
        block_id = request.match_info["block_id"]
        
        data = await self.storage_engine.retrieve_block(block_id)
        if data:
            return web.json_response({
                "block_id": block_id,
                "data": data.hex(),
                "size": len(data)
            })
        else:
            return web.json_response({"error": "Block not found"}, status=404)
    
    async def _handle_delete(self, request):
        """Handle deletion request"""
        block_id = request.match_info["block_id"]
        
        success = await self.storage_engine.delete_block(block_id)
        if success:
            return web.json_response({"status": "deleted"})
        else:
            return web.json_response({"error": "Block not found"}, status=404)
    
    async def _handle_replicate(self, request):
        """Handle replication request from peer"""
        data = await request.json()
        block_id = data.get("block_id")
        block_data = bytes.fromhex(data.get("data", ""))
        
        result = await self.storage_engine.store_block(block_id, block_data, "warm")
        return web.json_response(result)
    
    async def _handle_verify(self, request):
        """Handle verification request"""
        block_id = request.match_info["block_id"]
        
        if block_id in self.storage_engine.blocks:
            return web.json_response({"exists": True})
        else:
            return web.json_response({"exists": False}, status=404)
    
    async def _handle_stats(self, request):
        """Handle stats request"""
        stats = self.storage_engine.get_storage_stats()
        return web.json_response(stats)
    
    async def _cleanup(self):
        """Cleanup on shutdown"""
        self.running = False
        self.logger.info("Storage node shutting down")

async def main():
    node = OmegaStorageNode()
    await node.start()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
