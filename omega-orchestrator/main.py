"""
Omega Super Desktop Console - Ω-Orchestrator Service
Initial prototype node discovery, heartbeat, placement, and rolling upgrades.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import uuid
import hashlib

from fastapi import FastAPI, WebSocket, BackgroundTasks
from pydantic import BaseModel
import os as _os
_os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import aioredis
import asyncpg
from kubernetes import client, config
import etcd3

# Configuration
CLUSTER_NAME = "omega-cluster-01"
ETCD_ENDPOINTS = ["localhost:2379"]
REDIS_URL = "redis://localhost:6379"
POSTGRES_URL = "postgresql://omega:omega@localhost/omega_orchestrator"

@dataclass
class NodeSpec:
    node_id: str
    node_type: str  # cpu_node, gpu_node, storage_node, hybrid_node
    resources: Dict[str, Any]
    status: str  # active, inactive, draining, failed
    last_heartbeat: datetime
    labels: Dict[str, str]
    annotations: Dict[str, str]
    network_config: Dict[str, Any]
    
class PlacementRequest(BaseModel):
    session_id: str
    resource_requirements: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    preferences: List[Dict[str, Any]]
    
class PlacementDecision(BaseModel):
    session_id: str
    selected_nodes: List[str]
    resource_allocation: Dict[str, Dict[str, Any]]
    placement_score: float
    reasoning: str

class OmegaOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, NodeSpec] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.placement_history: List[PlacementDecision] = []
        self.cluster_state = "initializing"
        self.redis_client = None
        self.postgres_pool = None
        self.etcd_client = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize orchestrator components"""
        # Redis for fast lookups
        try:
            # Use centralized helper which tries aioredis, redis.asyncio, then an in-memory stub.
            from utils.redis_helper import get_redis_client
            self.redis_client = await get_redis_client(REDIS_URL)
        except Exception as e:
            self.logger.warning(f"Redis client helper failed, using in-memory fallback: {e}")
            class _InMemoryRedisStub:
                def __init__(self):
                    self._store = {}
                async def hset(self, key, mapping=None, **kwargs):
                    self._store[key] = mapping or kwargs
                async def delete(self, key):
                    self._store.pop(key, None)
                async def hgetall(self, key):
                    return self._store.get(key, {})
            self.redis_client = _InMemoryRedisStub()
        
        # PostgreSQL for persistent storage (fall back to SQLite if not available)
        try:
            self.postgres_pool = await asyncpg.create_pool(POSTGRES_URL)
        except Exception as e:
            self.logger.warning(f"Postgres unavailable ({e}), falling back to local SQLite")
            try:
                import aiosqlite
                # Use a simple aiosqlite-based thin pool wrapper
                class _SQLitePool:
                    def __init__(self, path):
                        self.path = path
                    async def acquire(self):
                        return await aiosqlite.connect(self.path)
                    async def __aenter__(self):
                        return await aiosqlite.connect(self.path)
                    async def __aexit__(self, exc_type, exc, tb):
                        pass
                sqlite_path = _os.path.join(_os.getcwd(), 'omega_orchestrator.db')
                self.postgres_pool = _SQLitePool(sqlite_path)
            except Exception:
                self.logger.error("No local DB available for orchestrator persistence")
                self.postgres_pool = None
        
        # etcd for distributed consensus
        self.etcd_client = etcd3.client(host='localhost', port=2379)
        
        # Initialize database schema
        await self._init_database()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._cluster_optimization())
        asyncio.create_task(self._metrics_collection())
        
        self.cluster_state = "active"
        self.logger.info("Omega Orchestrator initialized successfully")
    
    async def _init_database(self):
        """Initialize database schema"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id VARCHAR(64) PRIMARY KEY,
                    node_type VARCHAR(32) NOT NULL,
                    resources JSONB,
                    status VARCHAR(16),
                    last_heartbeat TIMESTAMP WITH TIME ZONE,
                    labels JSONB,
                    annotations JSONB,
                    network_config JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS placement_decisions (
                    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(64) NOT NULL,
                    selected_nodes JSONB,
                    resource_allocation JSONB,
                    placement_score FLOAT,
                    reasoning TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_placement_session ON placement_decisions(session_id);
            ''')
    
    async def register_node(self, node_spec: NodeSpec) -> bool:
        """Register a new node in the cluster"""
        try:
            # Validate node specification
            if not self._validate_node_spec(node_spec):
                return False
            
            # Store in memory
            self.nodes[node_spec.node_id] = node_spec
            
            # Store in Redis for fast access
            await self.redis_client.hset(
                f"node:{node_spec.node_id}",
                mapping=asdict(node_spec)
            )
            
            # Store in PostgreSQL for persistence
            async with self.postgres_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO nodes (node_id, node_type, resources, status, 
                                     last_heartbeat, labels, annotations, network_config)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (node_id) DO UPDATE SET
                        node_type = EXCLUDED.node_type,
                        resources = EXCLUDED.resources,
                        status = EXCLUDED.status,
                        last_heartbeat = EXCLUDED.last_heartbeat,
                        labels = EXCLUDED.labels,
                        annotations = EXCLUDED.annotations,
                        network_config = EXCLUDED.network_config,
                        updated_at = NOW()
                ''', node_spec.node_id, node_spec.node_type, 
                json.dumps(node_spec.resources), node_spec.status,
                node_spec.last_heartbeat, json.dumps(node_spec.labels),
                json.dumps(node_spec.annotations), json.dumps(node_spec.network_config))
            
            # Announce to cluster via etcd
            await self._announce_node_change("register", node_spec)
            
            self.logger.info(f"Node {node_spec.node_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_spec.node_id}: {e}")
            return False
    
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a node from the cluster"""
        try:
            if node_id not in self.nodes:
                return False
            
            node_spec = self.nodes[node_id]
            
            # Mark as draining first
            node_spec.status = "draining"
            await self._drain_node(node_id)
            
            # Remove from memory
            del self.nodes[node_id]
            
            # Remove from Redis
            await self.redis_client.delete(f"node:{node_id}")
            
            # Update status in PostgreSQL
            async with self.postgres_pool.acquire() as conn:
                await conn.execute('''
                    UPDATE nodes SET status = 'deregistered', updated_at = NOW()
                    WHERE node_id = $1
                ''', node_id)
            
            # Announce to cluster
            await self._announce_node_change("deregister", node_spec)
            
            self.logger.info(f"Node {node_id} deregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister node {node_id}: {e}")
            return False
    
    async def process_placement_request(self, request: PlacementRequest) -> PlacementDecision:
        """Process resource placement request using advanced algorithms"""
        try:
            # Calculate placement scores for all eligible nodes
            placement_scores = await self._calculate_placement_scores(request)
            
            # Apply bin-packing algorithm with latency weights
            selected_nodes = self._bin_pack_with_latency(request, placement_scores)
            
            # Generate resource allocation plan
            resource_allocation = await self._generate_allocation_plan(request, selected_nodes)
            
            # Calculate overall placement score
            overall_score = sum(placement_scores.get(node, 0) for node in selected_nodes)
            
            # Generate reasoning
            reasoning = self._generate_placement_reasoning(request, selected_nodes, placement_scores)
            
            decision = PlacementDecision(
                session_id=request.session_id,
                selected_nodes=selected_nodes,
                resource_allocation=resource_allocation,
                placement_score=overall_score,
                reasoning=reasoning
            )
            
            # Store decision for future optimization
            self.placement_history.append(decision)
            await self._store_placement_decision(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Failed to process placement request: {e}")
            raise
    
    async def _calculate_placement_scores(self, request: PlacementRequest) -> Dict[str, float]:
        """Calculate placement scores using multi-factor algorithm"""
        scores = {}
        
        for node_id, node in self.nodes.items():
            if node.status != "active":
                continue
            
            score = 0.0
            
            # Resource availability score (40% weight)
            resource_score = self._calculate_resource_score(request.resource_requirements, node.resources)
            score += resource_score * 0.4
            
            # Network latency score (30% weight)
            latency_score = await self._calculate_latency_score(node_id, request)
            score += latency_score * 0.3
            
            # Thermal headroom score (20% weight)
            thermal_score = await self._calculate_thermal_score(node_id)
            score += thermal_score * 0.2
            
            # Load balancing score (10% weight)
            load_score = self._calculate_load_score(node_id)
            score += load_score * 0.1
            
            scores[node_id] = score
        
        return scores
    
    def _bin_pack_with_latency(self, request: PlacementRequest, scores: Dict[str, float]) -> List[str]:
        """Bin-packing algorithm optimized for latency"""
        # Sort nodes by score (descending)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_nodes = []
        remaining_requirements = request.resource_requirements.copy()
        
        for node_id, score in sorted_nodes:
            if not remaining_requirements:
                break
            
            node = self.nodes[node_id]
            
            # Check if node can satisfy any remaining requirements
            if self._can_satisfy_requirements(node, remaining_requirements):
                selected_nodes.append(node_id)
                
                # Update remaining requirements
                remaining_requirements = self._subtract_resources(
                    remaining_requirements, node.resources
                )
        
        return selected_nodes
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats and handle failures"""
        while True:
            try:
                current_time = datetime.now()
                failed_nodes = []
                
                for node_id, node in self.nodes.items():
                    if node.status == "active":
                        time_since_heartbeat = current_time - node.last_heartbeat
                        
                        if time_since_heartbeat > timedelta(seconds=30):
                            self.logger.warning(f"Node {node_id} missed heartbeat")
                            node.status = "unhealthy"
                            
                        if time_since_heartbeat > timedelta(seconds=90):
                            self.logger.error(f"Node {node_id} failed - no heartbeat for 90s")
                            failed_nodes.append(node_id)
                
                # Handle failed nodes
                for node_id in failed_nodes:
                    await self._handle_node_failure(node_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(10)
    
    async def _cluster_optimization(self):
        """Continuously optimize cluster performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Analyze placement decisions
                recent_decisions = self.placement_history[-100:]  # Last 100 decisions
                
                # Check for suboptimal placements
                suboptimal_sessions = await self._identify_suboptimal_placements(recent_decisions)
                
                # Trigger rebalancing if needed
                if suboptimal_sessions:
                    await self._trigger_rebalancing(suboptimal_sessions)
                
                # Update ML models
                await self._update_prediction_models(recent_decisions)
                
            except Exception as e:
                self.logger.error(f"Error in cluster optimization: {e}")
    
    async def _metrics_collection(self):
        """Collect and export cluster metrics"""
        while True:
            try:
                metrics = {
                    "total_nodes": len(self.nodes),
                    "active_nodes": len([n for n in self.nodes.values() if n.status == "active"]),
                    "active_sessions": len(self.active_sessions),
                    "cluster_utilization": await self._calculate_cluster_utilization(),
                    "average_placement_score": self._calculate_avg_placement_score(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in Redis for real-time access
                await self.redis_client.set("cluster:metrics", json.dumps(metrics))
                
                # Export to Prometheus
                await self._export_prometheus_metrics(metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    def _validate_node_spec(self, node_spec: NodeSpec) -> bool:
        """Validate node specification"""
        required_fields = ["node_id", "node_type", "resources"]
        for field in required_fields:
            if not getattr(node_spec, field):
                return False
        
        valid_types = ["cpu_node", "gpu_node", "storage_node", "hybrid_node"]
        if node_spec.node_type not in valid_types:
            return False
        
        return True
    
    async def _announce_node_change(self, action: str, node_spec: NodeSpec):
        """Announce node changes to cluster via etcd"""
        try:
            announcement = {
                "action": action,
                "node_id": node_spec.node_id,
                "timestamp": datetime.now().isoformat(),
                "cluster_name": CLUSTER_NAME
            }
            
            self.etcd_client.put(
                f"/omega/cluster/{CLUSTER_NAME}/announcements/{uuid.uuid4()}",
                json.dumps(announcement)
            )
        except Exception as e:
            self.logger.error(f"Failed to announce node change: {e}")

# Create FastAPI app for orchestrator API
app = FastAPI(title="Omega Orchestrator", version="1.0.0")
orchestrator = OmegaOrchestrator()

@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize()

@app.post("/api/v1/nodes/register")
async def register_node(node_data: dict):
    node_spec = NodeSpec(**node_data)
    success = await orchestrator.register_node(node_spec)
    return {"success": success}

@app.delete("/api/v1/nodes/{node_id}")
async def deregister_node(node_id: str):
    success = await orchestrator.deregister_node(node_id)
    return {"success": success}

@app.post("/api/v1/placement/request")
async def placement_request(request: PlacementRequest):
    decision = await orchestrator.process_placement_request(request)
    return decision

@app.get("/api/v1/cluster/status")
async def get_cluster_status():
    return {
        "cluster_name": CLUSTER_NAME,
        "state": orchestrator.cluster_state,
        "total_nodes": len(orchestrator.nodes),
        "active_nodes": len([n for n in orchestrator.nodes.values() if n.status == "active"]),
        "active_sessions": len(orchestrator.active_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
