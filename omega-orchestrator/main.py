"""
Omega Super Desktop Console - Ω-Orchestrator Service
Initial prototype node discovery, heartbeat, placement, and rolling upgrades.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import json
import uuid
import hashlib

from fastapi import FastAPI, WebSocket, BackgroundTasks
from contextlib import asynccontextmanager
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
        """Initialize in-memory state containers and stat counters."""
        self.start_time = time.time()
        # Core in-memory state
        self.nodes = {}              # type: Dict[str, NodeSpec]
        self.active_sessions = {}    # type: Dict[str, Dict[str, Any]]
        self.placement_history = []  # type: List[PlacementDecision]
        self.cluster_state = "initializing"

        # External clients (lazy async init in initialize())
        self.redis_client = None
        self.postgres_pool = None
        self.etcd_client = None

        # Logging
        self.logger = logging.getLogger(__name__)

        # Scheduling / autoscaling stats (numeric counters)
        self._scheduling_stats = {
            'decisions': 0,
            'last_score': 0.0,
            'scales_out': 0,
            'scales_in': 0,
            'autoscale_iterations': 0
        }

        # Rolling samples for derived custom metrics
        self._seamlessness_samples = []  # type: List[float]
        self._scaling_eff_samples = []   # type: List[float]
        self._collab_coeff_samples = []  # type: List[float]

        # Recorded autoscaling actions (auditable). Each item: {ts, action, reason, util_before}
        self.autoscaling_events = []     # type: List[Dict[str, Any]]
        # Internal cache of whether autoscaling tables initialized (avoid repeat DDL attempts)
        self._autoscale_persistence_ready = False

    # --- Unified health schema ---
    def build_health(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        deps = {
            'redis': {'ok': self.redis_client is not None},
            'database': {'ok': self.postgres_pool is not None},
            'etcd': {'ok': self.etcd_client is not None},
        }
        degraded = [k for k, v in deps.items() if not v['ok']]
        return {
            'status': 'healthy' if not degraded else 'degraded',
            'version': '1.0.0',
            'uptime_seconds': int(time.time() - self.start_time),
            'cluster': {
                'state': self.cluster_state,
                'node_count': len(self.nodes),
                'active_sessions': len(self.active_sessions),
                'last_decision_score': self._scheduling_stats.get('last_score'),
                'autoscale': {
                    'iterations': self._scheduling_stats.get('autoscale_iterations'),
                    'scale_out': self._scheduling_stats.get('scales_out'),
                    'scale_in': self._scheduling_stats.get('scales_in')
                }
            },
            'dependencies': deps,
            'degraded': degraded,
            'timestamp': now.isoformat()
        }
        
    async def initialize(self):
        """Initialize orchestrator components"""
        # Redis for fast lookups
        try:
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

        # PostgreSQL (fallback to SQLite if unavailable)
        try:
            self.postgres_pool = await asyncpg.create_pool(POSTGRES_URL)
        except Exception as e:
            self.logger.warning(f"Postgres unavailable ({e}), falling back to local SQLite")
            try:
                try:
                    import aiosqlite  # type: ignore
                except Exception:
                    aiosqlite = None  # type: ignore
                if aiosqlite is None:
                    raise RuntimeError('aiosqlite not installed')
                class _SQLitePool:
                    def __init__(self, path):
                        self.path = path
                    async def acquire(self):
                        return await aiosqlite.connect(self.path)
                sqlite_path = _os.path.join(_os.getcwd(), 'omega_orchestrator.db')
                self.postgres_pool = _SQLitePool(sqlite_path)
            except Exception:
                self.logger.error("No local DB available for orchestrator persistence")
                self.postgres_pool = None

        # etcd client
        self.etcd_client = etcd3.client(host='localhost', port=2379)

        # Initialize schema
        await self._init_database()

        # Launch background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._cluster_optimization())
        asyncio.create_task(self._metrics_collection())
        asyncio.create_task(self._autoscaling_loop())

        self.cluster_state = "active"
        self.logger.info("Omega Orchestrator initialized successfully")
    
    async def _init_database(self):
        """Initialize database schema"""
        if not self.postgres_pool:
            self.logger.warning("Skipping database schema init - no persistence backend available")
            return
        try:
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
                CREATE TABLE IF NOT EXISTS autoscaling_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    ts TIMESTAMPTZ DEFAULT NOW(),
                    action VARCHAR(32) NOT NULL,
                    reason TEXT,
                    util_before FLOAT,
                    active_nodes INT
                );
            ''')
                self._autoscale_persistence_ready = True
        except Exception as e:
            self.logger.error(f"Database init failed (continuing without persistence): {e}")
    
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
            # Capability filters (CPU/GPU/memory tags & custom constraints)
            eligible = {
                nid: n for nid, n in self.nodes.items()
                if self._passes_capability_filters(n, request.resource_requirements, request.constraints)
            }
            if not eligible:
                raise RuntimeError('No eligible nodes for placement')
            # Calculate placement scores for all eligible nodes
            placement_scores = await self._calculate_placement_scores(request, candidates=eligible)
            
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
    
    async def _calculate_placement_scores(self, request: PlacementRequest, candidates: Optional[Dict[str, NodeSpec]] = None) -> Dict[str, float]:
        """Calculate placement scores using multi-factor algorithm"""
        scores = {}
        nodes_iter = (candidates or self.nodes)
        for node_id, node in nodes_iter.items():
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
                # Use timezone-aware UTC time for consistency
                current_time = datetime.now(timezone.utc)
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
                    "seamlessness_index": self._compute_seamlessness_index(),
                    "scaling_efficiency": self._compute_scaling_efficiency(),
                    "collaboration_coefficient": self._compute_collaboration_coefficient(),
                    "timestamp": datetime.now().isoformat()
                }
                # Store in Redis for real-time access
                try:
                    await self.redis_client.set("cluster:metrics", json.dumps(metrics))  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Export to Prometheus (placeholder hook)
                await self._export_prometheus_metrics(metrics)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
            await asyncio.sleep(15)
    async def _autoscaling_loop(self):
        """Periodic autoscaling decisions (hooks only / v1)."""
        interval = int(_os.getenv('OMEGA_AUTOSCALE_INTERVAL', '60'))
        while True:
            try:
                self._scheduling_stats['autoscale_iterations'] += 1
                util = await self._calculate_cluster_utilization()
                # Simple thresholds
                if util > 0.75:
                    await self._scale_out(util, reason="utilization_high")
                elif util < 0.25 and len(self.nodes) > 1:
                    await self._scale_in(util, reason="utilization_low")
            except Exception as e:
                self.logger.warning(f"Autoscale loop error: {e}")
            await asyncio.sleep(interval)
    # --- Autoscaling helpers ---
    def _record_autoscale_event(self, action: str, util_before: float, reason: str):
        evt = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'reason': reason,
            'util_before': round(util_before, 4),
            'active_nodes': len([n for n in self.nodes.values() if n.status == 'active'])
        }
        self.autoscaling_events.append(evt)
        # retain last 200 events to bound memory
        if len(self.autoscaling_events) > 200:
            self.autoscaling_events = self.autoscaling_events[-200:]
        self.logger.info(f"[autoscale] {action} recorded reason={reason} util={util_before:.2f}")
        # Best-effort persistence
        asyncio.create_task(self._persist_autoscale_event(evt))

    async def _scale_out(self, util_before: float, reason: str = "threshold", count: int = 1):
        """Provision simulated nodes (in-memory + persistence) to satisfy scale out."""
        self._scheduling_stats['scales_out'] += 1
        for _ in range(max(1, count)):
            node_id = f"auto-node-{uuid.uuid4().hex[:6]}"
            spec = NodeSpec(
                node_id=node_id,
                node_type='cpu_node',
                resources={'cpu': 8, 'memory': 32, 'gpu': 0},
                status='active',
                last_heartbeat=datetime.now(timezone.utc),
                labels={'autoscaled': 'true'},
                annotations={'reason': reason},
                network_config={}
            )
            # Reuse existing register logic for consistency
            try:
                await self.register_node(spec)
            except Exception as e:
                self.logger.error(f"Autoscale scale_out register failed for {node_id}: {e}")
        self._record_autoscale_event('scale_out', util_before, reason)

    async def _scale_in(self, util_before: float, reason: str = "threshold", count: int = 1):
        """Select candidate nodes (autoscaled) and remove them gracefully."""
        self._scheduling_stats['scales_in'] += 1
        removable = [n for n in self.nodes.values() if n.labels.get('autoscaled') == 'true' and n.status == 'active']
        # Fallback: include any active nodes except first if not enough
        if len(removable) < count:
            extra = [n for n in self.nodes.values() if n.status == 'active' and n not in removable]
            removable.extend(extra)
        removed = 0
        for node in removable:
            if len([n for n in self.nodes.values() if n.status == 'active']) <= 1:
                break  # never remove last node
            try:
                await self.deregister_node(node.node_id)
                removed += 1
            except Exception as e:
                self.logger.error(f"Autoscale scale_in deregister failed for {node.node_id}: {e}")
            if removed >= count:
                break
        self._record_autoscale_event('scale_in', util_before, reason + f" removed={removed}")

    async def _persist_autoscale_event(self, evt: Dict[str, Any]):
        if not self.postgres_pool or not self._autoscale_persistence_ready:
            return
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    'INSERT INTO autoscaling_events (action, reason, util_before, active_nodes, ts) VALUES ($1,$2,$3,$4,NOW())',
                    evt['action'], evt['reason'], float(evt['util_before']), int(evt['active_nodes'])
                )
        except Exception as e:
            # Suppress to avoid loop failure; log once
            self.logger.debug(f"Persist autoscale event failed: {e}")

    # --- Custom metrics computations ---
    def _compute_seamlessness_index(self) -> float:
        try:
            # Use variance of last placement scores (lower variance -> higher seamlessness)
            last_scores = [d.placement_score for d in self.placement_history[-20:]]
            if len(last_scores) < 2:
                return 100.0
            import statistics
            var = statistics.pvariance(last_scores)
            score = max(0.0, 100.0 - min(var, 100.0))
            self._seamlessness_samples.append(score)
            return round(score,2)
        except Exception:
            return 0.0

    def _compute_scaling_efficiency(self) -> float:
        try:
            active = len([n for n in self.nodes.values() if n.status=='active']) or 1
            util = self._calculate_avg_placement_score()/ (active*10) if active else 0
            eff = min(100.0, util * 100)
            self._scaling_eff_samples.append(eff)
            return round(eff,2)
        except Exception:
            return 0.0

    def _compute_collaboration_coefficient(self) -> float:
        try:
            sessions = len(self.active_sessions) or 1
            nodes = len(self.nodes) or 1
            coeff = min(100.0, (sessions / nodes) * 50 + 50)
            self._collab_coeff_samples.append(coeff)
            return round(coeff,2)
        except Exception:
            return 0.0

    # ---- Missing helper methods (added) ----
    async def _store_placement_decision(self, decision: PlacementDecision) -> None:
        if not self.postgres_pool:
            return
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO placement_decisions (session_id, selected_nodes, resource_allocation, placement_score, reasoning)
                    VALUES ($1, $2, $3, $4, $5)
                ''', decision.session_id, json.dumps(decision.selected_nodes), json.dumps(decision.resource_allocation), decision.placement_score, decision.reasoning)
        except Exception as e:
            self.logger.warning(f"Persist placement decision failed: {e}")

    async def _generate_allocation_plan(self, request: PlacementRequest, selected_nodes: List[str]) -> Dict[str, Dict[str, Any]]:
        plan: Dict[str, Dict[str, Any]] = {}
        if not selected_nodes:
            return plan
        # naive: evenly divide numeric resource requirements
        counts = len(selected_nodes)
        for node_id in selected_nodes:
            node = self.nodes.get(node_id)
            alloc: Dict[str, Any] = {}
            for k,v in request.resource_requirements.items():
                if isinstance(v,(int,float)):
                    alloc[k] = max(0, v / counts)
            plan[node_id] = alloc
        return plan

    def _generate_placement_reasoning(self, request: PlacementRequest, selected_nodes: List[str], scores: Dict[str,float]) -> str:
        if not selected_nodes:
            return "No nodes selected"
        parts = []
        for nid in selected_nodes:
            parts.append(f"node {nid} score={scores.get(nid,0):.2f}")
        return "; ".join(parts)

    def _calculate_resource_score(self, req: Dict[str,Any], available: Dict[str,Any]) -> float:
        if not req:
            return 1.0
        score = 0.0
        counted = 0
        for k,v in req.items():
            if isinstance(v,(int,float)) and isinstance(available.get(k),(int,float)):
                counted += 1
                have = available[k]
                score += min(1.0, have / (v if v else 1))
        return score / counted if counted else 1.0

    async def _calculate_latency_score(self, node_id: str, request: PlacementRequest) -> float:
        # placeholder heuristic: random-ish deterministic hash based
        h = int(hashlib.sha256(node_id.encode()).hexdigest(),16)
        return ((h % 50) / 50)  # 0..0.98

    async def _calculate_thermal_score(self, node_id: str) -> float:
        # stub: assume good thermal headroom
        return 0.9

    def _calculate_load_score(self, node_id: str) -> float:
        # low historical placement count yields higher score (spread load)
        count = sum(1 for d in self.placement_history if node_id in d.selected_nodes)
        return 1.0 / (1 + count)

    def _passes_capability_filters(self, node: NodeSpec, req: Dict[str,Any], constraints: List[Dict[str,Any]]) -> bool:
        # Basic CPU/GPU/memory filters
        try:
            if 'cpu' in req and node.resources.get('cpu',0) < req['cpu']:
                return False
            if 'memory' in req and node.resources.get('memory',0) < req['memory']:
                return False
            if req.get('gpu') and node.resources.get('gpu',0) < req['gpu']:
                return False
            # label constraints
            for c in constraints or []:
                key = c.get('key'); val = c.get('value')
                if key and val and node.labels.get(key) != val:
                    return False
            return True
        except Exception:
            return False

    def _can_satisfy_requirements(self, node: NodeSpec, remaining: Dict[str,Any]) -> bool:
        for k,v in remaining.items():
            if isinstance(v,(int,float)) and node.resources.get(k,0) <= 0:
                return False
        return True

    def _subtract_resources(self, remaining: Dict[str,Any], provided: Dict[str,Any]) -> Dict[str,Any]:
        new = {}
        for k,v in remaining.items():
            if isinstance(v,(int,float)):
                new_v = v - float(provided.get(k,0))
                if new_v > 0:
                    new[k] = new_v
            else:
                new[k] = v
        return new

    async def _calculate_cluster_utilization(self) -> float:
        # naive: average resource score across nodes
        if not self.nodes:
            return 0.0
        util = 0.0
        for n in self.nodes.values():
            util += sum(v for v in n.resources.values() if isinstance(v,(int,float)))
        return min(1.0, util / (len(self.nodes) * 100.0))

    def _calculate_avg_placement_score(self) -> float:
        if not self.placement_history:
            return 0.0
        return sum(d.placement_score for d in self.placement_history) / len(self.placement_history)

    async def _handle_node_failure(self, node_id: str):
        node = self.nodes.get(node_id)
        if not node:
            return
        node.status = 'failed'
        try:
            await self.redis_client.delete(f"node:{node_id}")  # type: ignore[attr-defined]
        except Exception:
            pass
        self.logger.error(f"Node {node_id} marked failed")

    async def _drain_node(self, node_id: str):
        # placeholder: mark draining then drained; real impl would migrate sessions
        node = self.nodes.get(node_id)
        if node:
            node.status = 'drained'

    async def _identify_suboptimal_placements(self, recent: List[PlacementDecision]) -> List[str]:
        # simple heuristic: pick sessions with score below median*0.5
        if not recent:
            return []
        scores = [d.placement_score for d in recent]
        median = sorted(scores)[len(scores)//2]
        return [d.session_id for d in recent if d.placement_score < median*0.5]

    async def _trigger_rebalancing(self, sessions: List[str]):
        if not sessions:
            return
        self.logger.info(f"Rebalancing sessions: {sessions}")

    async def _update_prediction_models(self, recent: List[PlacementDecision]):
        # stub hook for future ML model updates
        return

    async def _export_prometheus_metrics(self, metrics: Dict[str,Any]):
        # Expose a minimal Prometheus metrics integration using custom helper if available
        try:
            from utils import metrics as m
            # Gauges (cached by helper)
            g_total = m.create_gauge('omega_total_nodes', 'Total nodes in orchestrator')
            g_active = m.create_gauge('omega_active_nodes', 'Active nodes in orchestrator')
            g_util = m.create_gauge('omega_cluster_utilization', 'Cluster utilization (0-1)')
            g_score = m.create_gauge('omega_avg_placement_score', 'Average placement score')
            g_total.set(metrics.get('total_nodes',0))
            g_active.set(metrics.get('active_nodes',0))
            g_util.set(metrics.get('cluster_utilization',0.0))
            g_score.set(metrics.get('average_placement_score',0.0))
        except Exception:
            pass

    # Unified health endpoint builder (mirrors backend schema)
    def build_health(self) -> Dict[str, Any]:
        deps = {
            'redis': {'ok': self.redis_client is not None},
            'database': {'ok': self.postgres_pool is not None},
            'etcd': {'ok': self.etcd_client is not None},
            'scheduler': {'ok': True},
        }
        degraded = [k for k,v in deps.items() if not v['ok']]
        return {
            'status': 'healthy' if not degraded else 'degraded',
            'version': '1.0.0',
            'uptime_seconds': int(time.time()),
            'dependencies': deps,
            'degraded': degraded,
            'scheduling': self._scheduling_stats,
            # Use timezone-aware UTC timestamp instead of deprecated utcnow
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    # (FastAPI app defined after class)
    
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

# FastAPI app instantiation (single, after class definition)
orch = OmegaOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await orch.initialize()
    yield
    # Shutdown hooks (future): flush metrics, close DB pools
    try:
        if orch.postgres_pool:
            # asyncpg pool has close() coroutine; sqlite fallback has none
            close = getattr(orch.postgres_pool, 'close', None)
            if close:
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
    except Exception:
        pass

app = FastAPI(title="Omega Orchestrator", version="1.0.0", lifespan=lifespan)

@app.get('/health')
async def health():
    return orch.build_health()

@app.post('/schedule')
async def schedule(req: PlacementRequest):
    decision = await orch.process_placement_request(req)
    return decision

@app.get('/autoscaling/events')
async def autoscaling_events(limit: int = 50):
    limit = max(1, min(200, limit))
    return {'events': orch.autoscaling_events[-limit:]}

@app.post('/autoscaling/scale_out')
async def manual_scale_out(count: int = 1, reason: str = 'manual'):
    util = await orch._calculate_cluster_utilization()
    await orch._scale_out(util, reason=reason, count=count)
    return {'status':'ok','action':'scale_out','count':count,'nodes': list(orch.nodes.keys())}

@app.post('/autoscaling/scale_in')
async def manual_scale_in(count: int = 1, reason: str = 'manual'):
    util = await orch._calculate_cluster_utilization()
    await orch._scale_in(util, reason=reason, count=count)
    return {'status':'ok','action':'scale_in','count':count,'nodes': list(orch.nodes.keys())}

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
