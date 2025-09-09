"""
Omega Super Desktop Console v2.0 - Advanced Scheduling Engine
Production-grade scheduler with weighted least-loaded placement, capability filtering,
predictive placement, energy/thermal awareness, and workload migration capabilities.
"""

import asyncio
import logging
import time
import json
import math
import heapq
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class SchedulingStrategy(Enum):
    WEIGHTED_LEAST_LOADED = "weighted_least_loaded"
    CAPABILITY_AWARE = "capability_aware"
    ENERGY_EFFICIENT = "energy_efficient"
    PREDICTIVE_PLACEMENT = "predictive_placement"
    THERMAL_AWARE = "thermal_aware"
    LATENCY_OPTIMIZED = "latency_optimized"

class WorkloadType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    GPU_INTENSIVE = "gpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED = "mixed"

class NodeCapability(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    SPECIALIZED = "specialized"

@dataclass
class ResourceRequirement:
    """Resource requirements for a workload"""
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_units: int = 0
    storage_gb: float = 0.0
    network_mbps: float = 0.0
    capabilities: Set[NodeCapability] = field(default_factory=set)
    affinity_rules: Dict[str, Any] = field(default_factory=dict)
    anti_affinity_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeResource:
    """Current resource state of a node"""
    node_id: str
    total_cpu_cores: int
    available_cpu_cores: float
    total_memory_gb: float
    available_memory_gb: float
    total_gpu_units: int
    available_gpu_units: int
    total_storage_gb: float
    available_storage_gb: float
    network_bandwidth_mbps: float
    capabilities: Set[NodeCapability] = field(default_factory=set)
    
    # Performance characteristics
    cpu_performance_score: float = 1.0
    memory_bandwidth_score: float = 1.0
    storage_iops_score: float = 1.0
    network_latency_ms: float = 1.0
    
    # Energy and thermal
    power_consumption_watts: float = 100.0
    temperature_celsius: float = 35.0
    thermal_throttling: bool = False
    energy_efficiency_score: float = 1.0
    
    # Historical performance
    load_average_5min: float = 0.0
    load_average_15min: float = 0.0
    success_rate: float = 1.0
    completion_time_avg: float = 0.0
    
    # Workload history for predictive placement
    workload_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    maintenance_window: Optional[Tuple[float, float]] = None

@dataclass
class SchedulingJob:
    """A job to be scheduled"""
    job_id: str
    workload_type: WorkloadType
    requirements: ResourceRequirement
    priority: int = 0
    deadline: Optional[float] = None
    estimated_duration: float = 3600.0  # seconds
    user_id: str = "system"
    submitted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class SchedulingDecision:
    """Result of scheduling decision"""
    job_id: str
    target_node_id: Optional[str] = None
    score: float = 0.0
    reasoning: str = ""
    alternative_nodes: List[Tuple[str, float]] = field(default_factory=list)
    estimated_start_time: float = field(default_factory=time.time)
    estimated_completion_time: float = 0.0
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: str = ""

class AdvancedScheduler:
    """Production-grade scheduling engine with predictive placement and energy awareness"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.nodes: Dict[str, NodeResource] = {}
        self.job_queue: List[SchedulingJob] = []
        self.active_jobs: Dict[str, SchedulingJob] = {}
        self.completed_jobs: List[Tuple[SchedulingJob, SchedulingDecision, float]] = []
        
        # Scheduling state
        self.scheduling_history: List[SchedulingDecision] = []
        self.node_performance_cache: Dict[str, Dict[str, float]] = {}
        self.workload_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Strategy weights
        self.strategy_weights = {
            SchedulingStrategy.WEIGHTED_LEAST_LOADED: 0.3,
            SchedulingStrategy.CAPABILITY_AWARE: 0.25,
            SchedulingStrategy.ENERGY_EFFICIENT: 0.15,
            SchedulingStrategy.PREDICTIVE_PLACEMENT: 0.15,
            SchedulingStrategy.THERMAL_AWARE: 0.1,
            SchedulingStrategy.LATENCY_OPTIMIZED: 0.05
        }
        
        # Performance tracking
        self.metrics = {
            'total_jobs_scheduled': 0,
            'successful_placements': 0,
            'failed_placements': 0,
            'average_scheduling_time': 0.0,
            'resource_utilization': {},
            'energy_efficiency': 0.0,
            'thermal_violations': 0
        }
        
        # Predictive models (simplified ML)
        self.placement_success_model = {}
        self.completion_time_model = {}
        
        logger.info("Advanced Scheduler initialized")
    
    async def register_node(self, node: NodeResource) -> bool:
        """Register a new node with the scheduler"""
        try:
            self.nodes[node.node_id] = node
            self._update_node_performance_cache(node.node_id)
            logger.info(f"Node {node.node_id} registered with scheduler")
            return True
        except Exception as e:
            logger.error(f"Failed to register node {node.node_id}: {e}")
            return False
    
    async def update_node_resources(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node resource information"""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            for key, value in updates.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            
            node.last_heartbeat = time.time()
            self._update_node_performance_cache(node_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False
    
    async def submit_job(self, job: SchedulingJob) -> str:
        """Submit a job for scheduling"""
        try:
            heapq.heappush(self.job_queue, (-job.priority, job.submitted_at, job))
            logger.info(f"Job {job.job_id} submitted for scheduling")
            return job.job_id
        except Exception as e:
            logger.error(f"Failed to submit job {job.job_id}: {e}")
            raise
    
    async def schedule_job(self, job: SchedulingJob) -> SchedulingDecision:
        """Core scheduling logic with multi-strategy approach"""
        start_time = time.time()
        
        try:
            # Pre-flight checks
            if not self._validate_job(job):
                return SchedulingDecision(
                    job_id=job.job_id,
                    success=False,
                    error_message="Job validation failed"
                )
            
            # Get available nodes
            available_nodes = self._get_available_nodes(job)
            if not available_nodes:
                return SchedulingDecision(
                    job_id=job.job_id,
                    success=False,
                    error_message="No available nodes meet requirements"
                )
            
            # Multi-strategy scoring
            node_scores = {}
            for node_id in available_nodes:
                score = await self._calculate_node_score(job, node_id)
                node_scores[node_id] = score
            
            # Select best node
            best_node_id = max(node_scores.keys(), key=lambda n: node_scores[n])
            best_score = node_scores[best_node_id]
            
            # Create scheduling decision
            decision = SchedulingDecision(
                job_id=job.job_id,
                target_node_id=best_node_id,
                score=best_score,
                reasoning=self._generate_reasoning(job, best_node_id, node_scores),
                alternative_nodes=sorted(
                    [(n, s) for n, s in node_scores.items() if n != best_node_id],
                    key=lambda x: x[1], reverse=True
                )[:3],
                estimated_start_time=time.time(),
                estimated_completion_time=time.time() + job.estimated_duration,
                resource_allocation=self._calculate_resource_allocation(job, best_node_id),
                success=True
            )
            
            # Update metrics
            scheduling_time = time.time() - start_time
            self.metrics['total_jobs_scheduled'] += 1
            self.metrics['successful_placements'] += 1
            self.metrics['average_scheduling_time'] = (
                (self.metrics['average_scheduling_time'] * (self.metrics['total_jobs_scheduled'] - 1) + scheduling_time)
                / self.metrics['total_jobs_scheduled']
            )
            
            # Reserve resources
            await self._reserve_resources(job, best_node_id)
            
            # Add to active jobs
            self.active_jobs[job.job_id] = job
            
            # Update scheduling history
            self.scheduling_history.append(decision)
            if len(self.scheduling_history) > 1000:
                self.scheduling_history = self.scheduling_history[-500:]
            
            logger.info(f"Job {job.job_id} scheduled to node {best_node_id} (score: {best_score:.3f})")
            return decision
            
        except Exception as e:
            logger.error(f"Scheduling failed for job {job.job_id}: {e}")
            self.metrics['failed_placements'] += 1
            return SchedulingDecision(
                job_id=job.job_id,
                success=False,
                error_message=f"Scheduling error: {str(e)}"
            )
    
    async def _calculate_node_score(self, job: SchedulingJob, node_id: str) -> float:
        """Calculate composite score for node using multiple strategies"""
        node = self.nodes[node_id]
        total_score = 0.0
        
        # Weighted least-loaded score
        load_score = await self._calculate_load_score(job, node)
        total_score += load_score * self.strategy_weights[SchedulingStrategy.WEIGHTED_LEAST_LOADED]
        
        # Capability-aware score
        capability_score = await self._calculate_capability_score(job, node)
        total_score += capability_score * self.strategy_weights[SchedulingStrategy.CAPABILITY_AWARE]
        
        # Energy efficiency score
        energy_score = await self._calculate_energy_score(job, node)
        total_score += energy_score * self.strategy_weights[SchedulingStrategy.ENERGY_EFFICIENT]
        
        # Predictive placement score
        predictive_score = await self._calculate_predictive_score(job, node)
        total_score += predictive_score * self.strategy_weights[SchedulingStrategy.PREDICTIVE_PLACEMENT]
        
        # Thermal awareness score
        thermal_score = await self._calculate_thermal_score(job, node)
        total_score += thermal_score * self.strategy_weights[SchedulingStrategy.THERMAL_AWARE]
        
        # Latency optimization score
        latency_score = await self._calculate_latency_score(job, node)
        total_score += latency_score * self.strategy_weights[SchedulingStrategy.LATENCY_OPTIMIZED]
        
        return total_score
    
    async def _calculate_load_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on current load (higher available resources = higher score)"""
        try:
            # CPU utilization score
            cpu_util = 1.0 - (node.available_cpu_cores / max(node.total_cpu_cores, 1))
            cpu_score = 1.0 - cpu_util
            
            # Memory utilization score
            memory_util = 1.0 - (node.available_memory_gb / max(node.total_memory_gb, 1))
            memory_score = 1.0 - memory_util
            
            # GPU utilization score (if needed)
            gpu_score = 1.0
            if job.requirements.gpu_units > 0:
                gpu_util = 1.0 - (node.available_gpu_units / max(node.total_gpu_units, 1))
                gpu_score = 1.0 - gpu_util
            
            # Weighted average based on job requirements
            if job.workload_type == WorkloadType.CPU_INTENSIVE:
                return cpu_score * 0.7 + memory_score * 0.2 + gpu_score * 0.1
            elif job.workload_type == WorkloadType.GPU_INTENSIVE:
                return gpu_score * 0.7 + cpu_score * 0.2 + memory_score * 0.1
            elif job.workload_type == WorkloadType.MEMORY_INTENSIVE:
                return memory_score * 0.7 + cpu_score * 0.2 + gpu_score * 0.1
            else:
                return (cpu_score + memory_score + gpu_score) / 3.0
                
        except Exception as e:
            logger.error(f"Error calculating load score: {e}")
            return 0.0
    
    async def _calculate_capability_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on capability matching"""
        try:
            if not job.requirements.capabilities:
                return 1.0  # No specific requirements
            
            # Check if node has all required capabilities
            missing_capabilities = job.requirements.capabilities - node.capabilities
            if missing_capabilities:
                return 0.0  # Hard requirement not met
            
            # Bonus for having additional relevant capabilities
            extra_capabilities = node.capabilities - job.requirements.capabilities
            capability_bonus = min(len(extra_capabilities) * 0.1, 0.5)
            
            # Performance score based on capability utilization
            performance_score = 1.0
            if NodeCapability.CPU in job.requirements.capabilities:
                performance_score *= node.cpu_performance_score
            if NodeCapability.MEMORY in job.requirements.capabilities:
                performance_score *= node.memory_bandwidth_score
            if NodeCapability.STORAGE in job.requirements.capabilities:
                performance_score *= node.storage_iops_score
            
            return min(performance_score + capability_bonus, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating capability score: {e}")
            return 0.0
    
    async def _calculate_energy_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on energy efficiency"""
        try:
            # Higher efficiency score is better
            base_efficiency = node.energy_efficiency_score
            
            # Penalize high power consumption
            power_penalty = min(node.power_consumption_watts / 500.0, 1.0)  # Normalize to 500W max
            
            # Bonus for low current utilization (energy savings)
            utilization = 1.0 - (node.available_cpu_cores / max(node.total_cpu_cores, 1))
            efficiency_bonus = (1.0 - utilization) * 0.2
            
            return max(base_efficiency - power_penalty + efficiency_bonus, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating energy score: {e}")
            return 0.5
    
    async def _calculate_predictive_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on historical performance prediction"""
        try:
            # Use historical data to predict success probability
            workload_key = f"{job.workload_type.value}_{node.node_id}"
            
            if workload_key in self.workload_patterns and len(self.workload_patterns[workload_key]) > 3:
                # Simple success rate prediction
                recent_performance = self.workload_patterns[workload_key][-10:]
                success_rate = sum(1 for p in recent_performance if p > 0.8) / len(recent_performance)
                
                # Factor in node's historical success rate
                node_success_rate = node.success_rate
                
                return (success_rate + node_success_rate) / 2.0
            else:
                # Default to node's general success rate
                return node.success_rate
                
        except Exception as e:
            logger.error(f"Error calculating predictive score: {e}")
            return 0.5
    
    async def _calculate_thermal_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on thermal conditions"""
        try:
            # Penalize high temperatures
            temp_score = 1.0 - min(max(node.temperature_celsius - 30.0, 0.0) / 50.0, 1.0)
            
            # Heavy penalty for thermal throttling
            if node.thermal_throttling:
                temp_score *= 0.3
            
            # Consider estimated thermal impact of job
            thermal_impact = self._estimate_thermal_impact(job, node)
            projected_temp = node.temperature_celsius + thermal_impact
            
            if projected_temp > 80.0:  # Critical temperature
                temp_score *= 0.1
            elif projected_temp > 70.0:  # Warning temperature
                temp_score *= 0.5
            
            return max(temp_score, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating thermal score: {e}")
            return 0.5
    
    async def _calculate_latency_score(self, job: SchedulingJob, node: NodeResource) -> float:
        """Calculate score based on network latency optimization"""
        try:
            # Lower latency is better
            latency_score = 1.0 - min(node.network_latency_ms / 100.0, 1.0)  # Normalize to 100ms max
            
            # Consider job's network requirements
            if job.workload_type == WorkloadType.NETWORK_INTENSIVE:
                # Give higher weight to latency for network-intensive jobs
                return latency_score
            else:
                # Moderate impact for other job types
                return 0.5 + (latency_score * 0.5)
                
        except Exception as e:
            logger.error(f"Error calculating latency score: {e}")
            return 0.5
    
    def _get_available_nodes(self, job: SchedulingJob) -> List[str]:
        """Get list of nodes that can potentially handle the job"""
        available = []
        
        for node_id, node in self.nodes.items():
            if (node.status == "active" and
                not node.thermal_throttling and
                time.time() - node.last_heartbeat < 60 and  # Recent heartbeat
                self._node_can_handle_job(node, job)):
                available.append(node_id)
        
        return available
    
    def _node_can_handle_job(self, node: NodeResource, job: SchedulingJob) -> bool:
        """Check if node has sufficient resources for job"""
        req = job.requirements
        
        # Check hard resource requirements
        if (req.cpu_cores > node.available_cpu_cores or
            req.memory_gb > node.available_memory_gb or
            req.gpu_units > node.available_gpu_units or
            req.storage_gb > node.available_storage_gb):
            return False
        
        # Check capability requirements
        if req.capabilities and not req.capabilities.issubset(node.capabilities):
            return False
        
        # Check maintenance window
        if node.maintenance_window:
            start_time, end_time = node.maintenance_window
            current_time = time.time()
            if start_time <= current_time <= end_time:
                return False
        
        return True
    
    def _validate_job(self, job: SchedulingJob) -> bool:
        """Validate job requirements"""
        try:
            req = job.requirements
            
            # Basic validation
            if (req.cpu_cores < 0 or req.memory_gb < 0 or
                req.gpu_units < 0 or req.storage_gb < 0):
                return False
            
            # Check for reasonable resource requests
            if (req.cpu_cores > 128 or req.memory_gb > 1024 or
                req.gpu_units > 8 or req.storage_gb > 10240):
                logger.warning(f"Job {job.job_id} has unusually high resource requirements")
            
            return True
            
        except Exception as e:
            logger.error(f"Job validation error: {e}")
            return False
    
    def _generate_reasoning(self, job: SchedulingJob, node_id: str, all_scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for scheduling decision"""
        node = self.nodes[node_id]
        score = all_scores[node_id]
        
        reasons = []
        
        # Resource availability
        cpu_util = 1.0 - (node.available_cpu_cores / max(node.total_cpu_cores, 1))
        if cpu_util < 0.5:
            reasons.append("low CPU utilization")
        
        # Capability matching
        if job.requirements.capabilities.issubset(node.capabilities):
            reasons.append("capability match")
        
        # Performance history
        if node.success_rate > 0.9:
            reasons.append("high success rate")
        
        # Energy efficiency
        if node.energy_efficiency_score > 0.8:
            reasons.append("energy efficient")
        
        # Thermal conditions
        if node.temperature_celsius < 50:
            reasons.append("good thermal conditions")
        
        base_reason = f"Selected node {node_id} (score: {score:.3f})"
        if reasons:
            return f"{base_reason} - {', '.join(reasons)}"
        else:
            return base_reason
    
    def _calculate_resource_allocation(self, job: SchedulingJob, node_id: str) -> Dict[str, Any]:
        """Calculate specific resource allocation for the job"""
        req = job.requirements
        
        return {
            'cpu_cores': req.cpu_cores,
            'memory_gb': req.memory_gb,
            'gpu_units': req.gpu_units,
            'storage_gb': req.storage_gb,
            'network_mbps': req.network_mbps,
            'node_id': node_id,
            'allocation_time': time.time()
        }
    
    async def _reserve_resources(self, job: SchedulingJob, node_id: str):
        """Reserve resources on the selected node"""
        try:
            node = self.nodes[node_id]
            req = job.requirements
            
            # Update available resources
            node.available_cpu_cores -= req.cpu_cores
            node.available_memory_gb -= req.memory_gb
            node.available_gpu_units -= req.gpu_units
            node.available_storage_gb -= req.storage_gb
            
            logger.info(f"Reserved resources on {node_id} for job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to reserve resources: {e}")
            raise
    
    async def release_resources(self, job_id: str, node_id: str):
        """Release resources when job completes"""
        try:
            if job_id not in self.active_jobs:
                return
            
            job = self.active_jobs[job_id]
            node = self.nodes[node_id]
            req = job.requirements
            
            # Release resources
            node.available_cpu_cores = min(node.total_cpu_cores, node.available_cpu_cores + req.cpu_cores)
            node.available_memory_gb = min(node.total_memory_gb, node.available_memory_gb + req.memory_gb)
            node.available_gpu_units = min(node.total_gpu_units, node.available_gpu_units + req.gpu_units)
            node.available_storage_gb = min(node.total_storage_gb, node.available_storage_gb + req.storage_gb)
            
            # Move to completed jobs
            del self.active_jobs[job_id]
            
            logger.info(f"Released resources on {node_id} for completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to release resources: {e}")
    
    async def process_job_queue(self):
        """Process pending jobs in the queue"""
        try:
            while self.job_queue:
                _, _, job = heapq.heappop(self.job_queue)
                
                # Check if job hasn't expired
                if job.deadline and time.time() > job.deadline:
                    logger.warning(f"Job {job.job_id} expired, skipping")
                    continue
                
                decision = await self.schedule_job(job)
                
                if not decision.success and job.retry_count < job.max_retries:
                    # Retry failed job
                    job.retry_count += 1
                    await asyncio.sleep(1)  # Brief delay before retry
                    await self.submit_job(job)
                    logger.info(f"Retrying job {job.job_id} (attempt {job.retry_count + 1})")
                
        except Exception as e:
            logger.error(f"Error processing job queue: {e}")
    
    def _estimate_thermal_impact(self, job: SchedulingJob, node: NodeResource) -> float:
        """Estimate thermal impact of running job on node"""
        # Simplified thermal model
        base_impact = job.requirements.cpu_cores * 2.0  # 2°C per CPU core
        
        if job.workload_type == WorkloadType.CPU_INTENSIVE:
            base_impact *= 1.5
        elif job.workload_type == WorkloadType.GPU_INTENSIVE:
            base_impact += job.requirements.gpu_units * 5.0  # 5°C per GPU
        
        return base_impact
    
    def _update_node_performance_cache(self, node_id: str):
        """Update performance metrics cache for node"""
        try:
            node = self.nodes[node_id]
            
            self.node_performance_cache[node_id] = {
                'last_updated': time.time(),
                'cpu_score': node.cpu_performance_score,
                'memory_score': node.memory_bandwidth_score,
                'storage_score': node.storage_iops_score,
                'energy_score': node.energy_efficiency_score,
                'success_rate': node.success_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to update performance cache for {node_id}: {e}")
    
    async def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling metrics"""
        try:
            # Calculate current resource utilization
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for n in self.nodes.values() if n.status == "active")
            
            total_cpu = sum(n.total_cpu_cores for n in self.nodes.values())
            available_cpu = sum(n.available_cpu_cores for n in self.nodes.values())
            cpu_utilization = (total_cpu - available_cpu) / max(total_cpu, 1)
            
            total_memory = sum(n.total_memory_gb for n in self.nodes.values())
            available_memory = sum(n.available_memory_gb for n in self.nodes.values())
            memory_utilization = (total_memory - available_memory) / max(total_memory, 1)
            
            # Energy metrics
            total_power = sum(n.power_consumption_watts for n in self.nodes.values())
            avg_efficiency = sum(n.energy_efficiency_score for n in self.nodes.values()) / max(total_nodes, 1)
            
            # Thermal metrics
            avg_temperature = sum(n.temperature_celsius for n in self.nodes.values()) / max(total_nodes, 1)
            thermal_violations = sum(1 for n in self.nodes.values() if n.thermal_throttling)
            
            return {
                'cluster_overview': {
                    'total_nodes': total_nodes,
                    'active_nodes': active_nodes,
                    'cpu_utilization': cpu_utilization,
                    'memory_utilization': memory_utilization,
                    'active_jobs': len(self.active_jobs),
                    'queued_jobs': len(self.job_queue)
                },
                'scheduling_performance': self.metrics,
                'energy_metrics': {
                    'total_power_consumption': total_power,
                    'average_efficiency_score': avg_efficiency,
                    'energy_score': self.metrics.get('energy_efficiency', 0.0)
                },
                'thermal_metrics': {
                    'average_temperature': avg_temperature,
                    'thermal_violations': thermal_violations,
                    'nodes_throttling': thermal_violations
                },
                'predictive_insights': {
                    'placement_success_rate': self.metrics['successful_placements'] / max(self.metrics['total_jobs_scheduled'], 1),
                    'average_scheduling_latency': self.metrics['average_scheduling_time'],
                    'workload_patterns': len(self.workload_patterns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating scheduling metrics: {e}")
            return {}

# Global scheduler instance
_scheduler_instance = None

def get_scheduler() -> AdvancedScheduler:
    """Get or create global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AdvancedScheduler()
    return _scheduler_instance

async def initialize_scheduler(config: Dict[str, Any] = None):
    """Initialize the global scheduler"""
    global _scheduler_instance
    _scheduler_instance = AdvancedScheduler(config)
    logger.info("Global scheduler initialized")
    return _scheduler_instance
