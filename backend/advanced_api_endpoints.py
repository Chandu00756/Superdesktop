"""
Omega Super Desktop Console v2.0 - Advanced Engine Integration Endpoints
Production-grade API endpoints integrating scheduler, predictor, policy, health, and orchestrator engines
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import time
import json
import logging

# Import all advanced engines
from backend.scheduler_engine import get_scheduler, SchedulingJob, ResourceRequirement, WorkloadType, NodeCapability
from backend.resource_predictor import get_predictor, MetricDataPoint, MetricType, PredictionHorizon
from backend.policy_engine import get_policy_engine, Policy, PolicyType, PolicyScope, PolicyEnforcement, PolicyPriority, PolicyStatus
from backend.health_manager import get_health_manager, ServiceHealth, ServiceType, HealthStatus, HealthCheck, CheckType
from backend.orchestrator_persistence import get_orchestrator, OrchestratorTask, OrchestratorNode, TaskState, NodeState
from backend.webrtc_streaming import get_webrtc_engine, StreamConfiguration, StreamType, StreamQuality, SecurityLevel as WebRTCSecurityLevel
from backend.memory_fabric import get_memory_fabric, MemoryType, CompressionType, MemoryState
from backend.plugin_framework import get_plugin_framework, PluginType, PluginState, SecurityLevel as PluginSecurityLevel

logger = logging.getLogger(__name__)

# Create router for advanced engine endpoints
advanced_router = APIRouter(prefix="/api/v2", tags=["Advanced Engines"])

# ======================== SCHEDULER ENDPOINTS ========================

@advanced_router.post("/scheduler/jobs/submit")
async def submit_scheduling_job(
    job_data: Dict[str, Any] = Body(...),
    user_id: str = "system"
) -> JSONResponse:
    """Submit a job to the advanced scheduler"""
    try:
        scheduler = get_scheduler()
        
        # Parse job requirements
        requirements = ResourceRequirement(
            cpu_cores=job_data.get('cpu_cores', 1.0),
            memory_gb=job_data.get('memory_gb', 1.0),
            gpu_units=job_data.get('gpu_units', 0),
            storage_gb=job_data.get('storage_gb', 0.0),
            network_mbps=job_data.get('network_mbps', 0.0),
            capabilities=set(NodeCapability(cap) for cap in job_data.get('capabilities', []))
        )
        
        # Create scheduling job
        job = SchedulingJob(
            job_id=job_data['job_id'],
            workload_type=WorkloadType(job_data.get('workload_type', 'mixed')),
            requirements=requirements,
            priority=job_data.get('priority', 0),
            deadline=job_data.get('deadline'),
            estimated_duration=job_data.get('estimated_duration', 3600.0),
            user_id=user_id,
            metadata=job_data.get('metadata', {})
        )
        
        # Submit job
        job_id = await scheduler.submit_job(job)
        
        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "message": f"Job {job_id} submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@advanced_router.get("/scheduler/jobs/{job_id}/schedule")
async def schedule_job(job_id: str) -> JSONResponse:
    """Schedule a specific job"""
    try:
        scheduler = get_scheduler()
        
        # Find job in queue
        job = None
        for _, _, candidate_job in scheduler.job_queue:
            if candidate_job.job_id == job_id:
                job = candidate_job
                break
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Schedule the job
        decision = await scheduler.schedule_job(job)
        
        return JSONResponse({
            "status": "success",
            "scheduling_decision": {
                "job_id": decision.job_id,
                "target_node_id": decision.target_node_id,
                "score": decision.score,
                "reasoning": decision.reasoning,
                "alternative_nodes": decision.alternative_nodes,
                "estimated_start_time": decision.estimated_start_time,
                "estimated_completion_time": decision.estimated_completion_time,
                "resource_allocation": decision.resource_allocation,
                "success": decision.success,
                "error_message": decision.error_message
            }
        })
        
    except Exception as e:
        logger.error(f"Job scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")

@advanced_router.get("/scheduler/metrics")
async def get_scheduler_metrics() -> JSONResponse:
    """Get comprehensive scheduler metrics"""
    try:
        scheduler = get_scheduler()
        metrics = await scheduler.get_scheduling_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get scheduler metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== RESOURCE PREDICTOR ENDPOINTS ========================

@advanced_router.post("/predictor/metrics/record")
async def record_metric(
    node_id: str = Query(...),
    metric_type: str = Query(...),
    value: float = Query(...),
    metadata: Dict[str, Any] = Body(default_factory=dict)
) -> JSONResponse:
    """Record a metric data point for prediction"""
    try:
        predictor = get_predictor()
        
        # Create metric data point
        data_point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            node_id=node_id,
            metric_type=MetricType(metric_type),
            metadata=metadata
        )
        
        # Record the metric
        success = await predictor.record_metric(data_point)
        
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Metric recorded for {node_id}"
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to record metric")
        
    except Exception as e:
        logger.error(f"Metric recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metric recording failed: {str(e)}")

@advanced_router.get("/predictor/predict")
async def predict_resource_usage(
    node_id: str = Query(...),
    metric_type: str = Query(...),
    horizon: str = Query(default="medium_term"),
    force_retrain: bool = Query(default=False)
) -> JSONResponse:
    """Predict resource usage for a node"""
    try:
        predictor = get_predictor()
        
        # Generate prediction
        prediction = await predictor.predict_resource_usage(
            node_id=node_id,
            metric_type=MetricType(metric_type),
            horizon=PredictionHorizon[horizon.upper()],
            force_retrain=force_retrain
        )
        
        return JSONResponse({
            "status": "success",
            "prediction": {
                "metric_type": prediction.metric_type.value,
                "node_id": prediction.node_id,
                "horizon": prediction.horizon.value,
                "predicted_values": prediction.predicted_values,
                "timestamps": prediction.timestamps,
                "confidence_intervals": prediction.confidence_intervals,
                "model_accuracy": prediction.model_accuracy,
                "features_used": prediction.features_used,
                "anomaly_detected": prediction.anomaly_detected,
                "capacity_warning": prediction.capacity_warning,
                "recommendation": prediction.recommendation
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@advanced_router.get("/predictor/capacity/{node_id}")
async def predict_capacity_needs(
    node_id: str,
    resource_types: Optional[List[str]] = Query(default=None)
) -> JSONResponse:
    """Predict capacity planning needs for a node"""
    try:
        predictor = get_predictor()
        
        # Generate capacity predictions
        predictions = await predictor.predict_capacity_needs(
            node_id=node_id,
            resource_types=resource_types
        )
        
        return JSONResponse({
            "status": "success",
            "capacity_predictions": [
                {
                    "resource_type": p.resource_type,
                    "current_utilization": p.current_utilization,
                    "predicted_peak": p.predicted_peak,
                    "time_to_capacity": p.time_to_capacity,
                    "growth_rate": p.growth_rate,
                    "confidence": p.confidence,
                    "recommended_action": p.recommended_action,
                    "scaling_suggestions": p.scaling_suggestions
                }
                for p in predictions
            ]
        })
        
    except Exception as e:
        logger.error(f"Capacity prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capacity prediction failed: {str(e)}")

@advanced_router.get("/predictor/patterns/{node_id}")
async def detect_workload_patterns(node_id: str) -> JSONResponse:
    """Detect workload patterns for a node"""
    try:
        predictor = get_predictor()
        
        # Detect patterns
        patterns = await predictor.detect_workload_patterns(node_id)
        
        return JSONResponse({
            "status": "success",
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "amplitude": p.amplitude,
                    "confidence": p.confidence,
                    "first_observed": p.first_observed,
                    "last_observed": p.last_observed,
                    "occurrence_count": p.occurrence_count
                }
                for p in patterns
            ]
        })
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")

@advanced_router.get("/predictor/metrics")
async def get_predictor_metrics() -> JSONResponse:
    """Get comprehensive predictor metrics"""
    try:
        predictor = get_predictor()
        metrics = await predictor.get_prediction_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get predictor metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== POLICY ENGINE ENDPOINTS ========================

@advanced_router.post("/policies/create")
async def create_policy(policy_data: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Create a new policy"""
    try:
        policy_engine = get_policy_engine()
        
        # Parse policy data (simplified - in production you'd have proper validation)
        policy = Policy(
            policy_id=policy_data['policy_id'],
            name=policy_data['name'],
            description=policy_data.get('description', ''),
            policy_type=PolicyType(policy_data['policy_type']),
            scope=PolicyScope(policy_data['scope']),
            enforcement=PolicyEnforcement(policy_data['enforcement']),
            priority=PolicyPriority(policy_data['priority']),
            status=PolicyStatus(policy_data.get('status', 'active')),
            rules=[],  # Simplified - would parse rules from policy_data
            tags=policy_data.get('tags', []),
            metadata=policy_data.get('metadata', {})
        )
        
        # Create policy
        success = await policy_engine.create_policy(policy)
        
        if success:
            return JSONResponse({
                "status": "success",
                "policy_id": policy.policy_id,
                "message": f"Policy {policy.policy_id} created successfully"
            })
        else:
            raise HTTPException(status_code=400, detail="Policy creation failed")
        
    except Exception as e:
        logger.error(f"Policy creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy creation failed: {str(e)}")

@advanced_router.post("/policies/evaluate")
async def evaluate_policies(
    context: Dict[str, Any] = Body(...),
    scope: Optional[str] = Query(default=None),
    policy_types: Optional[List[str]] = Query(default=None)
) -> JSONResponse:
    """Evaluate policies against a context"""
    try:
        policy_engine = get_policy_engine()
        
        # Convert parameters
        scope_enum = PolicyScope(scope) if scope else None
        types_enum = [PolicyType(t) for t in policy_types] if policy_types else None
        
        # Evaluate policies
        evaluations = await policy_engine.evaluate_policies(
            context=context,
            scope=scope_enum,
            policy_types=types_enum
        )
        
        return JSONResponse({
            "status": "success",
            "evaluations": [
                {
                    "policy_id": e.policy_id,
                    "rule_id": e.rule_id,
                    "evaluation_time": e.evaluation_time,
                    "matched": e.matched,
                    "actions_triggered": e.actions_triggered,
                    "enforcement_action": e.enforcement_action.value,
                    "confidence": e.confidence,
                    "message": e.message
                }
                for e in evaluations
            ]
        })
        
    except Exception as e:
        logger.error(f"Policy evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy evaluation failed: {str(e)}")

@advanced_router.get("/policies")
async def get_policies(
    scope: Optional[str] = Query(default=None),
    policy_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Get policies with optional filters"""
    try:
        policy_engine = get_policy_engine()
        
        # Convert filters
        scope_enum = PolicyScope(scope) if scope else None
        type_enum = PolicyType(policy_type) if policy_type else None
        status_enum = PolicyStatus(status) if status else None
        
        # Get policies
        policies = await policy_engine.get_policies(
            scope=scope_enum,
            policy_type=type_enum,
            status=status_enum
        )
        
        return JSONResponse({
            "status": "success",
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "name": p.name,
                    "description": p.description,
                    "policy_type": p.policy_type.value,
                    "scope": p.scope.value,
                    "enforcement": p.enforcement.value,
                    "priority": p.priority.value,
                    "status": p.status.value,
                    "version": p.version,
                    "created_by": p.created_by,
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                    "tags": p.tags
                }
                for p in policies
            ]
        })
        
    except Exception as e:
        logger.error(f"Failed to get policies: {e}")
        raise HTTPException(status_code=500, detail=f"Policy retrieval failed: {str(e)}")

@advanced_router.get("/policies/violations")
async def get_policy_violations(
    policy_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    resolved: Optional[bool] = Query(default=None),
    limit: int = Query(default=100)
) -> JSONResponse:
    """Get policy violations"""
    try:
        policy_engine = get_policy_engine()
        
        # Get violations
        violations = await policy_engine.get_violations(
            policy_id=policy_id,
            user_id=user_id,
            node_id=node_id,
            resolved=resolved,
            limit=limit
        )
        
        return JSONResponse({
            "status": "success",
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "policy_id": v.policy_id,
                    "rule_id": v.rule_id,
                    "severity": v.severity,
                    "description": v.description,
                    "timestamp": v.timestamp,
                    "resolved": v.resolved,
                    "user_id": v.user_id,
                    "node_id": v.node_id
                }
                for v in violations
            ]
        })
        
    except Exception as e:
        logger.error(f"Failed to get violations: {e}")
        raise HTTPException(status_code=500, detail=f"Violations retrieval failed: {str(e)}")

@advanced_router.get("/policies/metrics")
async def get_policy_metrics() -> JSONResponse:
    """Get comprehensive policy engine metrics"""
    try:
        policy_engine = get_policy_engine()
        metrics = await policy_engine.get_policy_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get policy metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== HEALTH MANAGER ENDPOINTS ========================

@advanced_router.post("/health/services/register")
async def register_service_for_health_monitoring(
    service_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Register a service for health monitoring"""
    try:
        health_manager = get_health_manager()
        
        # Create service health object
        service = ServiceHealth(
            service_id=service_data['service_id'],
            service_name=service_data['service_name'],
            service_type=ServiceType(service_data['service_type']),
            node_id=service_data['node_id'],
            overall_status=HealthStatus.UNKNOWN,
            health_checks=[],
            version=service_data.get('version', '1.0.0'),
            process_id=service_data.get('process_id'),
            check_interval=service_data.get('check_interval', 60.0),
            alert_enabled=service_data.get('alert_enabled', True),
            auto_recovery=service_data.get('auto_recovery', False)
        )
        
        # Register service
        success = await health_manager.register_service(service)
        
        if success:
            return JSONResponse({
                "status": "success",
                "service_id": service.service_id,
                "message": f"Service {service.service_id} registered for health monitoring"
            })
        else:
            raise HTTPException(status_code=400, detail="Service registration failed")
        
    except Exception as e:
        logger.error(f"Service registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service registration failed: {str(e)}")

@advanced_router.get("/health/services/{service_id}")
async def get_service_health(service_id: str) -> JSONResponse:
    """Get health status for a specific service"""
    try:
        health_manager = get_health_manager()
        
        # Get service health
        service = await health_manager.get_service_health(service_id)
        
        if not service:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
        
        return JSONResponse({
            "status": "success",
            "service_health": {
                "service_id": service.service_id,
                "service_name": service.service_name,
                "service_type": service.service_type.value,
                "node_id": service.node_id,
                "overall_status": service.overall_status.value,
                "version": service.version,
                "uptime_seconds": service.uptime_seconds,
                "last_restart": service.last_restart,
                "cpu_percent": service.cpu_percent,
                "memory_mb": service.memory_mb,
                "memory_percent": service.memory_percent,
                "failure_count": service.failure_count,
                "success_count": service.success_count,
                "last_check": service.last_check,
                "health_checks": [
                    {
                        "check_id": check.check_id,
                        "name": check.name,
                        "check_type": check.check_type.value,
                        "status": check.status.value,
                        "message": check.message,
                        "timestamp": check.timestamp,
                        "duration_ms": check.duration_ms
                    }
                    for check in service.health_checks
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get service health: {e}")
        raise HTTPException(status_code=500, detail=f"Service health retrieval failed: {str(e)}")

@advanced_router.get("/health/system")
async def get_system_health() -> JSONResponse:
    """Get system-wide health status"""
    try:
        health_manager = get_health_manager()
        
        # Get system health
        system_health = await health_manager.get_system_health()
        
        if not system_health:
            raise HTTPException(status_code=404, detail="System health not available")
        
        return JSONResponse({
            "status": "success",
            "system_health": {
                "system_id": system_health.system_id,
                "timestamp": system_health.timestamp,
                "overall_status": system_health.overall_status.value,
                "total_services": system_health.total_services,
                "healthy_services": system_health.healthy_services,
                "warning_services": system_health.warning_services,
                "critical_services": system_health.critical_services,
                "system_load": system_health.system_load,
                "memory_utilization": system_health.memory_utilization,
                "disk_utilization": system_health.disk_utilization,
                "availability_percent": system_health.availability_percent,
                "health_trend": system_health.health_trend,
                "active_alerts": system_health.active_alerts
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"System health retrieval failed: {str(e)}")

@advanced_router.get("/health/metrics")
async def get_health_metrics() -> JSONResponse:
    """Get comprehensive health system metrics"""
    try:
        health_manager = get_health_manager()
        metrics = await health_manager.get_health_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get health metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== ORCHESTRATOR ENDPOINTS ========================

@advanced_router.post("/orchestrator/nodes/register")
async def register_orchestrator_node(
    node_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Register a node with the orchestrator"""
    try:
        orchestrator = get_orchestrator()
        
        # Create orchestrator node
        node = OrchestratorNode(
            node_id=node_data['node_id'],
            node_name=node_data['node_name'],
            endpoint=node_data['endpoint'],
            state=NodeState.REGISTERING,
            capabilities=node_data.get('capabilities', []),
            last_heartbeat=time.time(),
            total_cpu_cores=node_data.get('total_cpu_cores', 0),
            available_cpu_cores=node_data.get('available_cpu_cores', 0),
            total_memory_gb=node_data.get('total_memory_gb', 0.0),
            available_memory_gb=node_data.get('available_memory_gb', 0.0),
            health_score=1.0,
            metadata=node_data.get('metadata', {}),
            version=node_data.get('version', '1.0.0')
        )
        
        # Register node
        success = await orchestrator.register_node(node)
        
        if success:
            return JSONResponse({
                "status": "success",
                "node_id": node.node_id,
                "message": f"Node {node.node_id} registered successfully"
            })
        else:
            raise HTTPException(status_code=400, detail="Node registration failed")
        
    except Exception as e:
        logger.error(f"Node registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Node registration failed: {str(e)}")

@advanced_router.post("/orchestrator/tasks/submit")
async def submit_orchestrator_task(
    task_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Submit a task to the orchestrator"""
    try:
        orchestrator = get_orchestrator()
        
        # Create orchestrator task
        task = OrchestratorTask(
            task_id=task_data['task_id'],
            task_name=task_data['task_name'],
            task_type=task_data.get('task_type', 'general'),
            state=TaskState.PENDING,
            assigned_node_id=None,
            command=task_data['command'],
            arguments=task_data.get('arguments', []),
            environment=task_data.get('environment', {}),
            working_directory=task_data.get('working_directory', ''),
            cpu_cores=task_data.get('cpu_cores', 1.0),
            memory_gb=task_data.get('memory_gb', 1.0),
            timeout_seconds=task_data.get('timeout_seconds', 3600),
            priority=task_data.get('priority', 0),
            max_retries=task_data.get('max_retries', 3),
            dependencies=task_data.get('dependencies', []),
            user_id=task_data.get('user_id', 'system'),
            labels=task_data.get('labels', {}),
            annotations=task_data.get('annotations', {})
        )
        
        # Submit task
        success = await orchestrator.submit_task(task)
        
        if success:
            return JSONResponse({
                "status": "success",
                "task_id": task.task_id,
                "message": f"Task {task.task_id} submitted successfully"
            })
        else:
            raise HTTPException(status_code=400, detail="Task submission failed")
        
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")

@advanced_router.get("/orchestrator/nodes")
async def get_orchestrator_nodes() -> JSONResponse:
    """Get all orchestrator nodes"""
    try:
        orchestrator = get_orchestrator()
        
        # Get available nodes
        nodes = await orchestrator.get_available_nodes()
        
        return JSONResponse({
            "status": "success",
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_name": node.node_name,
                    "endpoint": node.endpoint,
                    "state": node.state.value,
                    "capabilities": node.capabilities,
                    "last_heartbeat": node.last_heartbeat,
                    "total_cpu_cores": node.total_cpu_cores,
                    "available_cpu_cores": node.available_cpu_cores,
                    "total_memory_gb": node.total_memory_gb,
                    "available_memory_gb": node.available_memory_gb,
                    "health_score": node.health_score,
                    "version": node.version
                }
                for node in nodes
            ]
        })
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Node retrieval failed: {str(e)}")

@advanced_router.get("/orchestrator/tasks")
async def get_orchestrator_tasks(
    state: Optional[str] = Query(default=None),
    limit: int = Query(default=100)
) -> JSONResponse:
    """Get orchestrator tasks"""
    try:
        orchestrator = get_orchestrator()
        
        # Get tasks based on state
        if state == "pending":
            tasks = await orchestrator.get_pending_tasks()
        else:
            # For other states, we'd need to implement a more general method
            tasks = await orchestrator.get_pending_tasks()  # Simplified for demo
        
        return JSONResponse({
            "status": "success",
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "task_type": task.task_type,
                    "state": task.state.value,
                    "assigned_node_id": task.assigned_node_id,
                    "command": task.command,
                    "cpu_cores": task.cpu_cores,
                    "memory_gb": task.memory_gb,
                    "priority": task.priority,
                    "created_at": task.created_at,
                    "user_id": task.user_id
                }
                for task in tasks[:limit]
            ]
        })
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Task retrieval failed: {str(e)}")

@advanced_router.get("/orchestrator/metrics")
async def get_orchestrator_metrics() -> JSONResponse:
    """Get comprehensive orchestrator metrics"""
    try:
        orchestrator = get_orchestrator()
        metrics = await orchestrator.get_orchestrator_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== WEBRTC STREAMING ENDPOINTS ========================

@advanced_router.post("/webrtc/sessions/create")
async def create_streaming_session(
    session_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Create a new WebRTC streaming session"""
    try:
        webrtc_engine = get_webrtc_engine()
        
        # Create stream configuration
        stream_config = StreamConfiguration(
            stream_id=session_data['stream_id'],
            stream_type=StreamType(session_data['stream_type']),
            quality=StreamQuality(session_data.get('quality', 'medium')),
            security_level=WebRTCSecurityLevel(session_data.get('security_level', 'standard')),
            adaptive_bitrate=session_data.get('adaptive_bitrate', True),
            audio_enabled=session_data.get('audio_enabled', True),
            video_enabled=session_data.get('video_enabled', True),
            framerate=session_data.get('framerate', 30)
        )
        
        # Create session
        session_id = await webrtc_engine.create_stream_session(
            host_user_id=session_data['host_user_id'],
            host_node_id=session_data['host_node_id'],
            stream_config=stream_config
        )
        
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "stream_id": stream_config.stream_id
        })
        
    except Exception as e:
        logger.error(f"Failed to create streaming session: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@advanced_router.post("/webrtc/sessions/{session_id}/join")
async def join_streaming_session(
    session_id: str,
    join_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Join a WebRTC streaming session"""
    try:
        webrtc_engine = get_webrtc_engine()
        
        peer_id = await webrtc_engine.join_session(
            session_id=session_id,
            user_id=join_data['user_id'],
            node_id=join_data['node_id'],
            permissions=join_data.get('permissions', ['view'])
        )
        
        return JSONResponse({
            "status": "success",
            "peer_id": peer_id,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Failed to join session: {e}")
        raise HTTPException(status_code=500, detail=f"Session join failed: {str(e)}")

@advanced_router.get("/webrtc/sessions/{session_id}")
async def get_session_info(session_id: str) -> JSONResponse:
    """Get information about a streaming session"""
    try:
        webrtc_engine = get_webrtc_engine()
        session_info = await webrtc_engine.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return JSONResponse({
            "status": "success",
            "session": session_info
        })
        
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=f"Session info retrieval failed: {str(e)}")

@advanced_router.get("/webrtc/metrics")
async def get_webrtc_metrics() -> JSONResponse:
    """Get WebRTC streaming metrics"""
    try:
        webrtc_engine = get_webrtc_engine()
        metrics = await webrtc_engine.get_streaming_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get WebRTC metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== MEMORY FABRIC ENDPOINTS ========================

@advanced_router.post("/memory/regions/allocate")
async def allocate_memory_region(
    allocation_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Allocate a memory region in the fabric"""
    try:
        memory_fabric = get_memory_fabric()
        
        region_id = await memory_fabric.allocate_region(
            size=allocation_data['size'],
            memory_type=MemoryType(allocation_data.get('memory_type', 'system')),
            permissions=set(allocation_data.get('permissions', [])),
            compression=CompressionType(allocation_data.get('compression', 'none'))
        )
        
        if region_id:
            return JSONResponse({
                "status": "success",
                "region_id": region_id,
                "size": allocation_data['size']
            })
        else:
            raise HTTPException(status_code=500, detail="Memory allocation failed")
        
    except Exception as e:
        logger.error(f"Memory allocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Allocation failed: {str(e)}")

@advanced_router.post("/memory/regions/{region_id}/write")
async def write_memory_block(
    region_id: str,
    write_data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Write data to a memory region"""
    try:
        memory_fabric = get_memory_fabric()
        
        # Convert data to bytes (in production, handle proper encoding)
        data = write_data['data'].encode() if isinstance(write_data['data'], str) else write_data['data']
        
        block_id = await memory_fabric.write_block(
            region_id=region_id,
            offset=write_data['offset'],
            data=data
        )
        
        if block_id:
            return JSONResponse({
                "status": "success",
                "block_id": block_id,
                "region_id": region_id
            })
        else:
            raise HTTPException(status_code=500, detail="Memory write failed")
        
    except Exception as e:
        logger.error(f"Memory write failed: {e}")
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}")

@advanced_router.get("/memory/regions/{region_id}/read")
async def read_memory_block(
    region_id: str,
    offset: int = Query(...),
    size: int = Query(...)
) -> JSONResponse:
    """Read data from a memory region"""
    try:
        memory_fabric = get_memory_fabric()
        
        data = await memory_fabric.read_block(
            region_id=region_id,
            offset=offset,
            size=size
        )
        
        if data is not None:
            return JSONResponse({
                "status": "success",
                "data": data.hex(),  # Return as hex string
                "size": len(data)
            })
        else:
            raise HTTPException(status_code=404, detail="Memory block not found")
        
    except Exception as e:
        logger.error(f"Memory read failed: {e}")
        raise HTTPException(status_code=500, detail=f"Read failed: {str(e)}")

@advanced_router.get("/memory/stats")
async def get_memory_stats() -> JSONResponse:
    """Get memory fabric statistics"""
    try:
        memory_fabric = get_memory_fabric()
        stats = await memory_fabric.get_memory_stats()
        
        return JSONResponse({
            "status": "success",
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

# ======================== PLUGIN FRAMEWORK ENDPOINTS ========================

@advanced_router.get("/plugins")
async def get_plugins(
    plugin_type: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Get all plugins with optional filters"""
    try:
        plugin_framework = get_plugin_framework()
        
        # Get all plugin info
        plugins = []
        for plugin_id in plugin_framework.plugins.keys():
            plugin_info = await plugin_framework.get_plugin_info(plugin_id)
            if plugin_info:
                # Apply filters
                if plugin_type and plugin_info['plugin_type'] != plugin_type:
                    continue
                if state and plugin_info['state'] != state:
                    continue
                plugins.append(plugin_info)
        
        return JSONResponse({
            "status": "success",
            "plugins": plugins
        })
        
    except Exception as e:
        logger.error(f"Failed to get plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Plugin retrieval failed: {str(e)}")

@advanced_router.post("/plugins/{plugin_id}/load")
async def load_plugin(plugin_id: str) -> JSONResponse:
    """Load a specific plugin"""
    try:
        plugin_framework = get_plugin_framework()
        
        success = await plugin_framework.load_plugin(plugin_id)
        
        if success:
            return JSONResponse({
                "status": "success",
                "plugin_id": plugin_id,
                "message": f"Plugin {plugin_id} loaded successfully"
            })
        else:
            raise HTTPException(status_code=500, detail="Plugin load failed")
        
    except Exception as e:
        logger.error(f"Plugin load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

@advanced_router.post("/plugins/{plugin_id}/start")
async def start_plugin(plugin_id: str) -> JSONResponse:
    """Start a loaded plugin"""
    try:
        plugin_framework = get_plugin_framework()
        
        success = await plugin_framework.start_plugin(plugin_id)
        
        if success:
            return JSONResponse({
                "status": "success",
                "plugin_id": plugin_id,
                "message": f"Plugin {plugin_id} started successfully"
            })
        else:
            raise HTTPException(status_code=500, detail="Plugin start failed")
        
    except Exception as e:
        logger.error(f"Plugin start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Start failed: {str(e)}")

@advanced_router.post("/plugins/{plugin_id}/stop")
async def stop_plugin(plugin_id: str) -> JSONResponse:
    """Stop an active plugin"""
    try:
        plugin_framework = get_plugin_framework()
        
        success = await plugin_framework.stop_plugin(plugin_id)
        
        if success:
            return JSONResponse({
                "status": "success",
                "plugin_id": plugin_id,
                "message": f"Plugin {plugin_id} stopped successfully"
            })
        else:
            raise HTTPException(status_code=500, detail="Plugin stop failed")
        
    except Exception as e:
        logger.error(f"Plugin stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")

@advanced_router.get("/plugins/framework/metrics")
async def get_plugin_framework_metrics() -> JSONResponse:
    """Get plugin framework metrics"""
    try:
        plugin_framework = get_plugin_framework()
        metrics = await plugin_framework.get_framework_metrics()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get plugin framework metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# ======================== UNIFIED DASHBOARD ENDPOINT ========================

@advanced_router.get("/dashboard/unified")
async def get_unified_dashboard() -> JSONResponse:
    """Get unified dashboard data from all advanced engines"""
    try:
        # Collect data from all engines
        scheduler = get_scheduler()
        predictor = get_predictor()
        policy_engine = get_policy_engine()
        health_manager = get_health_manager()
        orchestrator = get_orchestrator()
        webrtc_engine = get_webrtc_engine()
        memory_fabric = get_memory_fabric()
        plugin_framework = get_plugin_framework()
        
        # Get metrics from all engines
        scheduler_metrics = await scheduler.get_scheduling_metrics()
        predictor_metrics = await predictor.get_prediction_metrics()
        policy_metrics = await policy_engine.get_policy_metrics()
        health_metrics = await health_manager.get_health_metrics()
        orchestrator_metrics = await orchestrator.get_orchestrator_metrics()
        webrtc_metrics = await webrtc_engine.get_streaming_metrics()
        memory_stats = await memory_fabric.get_memory_stats()
        plugin_metrics = await plugin_framework.get_framework_metrics()
        
        # Get system health overview
        system_health = await health_manager.get_system_health()
        
        return JSONResponse({
            "status": "success",
            "dashboard": {
                "timestamp": time.time(),
                "system_overview": {
                    "overall_status": system_health.overall_status.value if system_health else "unknown",
                    "total_services": system_health.total_services if system_health else 0,
                    "availability_percent": system_health.availability_percent if system_health else 0.0
                },
                "scheduler": {
                    "active_jobs": len(scheduler.active_jobs),
                    "queued_jobs": len(scheduler.job_queue),
                    "success_rate": scheduler_metrics.get('cluster_overview', {}).get('success_rate', 0.0)
                },
                "predictor": {
                    "total_predictions": predictor_metrics.get('performance', {}).get('total_predictions', 0),
                    "model_accuracy": predictor_metrics.get('model_quality', {}).get('average_accuracy', 0.0),
                    "anomalies_detected": predictor_metrics.get('anomaly_detection', {}).get('anomalies_detected', 0)
                },
                "policies": {
                    "total_policies": policy_metrics.get('policy_overview', {}).get('total_policies', 0),
                    "active_policies": policy_metrics.get('policy_overview', {}).get('active_policies', 0),
                    "violations": policy_metrics.get('violations_and_conflicts', {}).get('unresolved_violations', 0)
                },
                "health": {
                    "total_services": health_metrics.get('monitoring_overview', {}).get('total_services', 0),
                    "monitoring_active": health_metrics.get('monitoring_overview', {}).get('monitoring_active', False),
                    "success_rate": health_metrics.get('check_performance', {}).get('success_rate', 0.0)
                },
                "orchestrator": {
                    "total_nodes": orchestrator_metrics.get('cluster_status', {}).get('total_nodes', 0),
                    "total_tasks": orchestrator_metrics.get('cluster_status', {}).get('total_tasks', 0),
                    "success_rate": orchestrator_metrics.get('performance_metrics', {}).get('success_rate', 0.0)
                },
                "webrtc": {
                    "active_sessions": webrtc_metrics.get('sessions', {}).get('active_sessions', 0),
                    "total_peers": webrtc_metrics.get('connections', {}).get('active_peers', 0),
                    "streaming_running": webrtc_metrics.get('engine_status', {}).get('running', False),
                    "bytes_transferred": webrtc_metrics.get('performance', {}).get('bytes_transferred', 0)
                },
                "memory_fabric": {
                    "total_regions": memory_stats.get('fabric_status', {}).get('total_regions', 0),
                    "memory_utilization": memory_stats.get('memory_usage', {}).get('utilization_percent', 0.0),
                    "cache_hit_ratio": memory_stats.get('cache_performance', {}).get('hit_ratio', 0.0),
                    "fragmentation": memory_stats.get('memory_usage', {}).get('fragmentation_ratio', 0.0)
                },
                "plugins": {
                    "total_plugins": plugin_metrics.get('framework_status', {}).get('total_plugins', 0),
                    "active_plugins": plugin_metrics.get('plugin_states', {}).get('active', 0),
                    "error_plugins": plugin_metrics.get('plugin_states', {}).get('error', 0),
                    "framework_running": plugin_metrics.get('framework_status', {}).get('running', False)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get unified dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard retrieval failed: {str(e)}")

# Export the router
__all__ = ['advanced_router']
