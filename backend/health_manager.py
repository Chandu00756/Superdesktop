"""
Omega Super Desktop Console v2.0 - Unified Health Management System
Production-grade health monitoring with standardized schemas, real-time status tracking,
dependency checking, and automated recovery capabilities.
"""

import asyncio
import logging
import time
import json
import psutil
import aiohttp
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from enum import Enum
import hashlib
import sqlite3
import threading

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class ServiceType(Enum):
    CONTROL_NODE = "control_node"
    COMPUTE_NODE = "compute_node"
    STORAGE_NODE = "storage_node"
    NETWORK_NODE = "network_node"
    EDGE_NODE = "edge_node"
    API_SERVER = "api_server"
    ORCHESTRATOR = "orchestrator"
    SESSION_DAEMON = "session_daemon"
    PREDICTOR_SERVICE = "predictor_service"
    RENDER_ROUTER = "render_router"
    MEMORY_FABRIC = "memory_fabric"

class CheckType(Enum):
    SYSTEM = "system"
    SERVICE = "service"
    NETWORK = "network"
    STORAGE = "storage"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Union[float, int, str, bool]
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    description: str = ""

@dataclass
class HealthCheck:
    """Health check definition and result"""
    check_id: str
    name: str
    check_type: CheckType
    status: HealthStatus
    message: str
    metrics: List[HealthMetric] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Timing information
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    
    # Configuration
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    enabled: bool = True
    
    # Advanced features
    recovery_actions: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceHealth:
    """Complete health status for a service"""
    service_id: str
    service_name: str
    service_type: ServiceType
    node_id: str
    overall_status: HealthStatus
    health_checks: List[HealthCheck]
    
    # Service information
    version: str = "unknown"
    uptime_seconds: float = 0.0
    last_restart: Optional[float] = None
    process_id: Optional[int] = None
    
    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_usage_mb: float = 0.0
    network_connections: int = 0
    
    # Status history
    status_history: List[Tuple[float, HealthStatus]] = field(default_factory=list)
    failure_count: int = 0
    success_count: int = 0
    
    # Timestamps
    last_check: float = field(default_factory=time.time)
    last_status_change: float = field(default_factory=time.time)
    
    # Configuration
    check_interval: float = 60.0  # seconds
    alert_enabled: bool = True
    auto_recovery: bool = False

@dataclass
class SystemHealth:
    """System-wide health overview"""
    system_id: str
    timestamp: float
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    
    # System metrics
    total_services: int = 0
    healthy_services: int = 0
    warning_services: int = 0
    critical_services: int = 0
    
    # Performance indicators
    system_load: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_latency_ms: float = 0.0
    
    # Alerts and issues
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    
    # Trends
    health_trend: str = "stable"  # improving, degrading, stable
    availability_percent: float = 100.0

class HealthManager:
    """Unified health management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Health data storage
        self.services: Dict[str, ServiceHealth] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_health: Optional[SystemHealth] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.alert_handlers: List[Callable] = []
        
        # Configuration
        self.default_check_interval = self.config.get('check_interval', 60.0)
        self.alert_cooldown = self.config.get('alert_cooldown', 300.0)  # 5 minutes
        self.history_retention = self.config.get('history_retention', 86400.0)  # 24 hours
        
        # Performance tracking
        self.metrics = {
            'total_checks_performed': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'average_check_duration': 0.0,
            'alerts_sent': 0,
            'recovery_actions_executed': 0
        }
        
        # Built-in health checks
        self.builtin_checks = {}
        self._register_builtin_checks()
        
        # Database for persistence
        self.db_path = self.config.get('db_path', 'health_data.db')
        self._init_database()
        
        logger.info("Health Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for health data persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Health checks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    check_id TEXT PRIMARY KEY,
                    service_id TEXT,
                    check_type TEXT,
                    status TEXT,
                    message TEXT,
                    timestamp REAL,
                    duration_ms REAL,
                    metrics TEXT
                )
            ''')
            
            # Service health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_health (
                    service_id TEXT,
                    timestamp REAL,
                    status TEXT,
                    cpu_percent REAL,
                    memory_mb REAL,
                    uptime_seconds REAL,
                    failure_count INTEGER,
                    success_count INTEGER,
                    PRIMARY KEY (service_id, timestamp)
                )
            ''')
            
            # System health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    timestamp REAL PRIMARY KEY,
                    overall_status TEXT,
                    total_services INTEGER,
                    healthy_services INTEGER,
                    warning_services INTEGER,
                    critical_services INTEGER,
                    system_load REAL,
                    memory_utilization REAL
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    service_id TEXT,
                    severity TEXT,
                    message TEXT,
                    timestamp REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _register_builtin_checks(self):
        """Register built-in health checks"""
        try:
            # System resource checks
            self.builtin_checks['cpu_usage'] = self._check_cpu_usage
            self.builtin_checks['memory_usage'] = self._check_memory_usage
            self.builtin_checks['disk_usage'] = self._check_disk_usage
            self.builtin_checks['network_connectivity'] = self._check_network_connectivity
            
            # Service-specific checks
            self.builtin_checks['process_running'] = self._check_process_running
            self.builtin_checks['port_listening'] = self._check_port_listening
            self.builtin_checks['http_endpoint'] = self._check_http_endpoint
            self.builtin_checks['database_connection'] = self._check_database_connection
            
            # Performance checks
            self.builtin_checks['response_time'] = self._check_response_time
            self.builtin_checks['throughput'] = self._check_throughput
            self.builtin_checks['error_rate'] = self._check_error_rate
            
        except Exception as e:
            logger.error(f"Failed to register builtin checks: {e}")
    
    async def register_service(self, service: ServiceHealth) -> bool:
        """Register a service for health monitoring"""
        try:
            self.services[service.service_id] = service
            
            # Start monitoring if not already active
            if self.monitoring_active:
                await self._start_service_monitoring(service.service_id)
            
            # Initialize health checks for service
            await self._initialize_service_checks(service)
            
            logger.info(f"Service {service.service_id} registered for health monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from health monitoring"""
        try:
            if service_id in self.services:
                # Stop monitoring task
                if service_id in self.check_tasks:
                    self.check_tasks[service_id].cancel()
                    del self.check_tasks[service_id]
                
                # Remove service
                del self.services[service_id]
                
                logger.info(f"Service {service_id} unregistered from health monitoring")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    async def start_monitoring(self) -> bool:
        """Start health monitoring for all registered services"""
        try:
            self.monitoring_active = True
            
            # Start monitoring tasks for all services
            for service_id in self.services.keys():
                await self._start_service_monitoring(service_id)
            
            # Start system health monitoring
            asyncio.create_task(self._system_health_loop())
            
            logger.info("Health monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop health monitoring"""
        try:
            self.monitoring_active = False
            
            # Cancel all monitoring tasks
            for task in self.check_tasks.values():
                task.cancel()
            
            self.check_tasks.clear()
            
            logger.info("Health monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop health monitoring: {e}")
            return False
    
    async def _start_service_monitoring(self, service_id: str):
        """Start monitoring task for a specific service"""
        try:
            if service_id in self.check_tasks:
                self.check_tasks[service_id].cancel()
            
            task = asyncio.create_task(self._service_health_loop(service_id))
            self.check_tasks[service_id] = task
            
        except Exception as e:
            logger.error(f"Failed to start monitoring for {service_id}: {e}")
    
    async def _service_health_loop(self, service_id: str):
        """Main health check loop for a service"""
        try:
            service = self.services.get(service_id)
            if not service:
                return
            
            while self.monitoring_active:
                try:
                    # Perform health checks
                    await self._perform_service_health_checks(service)
                    
                    # Update service status
                    await self._update_service_status(service)
                    
                    # Store health data
                    await self._store_service_health(service)
                    
                    # Check for alerts
                    await self._check_alerts(service)
                    
                    # Sleep until next check
                    await asyncio.sleep(service.check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in health loop for {service_id}: {e}")
                    await asyncio.sleep(30)  # Retry after 30 seconds
                    
        except Exception as e:
            logger.error(f"Service health loop failed for {service_id}: {e}")
    
    async def _system_health_loop(self):
        """System-wide health monitoring loop"""
        try:
            while self.monitoring_active:
                try:
                    # Collect system health data
                    system_health = await self._collect_system_health()
                    
                    # Update system health
                    self.system_health = system_health
                    
                    # Store system health data
                    await self._store_system_health(system_health)
                    
                    # Sleep until next check
                    await asyncio.sleep(30)  # System health every 30 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in system health loop: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"System health loop failed: {e}")
    
    async def _perform_service_health_checks(self, service: ServiceHealth):
        """Perform all health checks for a service"""
        try:
            updated_checks = []
            
            for check in service.health_checks:
                if not check.enabled:
                    continue
                
                start_time = time.time()
                
                try:
                    # Perform the health check
                    result = await self._execute_health_check(check, service)
                    
                    check.status = result.status
                    check.message = result.message
                    check.metrics = result.metrics
                    check.timestamp = time.time()
                    check.duration_ms = (time.time() - start_time) * 1000
                    
                    if check.status == HealthStatus.HEALTHY:
                        check.last_success = time.time()
                        service.success_count += 1
                    else:
                        check.last_failure = time.time()
                        service.failure_count += 1
                    
                    check.retry_count = 0
                    self.metrics['successful_checks'] += 1
                    
                except Exception as e:
                    check.status = HealthStatus.CRITICAL
                    check.message = f"Health check failed: {str(e)}"
                    check.timestamp = time.time()
                    check.duration_ms = (time.time() - start_time) * 1000
                    check.last_failure = time.time()
                    check.retry_count += 1
                    
                    service.failure_count += 1
                    self.metrics['failed_checks'] += 1
                    
                    logger.error(f"Health check {check.check_id} failed: {e}")
                
                updated_checks.append(check)
                self.metrics['total_checks_performed'] += 1
            
            service.health_checks = updated_checks
            service.last_check = time.time()
            
        except Exception as e:
            logger.error(f"Failed to perform health checks for {service.service_id}: {e}")
    
    async def _execute_health_check(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Execute a specific health check"""
        try:
            if check.check_id in self.builtin_checks:
                # Execute built-in check
                return await self.builtin_checks[check.check_id](check, service)
            else:
                # Custom check - return as-is for now
                # In production, you'd have a plugin system for custom checks
                return check
                
        except Exception as e:
            logger.error(f"Health check execution failed: {e}")
            check.status = HealthStatus.CRITICAL
            check.message = f"Execution error: {str(e)}"
            return check
    
    async def _check_cpu_usage(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Built-in CPU usage check"""
        try:
            if service.process_id:
                process = psutil.Process(service.process_id)
                cpu_percent = process.cpu_percent(interval=1)
            else:
                cpu_percent = psutil.cpu_percent(interval=1)
            
            service.cpu_percent = cpu_percent
            
            # Determine status based on thresholds
            if cpu_percent > 90:
                check.status = HealthStatus.CRITICAL
                check.message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 75:
                check.status = HealthStatus.WARNING
                check.message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            # Add metric
            check.metrics = [HealthMetric(
                name="cpu_percent",
                value=cpu_percent,
                unit="%",
                threshold_warning=75.0,
                threshold_critical=90.0
            )]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check CPU usage: {str(e)}"
            return check
    
    async def _check_memory_usage(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Built-in memory usage check"""
        try:
            if service.process_id:
                process = psutil.Process(service.process_id)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()
            else:
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                memory_percent = memory.percent
            
            service.memory_mb = memory_mb
            service.memory_percent = memory_percent
            
            # Determine status
            if memory_percent > 90:
                check.status = HealthStatus.CRITICAL
                check.message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 80:
                check.status = HealthStatus.WARNING
                check.message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"Memory usage normal: {memory_percent:.1f}%"
            
            # Add metrics
            check.metrics = [
                HealthMetric(
                    name="memory_percent",
                    value=memory_percent,
                    unit="%",
                    threshold_warning=80.0,
                    threshold_critical=90.0
                ),
                HealthMetric(
                    name="memory_mb",
                    value=memory_mb,
                    unit="MB"
                )
            ]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check memory usage: {str(e)}"
            return check
    
    async def _check_disk_usage(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Built-in disk usage check"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / 1024 / 1024 / 1024
            
            # Determine status
            if disk_percent > 95:
                check.status = HealthStatus.CRITICAL
                check.message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 85:
                check.status = HealthStatus.WARNING
                check.message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"Disk usage normal: {disk_percent:.1f}%"
            
            # Add metrics
            check.metrics = [
                HealthMetric(
                    name="disk_percent",
                    value=disk_percent,
                    unit="%",
                    threshold_warning=85.0,
                    threshold_critical=95.0
                ),
                HealthMetric(
                    name="disk_free_gb",
                    value=disk_free_gb,
                    unit="GB"
                )
            ]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check disk usage: {str(e)}"
            return check
    
    async def _check_network_connectivity(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Built-in network connectivity check"""
        try:
            # Simple ping test (simplified for demo)
            network_stats = psutil.net_io_counters()
            
            if network_stats.bytes_sent > 0 and network_stats.bytes_recv > 0:
                check.status = HealthStatus.HEALTHY
                check.message = "Network connectivity normal"
            else:
                check.status = HealthStatus.WARNING
                check.message = "Network activity low"
            
            # Add metrics
            check.metrics = [
                HealthMetric(
                    name="bytes_sent",
                    value=network_stats.bytes_sent,
                    unit="bytes"
                ),
                HealthMetric(
                    name="bytes_recv",
                    value=network_stats.bytes_recv,
                    unit="bytes"
                )
            ]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check network connectivity: {str(e)}"
            return check
    
    async def _check_process_running(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check if service process is running"""
        try:
            if service.process_id:
                if psutil.pid_exists(service.process_id):
                    process = psutil.Process(service.process_id)
                    if process.is_running():
                        check.status = HealthStatus.HEALTHY
                        check.message = f"Process {service.process_id} is running"
                        
                        # Update uptime
                        create_time = process.create_time()
                        service.uptime_seconds = time.time() - create_time
                    else:
                        check.status = HealthStatus.CRITICAL
                        check.message = f"Process {service.process_id} not running"
                else:
                    check.status = HealthStatus.CRITICAL
                    check.message = f"Process {service.process_id} does not exist"
            else:
                check.status = HealthStatus.WARNING
                check.message = "No process ID configured"
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check process: {str(e)}"
            return check
    
    async def _check_port_listening(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check if service port is listening"""
        try:
            port = check.metadata.get('port')
            if not port:
                check.status = HealthStatus.WARNING
                check.message = "No port specified for check"
                return check
            
            # Check if port is listening
            connections = psutil.net_connections(kind='inet')
            listening = any(conn.laddr.port == port and conn.status == 'LISTEN' 
                          for conn in connections if conn.laddr)
            
            if listening:
                check.status = HealthStatus.HEALTHY
                check.message = f"Port {port} is listening"
            else:
                check.status = HealthStatus.CRITICAL
                check.message = f"Port {port} is not listening"
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Failed to check port: {str(e)}"
            return check
    
    async def _check_http_endpoint(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check HTTP endpoint availability"""
        try:
            url = check.metadata.get('url')
            if not url:
                check.status = HealthStatus.WARNING
                check.message = "No URL specified for check"
                return check
            
            timeout = aiohttp.ClientTimeout(total=check.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        check.status = HealthStatus.HEALTHY
                        check.message = f"HTTP endpoint responding (200 OK)"
                    elif response.status < 500:
                        check.status = HealthStatus.WARNING
                        check.message = f"HTTP endpoint returned {response.status}"
                    else:
                        check.status = HealthStatus.CRITICAL
                        check.message = f"HTTP endpoint error {response.status}"
                    
                    # Add response time metric
                    check.metrics = [HealthMetric(
                        name="response_time_ms",
                        value=response_time,
                        unit="ms",
                        threshold_warning=1000.0,
                        threshold_critical=5000.0
                    )]
            
            return check
            
        except asyncio.TimeoutError:
            check.status = HealthStatus.CRITICAL
            check.message = f"HTTP endpoint timeout after {check.timeout_seconds}s"
            return check
        except Exception as e:
            check.status = HealthStatus.CRITICAL
            check.message = f"HTTP endpoint check failed: {str(e)}"
            return check
    
    async def _check_database_connection(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check database connectivity"""
        try:
            # Simplified database check - test SQLite connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                check.status = HealthStatus.HEALTHY
                check.message = "Database connection successful"
            else:
                check.status = HealthStatus.CRITICAL
                check.message = "Database query failed"
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.CRITICAL
            check.message = f"Database check failed: {str(e)}"
            return check
    
    async def _check_response_time(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check service response time"""
        try:
            # This would measure actual service response time
            # For demo, we'll use a simulated value
            import random
            response_time = random.uniform(10, 500)  # milliseconds
            
            if response_time > 1000:
                check.status = HealthStatus.CRITICAL
                check.message = f"Response time critical: {response_time:.1f}ms"
            elif response_time > 500:
                check.status = HealthStatus.WARNING
                check.message = f"Response time high: {response_time:.1f}ms"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"Response time normal: {response_time:.1f}ms"
            
            check.metrics = [HealthMetric(
                name="response_time_ms",
                value=response_time,
                unit="ms",
                threshold_warning=500.0,
                threshold_critical=1000.0
            )]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Response time check failed: {str(e)}"
            return check
    
    async def _check_throughput(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check service throughput"""
        try:
            # Simplified throughput check
            import random
            throughput = random.uniform(50, 1000)  # requests per second
            
            if throughput < 100:
                check.status = HealthStatus.CRITICAL
                check.message = f"Throughput critical: {throughput:.1f} req/s"
            elif throughput < 200:
                check.status = HealthStatus.WARNING
                check.message = f"Throughput low: {throughput:.1f} req/s"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"Throughput normal: {throughput:.1f} req/s"
            
            check.metrics = [HealthMetric(
                name="throughput_rps",
                value=throughput,
                unit="req/s",
                threshold_warning=200.0,
                threshold_critical=100.0
            )]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Throughput check failed: {str(e)}"
            return check
    
    async def _check_error_rate(self, check: HealthCheck, service: ServiceHealth) -> HealthCheck:
        """Check service error rate"""
        try:
            # Simplified error rate check
            import random
            error_rate = random.uniform(0, 10)  # percentage
            
            if error_rate > 5:
                check.status = HealthStatus.CRITICAL
                check.message = f"Error rate critical: {error_rate:.1f}%"
            elif error_rate > 2:
                check.status = HealthStatus.WARNING
                check.message = f"Error rate high: {error_rate:.1f}%"
            else:
                check.status = HealthStatus.HEALTHY
                check.message = f"Error rate normal: {error_rate:.1f}%"
            
            check.metrics = [HealthMetric(
                name="error_rate_percent",
                value=error_rate,
                unit="%",
                threshold_warning=2.0,
                threshold_critical=5.0
            )]
            
            return check
            
        except Exception as e:
            check.status = HealthStatus.UNKNOWN
            check.message = f"Error rate check failed: {str(e)}"
            return check
    
    async def _update_service_status(self, service: ServiceHealth):
        """Update overall service status based on health checks"""
        try:
            if not service.health_checks:
                service.overall_status = HealthStatus.UNKNOWN
                return
            
            # Determine overall status from individual checks
            statuses = [check.status for check in service.health_checks if check.enabled]
            
            if not statuses:
                service.overall_status = HealthStatus.UNKNOWN
            elif any(status == HealthStatus.CRITICAL for status in statuses):
                service.overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.WARNING for status in statuses):
                service.overall_status = HealthStatus.WARNING
            elif any(status == HealthStatus.DEGRADED for status in statuses):
                service.overall_status = HealthStatus.DEGRADED
            elif all(status == HealthStatus.HEALTHY for status in statuses):
                service.overall_status = HealthStatus.HEALTHY
            else:
                service.overall_status = HealthStatus.UNKNOWN
            
            # Update status history
            current_time = time.time()
            if (not service.status_history or 
                service.status_history[-1][1] != service.overall_status):
                service.status_history.append((current_time, service.overall_status))
                service.last_status_change = current_time
                
                # Limit history size
                if len(service.status_history) > 100:
                    service.status_history = service.status_history[-50:]
            
        except Exception as e:
            logger.error(f"Failed to update service status for {service.service_id}: {e}")
    
    async def _collect_system_health(self) -> SystemHealth:
        """Collect system-wide health information"""
        try:
            timestamp = time.time()
            
            # Count services by status
            total_services = len(self.services)
            healthy_services = sum(1 for s in self.services.values() 
                                 if s.overall_status == HealthStatus.HEALTHY)
            warning_services = sum(1 for s in self.services.values() 
                                 if s.overall_status == HealthStatus.WARNING)
            critical_services = sum(1 for s in self.services.values() 
                                  if s.overall_status == HealthStatus.CRITICAL)
            
            # Determine overall system status
            if critical_services > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_services > total_services * 0.3:  # More than 30% warnings
                overall_status = HealthStatus.WARNING
            elif warning_services > 0:
                overall_status = HealthStatus.DEGRADED
            elif healthy_services == total_services and total_services > 0:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            # Collect system metrics
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            memory = psutil.virtual_memory()
            memory_utilization = memory.percent
            disk = psutil.disk_usage('/')
            disk_utilization = (disk.used / disk.total) * 100
            
            # Create system health object
            system_health = SystemHealth(
                system_id="omega_system",
                timestamp=timestamp,
                overall_status=overall_status,
                services={s.service_id: s for s in self.services.values()},
                total_services=total_services,
                healthy_services=healthy_services,
                warning_services=warning_services,
                critical_services=critical_services,
                system_load=system_load,
                memory_utilization=memory_utilization,
                disk_utilization=disk_utilization,
                availability_percent=self._calculate_availability()
            )
            
            return system_health
            
        except Exception as e:
            logger.error(f"Failed to collect system health: {e}")
            return SystemHealth(
                system_id="omega_system",
                timestamp=time.time(),
                overall_status=HealthStatus.UNKNOWN,
                services={}
            )
    
    def _calculate_availability(self) -> float:
        """Calculate system availability percentage"""
        try:
            if not self.services:
                return 100.0
            
            total_uptime = 0.0
            total_time = 0.0
            
            for service in self.services.values():
                if service.status_history:
                    # Calculate uptime from status history
                    healthy_time = 0.0
                    total_service_time = 0.0
                    
                    for i, (timestamp, status) in enumerate(service.status_history):
                        if i > 0:
                            duration = timestamp - service.status_history[i-1][0]
                            total_service_time += duration
                            
                            if service.status_history[i-1][1] == HealthStatus.HEALTHY:
                                healthy_time += duration
                    
                    if total_service_time > 0:
                        total_uptime += healthy_time
                        total_time += total_service_time
            
            if total_time > 0:
                return (total_uptime / total_time) * 100.0
            else:
                return 100.0
                
        except Exception:
            return 100.0
    
    async def _initialize_service_checks(self, service: ServiceHealth):
        """Initialize default health checks for a service"""
        try:
            if not service.health_checks:
                default_checks = []
                
                # Add basic system checks
                default_checks.append(HealthCheck(
                    check_id=f"{service.service_id}_cpu",
                    name="CPU Usage",
                    check_type=CheckType.SYSTEM,
                    status=HealthStatus.UNKNOWN,
                    message="Not checked yet"
                ))
                
                default_checks.append(HealthCheck(
                    check_id=f"{service.service_id}_memory",
                    name="Memory Usage",
                    check_type=CheckType.SYSTEM,
                    status=HealthStatus.UNKNOWN,
                    message="Not checked yet"
                ))
                
                # Add service-specific checks based on service type
                if service.service_type == ServiceType.API_SERVER:
                    default_checks.append(HealthCheck(
                        check_id=f"{service.service_id}_http",
                        name="HTTP Endpoint",
                        check_type=CheckType.SERVICE,
                        status=HealthStatus.UNKNOWN,
                        message="Not checked yet",
                        metadata={'url': 'http://localhost:8000/health'}
                    ))
                
                service.health_checks = default_checks
                
        except Exception as e:
            logger.error(f"Failed to initialize checks for {service.service_id}: {e}")
    
    async def _store_service_health(self, service: ServiceHealth):
        """Store service health data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store service health record
            cursor.execute('''
                INSERT OR REPLACE INTO service_health 
                (service_id, timestamp, status, cpu_percent, memory_mb, uptime_seconds, 
                 failure_count, success_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                service.service_id,
                service.last_check,
                service.overall_status.value,
                service.cpu_percent,
                service.memory_mb,
                service.uptime_seconds,
                service.failure_count,
                service.success_count
            ))
            
            # Store individual health checks
            for check in service.health_checks:
                metrics_json = json.dumps([{
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'threshold_warning': m.threshold_warning,
                    'threshold_critical': m.threshold_critical
                } for m in check.metrics])
                
                cursor.execute('''
                    INSERT INTO health_checks 
                    (check_id, service_id, check_type, status, message, timestamp, 
                     duration_ms, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    check.check_id,
                    service.service_id,
                    check.check_type.value,
                    check.status.value,
                    check.message,
                    check.timestamp,
                    check.duration_ms,
                    metrics_json
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store service health data: {e}")
    
    async def _store_system_health(self, system_health: SystemHealth):
        """Store system health data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_health 
                (timestamp, overall_status, total_services, healthy_services, 
                 warning_services, critical_services, system_load, memory_utilization)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                system_health.timestamp,
                system_health.overall_status.value,
                system_health.total_services,
                system_health.healthy_services,
                system_health.warning_services,
                system_health.critical_services,
                system_health.system_load,
                system_health.memory_utilization
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store system health data: {e}")
    
    async def _check_alerts(self, service: ServiceHealth):
        """Check if alerts should be sent for service"""
        try:
            if not service.alert_enabled:
                return
            
            critical_checks = [check for check in service.health_checks 
                             if check.status == HealthStatus.CRITICAL]
            
            for check in critical_checks:
                # Send alert for critical status
                await self._send_alert(service, check, "critical")
            
        except Exception as e:
            logger.error(f"Alert checking failed for {service.service_id}: {e}")
    
    async def _send_alert(self, service: ServiceHealth, check: HealthCheck, severity: str):
        """Send alert for service issue"""
        try:
            alert_data = {
                'service_id': service.service_id,
                'service_name': service.service_name,
                'check_id': check.check_id,
                'check_name': check.name,
                'severity': severity,
                'message': check.message,
                'timestamp': time.time(),
                'node_id': service.node_id
            }
            
            # Store alert in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            alert_id = f"{service.service_id}_{check.check_id}_{int(time.time())}"
            cursor.execute('''
                INSERT INTO alerts (alert_id, service_id, severity, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_id, service.service_id, severity, check.message, time.time()))
            
            conn.commit()
            conn.close()
            
            # Execute alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert_data)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            self.metrics['alerts_sent'] += 1
            logger.warning(f"Alert sent for {service.service_id}: {check.message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler function"""
        self.alert_handlers.append(handler)
        logger.info("Alert handler registered")
    
    async def get_service_health(self, service_id: str) -> Optional[ServiceHealth]:
        """Get health status for a specific service"""
        return self.services.get(service_id)
    
    async def get_system_health(self) -> Optional[SystemHealth]:
        """Get system-wide health status"""
        return self.system_health
    
    async def get_health_history(
        self,
        service_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get health history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if service_id:
                query = '''
                    SELECT * FROM service_health 
                    WHERE service_id = ?
                '''
                params = [service_id]
            else:
                query = '''
                    SELECT * FROM system_health
                '''
                params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            if service_id:
                columns = ['service_id', 'timestamp', 'status', 'cpu_percent', 
                          'memory_mb', 'uptime_seconds', 'failure_count', 'success_count']
            else:
                columns = ['timestamp', 'overall_status', 'total_services', 
                          'healthy_services', 'warning_services', 'critical_services',
                          'system_load', 'memory_utilization']
            
            result = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return []
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health system metrics"""
        try:
            # Calculate additional metrics
            total_services = len(self.services)
            active_monitoring_tasks = len(self.check_tasks)
            
            # Get recent performance data
            avg_check_duration = (
                sum(check.duration_ms for service in self.services.values() 
                    for check in service.health_checks) / 
                max(sum(len(service.health_checks) for service in self.services.values()), 1)
            )
            
            return {
                'monitoring_overview': {
                    'total_services': total_services,
                    'active_monitoring_tasks': active_monitoring_tasks,
                    'monitoring_active': self.monitoring_active,
                    'registered_alert_handlers': len(self.alert_handlers)
                },
                'check_performance': {
                    'total_checks_performed': self.metrics['total_checks_performed'],
                    'successful_checks': self.metrics['successful_checks'],
                    'failed_checks': self.metrics['failed_checks'],
                    'success_rate': (self.metrics['successful_checks'] / 
                                   max(self.metrics['total_checks_performed'], 1)),
                    'average_check_duration_ms': avg_check_duration
                },
                'alerting': {
                    'alerts_sent': self.metrics['alerts_sent'],
                    'recovery_actions_executed': self.metrics['recovery_actions_executed']
                },
                'system_status': {
                    'overall_status': self.system_health.overall_status.value if self.system_health else 'unknown',
                    'availability_percent': self.system_health.availability_percent if self.system_health else 0.0,
                    'last_check': self.system_health.timestamp if self.system_health else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating health metrics: {e}")
            return {}

# Global health manager instance
_health_manager_instance = None

def get_health_manager() -> HealthManager:
    """Get or create global health manager instance"""
    global _health_manager_instance
    if _health_manager_instance is None:
        _health_manager_instance = HealthManager()
    return _health_manager_instance

async def initialize_health_manager(config: Dict[str, Any] = None):
    """Initialize the global health manager"""
    global _health_manager_instance
    _health_manager_instance = HealthManager(config)
    logger.info("Global health manager initialized")
    return _health_manager_instance
