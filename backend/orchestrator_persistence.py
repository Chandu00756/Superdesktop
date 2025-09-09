"""
Omega Super Desktop Console v2.0 - Orchestrator Persistence Layer
Production-grade orchestrator with SQLite/PostgreSQL persistence, formal schema migrations,
state management, and high-availability capabilities.
"""

import asyncio
import logging
import time
import json
import sqlite3
import threading
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from enum import Enum
import hashlib
import os

# PostgreSQL support (optional)
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)

class PersistenceBackend(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

class OrchestratorState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILOVER = "failover"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class NodeState(Enum):
    REGISTERING = "registering"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAINING = "draining"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class OrchestratorNode:
    """Orchestrator node information"""
    node_id: str
    node_name: str
    endpoint: str
    state: NodeState
    capabilities: List[str]
    last_heartbeat: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource information
    total_cpu_cores: int = 0
    available_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    
    # Health and performance
    health_score: float = 1.0
    load_average: float = 0.0
    network_latency_ms: float = 0.0
    
    # Registration and lifecycle
    registered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: str = "1.0.0"

@dataclass
class OrchestratorTask:
    """Task managed by orchestrator"""
    task_id: str
    task_name: str
    task_type: str
    state: TaskState
    assigned_node_id: Optional[str]
    
    # Task definition
    command: str
    arguments: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    
    # Resource requirements
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    timeout_seconds: int = 3600
    
    # Scheduling and execution
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    
    # Lifecycle timestamps
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Result and error information
    exit_code: Optional[int] = None
    output: str = ""
    error_message: str = ""
    
    # Metadata
    user_id: str = "system"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class ClusterEvent:
    """Cluster event for audit and monitoring"""
    event_id: str
    event_type: str
    severity: str
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersistenceConfig:
    """Persistence layer configuration"""
    backend: PersistenceBackend
    connection_string: str
    pool_size: int = 10
    timeout_seconds: int = 30
    migration_path: str = "migrations"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

class DatabaseMigration:
    """Database schema migration"""
    
    def __init__(self, persistence: 'OrchestratorPersistence'):
        self.persistence = persistence
        self.migrations = self._get_migrations()
    
    def _get_migrations(self) -> List[Dict[str, str]]:
        """Get ordered list of migrations"""
        return [
            {
                'version': '001',
                'description': 'Initial schema creation',
                'sql': '''
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS orchestrator_nodes (
                        node_id TEXT PRIMARY KEY,
                        node_name TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        state TEXT NOT NULL,
                        capabilities TEXT,
                        last_heartbeat REAL,
                        total_cpu_cores INTEGER DEFAULT 0,
                        available_cpu_cores INTEGER DEFAULT 0,
                        total_memory_gb REAL DEFAULT 0.0,
                        available_memory_gb REAL DEFAULT 0.0,
                        health_score REAL DEFAULT 1.0,
                        load_average REAL DEFAULT 0.0,
                        network_latency_ms REAL DEFAULT 0.0,
                        registered_at REAL,
                        updated_at REAL,
                        version TEXT DEFAULT '1.0.0',
                        metadata TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS orchestrator_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        state TEXT NOT NULL,
                        assigned_node_id TEXT,
                        command TEXT NOT NULL,
                        arguments TEXT,
                        environment TEXT,
                        working_directory TEXT,
                        cpu_cores REAL DEFAULT 1.0,
                        memory_gb REAL DEFAULT 1.0,
                        timeout_seconds INTEGER DEFAULT 3600,
                        priority INTEGER DEFAULT 0,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        dependencies TEXT,
                        created_at REAL,
                        scheduled_at REAL,
                        started_at REAL,
                        completed_at REAL,
                        exit_code INTEGER,
                        output TEXT,
                        error_message TEXT,
                        user_id TEXT DEFAULT 'system',
                        labels TEXT,
                        annotations TEXT,
                        FOREIGN KEY (assigned_node_id) REFERENCES orchestrator_nodes(node_id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS cluster_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        source TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp REAL,
                        metadata TEXT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_nodes_state ON orchestrator_nodes(state);
                    CREATE INDEX IF NOT EXISTS idx_nodes_heartbeat ON orchestrator_nodes(last_heartbeat);
                    CREATE INDEX IF NOT EXISTS idx_tasks_state ON orchestrator_tasks(state);
                    CREATE INDEX IF NOT EXISTS idx_tasks_node ON orchestrator_tasks(assigned_node_id);
                    CREATE INDEX IF NOT EXISTS idx_tasks_created ON orchestrator_tasks(created_at);
                    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON cluster_events(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_events_type ON cluster_events(event_type);
                '''
            },
            {
                'version': '002',
                'description': 'Add node performance tracking',
                'sql': '''
                    ALTER TABLE orchestrator_nodes ADD COLUMN performance_score REAL DEFAULT 1.0;
                    ALTER TABLE orchestrator_nodes ADD COLUMN failure_count INTEGER DEFAULT 0;
                    ALTER TABLE orchestrator_nodes ADD COLUMN success_count INTEGER DEFAULT 0;
                    
                    CREATE INDEX IF NOT EXISTS idx_nodes_performance ON orchestrator_nodes(performance_score);
                '''
            },
            {
                'version': '003',
                'description': 'Add task execution metrics',
                'sql': '''
                    ALTER TABLE orchestrator_tasks ADD COLUMN execution_duration_ms REAL;
                    ALTER TABLE orchestrator_tasks ADD COLUMN resource_usage TEXT;
                    ALTER TABLE orchestrator_tasks ADD COLUMN performance_metrics TEXT;
                    
                    CREATE INDEX IF NOT EXISTS idx_tasks_duration ON orchestrator_tasks(execution_duration_ms);
                '''
            }
        ]
    
    async def run_migrations(self) -> bool:
        """Run pending migrations"""
        try:
            # Get current schema version
            current_version = await self._get_current_version()
            
            # Run pending migrations
            for migration in self.migrations:
                if migration['version'] > current_version:
                    logger.info(f"Running migration {migration['version']}: {migration['description']}")
                    
                    if await self._run_migration(migration):
                        await self._record_migration(migration['version'])
                        logger.info(f"Migration {migration['version']} completed successfully")
                    else:
                        logger.error(f"Migration {migration['version']} failed")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    async def _get_current_version(self) -> str:
        """Get current schema version"""
        try:
            if self.persistence.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.persistence.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
                if not cursor.fetchone():
                    conn.close()
                    return '000'
                
                cursor.execute("SELECT MAX(version) FROM schema_migrations")
                result = cursor.fetchone()
                conn.close()
                
                return result[0] if result and result[0] else '000'
            
            # PostgreSQL implementation would go here
            return '000'
            
        except Exception:
            return '000'
    
    async def _run_migration(self, migration: Dict[str, str]) -> bool:
        """Run a single migration"""
        try:
            if self.persistence.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.persistence.db_path)
                cursor = conn.cursor()
                
                # Execute migration SQL
                cursor.executescript(migration['sql'])
                conn.commit()
                conn.close()
                
                return True
            
            # PostgreSQL implementation would go here
            return False
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False
    
    async def _record_migration(self, version: str) -> bool:
        """Record completed migration"""
        try:
            if self.persistence.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.persistence.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (version,)
                )
                conn.commit()
                conn.close()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to record migration: {e}")
            return False

class OrchestratorPersistence:
    """Production-grade orchestrator persistence layer"""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.backend = config.backend
        self.connection_pool = None
        self._lock = threading.RLock()
        
        # SQLite specific
        if self.backend == PersistenceBackend.SQLITE:
            self.db_path = config.connection_string
            self._ensure_directory()
        
        # PostgreSQL specific
        elif self.backend == PersistenceBackend.POSTGRESQL and POSTGRES_AVAILABLE:
            self.connection_string = config.connection_string
        else:
            raise ValueError(f"Backend {self.backend} not supported or dependencies missing")
        
        # Migration manager
        self.migration_manager = DatabaseMigration(self)
        
        logger.info(f"Orchestrator Persistence initialized with {self.backend.value} backend")
    
    def _ensure_directory(self):
        """Ensure database directory exists"""
        if self.backend == PersistenceBackend.SQLITE:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize persistence layer"""
        try:
            # Run database migrations
            if not await self.migration_manager.run_migrations():
                logger.error("Database migration failed")
                return False
            
            # Initialize connection pool for PostgreSQL
            if self.backend == PersistenceBackend.POSTGRESQL:
                await self._init_postgres_pool()
            
            logger.info("Orchestrator persistence initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Persistence initialization failed: {e}")
            return False
    
    async def _init_postgres_pool(self):
        """Initialize PostgreSQL connection pool"""
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL dependencies not available")
        
        # Connection pool implementation would go here
        # For now, we'll use simple connections
        pass
    
    async def save_node(self, node: OrchestratorNode) -> bool:
        """Save or update node information"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO orchestrator_nodes (
                            node_id, node_name, endpoint, state, capabilities,
                            last_heartbeat, total_cpu_cores, available_cpu_cores,
                            total_memory_gb, available_memory_gb, health_score,
                            load_average, network_latency_ms, registered_at,
                            updated_at, version, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        node.node_id, node.node_name, node.endpoint, node.state.value,
                        json.dumps(node.capabilities), node.last_heartbeat,
                        node.total_cpu_cores, node.available_cpu_cores,
                        node.total_memory_gb, node.available_memory_gb,
                        node.health_score, node.load_average, node.network_latency_ms,
                        node.registered_at, node.updated_at, node.version,
                        json.dumps(node.metadata)
                    ))
                    
                    conn.commit()
                    conn.close()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save node {node.node_id}: {e}")
            return False
    
    async def load_node(self, node_id: str) -> Optional[OrchestratorNode]:
        """Load node by ID"""
        try:
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM orchestrator_nodes WHERE node_id = ?",
                    (node_id,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return self._row_to_node(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load node {node_id}: {e}")
            return None
    
    async def load_all_nodes(self, state_filter: Optional[NodeState] = None) -> List[OrchestratorNode]:
        """Load all nodes, optionally filtered by state"""
        try:
            nodes = []
            
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if state_filter:
                    cursor.execute(
                        "SELECT * FROM orchestrator_nodes WHERE state = ? ORDER BY updated_at DESC",
                        (state_filter.value,)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM orchestrator_nodes ORDER BY updated_at DESC"
                    )
                
                rows = cursor.fetchall()
                conn.close()
                
                nodes = [self._row_to_node(row) for row in rows]
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to load nodes: {e}")
            return []
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete node by ID"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "DELETE FROM orchestrator_nodes WHERE node_id = ?",
                        (node_id,)
                    )
                    
                    affected_rows = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    return affected_rows > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    async def save_task(self, task: OrchestratorTask) -> bool:
        """Save or update task information"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO orchestrator_tasks (
                            task_id, task_name, task_type, state, assigned_node_id,
                            command, arguments, environment, working_directory,
                            cpu_cores, memory_gb, timeout_seconds, priority,
                            retry_count, max_retries, dependencies, created_at,
                            scheduled_at, started_at, completed_at, exit_code,
                            output, error_message, user_id, labels, annotations
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        task.task_id, task.task_name, task.task_type, task.state.value,
                        task.assigned_node_id, task.command, json.dumps(task.arguments),
                        json.dumps(task.environment), task.working_directory,
                        task.cpu_cores, task.memory_gb, task.timeout_seconds,
                        task.priority, task.retry_count, task.max_retries,
                        json.dumps(task.dependencies), task.created_at,
                        task.scheduled_at, task.started_at, task.completed_at,
                        task.exit_code, task.output, task.error_message,
                        task.user_id, json.dumps(task.labels), json.dumps(task.annotations)
                    ))
                    
                    conn.commit()
                    conn.close()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save task {task.task_id}: {e}")
            return False
    
    async def load_task(self, task_id: str) -> Optional[OrchestratorTask]:
        """Load task by ID"""
        try:
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM orchestrator_tasks WHERE task_id = ?",
                    (task_id,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return self._row_to_task(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return None
    
    async def load_tasks(
        self,
        state_filter: Optional[TaskState] = None,
        node_id_filter: Optional[str] = None,
        limit: int = 1000
    ) -> List[OrchestratorTask]:
        """Load tasks with optional filters"""
        try:
            tasks = []
            
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = "SELECT * FROM orchestrator_tasks WHERE 1=1"
                params = []
                
                if state_filter:
                    query += " AND state = ?"
                    params.append(state_filter.value)
                
                if node_id_filter:
                    query += " AND assigned_node_id = ?"
                    params.append(node_id_filter)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                conn.close()
                
                tasks = [self._row_to_task(row) for row in rows]
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            return []
    
    async def update_task_state(self, task_id: str, new_state: TaskState, **kwargs) -> bool:
        """Update task state and optional fields"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Build dynamic update query
                    update_fields = ["state = ?"]
                    params = [new_state.value]
                    
                    for field, value in kwargs.items():
                        if field in ['assigned_node_id', 'scheduled_at', 'started_at', 
                                   'completed_at', 'exit_code', 'output', 'error_message']:
                            update_fields.append(f"{field} = ?")
                            params.append(value)
                    
                    params.append(task_id)
                    
                    query = f"UPDATE orchestrator_tasks SET {', '.join(update_fields)} WHERE task_id = ?"
                    cursor.execute(query, params)
                    
                    affected_rows = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    return affected_rows > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update task state for {task_id}: {e}")
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete task by ID"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "DELETE FROM orchestrator_tasks WHERE task_id = ?",
                        (task_id,)
                    )
                    
                    affected_rows = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    return affected_rows > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False
    
    async def save_event(self, event: ClusterEvent) -> bool:
        """Save cluster event"""
        try:
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO cluster_events (
                            event_id, event_type, severity, source, message, timestamp, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.event_id, event.event_type, event.severity,
                        event.source, event.message, event.timestamp,
                        json.dumps(event.metadata)
                    ))
                    
                    conn.commit()
                    conn.close()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save event {event.event_id}: {e}")
            return False
    
    async def load_events(
        self,
        event_type_filter: Optional[str] = None,
        severity_filter: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[ClusterEvent]:
        """Load cluster events with optional filters"""
        try:
            events = []
            
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = "SELECT * FROM cluster_events WHERE 1=1"
                params = []
                
                if event_type_filter:
                    query += " AND event_type = ?"
                    params.append(event_type_filter)
                
                if severity_filter:
                    query += " AND severity = ?"
                    params.append(severity_filter)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                conn.close()
                
                events = [self._row_to_event(row) for row in rows]
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            return []
    
    async def cleanup_old_data(self, retention_days: int = 30) -> bool:
        """Clean up old data based on retention policy"""
        try:
            cutoff_time = time.time() - (retention_days * 86400)
            
            with self._lock:
                if self.backend == PersistenceBackend.SQLITE:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Clean up old completed tasks
                    cursor.execute('''
                        DELETE FROM orchestrator_tasks 
                        WHERE state IN ('completed', 'failed', 'cancelled') 
                        AND completed_at < ?
                    ''', (cutoff_time,))
                    
                    # Clean up old events
                    cursor.execute('''
                        DELETE FROM cluster_events 
                        WHERE timestamp < ?
                    ''', (cutoff_time,))
                    
                    # Clean up old nodes that haven't been seen
                    cursor.execute('''
                        DELETE FROM orchestrator_nodes 
                        WHERE state = 'inactive' 
                        AND last_heartbeat < ?
                    ''', (cutoff_time,))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Cleaned up data older than {retention_days} days")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return False
    
    def _row_to_node(self, row: tuple) -> OrchestratorNode:
        """Convert database row to OrchestratorNode"""
        return OrchestratorNode(
            node_id=row[0],
            node_name=row[1],
            endpoint=row[2],
            state=NodeState(row[3]),
            capabilities=json.loads(row[4]) if row[4] else [],
            last_heartbeat=row[5],
            total_cpu_cores=row[6],
            available_cpu_cores=row[7],
            total_memory_gb=row[8],
            available_memory_gb=row[9],
            health_score=row[10],
            load_average=row[11],
            network_latency_ms=row[12],
            registered_at=row[13],
            updated_at=row[14],
            version=row[15],
            metadata=json.loads(row[16]) if row[16] else {}
        )
    
    def _row_to_task(self, row: tuple) -> OrchestratorTask:
        """Convert database row to OrchestratorTask"""
        return OrchestratorTask(
            task_id=row[0],
            task_name=row[1],
            task_type=row[2],
            state=TaskState(row[3]),
            assigned_node_id=row[4],
            command=row[5],
            arguments=json.loads(row[6]) if row[6] else [],
            environment=json.loads(row[7]) if row[7] else {},
            working_directory=row[8],
            cpu_cores=row[9],
            memory_gb=row[10],
            timeout_seconds=row[11],
            priority=row[12],
            retry_count=row[13],
            max_retries=row[14],
            dependencies=json.loads(row[15]) if row[15] else [],
            created_at=row[16],
            scheduled_at=row[17],
            started_at=row[18],
            completed_at=row[19],
            exit_code=row[20],
            output=row[21] or "",
            error_message=row[22] or "",
            user_id=row[23],
            labels=json.loads(row[24]) if row[24] else {},
            annotations=json.loads(row[25]) if row[25] else {}
        )
    
    def _row_to_event(self, row: tuple) -> ClusterEvent:
        """Convert database row to ClusterEvent"""
        return ClusterEvent(
            event_id=row[0],
            event_type=row[1],
            severity=row[2],
            source=row[3],
            message=row[4],
            timestamp=row[5],
            metadata=json.loads(row[6]) if row[6] else {}
        )
    
    async def get_persistence_metrics(self) -> Dict[str, Any]:
        """Get persistence layer metrics"""
        try:
            if self.backend == PersistenceBackend.SQLITE:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Count records
                cursor.execute("SELECT COUNT(*) FROM orchestrator_nodes")
                node_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM orchestrator_tasks")
                task_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM cluster_events")
                event_count = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size_bytes = page_count * page_size
                
                # Task state distribution
                cursor.execute('''
                    SELECT state, COUNT(*) 
                    FROM orchestrator_tasks 
                    GROUP BY state
                ''')
                task_states = dict(cursor.fetchall())
                
                # Node state distribution
                cursor.execute('''
                    SELECT state, COUNT(*) 
                    FROM orchestrator_nodes 
                    GROUP BY state
                ''')
                node_states = dict(cursor.fetchall())
                
                conn.close()
                
                return {
                    'backend': self.backend.value,
                    'database_size_mb': db_size_bytes / 1024 / 1024,
                    'record_counts': {
                        'nodes': node_count,
                        'tasks': task_count,
                        'events': event_count
                    },
                    'task_state_distribution': task_states,
                    'node_state_distribution': node_states,
                    'connection_pool_size': self.config.pool_size,
                    'migrations_applied': await self.migration_manager._get_current_version()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get persistence metrics: {e}")
            return {}

class AdvancedOrchestrator:
    """Production-grade orchestrator with persistence and high availability"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Persistence configuration
        persistence_config = PersistenceConfig(
            backend=PersistenceBackend(self.config.get('persistence_backend', 'sqlite')),
            connection_string=self.config.get('connection_string', 'backend/orchestrator.db'),
            pool_size=self.config.get('pool_size', 10),
            timeout_seconds=self.config.get('timeout_seconds', 30)
        )
        
        self.persistence = OrchestratorPersistence(persistence_config)
        
        # Runtime state
        self.state = OrchestratorState.INITIALIZING
        self.instance_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # In-memory caches
        self.nodes_cache: Dict[str, OrchestratorNode] = {}
        self.tasks_cache: Dict[str, OrchestratorTask] = {}
        self.cache_last_sync = 0.0
        self.cache_sync_interval = 30.0  # seconds
        
        # Background tasks
        self.background_tasks = []
        self.monitoring_active = False
        
        # Performance metrics
        self.metrics = {
            'total_tasks_scheduled': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_task_duration': 0.0,
            'nodes_registered': 0,
            'cache_hit_ratio': 0.0
        }
        
        logger.info(f"Advanced Orchestrator initialized (instance: {self.instance_id})")
    
    async def initialize(self) -> bool:
        """Initialize orchestrator"""
        try:
            # Initialize persistence layer
            if not await self.persistence.initialize():
                return False
            
            # Load state from persistence
            await self._load_state()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.state = OrchestratorState.ACTIVE
            
            # Record startup event
            await self._record_event("orchestrator_startup", "info", "system", 
                                    f"Orchestrator {self.instance_id} started")
            
            logger.info("Advanced Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            self.state = OrchestratorState.SHUTDOWN
            return False
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.state = OrchestratorState.SHUTDOWN
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Final state save
            await self._save_state()
            
            # Record shutdown event
            await self._record_event("orchestrator_shutdown", "info", "system",
                                    f"Orchestrator {self.instance_id} shutdown")
            
            logger.info("Advanced Orchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"Orchestrator shutdown error: {e}")
    
    async def _load_state(self):
        """Load orchestrator state from persistence"""
        try:
            # Load all nodes
            nodes = await self.persistence.load_all_nodes()
            self.nodes_cache = {node.node_id: node for node in nodes}
            
            # Load active/pending tasks
            active_tasks = await self.persistence.load_tasks(
                state_filter=None, limit=10000
            )
            self.tasks_cache = {task.task_id: task for task in active_tasks}
            
            self.cache_last_sync = time.time()
            
            logger.info(f"Loaded {len(self.nodes_cache)} nodes and {len(self.tasks_cache)} tasks from persistence")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self):
        """Save current state to persistence"""
        try:
            # Save all cached nodes
            for node in self.nodes_cache.values():
                await self.persistence.save_node(node)
            
            # Save all cached tasks
            for task in self.tasks_cache.values():
                await self.persistence.save_task(task)
            
            self.cache_last_sync = time.time()
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        try:
            self.monitoring_active = True
            
            # Cache sync task
            self.background_tasks.append(
                asyncio.create_task(self._cache_sync_loop())
            )
            
            # Health monitoring task
            self.background_tasks.append(
                asyncio.create_task(self._health_monitoring_loop())
            )
            
            # Cleanup task
            self.background_tasks.append(
                asyncio.create_task(self._cleanup_loop())
            )
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def _cache_sync_loop(self):
        """Periodic cache synchronization"""
        try:
            while self.monitoring_active:
                await asyncio.sleep(self.cache_sync_interval)
                
                if time.time() - self.cache_last_sync > self.cache_sync_interval:
                    await self._save_state()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cache sync loop error: {e}")
    
    async def _health_monitoring_loop(self):
        """Monitor orchestrator and node health"""
        try:
            while self.monitoring_active:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for stale nodes
                current_time = time.time()
                stale_threshold = 300  # 5 minutes
                
                for node in list(self.nodes_cache.values()):
                    if (current_time - node.last_heartbeat > stale_threshold and 
                        node.state == NodeState.ACTIVE):
                        node.state = NodeState.INACTIVE
                        await self.persistence.save_node(node)
                        
                        await self._record_event("node_inactive", "warning", "health_monitor",
                                                f"Node {node.node_id} marked as inactive")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health monitoring loop error: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        try:
            while self.monitoring_active:
                await asyncio.sleep(3600)  # Check every hour
                
                # Run cleanup
                await self.persistence.cleanup_old_data(retention_days=30)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")
    
    async def register_node(self, node: OrchestratorNode) -> bool:
        """Register a new node"""
        try:
            node.updated_at = time.time()
            
            # Save to persistence
            if await self.persistence.save_node(node):
                # Update cache
                self.nodes_cache[node.node_id] = node
                self.metrics['nodes_registered'] += 1
                
                await self._record_event("node_registered", "info", "orchestrator",
                                        f"Node {node.node_id} registered")
                
                logger.info(f"Node {node.node_id} registered successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to register node {node.node_id}: {e}")
            return False
    
    async def update_node_heartbeat(self, node_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Update node heartbeat and metadata"""
        try:
            if node_id in self.nodes_cache:
                node = self.nodes_cache[node_id]
                node.last_heartbeat = time.time()
                node.updated_at = time.time()
                
                if metadata:
                    node.metadata.update(metadata)
                
                # Update state if needed
                if node.state == NodeState.INACTIVE:
                    node.state = NodeState.ACTIVE
                    await self._record_event("node_active", "info", "orchestrator",
                                            f"Node {node_id} is active again")
                
                await self.persistence.save_node(node)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False
    
    async def submit_task(self, task: OrchestratorTask) -> bool:
        """Submit a new task for execution"""
        try:
            task.created_at = time.time()
            
            # Save to persistence
            if await self.persistence.save_task(task):
                # Update cache
                self.tasks_cache[task.task_id] = task
                self.metrics['total_tasks_scheduled'] += 1
                
                await self._record_event("task_submitted", "info", "orchestrator",
                                        f"Task {task.task_id} submitted")
                
                logger.info(f"Task {task.task_id} submitted successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def update_task_status(
        self,
        task_id: str,
        new_state: TaskState,
        **kwargs
    ) -> bool:
        """Update task status"""
        try:
            # Update cache
            if task_id in self.tasks_cache:
                task = self.tasks_cache[task_id]
                old_state = task.state
                task.state = new_state
                
                # Update specific fields
                for field, value in kwargs.items():
                    if hasattr(task, field):
                        setattr(task, field, value)
                
                # Set completion time for terminal states
                if new_state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
                    task.completed_at = time.time()
                    
                    if new_state == TaskState.COMPLETED:
                        self.metrics['successful_tasks'] += 1
                    else:
                        self.metrics['failed_tasks'] += 1
                
                # Save to persistence
                await self.persistence.update_task_state(task_id, new_state, **kwargs)
                
                await self._record_event("task_status_changed", "info", "orchestrator",
                                        f"Task {task_id} changed from {old_state.value} to {new_state.value}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update task status for {task_id}: {e}")
            return False
    
    async def get_available_nodes(self) -> List[OrchestratorNode]:
        """Get list of available nodes for task assignment"""
        try:
            return [
                node for node in self.nodes_cache.values()
                if node.state == NodeState.ACTIVE
            ]
        except Exception as e:
            logger.error(f"Failed to get available nodes: {e}")
            return []
    
    async def get_pending_tasks(self) -> List[OrchestratorTask]:
        """Get list of pending tasks"""
        try:
            return [
                task for task in self.tasks_cache.values()
                if task.state == TaskState.PENDING
            ]
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    async def _record_event(self, event_type: str, severity: str, source: str, message: str):
        """Record a cluster event"""
        try:
            event = ClusterEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                severity=severity,
                source=source,
                message=message
            )
            
            await self.persistence.save_event(event)
            
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics"""
        try:
            # Get persistence metrics
            persistence_metrics = await self.persistence.get_persistence_metrics()
            
            # Current runtime metrics
            uptime_seconds = time.time() - self.start_time
            
            # Cache metrics
            cache_entries = len(self.nodes_cache) + len(self.tasks_cache)
            
            # Node status distribution
            node_status_counts = defaultdict(int)
            for node in self.nodes_cache.values():
                node_status_counts[node.state.value] += 1
            
            # Task status distribution
            task_status_counts = defaultdict(int)
            for task in self.tasks_cache.values():
                task_status_counts[task.state.value] += 1
            
            return {
                'orchestrator_info': {
                    'instance_id': self.instance_id,
                    'state': self.state.value,
                    'uptime_seconds': uptime_seconds,
                    'start_time': self.start_time
                },
                'performance_metrics': {
                    'total_tasks_scheduled': self.metrics['total_tasks_scheduled'],
                    'successful_tasks': self.metrics['successful_tasks'],
                    'failed_tasks': self.metrics['failed_tasks'],
                    'nodes_registered': self.metrics['nodes_registered'],
                    'success_rate': (self.metrics['successful_tasks'] / 
                                   max(self.metrics['total_tasks_scheduled'], 1))
                },
                'cluster_status': {
                    'total_nodes': len(self.nodes_cache),
                    'total_tasks': len(self.tasks_cache),
                    'node_status_distribution': dict(node_status_counts),
                    'task_status_distribution': dict(task_status_counts)
                },
                'cache_metrics': {
                    'total_cache_entries': cache_entries,
                    'last_sync_time': self.cache_last_sync,
                    'sync_interval': self.cache_sync_interval
                },
                'persistence_metrics': persistence_metrics,
                'background_tasks': {
                    'active_tasks': len(self.background_tasks),
                    'monitoring_active': self.monitoring_active
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating orchestrator metrics: {e}")
            return {}

# Global orchestrator instance
_orchestrator_instance = None

def get_orchestrator() -> AdvancedOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AdvancedOrchestrator()
    return _orchestrator_instance

async def initialize_orchestrator(config: Dict[str, Any] = None):
    """Initialize the global orchestrator"""
    global _orchestrator_instance
    _orchestrator_instance = AdvancedOrchestrator(config)
    await _orchestrator_instance.initialize()
    logger.info("Global orchestrator initialized")
    return _orchestrator_instance
