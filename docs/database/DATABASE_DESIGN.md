# Omega Super Desktop Console - Database Design

## Overview

The Omega Super Desktop Console uses a multi-database architecture designed for high performance, scalability, and reliability in distributed supercomputing environments. This document outlines the complete database design for the open-source project.

## Database Architecture

### Primary Technologies

- **PostgreSQL 15+**: Primary relational database for core system data
- **TimescaleDB**: Time-series extension for metrics and monitoring data
- **Redis**: High-performance caching and real-time session state
- **SQLite**: Embedded database for local node storage
- **InfluxDB**: Optional time-series database for advanced analytics

### Architecture Pattern

``
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Application   │    │   Application   │
│   Services      │    │   Services      │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │    SQLite       │
│   (Primary DB)  │    │   (Cache/RT)    │    │  (Local Node)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
┌─────────────────┐
│   TimescaleDB   │
│  (Time Series)  │
└─────────────────┘
``

## Core Database Schemas

### 1. Cluster Management Schema

#### Nodes Table

```sql
CREATE TABLE nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('control', 'compute', 'storage', 'gpu', 'hybrid')),
    status VARCHAR(20) NOT NULL DEFAULT 'inactive' CHECK (status IN ('active', 'inactive', 'draining', 'failed', 'maintenance')),
    total_cpu_cores INTEGER NOT NULL,
    total_memory_gb DECIMAL(10,2) NOT NULL,
    total_storage_gb DECIMAL(12,2) NOT NULL,
    gpu_count INTEGER DEFAULT 0,
    gpu_memory_gb DECIMAL(10,2) DEFAULT 0,
    network_bandwidth_mbps INTEGER DEFAULT 1000,
    architecture VARCHAR(20) NOT NULL DEFAULT 'x86_64',
    os_version VARCHAR(100),
    kernel_version VARCHAR(100),
    docker_version VARCHAR(50),
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    capabilities JSONB DEFAULT '{}',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_last_heartbeat ON nodes(last_heartbeat);
CREATE INDEX idx_nodes_labels ON nodes USING GIN(labels);
```

#### Node Resources Table

```sql
CREATE TABLE node_resources (
    resource_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL CHECK (resource_type IN ('cpu', 'memory', 'storage', 'gpu', 'network')),
    total_capacity DECIMAL(15,4) NOT NULL,
    allocated_capacity DECIMAL(15,4) DEFAULT 0,
    available_capacity DECIMAL(15,4) GENERATED ALWAYS AS (total_capacity - allocated_capacity) STORED,
    reservation_capacity DECIMAL(15,4) DEFAULT 0,
    unit VARCHAR(20) NOT NULL,
    constraints JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_node_resources_unique ON node_resources(node_id, resource_type);
CREATE INDEX idx_node_resources_type ON node_resources(resource_type);
```

### 2. Session Management Schema

#### Sessions Table

```sql
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    session_name VARCHAR(255) NOT NULL,
    session_type VARCHAR(50) NOT NULL CHECK (session_type IN ('interactive', 'batch', 'streaming', 'ai_training', 'rendering')),
    application VARCHAR(100),
    container_image VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'scheduled', 'running', 'paused', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 10),
    max_runtime_minutes INTEGER,
    cpu_request DECIMAL(8,2) NOT NULL,
    cpu_limit DECIMAL(8,2),
    memory_request_gb DECIMAL(10,2) NOT NULL,
    memory_limit_gb DECIMAL(10,2),
    gpu_request INTEGER DEFAULT 0,
    gpu_type VARCHAR(50),
    storage_request_gb DECIMAL(12,2) DEFAULT 0,
    network_policy VARCHAR(50) DEFAULT 'default',
    environment_variables JSONB DEFAULT '{}',
    resource_constraints JSONB DEFAULT '{}',
    placement_preferences JSONB DEFAULT '{}',
    scheduled_node_id UUID REFERENCES nodes(node_id),
    actual_node_ids UUID[],
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_type ON sessions(session_type);
CREATE INDEX idx_sessions_scheduled_node ON sessions(scheduled_node_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);
```

#### Session Allocations Table

```sql
CREATE TABLE session_allocations (
    allocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(node_id),
    resource_type VARCHAR(50) NOT NULL,
    allocated_amount DECIMAL(15,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    allocation_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deallocation_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'released', 'failed'))
);

CREATE INDEX idx_session_allocations_session ON session_allocations(session_id);
CREATE INDEX idx_session_allocations_node ON session_allocations(node_id);
CREATE INDEX idx_session_allocations_status ON session_allocations(status);
```

### 3. User Management Schema

#### Users Table

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'operator', 'user', 'readonly')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    account_locked_until TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}',
    quota_cpu_hours DECIMAL(10,2) DEFAULT 100,
    quota_memory_gb_hours DECIMAL(12,2) DEFAULT 1000,
    quota_storage_gb DECIMAL(12,2) DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_status ON users(status);
```

#### User Sessions Table

```sql
CREATE TABLE user_sessions (
    token_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_token VARCHAR(512) NOT NULL UNIQUE,
    refresh_token VARCHAR(512),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    client_ip INET,
    user_agent TEXT
);

CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
```

### 4. Monitoring and Metrics Schema (TimescaleDB)

#### Node Metrics Table

```sql
CREATE TABLE node_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id UUID NOT NULL REFERENCES nodes(node_id),
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'
);

SELECT create_hypertable('node_metrics', 'time');
CREATE INDEX idx_node_metrics_node_time ON node_metrics(node_id, time DESC);
CREATE INDEX idx_node_metrics_type ON node_metrics(metric_type);
```

#### Session Metrics Table

```sql
CREATE TABLE session_metrics (
    time TIMESTAMPTZ NOT NULL,
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'
);

SELECT create_hypertable('session_metrics', 'time');
CREATE INDEX idx_session_metrics_session_time ON session_metrics(session_id, time DESC);
CREATE INDEX idx_session_metrics_type ON session_metrics(metric_type);
```

#### System Events Table

```sql
CREATE TABLE system_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'error', 'warning', 'info', 'debug')),
    source_component VARCHAR(100) NOT NULL,
    source_node_id UUID REFERENCES nodes(node_id),
    session_id UUID REFERENCES sessions(session_id),
    user_id UUID REFERENCES users(user_id),
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    resolved_at TIMESTAMPTZ,
    resolved_by UUID REFERENCES users(user_id)
);

SELECT create_hypertable('system_events', 'timestamp');
CREATE INDEX idx_system_events_type ON system_events(event_type);
CREATE INDEX idx_system_events_severity ON system_events(severity);
CREATE INDEX idx_system_events_timestamp ON system_events(timestamp DESC);
```

### 5. Storage Management Schema

#### Storage Pools Table

```sql
CREATE TABLE storage_pools (
    pool_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_name VARCHAR(255) NOT NULL UNIQUE,
    pool_type VARCHAR(50) NOT NULL CHECK (pool_type IN ('local', 'distributed', 'replicated', 'erasure_coded')),
    total_capacity_gb DECIMAL(15,2) NOT NULL,
    used_capacity_gb DECIMAL(15,2) DEFAULT 0,
    available_capacity_gb DECIMAL(15,2) GENERATED ALWAYS AS (total_capacity_gb - used_capacity_gb) STORED,
    replication_factor INTEGER DEFAULT 1,
    compression_enabled BOOLEAN DEFAULT true,
    encryption_enabled BOOLEAN DEFAULT true,
    performance_tier VARCHAR(20) DEFAULT 'standard' CHECK (performance_tier IN ('nvme', 'ssd', 'standard', 'cold')),
    node_ids UUID[] NOT NULL,
    policies JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_storage_pools_type ON storage_pools(pool_type);
CREATE INDEX idx_storage_pools_tier ON storage_pools(performance_tier);
```

#### Storage Volumes Table

```sql
CREATE TABLE storage_volumes (
    volume_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_id UUID NOT NULL REFERENCES storage_pools(pool_id),
    session_id UUID REFERENCES sessions(session_id),
    volume_name VARCHAR(255) NOT NULL,
    size_gb DECIMAL(12,2) NOT NULL,
    mount_path VARCHAR(500),
    access_mode VARCHAR(20) DEFAULT 'ReadWriteOnce' CHECK (access_mode IN ('ReadWriteOnce', 'ReadOnlyMany', 'ReadWriteMany')),
    storage_class VARCHAR(100),
    snapshot_policy VARCHAR(100),
    backup_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_storage_volumes_pool ON storage_volumes(pool_id);
CREATE INDEX idx_storage_volumes_session ON storage_volumes(session_id);
CREATE INDEX idx_storage_volumes_name ON storage_volumes(volume_name);
```

### 6. Network Management Schema

#### Network Topologies Table

```sql
CREATE TABLE network_topologies (
    topology_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topology_name VARCHAR(255) NOT NULL,
    network_type VARCHAR(50) NOT NULL CHECK (network_type IN ('ethernet', 'infiniband', 'roce', 'omni_path')),
    subnet CIDR NOT NULL,
    vlan_id INTEGER,
    bandwidth_mbps INTEGER NOT NULL,
    latency_ms DECIMAL(8,4),
    mtu INTEGER DEFAULT 1500,
    qos_policies JSONB DEFAULT '{}',
    security_policies JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_network_topologies_type ON network_topologies(network_type);
CREATE INDEX idx_network_topologies_subnet ON network_topologies(subnet);
```

#### Node Network Interfaces Table

```sql
CREATE TABLE node_network_interfaces (
    interface_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    topology_id UUID REFERENCES network_topologies(topology_id),
    interface_name VARCHAR(50) NOT NULL,
    mac_address MACADDR NOT NULL,
    ip_address INET,
    interface_type VARCHAR(20) NOT NULL CHECK (interface_type IN ('management', 'data', 'storage', 'cluster')),
    speed_mbps INTEGER NOT NULL,
    duplex VARCHAR(10) DEFAULT 'full',
    status VARCHAR(20) DEFAULT 'up' CHECK (status IN ('up', 'down', 'unknown')),
    driver VARCHAR(100),
    firmware_version VARCHAR(100),
    pci_slot VARCHAR(20),
    numa_node INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_node_interfaces_node_name ON node_network_interfaces(node_id, interface_name);
CREATE INDEX idx_node_interfaces_topology ON node_network_interfaces(topology_id);
CREATE INDEX idx_node_interfaces_type ON node_network_interfaces(interface_type);
```

## Redis Schema Design

### Key Patterns

``

## Node heartbeats (TTL: 30 seconds)

node:heartbeat:{node_id} = {timestamp, status, load}

## Session state cache (TTL: session duration)

session:state:{session_id} = {status, allocated_resources, start_time}

## Real-time metrics (TTL: 5 minutes)

metrics:node:{node_id}:{metric_type} = {value, timestamp, unit}

## Active connections

connections:websocket:{connection_id} = {user_id, node_id, session_id}

## Resource locks (TTL: 60 seconds)

lock:resource:{node_id}:{resource_type} = {session_id, lock_time}

## Queue management

queue:pending_sessions = [session_id, session_id, ...]
queue:placement_decisions = [decision_data, ...]

## User sessions (TTL: 24 hours)

user:session:{token} = {user_id, permissions, expires_at}

## Cluster state

cluster:state = {total_nodes, active_nodes, total_sessions, cluster_load}
``

## SQLite Schema (Local Node Storage)

### Node-specific tables for local operations

```sql
-- Local node configuration
CREATE TABLE local_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Local resource tracking
CREATE TABLE local_resources (
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage REAL,
    memory_usage REAL,
    disk_usage REAL,
    network_rx_bytes INTEGER,
    network_tx_bytes INTEGER,
    gpu_usage REAL,
    temperature REAL
);

-- Local session cache
CREATE TABLE local_sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    container_id TEXT,
    pid INTEGER,
    started_at TIMESTAMP,
    resource_limits TEXT -- JSON
);

-- Local events log
CREATE TABLE local_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT -- JSON
);
```

## Performance Optimizations

### Indexing Strategy

1. **Primary Keys**: All tables use UUID primary keys for distributed uniqueness
2. **Foreign Key Indexes**: Automatic indexes on all foreign key columns
3. **Composite Indexes**: Multi-column indexes for common query patterns
4. **Partial Indexes**: Indexes on filtered subsets of data
5. **GIN Indexes**: For JSONB columns with complex queries

### Partitioning Strategy

```sql
-- Partition large tables by time
CREATE TABLE node_metrics_y2025 PARTITION OF node_metrics
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Partition by node_id for horizontal scaling
CREATE TABLE sessions_node_1 PARTITION OF sessions
FOR VALUES WITH (MODULUS 4, REMAINDER 0);
```

### Materialized Views

```sql
-- Cluster summary view
CREATE MATERIALIZED VIEW cluster_summary AS
SELECT 
    COUNT(*) as total_nodes,
    COUNT(*) FILTER (WHERE status = 'active') as active_nodes,
    SUM(total_cpu_cores) as total_cpu_cores,
    SUM(total_memory_gb) as total_memory_gb,
    AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))) as avg_heartbeat_age
FROM nodes;

-- Session statistics view
CREATE MATERIALIZED VIEW session_stats AS
SELECT 
    session_type,
    status,
    COUNT(*) as session_count,
    AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) as avg_duration_minutes,
    SUM(cpu_request) as total_cpu_request,
    SUM(memory_request_gb) as total_memory_request
FROM sessions
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY session_type, status;
```

## Data Retention Policies

### Time-series Data Retention

```sql
-- Retain detailed metrics for 30 days, aggregated for 1 year
SELECT add_retention_policy('node_metrics', INTERVAL '30 days');
SELECT add_retention_policy('session_metrics', INTERVAL '30 days');

-- Aggregate metrics by hour after 24 hours
SELECT add_continuous_aggregate_policy('node_metrics_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

### Archival Strategy

```sql
-- Archive completed sessions older than 90 days
CREATE TABLE sessions_archive (LIKE sessions INCLUDING ALL);

-- Archive system events older than 6 months
CREATE TABLE system_events_archive (LIKE system_events INCLUDING ALL);
```

## Security Considerations

### Data Encryption

- **Encryption at Rest**: PostgreSQL with TDE (Transparent Data Encryption)
- **Encryption in Transit**: TLS 1.3 for all database connections
- **Column-level Encryption**: Sensitive data encrypted using `pgcrypto`

### Access Control

```sql
-- Role-based access control
CREATE ROLE omega_admin;
CREATE ROLE omega_operator;
CREATE ROLE omega_user;
CREATE ROLE omega_readonly;

-- Grant appropriate permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO omega_admin;
GRANT SELECT, INSERT, UPDATE ON sessions TO omega_operator;
GRANT SELECT ON sessions TO omega_user WHERE user_id = current_user_id();
GRANT SELECT ON ALL TABLES IN SCHEMA public TO omega_readonly;
```

### Row Level Security

```sql
-- Enable RLS for user data isolation
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_sessions_policy ON sessions
    USING (user_id = current_setting('app.current_user_id')::UUID)
    WITH CHECK (user_id = current_setting('app.current_user_id')::UUID);
```

## Monitoring and Observability

### Database Metrics

- Connection pool utilization
- Query execution times
- Index usage statistics
- Replication lag
- Cache hit ratios
- Lock wait times

### Health Checks

```sql
-- Database health check query
SELECT 
    'database' as component,
    CASE 
        WHEN pg_is_in_recovery() THEN 'replica'
        ELSE 'primary'
    END as role,
    pg_database_size(current_database()) as size_bytes,
    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
    pg_postmaster_start_time() as started_at;
```

## Backup and Recovery

### Backup Strategy

1. **Continuous WAL Archiving**: Real-time transaction log backup
2. **Daily Base Backups**: Full database backup every 24 hours
3. **Point-in-Time Recovery**: Ability to restore to any point in time
4. **Cross-Region Replication**: Async replication to disaster recovery site

### Recovery Procedures

```bash
# Point-in-time recovery example
pg_basebackup -h primary -D /backup/base -U replica -v -P -W
pg_ctl start -D /backup/base -o "-c archive_command='cp %p /archive/%f'"
```

This database design provides a robust foundation for the Omega Super Desktop Console, supporting high-performance distributed computing workloads while maintaining data integrity, security, and observability.
