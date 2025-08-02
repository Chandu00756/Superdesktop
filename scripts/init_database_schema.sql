-- Omega Super Desktop Console - Database Schema Initialization
-- This script creates the complete database schema for the open-source project
-- Compatible with PostgreSQL 15+ and TimescaleDB 2.0+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Create custom types
CREATE TYPE node_type AS ENUM ('control', 'compute', 'storage', 'gpu', 'hybrid');
CREATE TYPE node_status AS ENUM ('active', 'inactive', 'draining', 'failed', 'maintenance');
CREATE TYPE session_type AS ENUM ('interactive', 'batch', 'streaming', 'ai_training', 'rendering');
CREATE TYPE session_status AS ENUM ('pending', 'scheduled', 'running', 'paused', 'completed', 'failed', 'cancelled');
CREATE TYPE user_role AS ENUM ('admin', 'operator', 'user', 'readonly');
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE resource_type AS ENUM ('cpu', 'memory', 'storage', 'gpu', 'network');
CREATE TYPE allocation_status AS ENUM ('active', 'released', 'failed');
CREATE TYPE event_severity AS ENUM ('critical', 'error', 'warning', 'info', 'debug');
CREATE TYPE storage_pool_type AS ENUM ('local', 'distributed', 'replicated', 'erasure_coded');
CREATE TYPE performance_tier AS ENUM ('nvme', 'ssd', 'standard', 'cold');
CREATE TYPE access_mode AS ENUM ('ReadWriteOnce', 'ReadOnlyMany', 'ReadWriteMany');
CREATE TYPE network_type AS ENUM ('ethernet', 'infiniband', 'roce', 'omni_path');
CREATE TYPE interface_type AS ENUM ('management', 'data', 'storage', 'cluster');

-- ===========================================
-- USER MANAGEMENT SCHEMA
-- ===========================================

-- Users table for authentication and authorization
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'user',
    status user_status DEFAULT 'active',
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

-- User session tokens for API authentication
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

-- ===========================================
-- CLUSTER INFRASTRUCTURE SCHEMA
-- ===========================================

-- Node registry for cluster hardware
CREATE TABLE nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    node_type node_type NOT NULL,
    status node_status NOT NULL DEFAULT 'inactive',
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

-- Node resource capacity and allocation tracking
CREATE TABLE node_resources (
    resource_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    resource_type resource_type NOT NULL,
    total_capacity DECIMAL(15,4) NOT NULL,
    allocated_capacity DECIMAL(15,4) DEFAULT 0,
    available_capacity DECIMAL(15,4) GENERATED ALWAYS AS (total_capacity - allocated_capacity) STORED,
    reservation_capacity DECIMAL(15,4) DEFAULT 0,
    unit VARCHAR(20) NOT NULL,
    constraints JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- SESSION MANAGEMENT SCHEMA
-- ===========================================

-- Compute sessions for workload tracking
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    session_name VARCHAR(255) NOT NULL,
    session_type session_type NOT NULL,
    application VARCHAR(100),
    container_image VARCHAR(500),
    status session_status NOT NULL DEFAULT 'pending',
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

-- Resource allocations for sessions
CREATE TABLE session_allocations (
    allocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(node_id),
    resource_type resource_type NOT NULL,
    allocated_amount DECIMAL(15,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    allocation_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deallocation_time TIMESTAMP WITH TIME ZONE,
    status allocation_status DEFAULT 'active'
);

-- ===========================================
-- NETWORK INFRASTRUCTURE SCHEMA
-- ===========================================

-- Network topology definitions
CREATE TABLE network_topologies (
    topology_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topology_name VARCHAR(255) NOT NULL,
    network_type network_type NOT NULL,
    subnet CIDR NOT NULL,
    vlan_id INTEGER,
    bandwidth_mbps INTEGER NOT NULL,
    latency_ms DECIMAL(8,4),
    mtu INTEGER DEFAULT 1500,
    qos_policies JSONB DEFAULT '{}',
    security_policies JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Node network interface configuration
CREATE TABLE node_network_interfaces (
    interface_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    topology_id UUID REFERENCES network_topologies(topology_id),
    interface_name VARCHAR(50) NOT NULL,
    mac_address MACADDR NOT NULL,
    ip_address INET,
    interface_type interface_type NOT NULL,
    speed_mbps INTEGER NOT NULL,
    duplex VARCHAR(10) DEFAULT 'full',
    status VARCHAR(20) DEFAULT 'up' CHECK (status IN ('up', 'down', 'unknown')),
    driver VARCHAR(100),
    firmware_version VARCHAR(100),
    pci_slot VARCHAR(20),
    numa_node INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- STORAGE INFRASTRUCTURE SCHEMA
-- ===========================================

-- Storage pool definitions
CREATE TABLE storage_pools (
    pool_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_name VARCHAR(255) NOT NULL UNIQUE,
    pool_type storage_pool_type NOT NULL,
    total_capacity_gb DECIMAL(15,2) NOT NULL,
    used_capacity_gb DECIMAL(15,2) DEFAULT 0,
    available_capacity_gb DECIMAL(15,2) GENERATED ALWAYS AS (total_capacity_gb - used_capacity_gb) STORED,
    replication_factor INTEGER DEFAULT 1,
    compression_enabled BOOLEAN DEFAULT true,
    encryption_enabled BOOLEAN DEFAULT true,
    performance_tier performance_tier DEFAULT 'standard',
    node_ids UUID[] NOT NULL,
    policies JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Storage volumes for sessions
CREATE TABLE storage_volumes (
    volume_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_id UUID NOT NULL REFERENCES storage_pools(pool_id),
    session_id UUID REFERENCES sessions(session_id),
    volume_name VARCHAR(255) NOT NULL,
    size_gb DECIMAL(12,2) NOT NULL,
    mount_path VARCHAR(500),
    access_mode access_mode DEFAULT 'ReadWriteOnce',
    storage_class VARCHAR(100),
    snapshot_policy VARCHAR(100),
    backup_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- ===========================================
-- TIME-SERIES TABLES (TIMESCALEDB)
-- ===========================================

-- Node performance metrics
CREATE TABLE node_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id UUID NOT NULL REFERENCES nodes(node_id),
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'
);

-- Session performance metrics
CREATE TABLE session_metrics (
    time TIMESTAMPTZ NOT NULL,
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'
);

-- System events and audit logs
CREATE TABLE system_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    severity event_severity NOT NULL,
    source_component VARCHAR(100) NOT NULL,
    source_node_id UUID REFERENCES nodes(node_id),
    session_id UUID REFERENCES sessions(session_id),
    user_id UUID REFERENCES users(user_id),
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    resolved_at TIMESTAMPTZ,
    resolved_by UUID REFERENCES users(user_id)
);

-- ===========================================
-- CREATE HYPERTABLES FOR TIME-SERIES DATA
-- ===========================================

SELECT create_hypertable('node_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('session_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('system_events', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ===========================================
-- CREATE INDEXES FOR PERFORMANCE
-- ===========================================

-- User indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_status ON users(status);

-- User session indexes
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Node indexes
CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_last_heartbeat ON nodes(last_heartbeat);
CREATE INDEX idx_nodes_labels ON nodes USING GIN(labels);
CREATE INDEX idx_nodes_hostname ON nodes(hostname);
CREATE INDEX idx_nodes_ip_address ON nodes(ip_address);

-- Node resource indexes
CREATE UNIQUE INDEX idx_node_resources_unique ON node_resources(node_id, resource_type);
CREATE INDEX idx_node_resources_type ON node_resources(resource_type);
CREATE INDEX idx_node_resources_available ON node_resources(available_capacity);

-- Session indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_type ON sessions(session_type);
CREATE INDEX idx_sessions_scheduled_node ON sessions(scheduled_node_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);
CREATE INDEX idx_sessions_priority_status ON sessions(priority, status);

-- Session allocation indexes
CREATE INDEX idx_session_allocations_session ON session_allocations(session_id);
CREATE INDEX idx_session_allocations_node ON session_allocations(node_id);
CREATE INDEX idx_session_allocations_status ON session_allocations(status);

-- Network indexes
CREATE INDEX idx_network_topologies_type ON network_topologies(network_type);
CREATE INDEX idx_network_topologies_subnet ON network_topologies USING GIST(subnet);
CREATE UNIQUE INDEX idx_node_interfaces_node_name ON node_network_interfaces(node_id, interface_name);
CREATE INDEX idx_node_interfaces_topology ON node_network_interfaces(topology_id);
CREATE INDEX idx_node_interfaces_type ON node_network_interfaces(interface_type);

-- Storage indexes
CREATE INDEX idx_storage_pools_type ON storage_pools(pool_type);
CREATE INDEX idx_storage_pools_tier ON storage_pools(performance_tier);
CREATE INDEX idx_storage_volumes_pool ON storage_volumes(pool_id);
CREATE INDEX idx_storage_volumes_session ON storage_volumes(session_id);
CREATE INDEX idx_storage_volumes_name ON storage_volumes(volume_name);

-- Time-series indexes
CREATE INDEX idx_node_metrics_node_time ON node_metrics(node_id, time DESC);
CREATE INDEX idx_node_metrics_type ON node_metrics(metric_type);
CREATE INDEX idx_session_metrics_session_time ON session_metrics(session_id, time DESC);
CREATE INDEX idx_session_metrics_type ON session_metrics(metric_type);
CREATE INDEX idx_system_events_type ON system_events(event_type);
CREATE INDEX idx_system_events_severity ON system_events(severity);
CREATE INDEX idx_system_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX idx_system_events_node ON system_events(source_node_id);
CREATE INDEX idx_system_events_session ON system_events(session_id);
CREATE INDEX idx_system_events_user ON system_events(user_id);

-- ===========================================
-- CREATE MATERIALIZED VIEWS
-- ===========================================

-- Cluster summary view
CREATE MATERIALIZED VIEW cluster_summary AS
SELECT 
    COUNT(*) as total_nodes,
    COUNT(*) FILTER (WHERE status = 'active') as active_nodes,
    COUNT(*) FILTER (WHERE status = 'inactive') as inactive_nodes,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_nodes,
    SUM(total_cpu_cores) as total_cpu_cores,
    SUM(total_memory_gb) as total_memory_gb,
    SUM(total_storage_gb) as total_storage_gb,
    SUM(gpu_count) as total_gpus,
    AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))) as avg_heartbeat_age_seconds
FROM nodes
WHERE status != 'maintenance';

-- Session statistics view
CREATE MATERIALIZED VIEW session_stats_24h AS
SELECT 
    session_type,
    status,
    COUNT(*) as session_count,
    AVG(EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time))/60) as avg_duration_minutes,
    SUM(cpu_request) as total_cpu_request,
    SUM(memory_request_gb) as total_memory_request_gb,
    SUM(gpu_request) as total_gpu_request
FROM sessions
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY session_type, status;

-- Resource utilization view
CREATE MATERIALIZED VIEW resource_utilization AS
SELECT 
    n.node_id,
    n.hostname,
    n.node_type,
    n.status,
    nr_cpu.total_capacity as total_cpu,
    nr_cpu.allocated_capacity as allocated_cpu,
    nr_cpu.available_capacity as available_cpu,
    nr_mem.total_capacity as total_memory_gb,
    nr_mem.allocated_capacity as allocated_memory_gb,
    nr_mem.available_capacity as available_memory_gb,
    ROUND((nr_cpu.allocated_capacity / NULLIF(nr_cpu.total_capacity, 0) * 100)::numeric, 2) as cpu_utilization_percent,
    ROUND((nr_mem.allocated_capacity / NULLIF(nr_mem.total_capacity, 0) * 100)::numeric, 2) as memory_utilization_percent
FROM nodes n
LEFT JOIN node_resources nr_cpu ON n.node_id = nr_cpu.node_id AND nr_cpu.resource_type = 'cpu'
LEFT JOIN node_resources nr_mem ON n.node_id = nr_mem.node_id AND nr_mem.resource_type = 'memory'
WHERE n.status = 'active';

-- ===========================================
-- CREATE REFRESH POLICIES FOR MATERIALIZED VIEWS
-- ===========================================

-- Refresh cluster summary every 5 minutes
CREATE OR REPLACE FUNCTION refresh_cluster_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW cluster_summary;
END;
$$ LANGUAGE plpgsql;

-- Refresh session stats every 15 minutes
CREATE OR REPLACE FUNCTION refresh_session_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW session_stats_24h;
END;
$$ LANGUAGE plpgsql;

-- Refresh resource utilization every 1 minute
CREATE OR REPLACE FUNCTION refresh_resource_utilization()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW resource_utilization;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- CREATE DATA RETENTION POLICIES
-- ===========================================

-- Retain detailed metrics for 30 days
SELECT add_retention_policy('node_metrics', INTERVAL '30 days');
SELECT add_retention_policy('session_metrics', INTERVAL '30 days');

-- Retain system events for 90 days
SELECT add_retention_policy('system_events', INTERVAL '90 days');

-- Create continuous aggregates for long-term storage
CREATE MATERIALIZED VIEW node_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    node_id,
    metric_type,
    AVG(value) as avg_value,
    MAX(value) as max_value,
    MIN(value) as min_value,
    COUNT(*) as sample_count
FROM node_metrics
GROUP BY bucket, node_id, metric_type
WITH NO DATA;

-- Enable compression for older data
SELECT add_compression_policy('node_metrics', INTERVAL '7 days');
SELECT add_compression_policy('session_metrics', INTERVAL '7 days');
SELECT add_compression_policy('system_events', INTERVAL '7 days');

-- ===========================================
-- CREATE FUNCTIONS AND TRIGGERS
-- ===========================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at columns
CREATE TRIGGER update_users_modtime BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_modified_column();
CREATE TRIGGER update_nodes_modtime BEFORE UPDATE ON nodes FOR EACH ROW EXECUTE FUNCTION update_modified_column();
CREATE TRIGGER update_sessions_modtime BEFORE UPDATE ON sessions FOR EACH ROW EXECUTE FUNCTION update_modified_column();
CREATE TRIGGER update_node_resources_modtime BEFORE UPDATE ON node_resources FOR EACH ROW EXECUTE FUNCTION update_modified_column();
CREATE TRIGGER update_storage_pools_modtime BEFORE UPDATE ON storage_pools FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Function to automatically create node resources when a node is added
CREATE OR REPLACE FUNCTION create_default_node_resources()
RETURNS TRIGGER AS $$
BEGIN
    -- Create CPU resource entry
    INSERT INTO node_resources (node_id, resource_type, total_capacity, unit)
    VALUES (NEW.node_id, 'cpu', NEW.total_cpu_cores, 'cores');
    
    -- Create memory resource entry
    INSERT INTO node_resources (node_id, resource_type, total_capacity, unit)
    VALUES (NEW.node_id, 'memory', NEW.total_memory_gb, 'GB');
    
    -- Create storage resource entry
    INSERT INTO node_resources (node_id, resource_type, total_capacity, unit)
    VALUES (NEW.node_id, 'storage', NEW.total_storage_gb, 'GB');
    
    -- Create GPU resource entry if GPU count > 0
    IF NEW.gpu_count > 0 THEN
        INSERT INTO node_resources (node_id, resource_type, total_capacity, unit)
        VALUES (NEW.node_id, 'gpu', NEW.gpu_count, 'units');
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER create_node_resources_trigger 
    AFTER INSERT ON nodes 
    FOR EACH ROW 
    EXECUTE FUNCTION create_default_node_resources();

-- ===========================================
-- CREATE ROLES AND SECURITY
-- ===========================================

-- Create application roles
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'omega_admin') THEN
        CREATE ROLE omega_admin;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'omega_operator') THEN
        CREATE ROLE omega_operator;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'omega_user') THEN
        CREATE ROLE omega_user;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'omega_readonly') THEN
        CREATE ROLE omega_readonly;
    END IF;
END
$$;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO omega_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO omega_admin;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO omega_admin;

GRANT SELECT, INSERT, UPDATE, DELETE ON sessions, session_allocations, session_metrics TO omega_operator;
GRANT SELECT, INSERT, UPDATE ON nodes, node_resources, node_metrics TO omega_operator;
GRANT SELECT, INSERT ON system_events TO omega_operator;
GRANT SELECT ON users, storage_pools, storage_volumes, network_topologies TO omega_operator;

GRANT SELECT, INSERT, UPDATE ON sessions TO omega_user;
GRANT SELECT ON nodes, node_resources, storage_pools, storage_volumes TO omega_user;
GRANT INSERT ON system_events TO omega_user;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO omega_readonly;

-- Enable row level security
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY user_sessions_policy ON sessions
    FOR ALL
    TO omega_user
    USING (user_id = current_setting('app.current_user_id', true)::UUID)
    WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);

CREATE POLICY user_session_tokens_policy ON user_sessions
    FOR ALL
    TO omega_user
    USING (user_id = current_setting('app.current_user_id', true)::UUID)
    WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);

-- ===========================================
-- INSERT INITIAL DATA
-- ===========================================

-- Create default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, full_name, role, status)
VALUES ('admin', 'admin@omega.local', crypt('admin123', gen_salt('bf')), 'System Administrator', 'admin', 'active')
ON CONFLICT (username) DO NOTHING;

-- Create default storage pool
INSERT INTO storage_pools (pool_name, pool_type, total_capacity_gb, performance_tier, node_ids)
VALUES ('default-pool', 'local', 1000.0, 'standard', '{}')
ON CONFLICT (pool_name) DO NOTHING;

-- Create default network topology
INSERT INTO network_topologies (topology_name, network_type, subnet, bandwidth_mbps)
VALUES ('default-network', 'ethernet', '192.168.1.0/24', 1000)
ON CONFLICT DO NOTHING;

-- ===========================================
-- SCHEMA VALIDATION
-- ===========================================

-- Verify all tables were created
DO $$
DECLARE
    table_count INTEGER;
    expected_tables TEXT[] := ARRAY[
        'users', 'user_sessions', 'nodes', 'node_resources', 'sessions', 
        'session_allocations', 'network_topologies', 'node_network_interfaces',
        'storage_pools', 'storage_volumes', 'node_metrics', 'session_metrics', 
        'system_events'
    ];
    table_name TEXT;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
      AND table_name = ANY(expected_tables);
    
    IF table_count != array_length(expected_tables, 1) THEN
        RAISE EXCEPTION 'Expected % tables, found %', array_length(expected_tables, 1), table_count;
    END IF;
    
    RAISE NOTICE 'Database schema initialized successfully with % tables', table_count;
END
$$;

-- Show database statistics
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    (SELECT count(*) FROM information_schema.columns WHERE table_name = t.tablename) as columns
FROM pg_tables t
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

COMMIT;
