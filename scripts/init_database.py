"""
Omega Super Desktop Console - Database Initialization Script
Creates all required database tables and indexes for the distributed system

This script sets up the PostgreSQL database schema for:
- Session management and tracking
- Performance metrics and monitoring
- Memory allocation records
- Security and authentication
- Orchestration and placement decisions
"""

import asyncio
import asyncpg
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_database_schema():
    """Create all required database tables and indexes"""
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'omega'),
        'password': os.getenv('POSTGRES_PASSWORD', 'omega_secure_2025'),
        'database': os.getenv('POSTGRES_DB', 'omega_sessions')
    }
    
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(**db_config)
        logger.info("Connected to PostgreSQL database")
        
        # Create sessions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                session_type VARCHAR(50) NOT NULL,
                state VARCHAR(50) NOT NULL,
                assigned_node VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                started_at TIMESTAMP WITH TIME ZONE,
                ended_at TIMESTAMP WITH TIME ZONE,
                session_data JSONB,
                performance_tier VARCHAR(50),
                target_latency_ms REAL,
                
                -- Indexes
                INDEX idx_sessions_user_id (user_id),
                INDEX idx_sessions_state (state),
                INDEX idx_sessions_assigned_node (assigned_node),
                INDEX idx_sessions_created_at (created_at),
                INDEX idx_sessions_session_type (session_type)
            )
        """)
        logger.info("Sessions table created")
        
        # Create performance_metrics table with TimescaleDB support
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                node_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                session_type VARCHAR(50) NOT NULL,
                
                -- Performance Metrics
                latency_ms REAL NOT NULL,
                throughput_mbps REAL NOT NULL,
                fps INTEGER NOT NULL,
                frame_drops INTEGER DEFAULT 0,
                jitter_ms REAL DEFAULT 0.0,
                packet_loss_percent REAL DEFAULT 0.0,
                
                -- Resource Utilization
                cpu_utilization_percent REAL NOT NULL,
                memory_utilization_percent REAL NOT NULL,
                gpu_utilization_percent REAL NOT NULL,
                gpu_memory_utilization_percent REAL DEFAULT 0.0,
                network_utilization_percent REAL DEFAULT 0.0,
                storage_io_mbps REAL DEFAULT 0.0,
                
                -- System Context
                total_sessions_on_node INTEGER DEFAULT 1,
                node_load_score REAL DEFAULT 0.5,
                network_congestion_score REAL DEFAULT 0.1,
                temperature_celsius INTEGER DEFAULT 60,
                power_draw_watts INTEGER DEFAULT 200,
                
                -- Application Context
                application_profile VARCHAR(100) DEFAULT 'standard',
                resolution VARCHAR(20) DEFAULT '1920x1080',
                color_depth INTEGER DEFAULT 8,
                encoding_preset VARCHAR(20) DEFAULT 'medium',
                bitrate_mbps REAL DEFAULT 50.0,
                
                -- External Factors
                time_of_day INTEGER NOT NULL,
                day_of_week INTEGER NOT NULL,
                concurrent_users INTEGER DEFAULT 5,
                
                -- Indexes
                INDEX idx_perf_timestamp (timestamp),
                INDEX idx_perf_session_id (session_id),
                INDEX idx_perf_node_id (node_id),
                INDEX idx_perf_user_id (user_id),
                INDEX idx_perf_latency (latency_ms),
                INDEX idx_perf_compound (session_id, timestamp)
            )
        """)
        logger.info("Performance metrics table created")
        
        # Create memory_allocations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_allocations (
                allocation_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                fabric_node_id VARCHAR(255) NOT NULL,
                size_bytes BIGINT NOT NULL,
                numa_node INTEGER NOT NULL,
                allocation_strategy VARCHAR(50) NOT NULL,
                access_pattern VARCHAR(50) DEFAULT 'mixed',
                priority_level INTEGER DEFAULT 1,
                compression_enabled BOOLEAN DEFAULT TRUE,
                encryption_enabled BOOLEAN DEFAULT TRUE,
                allocated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                deallocated_at TIMESTAMP WITH TIME ZONE,
                allocation_data JSONB,
                
                -- Indexes
                INDEX idx_alloc_session_id (session_id),
                INDEX idx_alloc_fabric_node (fabric_node_id),
                INDEX idx_alloc_numa_node (numa_node),
                INDEX idx_alloc_allocated_at (allocated_at),
                INDEX idx_alloc_size (size_bytes)
            )
        """)
        logger.info("Memory allocations table created")
        
        # Create placement_decisions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS placement_decisions (
                decision_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                placement_algorithm VARCHAR(100) NOT NULL,
                selected_node VARCHAR(255) NOT NULL,
                decision_score REAL NOT NULL,
                alternative_nodes JSONB,
                constraints JSONB,
                resource_requirements JSONB,
                decision_made_at TIMESTAMP WITH TIME ZONE NOT NULL,
                execution_time_ms REAL,
                
                -- Indexes
                INDEX idx_placement_session_id (session_id),
                INDEX idx_placement_node (selected_node),
                INDEX idx_placement_algorithm (placement_algorithm),
                INDEX idx_placement_timestamp (decision_made_at),
                INDEX idx_placement_score (decision_score)
            )
        """)
        logger.info("Placement decisions table created")
        
        # Create node_resources table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS node_resources (
                node_id VARCHAR(255) PRIMARY KEY,
                node_name VARCHAR(255) NOT NULL,
                node_type VARCHAR(50) NOT NULL,
                cluster_region VARCHAR(100),
                
                -- Hardware Specifications
                cpu_cores INTEGER NOT NULL,
                cpu_model VARCHAR(255),
                cpu_frequency_mhz INTEGER,
                memory_total_gb INTEGER NOT NULL,
                memory_available_gb INTEGER NOT NULL,
                storage_total_gb INTEGER,
                storage_available_gb INTEGER,
                
                -- GPU Resources
                gpu_count INTEGER DEFAULT 0,
                gpu_models JSONB,
                gpu_memory_total_mb INTEGER DEFAULT 0,
                gpu_memory_available_mb INTEGER DEFAULT 0,
                
                -- Network
                network_bandwidth_gbps REAL DEFAULT 1.0,
                network_latency_ms REAL DEFAULT 1.0,
                
                -- Status
                status VARCHAR(50) DEFAULT 'active',
                health_score REAL DEFAULT 1.0,
                last_heartbeat TIMESTAMP WITH TIME ZONE,
                resource_data JSONB,
                
                -- Indexes
                INDEX idx_nodes_status (status),
                INDEX idx_nodes_type (node_type),
                INDEX idx_nodes_region (cluster_region),
                INDEX idx_nodes_health (health_score),
                INDEX idx_nodes_heartbeat (last_heartbeat)
            )
        """)
        logger.info("Node resources table created")
        
        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                
                -- Profile Information
                display_name VARCHAR(255),
                avatar_url VARCHAR(500),
                timezone VARCHAR(50) DEFAULT 'UTC',
                language VARCHAR(10) DEFAULT 'en',
                
                -- Subscription and Limits
                subscription_tier VARCHAR(50) DEFAULT 'standard',
                max_concurrent_sessions INTEGER DEFAULT 3,
                max_session_duration_hours INTEGER DEFAULT 8,
                max_gpu_allocation INTEGER DEFAULT 1,
                storage_quota_gb INTEGER DEFAULT 100,
                
                -- Security
                mfa_enabled BOOLEAN DEFAULT FALSE,
                mfa_secret VARCHAR(255),
                security_level VARCHAR(20) DEFAULT 'standard',
                
                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE,
                
                -- Preferences
                preferences JSONB DEFAULT '{}',
                
                -- Indexes
                INDEX idx_users_username (username),
                INDEX idx_users_email (email),
                INDEX idx_users_subscription (subscription_tier),
                INDEX idx_users_created_at (created_at),
                INDEX idx_users_last_login (last_login)
            )
        """)
        logger.info("Users table created")
        
        # Create audit_logs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                user_id VARCHAR(255),
                session_id VARCHAR(255),
                action VARCHAR(100) NOT NULL,
                resource_type VARCHAR(50),
                resource_id VARCHAR(255),
                source_ip INET,
                user_agent TEXT,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                additional_data JSONB,
                
                -- Indexes
                INDEX idx_audit_timestamp (timestamp),
                INDEX idx_audit_user_id (user_id),
                INDEX idx_audit_session_id (session_id),
                INDEX idx_audit_action (action),
                INDEX idx_audit_success (success),
                INDEX idx_audit_compound (user_id, timestamp)
            )
        """)
        logger.info("Audit logs table created")
        
        # Create api_keys table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                key_name VARCHAR(255) NOT NULL,
                key_hash VARCHAR(255) NOT NULL,
                permissions JSONB NOT NULL DEFAULT '[]',
                
                -- Limitations
                rate_limit_per_minute INTEGER DEFAULT 100,
                allowed_ips JSONB,
                
                -- Status
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE,
                last_used TIMESTAMP WITH TIME ZONE,
                usage_count BIGINT DEFAULT 0,
                
                -- Indexes
                INDEX idx_apikeys_user_id (user_id),
                INDEX idx_apikeys_hash (key_hash),
                INDEX idx_apikeys_active (is_active),
                INDEX idx_apikeys_expires (expires_at)
            )
        """)
        logger.info("API keys table created")
        
        # Create render_tasks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS render_tasks (
                task_id VARCHAR(255) PRIMARY KEY,
                stream_id VARCHAR(255) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                frame_number BIGINT NOT NULL,
                
                -- Task Details
                gpu_id VARCHAR(255) NOT NULL,
                viewport_x INTEGER NOT NULL,
                viewport_y INTEGER NOT NULL,
                viewport_width INTEGER NOT NULL,
                viewport_height INTEGER NOT NULL,
                rendering_api VARCHAR(50) NOT NULL,
                
                -- Timing
                submitted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                render_duration_ms REAL,
                encoding_duration_ms REAL,
                
                -- Results
                frame_size_bytes INTEGER,
                quality_score REAL,
                success BOOLEAN,
                error_message TEXT,
                
                -- Performance
                gpu_utilization_percent REAL,
                memory_used_mb INTEGER,
                task_data JSONB,
                
                -- Indexes
                INDEX idx_render_stream_id (stream_id),
                INDEX idx_render_session_id (session_id),
                INDEX idx_render_gpu_id (gpu_id),
                INDEX idx_render_submitted_at (submitted_at),
                INDEX idx_render_frame_number (frame_number),
                INDEX idx_render_compound (stream_id, frame_number)
            )
        """)
        logger.info("Render tasks table created")
        
        # Create ml_predictions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                prediction_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                prediction_type VARCHAR(50) NOT NULL,
                
                -- Input Features
                input_features JSONB NOT NULL,
                current_metrics JSONB NOT NULL,
                
                -- Prediction Results
                predicted_value REAL NOT NULL,
                confidence_score REAL NOT NULL,
                prediction_horizon_seconds INTEGER NOT NULL,
                
                -- Model Information
                model_version VARCHAR(50),
                model_accuracy REAL,
                
                -- Validation
                actual_value REAL,
                prediction_error REAL,
                
                -- Timestamps
                predicted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                validated_at TIMESTAMP WITH TIME ZONE,
                
                -- Additional Data
                recommendations JSONB,
                risk_factors JSONB,
                
                -- Indexes
                INDEX idx_predictions_session_id (session_id),
                INDEX idx_predictions_model_type (model_type),
                INDEX idx_predictions_prediction_type (prediction_type),
                INDEX idx_predictions_predicted_at (predicted_at),
                INDEX idx_predictions_confidence (confidence_score)
            )
        """)
        logger.info("ML predictions table created")
        
        # Create system_events table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                event_id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                event_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                source_component VARCHAR(100) NOT NULL,
                source_node VARCHAR(255),
                
                -- Event Details
                title VARCHAR(255) NOT NULL,
                description TEXT,
                affected_resources JSONB,
                
                -- Context
                session_id VARCHAR(255),
                user_id VARCHAR(255),
                correlation_id VARCHAR(255),
                
                -- Resolution
                status VARCHAR(50) DEFAULT 'open',
                resolved_at TIMESTAMP WITH TIME ZONE,
                resolution_notes TEXT,
                
                -- Additional Data
                event_data JSONB,
                
                -- Indexes
                INDEX idx_events_timestamp (timestamp),
                INDEX idx_events_type (event_type),
                INDEX idx_events_severity (severity),
                INDEX idx_events_component (source_component),
                INDEX idx_events_status (status),
                INDEX idx_events_session_id (session_id)
            )
        """)
        logger.info("System events table created")
        
        # Create configuration table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_configuration (
                config_key VARCHAR(255) PRIMARY KEY,
                config_value JSONB NOT NULL,
                config_type VARCHAR(50) NOT NULL,
                description TEXT,
                
                -- Metadata
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_by VARCHAR(255),
                
                -- Validation
                schema_version VARCHAR(20) DEFAULT '1.0',
                is_sensitive BOOLEAN DEFAULT FALSE,
                
                -- Indexes
                INDEX idx_config_type (config_type),
                INDEX idx_config_updated_at (updated_at)
            )
        """)
        logger.info("System configuration table created")
        
        # Insert default configuration values
        await conn.execute("""
            INSERT INTO system_configuration (config_key, config_value, config_type, description)
            VALUES 
                ('cluster.max_sessions_per_node', '10', 'integer', 'Maximum sessions per compute node'),
                ('cluster.default_session_timeout_minutes', '480', 'integer', 'Default session timeout in minutes'),
                ('orchestrator.placement_algorithm', '"predictive_binpack"', 'string', 'Default placement algorithm'),
                ('orchestrator.migration_threshold_ms', '25.0', 'float', 'Latency threshold for session migration'),
                ('security.encryption_required', 'true', 'boolean', 'Require encryption for all sessions'),
                ('security.mfa_enforcement', 'false', 'boolean', 'Enforce multi-factor authentication'),
                ('performance.target_latency_ms', '16.67', 'float', 'Target latency for 60fps'),
                ('performance.quality_threshold', '0.8', 'float', 'Minimum quality threshold'),
                ('monitoring.metrics_retention_days', '30', 'integer', 'Performance metrics retention period'),
                ('monitoring.log_retention_days', '90', 'integer', 'Audit log retention period')
            ON CONFLICT (config_key) DO NOTHING
        """)
        logger.info("Default configuration inserted")
        
        # Create indexes for better performance
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_metrics_time_bucket ON performance_metrics USING BRIN (timestamp)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active ON sessions (state) WHERE state IN ('active', 'initializing')")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_recent ON audit_logs (timestamp) WHERE timestamp > NOW() - INTERVAL '7 days'")
        
        # Create views for common queries
        await conn.execute("""
            CREATE OR REPLACE VIEW active_sessions AS
            SELECT 
                s.*,
                u.username,
                u.display_name,
                nr.node_name,
                nr.cluster_region
            FROM sessions s
            LEFT JOIN users u ON s.user_id = u.user_id
            LEFT JOIN node_resources nr ON s.assigned_node = nr.node_id
            WHERE s.state IN ('active', 'initializing', 'migrating')
        """)
        
        await conn.execute("""
            CREATE OR REPLACE VIEW session_performance_summary AS
            SELECT 
                session_id,
                COUNT(*) as metric_count,
                AVG(latency_ms) as avg_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
                AVG(fps) as avg_fps,
                AVG(cpu_utilization_percent) as avg_cpu_util,
                AVG(gpu_utilization_percent) as avg_gpu_util,
                MIN(timestamp) as first_metric,
                MAX(timestamp) as last_metric
            FROM performance_metrics
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY session_id
        """)
        
        await conn.execute("""
            CREATE OR REPLACE VIEW node_utilization_summary AS
            SELECT 
                nr.node_id,
                nr.node_name,
                nr.cpu_cores,
                nr.memory_total_gb,
                nr.gpu_count,
                COUNT(s.session_id) as active_sessions,
                AVG(pm.cpu_utilization_percent) as avg_cpu_util,
                AVG(pm.gpu_utilization_percent) as avg_gpu_util,
                nr.health_score,
                nr.last_heartbeat
            FROM node_resources nr
            LEFT JOIN sessions s ON nr.node_id = s.assigned_node AND s.state = 'active'
            LEFT JOIN performance_metrics pm ON nr.node_id = pm.node_id 
                AND pm.timestamp > NOW() - INTERVAL '5 minutes'
            GROUP BY nr.node_id, nr.node_name, nr.cpu_cores, nr.memory_total_gb, 
                     nr.gpu_count, nr.health_score, nr.last_heartbeat
        """)
        
        logger.info("Database views created")
        
        # Close connection
        await conn.close()
        logger.info("Database schema creation completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        raise

async def create_sample_data():
    """Create sample data for testing"""
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'omega'),
        'password': os.getenv('POSTGRES_PASSWORD', 'omega_secure_2025'),
        'database': os.getenv('POSTGRES_DB', 'omega_sessions')
    }
    
    try:
        conn = await asyncpg.connect(**db_config)
        
        # Create sample users
        await conn.execute("""
            INSERT INTO users (
                user_id, username, email, password_hash, display_name,
                subscription_tier, max_concurrent_sessions, created_at
            ) VALUES 
                ('user-001', 'admin', 'admin@omega-desktop.io', '$2b$12$dummy_hash_for_testing', 'System Administrator', 'enterprise', 10, NOW()),
                ('user-002', 'developer', 'dev@omega-desktop.io', '$2b$12$dummy_hash_for_testing', 'Developer User', 'professional', 5, NOW()),
                ('user-003', 'gamer', 'gamer@omega-desktop.io', '$2b$12$dummy_hash_for_testing', 'Gaming User', 'standard', 3, NOW())
            ON CONFLICT (user_id) DO NOTHING
        """)
        
        # Create sample nodes
        await conn.execute("""
            INSERT INTO node_resources (
                node_id, node_name, node_type, cluster_region,
                cpu_cores, cpu_model, memory_total_gb, memory_available_gb,
                gpu_count, gpu_memory_total_mb, gpu_memory_available_mb,
                network_bandwidth_gbps, status, health_score, last_heartbeat
            ) VALUES 
                ('node-compute-001', 'Gaming Compute Node 1', 'compute', 'us-west-1', 16, 'Intel Xeon Gold 6348', 128, 100, 2, 49152, 40960, 100.0, 'active', 0.95, NOW()),
                ('node-compute-002', 'Gaming Compute Node 2', 'compute', 'us-west-1', 16, 'Intel Xeon Gold 6348', 128, 110, 2, 49152, 45056, 100.0, 'active', 0.98, NOW()),
                ('node-storage-001', 'Storage Node 1', 'storage', 'us-west-1', 8, 'Intel Xeon Silver 4314', 64, 50, 0, 0, 0, 25.0, 'active', 0.92, NOW())
            ON CONFLICT (node_id) DO NOTHING
        """)
        
        # Create sample API key
        await conn.execute("""
            INSERT INTO api_keys (
                key_id, user_id, key_name, key_hash, permissions, created_at
            ) VALUES 
                ('key-001', 'user-001', 'Admin API Key', '$2b$12$dummy_api_key_hash', '["session:create", "session:delete", "node:manage"]', NOW())
            ON CONFLICT (key_id) DO NOTHING
        """)
        
        await conn.close()
        logger.info("Sample data created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")

async def main():
    """Main function to initialize database"""
    try:
        logger.info("Starting database initialization...")
        
        # Create schema
        await create_database_schema()
        
        # Create sample data
        await create_sample_data()
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
