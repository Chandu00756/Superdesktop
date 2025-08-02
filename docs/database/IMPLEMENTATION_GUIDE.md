# Omega Super Desktop Console - Database Implementation Guide

## Quick Start

This guide provides step-by-step instructions for setting up the Omega database infrastructure for development and initial prototype environments.

## Prerequisites

- PostgreSQL 15+ with TimescaleDB extension
- Redis 7+
- Python 3.9+ with asyncpg, redis, and sqlalchemy
- Docker (optional, for containerized deployment)

## Development Setup

### 1. Install Database Dependencies

```bash
# macOS
brew install postgresql redis timescaledb

# Ubuntu/Debian
sudo apt-get install postgresql-15 redis-server postgresql-15-timescaledb

# Enable TimescaleDB
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

### 2. Create Development Database

```bash
# Create database and user
sudo -u postgres createdb omega_dev
sudo -u postgres createuser omega_dev
sudo -u postgres psql -c "ALTER USER omega_dev WITH PASSWORD 'omega_dev_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE omega_dev TO omega_dev;"

# Enable TimescaleDB extension
sudo -u postgres psql omega_dev -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
sudo -u postgres psql omega_dev -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
```

### 3. Initialize Schema

```bash
cd /path/to/omega-console
python scripts/init_database.py --env development
```

## Initial Prototype Setup

### 1. Database Configuration

Create `config/database.yaml`:

```yaml
databases:
  primary:
    type: postgresql
    host: ${POSTGRES_HOST}
    port: ${POSTGRES_PORT}
    database: ${POSTGRES_DB}
    username: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    ssl_mode: require
    
  cache:
    type: redis
    host: ${REDIS_HOST}
    port: ${REDIS_PORT}
    password: ${REDIS_PASSWORD}
    db: 0
    max_connections: 100
    
  timeseries:
    type: timescaledb
    host: ${TIMESCALEDB_HOST}
    port: ${TIMESCALEDB_PORT}
    database: ${TIMESCALEDB_DB}
    username: ${TIMESCALEDB_USER}
    password: ${TIMESCALEDB_PASSWORD}
```

### 2. Environment Variables

```bash
# PostgreSQL Configuration
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=omega_prototype
export POSTGRES_USER=omega_prod
export POSTGRES_PASSWORD=secure_password_here

# Redis Configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=redis_password_here

# TimescaleDB Configuration (can be same as PostgreSQL)
export TIMESCALEDB_HOST=localhost
export TIMESCALEDB_PORT=5432
export TIMESCALEDB_DB=omega_prototype
export TIMESCALEDB_USER=omega_prod
export TIMESCALEDB_PASSWORD=secure_password_here
```

## Database Schema Overview

### Core Tables

``
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     USERS       │    │     NODES       │    │   SESSIONS      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ user_id (PK)    │    │ node_id (PK)    │    │ session_id (PK) │
│ username        │    │ hostname        │    │ user_id (FK)    │
│ email           │    │ ip_address      │    │ session_name    │
│ password_hash   │    │ node_type       │    │ session_type    │
│ role            │    │ status          │    │ status          │
│ quotas          │    │ resources       │    │ resources       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ USER_SESSIONS   │    │ NODE_RESOURCES  │    │SESSION_ALLOCS   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ token_id (PK)   │    │ resource_id(PK) │    │ allocation_id   │
│ user_id (FK)    │    │ node_id (FK)    │    │ session_id (FK) │
│ session_token   │    │ resource_type   │    │ node_id (FK)    │
│ expires_at      │    │ capacity        │    │ allocated_amt   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
``

### Time-Series Tables (TimescaleDB)

``
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  NODE_METRICS   │    │SESSION_METRICS  │    │ SYSTEM_EVENTS   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ time (PK)       │    │ time (PK)       │    │ event_id (PK)   │
│ node_id (FK)    │    │ session_id (FK) │    │ timestamp       │
│ metric_type     │    │ metric_type     │    │ event_type      │
│ value           │    │ value           │    │ severity        │
│ unit            │    │ unit            │    │ message         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
``

### Storage Tables

``
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ STORAGE_POOLS   │    │STORAGE_VOLUMES  │    │NETWORK_TOPOLOGY │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ pool_id (PK)    │    │ volume_id (PK)  │    │ topology_id(PK) │
│ pool_name       │    │ pool_id (FK)    │    │ topology_name   │
│ pool_type       │    │ session_id (FK) │    │ network_type    │
│ capacity        │    │ volume_name     │    │ subnet          │
│ replication     │    │ size_gb         │    │ bandwidth       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
``

## Database Operations

### Connection Management

```python
# Database connection example
import asyncpg
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine

class DatabaseManager:
    def __init__(self):
        self.postgres_pool = None
        self.redis_client = None
        self.engine = None
    
    async def initialize(self):
        # PostgreSQL connection pool
        self.postgres_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB'),
            min_size=5,
            max_size=20
        )
        
        # Redis connection
        self.redis_client = await aioredis.from_url(
            f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
            password=os.getenv('REDIS_PASSWORD'),
            encoding='utf-8',
            decode_responses=True
        )
        
        # SQLAlchemy async engine
        database_url = f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        self.engine = create_async_engine(database_url, echo=False)
```

### Common Queries

```python
# Node management queries
class NodeQueries:
    @staticmethod
    async def register_node(pool, node_data):
        async with pool.acquire() as conn:
            return await conn.fetchrow("""
                INSERT INTO nodes (hostname, ip_address, node_type, total_cpu_cores, 
                                 total_memory_gb, total_storage_gb, gpu_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING node_id
            """, node_data['hostname'], node_data['ip_address'], 
                node_data['node_type'], node_data['cpu_cores'],
                node_data['memory_gb'], node_data['storage_gb'], 
                node_data.get('gpu_count', 0))
    
    @staticmethod
    async def update_node_heartbeat(pool, node_id):
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE nodes 
                SET last_heartbeat = NOW(), status = 'active'
                WHERE node_id = $1
            """, node_id)
    
    @staticmethod
    async def get_available_nodes(pool, resource_requirements):
        async with pool.acquire() as conn:
            return await conn.fetch("""
                SELECT n.*, 
                       COALESCE(SUM(nr.available_capacity) FILTER (WHERE nr.resource_type = 'cpu'), 0) as available_cpu,
                       COALESCE(SUM(nr.available_capacity) FILTER (WHERE nr.resource_type = 'memory'), 0) as available_memory
                FROM nodes n
                LEFT JOIN node_resources nr ON n.node_id = nr.node_id
                WHERE n.status = 'active'
                  AND n.last_heartbeat > NOW() - INTERVAL '60 seconds'
                GROUP BY n.node_id
                HAVING COALESCE(SUM(nr.available_capacity) FILTER (WHERE nr.resource_type = 'cpu'), 0) >= $1
                   AND COALESCE(SUM(nr.available_capacity) FILTER (WHERE nr.resource_type = 'memory'), 0) >= $2
                ORDER BY available_cpu DESC, available_memory DESC
            """, resource_requirements['cpu'], resource_requirements['memory'])

# Session management queries
class SessionQueries:
    @staticmethod
    async def create_session(pool, session_data):
        async with pool.acquire() as conn:
            return await conn.fetchrow("""
                INSERT INTO sessions (user_id, session_name, session_type, application,
                                    cpu_request, memory_request_gb, gpu_request, priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING session_id
            """, session_data['user_id'], session_data['session_name'],
                session_data['session_type'], session_data['application'],
                session_data['cpu_request'], session_data['memory_request_gb'],
                session_data.get('gpu_request', 0), session_data.get('priority', 1))
    
    @staticmethod
    async def allocate_resources(pool, session_id, node_id, allocations):
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Update session with allocated node
                await conn.execute("""
                    UPDATE sessions 
                    SET scheduled_node_id = $1, status = 'scheduled'
                    WHERE session_id = $2
                """, node_id, session_id)
                
                # Create allocation records
                for resource_type, amount in allocations.items():
                    await conn.execute("""
                        INSERT INTO session_allocations (session_id, node_id, resource_type, allocated_amount, unit)
                        VALUES ($1, $2, $3, $4, $5)
                    """, session_id, node_id, resource_type, amount['value'], amount['unit'])
                    
                    # Update node resource availability
                    await conn.execute("""
                        UPDATE node_resources 
                        SET allocated_capacity = allocated_capacity + $1
                        WHERE node_id = $2 AND resource_type = $3
                    """, amount['value'], node_id, resource_type)

# Metrics queries
class MetricsQueries:
    @staticmethod
    async def insert_node_metrics(pool, node_id, metrics):
        async with pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO node_metrics (time, node_id, metric_type, value, unit)
                VALUES ($1, $2, $3, $4, $5)
            """, [(metric['timestamp'], node_id, metric['type'], 
                   metric['value'], metric['unit']) for metric in metrics])
    
    @staticmethod
    async def get_cluster_metrics(pool, time_range='1 hour'):
        async with pool.acquire() as conn:
            return await conn.fetch(f"""
                SELECT 
                    metric_type,
                    AVG(value) as avg_value,
                    MAX(value) as max_value,
                    MIN(value) as min_value,
                    COUNT(*) as sample_count
                FROM node_metrics 
                WHERE time >= NOW() - INTERVAL '{time_range}'
                GROUP BY metric_type
                ORDER BY metric_type
            """)
```

### Redis Operations

```python
class CacheOperations:
    @staticmethod
    async def set_node_heartbeat(redis_client, node_id, heartbeat_data):
        key = f"node:heartbeat:{node_id}"
        await redis_client.setex(key, 30, json.dumps(heartbeat_data))
    
    @staticmethod
    async def get_active_nodes(redis_client):
        keys = await redis_client.keys("node:heartbeat:*")
        active_nodes = []
        for key in keys:
            data = await redis_client.get(key)
            if data:
                active_nodes.append(json.loads(data))
        return active_nodes
    
    @staticmethod
    async def cache_session_state(redis_client, session_id, state_data):
        key = f"session:state:{session_id}"
        ttl = 3600  # 1 hour
        await redis_client.setex(key, ttl, json.dumps(state_data))
    
    @staticmethod
    async def get_cluster_state(redis_client):
        return await redis_client.hgetall("cluster:state")
    
    @staticmethod
    async def update_cluster_state(redis_client, updates):
        await redis_client.hmset("cluster:state", updates)
```

## Performance Tuning

### PostgreSQL Configuration

Add to `postgresql.conf`:

```conf
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Connection settings
max_connections = 200

# TimescaleDB settings
timescaledb.max_background_workers = 8
```

### Monitoring Queries

```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup_omega_db.sh

DB_NAME="omega_prototype"
BACKUP_DIR="/backup/omega"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Full database backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $DB_NAME \
    --format=custom --compress=9 \
    --file=$BACKUP_DIR/omega_backup_$TIMESTAMP.dump

# Backup Redis data
redis-cli --rdb $BACKUP_DIR/redis_backup_$TIMESTAMP.rdb

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete

echo "Backup completed: $TIMESTAMP"
```

### Recovery Script

```bash
#!/bin/bash
# restore_omega_db.sh

BACKUP_FILE=$1
DB_NAME="omega_prototype"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application services
systemctl stop omega-control-node
systemctl stop omega-compute-node

# Restore database
pg_restore -h $POSTGRES_HOST -U $POSTGRES_USER -d $DB_NAME \
    --clean --if-exists --verbose $BACKUP_FILE

# Restart services
systemctl start omega-control-node
systemctl start omega-compute-node

echo "Recovery completed"
```

## Migration Management

### Database Migration Script

```python
# migrations/001_initial_schema.py
import asyncpg

async def upgrade(connection):
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Apply schema changes
    with open('sql/001_create_tables.sql', 'r') as f:
        sql = f.read()
        await connection.execute(sql)
    
    # Record migration
    await connection.execute("""
        INSERT INTO schema_version (version) VALUES (1)
        ON CONFLICT (version) DO NOTHING;
    """)

async def downgrade(connection):
    # Rollback changes
    await connection.execute("DROP TABLE IF EXISTS nodes CASCADE;")
    await connection.execute("DELETE FROM schema_version WHERE version = 1;")
```

This database design provides a solid foundation for the open-source Omega Super Desktop Console project, with comprehensive schemas, performance optimizations, and operational procedures.
