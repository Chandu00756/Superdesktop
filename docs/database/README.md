# Omega Database Architecture

[![License: MIT](https://img## Database Components

| Table | Purpose | Records | Growth Rate |
|-------|---------|---------|-------------|
| `nodes` | Cluster node registry | ~1K-10K | Low |
| `sessions` | Compute session tracking | ~100K-1M | High |
| `users` | User management | ~1K-100K | Medium |
| `node_metrics` | Performance metrics | ~100M+ | Very High |
| `system_events` | Audit and logging | ~10M+ | High |

## Key Features

- **Security First**: Row-level security, encryption at rest, audit logging
- **High Performance**: Optimized indexing, partitioning, connection pooling
- **Real-time**: Redis caching, WebSocket updates, sub-second queries
- **Time-Series**: TimescaleDB for metrics, automatic retention policies
- **Distributed**: Multi-node architecture, replication, failover

## Schema Documentationicense-MIT-yellow.svg)](<https://opensource.org/licenses/MIT>)

[![Database: PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](https://postgresql.org)
[![Extension: TimescaleDB](https://img.shields.io/badge/Extension-TimescaleDB-orange.svg)](https://timescale.com)
[![Cache: Redis](https://img.shields.io/badge/Cache-Redis-red.svg)](https://redis.io)

> **Enterprise-grade database design for distributed supercomputing infrastructure**

The Omega Super Desktop Console uses a sophisticated multi-database architecture designed for high-performance distributed computing environments. This repository contains the complete database schemas, implementation guides, and operational procedures for the open-source project.

## Architecture Overview

``
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  Control Node  │  Compute Nodes  │  Storage Nodes  │  UI   │
└─────────────────┬───────────────────────────────────┬───────┘
                  │                                   │
┌─────────────────┴───────────────────────────────────┴───────┐
│                    DATABASE LAYER                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│  PostgreSQL 15+ │    Redis 7+     │     SQLite (Local)      │
│  (Primary DB)   │  (Cache/RT)     │   (Node Storage)        │
├─────────────────┼─────────────────┼─────────────────────────┤
│  TimescaleDB    │                 │                         │
│ (Time Series)   │                 │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
``

## Quick Start

### Prerequisites

- PostgreSQL 15+ with TimescaleDB extension
- Redis 7+
- Python 3.9+ with asyncpg, redis, sqlalchemy
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/omega-team/omega-console.git
cd omega-console

# Install dependencies
pip install -r requirements.txt

# Setup development database
./scripts/setup_dev_db.sh

# Initialize schema
python scripts/init_database.py --env development
```

## Database Components

### Core Tables

| Table | Purpose | Records | Growth Rate |
|-------|---------|---------|-------------|
| `nodes` | Cluster node registry | ~1K-10K | Low |
| `sessions` | Compute session tracking | ~100K-1M | High |
| `users` | User management | ~1K-100K | Medium |
| `node_metrics` | Performance metrics | ~100M+ | Very High |
| `system_events` | Audit and logging | ~10M+ | High |

### Feature Highlights

- **Security First**: Row-level security, encryption at rest, audit logging
- **High Performance**: Optimized indexing, partitioning, connection pooling
- **Real-time**: Redis caching, WebSocket updates, sub-second queries
- **Time-Series**: TimescaleDB for metrics, automatic retention policies
- **Distributed**: Multi-node architecture, replication, failover

## Schema Documentation

### Entity Relationship Diagram

![Database Schema](docs/database/schema-diagram.png)

### Core Entities

#### Nodes

- **Purpose**: Hardware inventory and resource tracking
- **Key Fields**: `node_id`, `hostname`, `ip_address`, `node_type`, `resources`
- **Relationships**: One-to-many with sessions, metrics, allocations

#### Sessions

- **Purpose**: Compute workload lifecycle management
- **Key Fields**: `session_id`, `user_id`, `resource_requirements`, `status`
- **Relationships**: Belongs to user, allocated to nodes

#### Users

- **Purpose**: Authentication, authorization, quota management
- **Key Fields**: `user_id`, `username`, `role`, `quotas`
- **Relationships**: One-to-many with sessions, audit logs

### Performance Metrics

```sql
-- Example: Get cluster utilization
SELECT 
    COUNT(*) as total_nodes,
    COUNT(*) FILTER (WHERE status = 'active') as active_nodes,
    AVG(cpu_usage) as avg_cpu_utilization,
    AVG(memory_usage) as avg_memory_utilization
FROM nodes n
LEFT JOIN LATERAL (
    SELECT value as cpu_usage 
    FROM node_metrics 
    WHERE node_id = n.node_id 
      AND metric_type = 'cpu_usage' 
      AND time >= NOW() - INTERVAL '5 minutes'
    ORDER BY time DESC 
    LIMIT 1
) cpu ON true
LEFT JOIN LATERAL (
    SELECT value as memory_usage 
    FROM node_metrics 
    WHERE node_id = n.node_id 
      AND metric_type = 'memory_usage' 
      AND time >= NOW() - INTERVAL '5 minutes'
    ORDER BY time DESC 
    LIMIT 1
) mem ON true;
```

## Operations

### Backup & Recovery

```bash
# Automated daily backup
0 2 * * * /opt/omega/scripts/backup_omega_db.sh

# Point-in-time recovery
./scripts/restore_omega_db.sh /backup/omega_backup_20250801_020000.dump
```

### Monitoring

```sql
-- Database health check
SELECT 
    'postgresql' as service,
    pg_is_in_recovery() as is_replica,
    pg_database_size(current_database()) as size_bytes,
    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections;
```

### Performance Tuning

Key configuration parameters:

```conf
# postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
max_connections = 200
checkpoint_completion_target = 0.9
```

## Scalability

### Horizontal Scaling

- **Read Replicas**: PostgreSQL streaming replication
- **Sharding**: Partition large tables by node_id or time
- **Caching**: Redis cluster for session state and metrics

### Vertical Scaling

- **Memory**: Optimized for 32GB+ RAM systems
- **Storage**: NVMe SSD recommended for time-series data
- **CPU**: Benefits from 8+ core systems

### Growth Projections

| Scale | Nodes | Sessions/Day | DB Size | Memory Req |
|-------|-------|--------------|---------|------------|
| Small | 10-50 | 100-1K | 10GB | 8GB |
| Medium | 50-500 | 1K-10K | 100GB | 32GB |
| Large | 500-5K | 10K-100K | 1TB | 128GB |
| Enterprise | 5K+ | 100K+ | 10TB+ | 512GB+ |

## Security

### Access Control

```sql
-- Role-based permissions
CREATE ROLE omega_admin;    -- Full access
CREATE ROLE omega_operator; -- Read/write operations
CREATE ROLE omega_user;     -- Limited to own resources
CREATE ROLE omega_readonly; -- Read-only access
```

### Data Protection

- **Encryption**: AES-256 for sensitive columns
- **Audit Logging**: All operations logged to `system_events`
- **Row-Level Security**: Users can only access their own data
- **Connection Security**: TLS 1.3 required for all connections

## Testing

### Test Data Generation

```python
# Generate realistic test data
python scripts/generate_test_data.py --nodes 100 --sessions 1000 --days 30
```

### Performance Testing

```bash
# Database load testing
pgbench -h localhost -U omega_test -d omega_test -c 10 -j 2 -T 60
```

### Integration Tests

```bash
# Run full test suite
pytest tests/database/ -v
```

## Documentation

- [Complete Database Design](docs/database/DATABASE_DESIGN.md) - Detailed schema documentation
- [Implementation Guide](docs/database/IMPLEMENTATION_GUIDE.md) - Setup and operations
- [Schema Diagram](docs/database/SCHEMA_DIAGRAM.md) - Visual entity relationships
- [API Reference](docs/api/) - Database API documentation
- [Migration Guide](docs/database/MIGRATIONS.md) - Schema upgrade procedures

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork the repository
git clone https://github.com/your-username/omega-console.git

# Create feature branch
git checkout -b feature/database-enhancement

# Make changes and test
python -m pytest tests/

# Submit pull request
```

### Code Style

- SQL: Use lowercase with underscores
- Python: Follow PEP 8
- Documentation: Markdown with proper formatting

## Metrics & Monitoring

### Key Performance Indicators

- **Query Response Time**: < 100ms for 95th percentile
- **Throughput**: > 10K transactions/second
- **Availability**: 99.9% uptime target
- **Data Consistency**: Zero data loss tolerance

### Monitoring Stack

- **Metrics**: Prometheus + TimescaleDB
- **Visualization**: Grafana dashboards
- **Alerting**: AlertManager integration
- **Logging**: Structured JSON logs

## Roadmap

### Version 2.0 (Q3 2025)

- [ ] Multi-tenant architecture
- [ ] Advanced analytics with ML
- [ ] Cross-region replication
- [ ] Kubernetes native deployment

### Version 3.0 (Q1 2026)

- [ ] Serverless computing support
- [ ] Edge node integration
- [ ] Blockchain-based resource accounting
- [ ] AI-driven optimization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PostgreSQL community for the robust database engine
- TimescaleDB team for time-series capabilities
- Redis community for high-performance caching
- All contributors to the Omega project

---

### Built with love by the Omega Team

For support, please [open an issue](https://github.com/omega-team/omega-console/issues) or join our [Discord community](https://discord.gg/omega).
