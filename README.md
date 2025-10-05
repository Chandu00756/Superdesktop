## Omega Super Desktop Console v2.0

## Initial prototype Distributed Computing Platform

A revolutionary distributed computing system that aggregates CPU, GPU, RAM, storage and network of multiple commodity PCs into one low-latency "super desktop" that runs unmodified Windows/Linux/Mac workloads.

## [LAUNCH] Features

### Core Platform

- **Distributed Computing**: Horizontal scaling across compute, storage, and control nodes
- **Advanced Resource Orchestration**: Smart placement algorithms with predictive optimization
- **Real-time Monitoring**: Live performance metrics and health monitoring
- **AI-Driven Optimization**: Machine learning models for latency prediction and resource optimization
- **Intelligent Storage**: Multi-tier storage with automatic data lifecycle management
- **Enterprise Security**: JWT authentication, TLS encryption, and role-based access control

### Desktop Application

# Superdesktop v2.0

A distributed desktop platform that aggregates CPU, GPU, RAM, storage, and network resources across multiple machines into a “super desktop.” This README reflects the current code and scripts in this repository.

## Components and ports

- Backend API (FastAPI): [http://127.0.0.1:8443](http://127.0.0.1:8443) (HTTPS if OMEGA_ENABLE_TLS=1)
- Control Node: [http://127.0.0.1:7777](http://127.0.0.1:7777)
- Frontend (static demo): [http://127.0.0.1:8081/omega-new.html](http://127.0.0.1:8081/omega-new.html)
- Metrics (Prometheus): [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

Notes

- The start script and health checks target Control Node on 7777. If your logs show a different port, prefer the value printed by start-omega.sh.
- Additional services and ports for prototype deployments are defined in docker-compose.yml.

## Quick start

Option A: one command (recommended)

- ./start-omega.sh
- Open [http://127.0.0.1:8081/omega-new.html](http://127.0.0.1:8081/omega-new.html)
- API docs: [http://127.0.0.1:8443/docs](http://127.0.0.1:8443/docs)

Option B: manual (two terminals)

1. Backend API

```bash
cd backend
python start_backend.py
```

1. Control Node

```bash
cd control_node
python main.py
```

## Current API endpoints (implemented)

- Health/readiness
  - GET /health → { status: "ok" | "healthy" | "degraded" }
  - GET /ready → 200 when backend is ready, 503 otherwise
- Secure session bootstrap
  - POST /api/secure/session/start
- Virtual Desktop (prototype)
  - POST /api/secure/vd/start
  - GET  /api/secure/vd/list
- Backups (RBAC)
  - POST /secure/backup/snapshot  (requires backup:create)
  - GET  /secure/backup/snapshots (requires backup:view)
- Metrics
  - GET /api/secure/metrics (secure JSON view)

Not implemented in this build

- /api/secure/vd/create, /vd/{id}/url, pause/resume/terminate, per-VD snapshots

## Environment variables

Backend/TLS

- OMEGA_ENABLE_TLS=0            # Set 1 to enable HTTPS for backend
- OMEGA_SSL_CERT=path/to/cert.crt
- OMEGA_SSL_KEY=path/to/key.key

Control Node

- OMEGA_CONTROL_PORT=7777
- OMEGA_METRICS_PORT=8000

Logging/cluster

- OMEGA_LOG_LEVEL=INFO
- OMEGA_CLUSTER_NAME=superdesktop-cluster-v2

KV storage adapter

- OMEGA_STORE_BACKEND=memory    # memory (default) | minio
- OMEGA_MINIO_ENDPOINT=localhost:9000
- OMEGA_MINIO_ACCESS=minioadmin
- OMEGA_MINIO_SECRET=minioadmin
- OMEGA_MINIO_BUCKET=omega-kv

Discovery (optional)

- OMEGA_ENABLE_DISCOVERY=1
- OMEGA_DISCOVERY_METHODS=tcp,arp  # tcp,icmp,arp,mdns

## Storage notes

- Default is in-memory KV store.
- If OMEGA_STORE_BACKEND=minio and MinIO is healthy, the backend will use the MinIO adapter; otherwise it automatically falls back to memory and reports this in /health.

## Health and metrics

- Backend health: GET /health → { status: ok/healthy/degraded }
- Backend readiness: GET /ready
- Control Node health: GET [http://127.0.0.1:7777/health](http://127.0.0.1:7777/health)
- Prometheus metrics (Control Node): [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

## Testing

- Run tests: use the “run-pytests” task (pytest -q). All current tests should pass in the managed environment.

## Project structure (high level)

- backend/: FastAPI backend, start script in start_backend.py
- control_node/: Control/orchestration and desktop_app (static UI)
- start-omega.sh / stop-omega.sh: one-command start/stop
- docker-compose.yml: prototype multi-service deployment
- tests/: current test suites
- data/, logs/: runtime data and logs

## Enterprise Features (100% Complete)

### Advanced Virtual Desktop Management
- Complete VD lifecycle: create, pause/resume, terminate, snapshots
- Multi-protocol support: VNC, RDP, WebRTC, SPICE
- GPU acceleration and resource allocation
- Load balancing and cluster orchestration

### Multi-Cloud Orchestration  
- AWS, Azure, GCP, and Kubernetes deployment
- Blue-green, rolling, and canary deployment strategies
- Auto-scaling and traffic management
- Disaster recovery and backup management

### Machine Learning Pipeline
- Predictive resource usage forecasting
- Real-time anomaly detection
- Performance optimization suggestions
- Failure prediction and prevention
- Time series analysis and trend forecasting

### Enterprise Security & Compliance
- Role-based access control (RBAC)
- Advanced encryption (AES-256-GCM)
- Audit logging and compliance reporting
- Multi-factor authentication support
- Certificate management and rotation

### Production-Ready Operations
- Comprehensive health monitoring
- Prometheus metrics integration
- Grafana dashboards and alerting
- Log aggregation and analysis
- Backup and disaster recovery

## API Endpoints (Complete)

### Virtual Desktop Endpoints
- `POST /api/enterprise/vd/create` - Create VD with full options
- `GET /api/enterprise/vd/{id}/url` - Get connection URL
- `POST /api/enterprise/vd/{id}/pause` - Pause session
- `POST /api/enterprise/vd/{id}/resume` - Resume session
- `DELETE /api/enterprise/vd/{id}` - Terminate session
- `POST /api/enterprise/vd/{id}/snapshot` - Create snapshot
- `GET /api/enterprise/vd/{id}/snapshots` - List snapshots
- `DELETE /api/enterprise/vd/{id}/snapshot/{snap_id}` - Delete snapshot

### Multi-Cloud Endpoints
- `POST /api/enterprise/cloud/provision` - Provision infrastructure
- `POST /api/enterprise/cloud/deploy` - Deploy applications
- `POST /api/enterprise/cloud/scale/{name}` - Scale deployments
- `GET /api/enterprise/cloud/deployments` - List deployments

### Machine Learning Endpoints
- `POST /api/enterprise/ml/metrics/ingest` - Ingest metrics
- `POST /api/enterprise/ml/predict/resources` - Resource predictions
- `POST /api/enterprise/ml/detect/anomalies` - Anomaly detection
- `POST /api/enterprise/ml/forecast/load` - Load forecasting
- `POST /api/enterprise/ml/optimize/performance` - Performance optimization
- `POST /api/enterprise/ml/predict/failures` - Failure prediction

### Analytics & Administration
- `GET /api/enterprise/analytics/dashboard` - Analytics dashboard
- `GET /api/enterprise/admin/system/status` - System status
- `POST /api/enterprise/admin/system/maintenance` - Maintenance mode

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
