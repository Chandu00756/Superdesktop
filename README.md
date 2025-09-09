## Omega Super Desktop Console v2.0

## Initial prototype Distributed Computing Platform

A revolutionary distributed computing system that aggregates CPU, GPU, RAM, storage and network of multiple commodity PCs into one low-latency "super desktop" that runs unmodified Windows/Linux/Mac workloads.

##  Features

### Core Platform

- **Distributed Computing**: Horizontal scaling across compute, storage, and control nodes
- **Advanced Resource Orchestration**: Smart placement algorithms with predictive optimization
- **Real-time Monitoring**: Live performance metrics and health monitoring
- **AI-Driven Optimization**: Machine learning models for latency prediction and resource optimization
- **Intelligent Storage**: Multi-tier storage with automatic data lifecycle management
- **Enterprise Security**: JWT authentication, TLS encryption, and role-based access control

### Desktop Application

- **Professional UI**: Modern Electron-based interface with responsive design
- **Real-time Dashboards**: Live charts and metrics using Chart.js
- **Session Management**: Create and manage compute sessions with custom resource allocation
- **Node Monitoring**: Visual node status and resource utilization
- **Performance Analytics**: Historical performance data and trend analysis

# Virtual Desktop Sessions (NoVNC over VNC)

This version adds a production-ready Virtual Desktop under the Sessions tab.

Key points:

- Backend secure API on <https://127.0.0.1:8443> under `/api/secure/*`.
- Handshake endpoint: `POST /api/secure/session/start` for AES-256-GCM session bootstrap.
- Create desktop: `POST /api/secure/vd/create` with JSON body `{ user_id, os_image, cpu_cores, memory_gb }`.
- Get connection URL: `GET /api/secure/vd/{session_id}/url` returns a noVNC URL.
- Pause/Resume: `POST /api/secure/vd/{session_id}/pause` and `POST /api/secure/vd/{session_id}/resume`.
- Terminate: `DELETE /api/secure/vd/{session_id}`.

Snapshots and RDP:

- Snapshot: `POST /api/secure/vd/{session_id}/snapshot`
- List snapshots: `GET /api/secure/vd/{session_id}/snapshots`
- Delete snapshot: `POST /api/secure/vd/{session_id}/snapshot/delete` with `{ tag }`
- Restore snapshot: `POST /api/secure/vd/{session_id}/snapshot/restore` with `{ tag }` (restores to a new container and returns connect_url)
- RDP sessions (Windows):
  - Create: `POST /api/secure/rdp/create` with `{ user_id, host, port, username, password, domain? }`
  - Get URL: `GET /api/secure/rdp/{session_id}/url`
  - Delete: `DELETE /api/secure/rdp/{session_id}`

RBAC enforcement:

- Destructive operations (kill processes, delete/restore snapshots) require `admin` role.
- Session deletion allowed to the owner or an admin.

Images:

- ubuntu-xfce: `dorowu/ubuntu-desktop-lxde-vnc` (default)
- debian-xfce: `accetto/debian-vnc-xfce`
- kali-xfce: `lscr.io/linuxserver/kali-linux:latest` (may need extra flags)
- Windows desktops require licensed VM and use RDP; not supported via the NoVNC container path.
- Qubes OS is provided as an external OS option (qubes-os) in the catalog. It cannot run inside Docker; use RDP/VNC to connect to an existing Qubes VM managed outside this platform, or register a connector via `/api/secure/vd/os-register`.

Storage:

- Default per-session path: `./data/object_storage/sessions/<session_id>`
- Override with `OMEGA_SESSION_BASE` env var.

TLS:

- If `security/certs/control_node.crt|.key` exist or `OMEGA_SSL_CERT`/`OMEGA_SSL_KEY` env provided, the backend serves HTTPS on 8443.

Notes:

- Requires Docker installed and available on host for managing desktop containers.

## [ARCHITECTURE] Architecture

## SuperDesktop v2.0 - Distributed Desktop Environment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/Chandu00756/Superdesktop)

> ** Latest Update**: Complete system v2.0 with unified architecture, modular frontend system, and enterprise-grade security!

##  Overview

**SuperDesktop v2.0** is a revolutionary distributed desktop environment that combines the power of cloud computing with the familiarity of traditional desktop interfaces. Built from the ground up with modern web technologies and distributed systems architecture, it provides a seamless, AI-powered desktop experience that can scale from personal use to enterprise deployments.

###  Key Features

- ** Distributed Desktop Environment**: Full desktop experience running across multiple nodes
- ** AI-Powered Intelligence**: Built-in AI assistance for system management and optimization
- ** Enterprise Security**: AES-256-GCM encryption, JWT authentication, and secure communication
- ** Real-time Monitoring**: Live system metrics, performance tracking, and health monitoring
- ** Modern Frontend Interface**: Modular ES6-based interface with responsive design
- ** Fault Tolerance**: Self-healing architecture with automatic failover and recovery
- ** Containerized Deployment**: Docker support for easy deployment and scaling
- ** Interactive Interface**: Modern, responsive UI with real-time data updates

##  Quick Start

### Option 1: One-Command Startup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Chandu00756/Superdesktop.git
cd Superdesktop

# Start the complete system
./start-omega.sh
```

The startup script will automatically:

-  Check system requirements (Python 3.11+, Node.js 18+)
-  Create and activate virtual environment
-  Install all dependencies automatically
-  Start Backend API Server (Port 8443)
-  Start Frontend HTTP Server (Port 8081)
-  Start all node services (Control, Storage, Compute, etc.)
-  Open the desktop interface in your browser

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv omega_env
source omega_env/bin/activate  # Linux/macOS
# or
omega_env\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start backend services
python backend/api_server.py &

# 4. Start frontend HTTP server
python -m http.server 8081 --directory control_node/desktop_app &

# 5. Start control node
python control_node/main.py &

# 6. Open the desktop interface
open http://localhost:8081/omega-new.html  # macOS
# or navigate to the URL in your browser
```

###  Access the System

After startup, access the system through:

- ** Main Desktop Interface**: `http://localhost:8081/omega-new.html` (Modern modular interface)
- ** Backend API**: `http://localhost:8443/docs` (FastAPI documentation)
- ** Control Node**: `http://localhost:7777/health` (Node management)
- ** Metrics Dashboard**: `http://localhost:8000/metrics` (System monitoring)

###  Stop the System

```bash
# Stop all services gracefully
./stop-omega.sh
```

##  **What You Get**

### ** Desktop Interface**

- **URL**: `http://localhost:8081/omega-new.html`
- **Features**: Real-time monitoring, node management, performance analytics
- **Data**: 100% real backend integration (no simulation)

### ** Backend Services**

- **API Server**: `http://127.0.0.1:8443` (encrypted endpoints)
- **Control Node**: `http://127.0.0.1:7777` (orchestration)
- **Metrics**: `http://127.0.0.1:8000/metrics` (Prometheus-compatible)

### ** Key Features**

-  Real-time Performance Monitoring**
-  AI-Powered Resource Optimization**
-  Fault-Tolerant Multi-Master Architecture**
-  Heterogeneous Hardware Support** (CPU/GPU/NPU/FPGA)
-  Tiered Storage Management**
-  Secure Encrypted Communication**
-  Hot-Swappable Components**
-  Auto-Discovery & Self-Registration**

---

## Architecture Overview**

---

## Project Structure**

```text
SuperDesktop/
├──  start-omega.sh               # Main startup script
├──  stop-omega.sh                # System shutdown script
├──  requirements.txt              # Python dependencies
├──  README.md                     # This file
│
├── 🔧 backend/                      # Backend API services
│   ├── api_server.py               # FastAPI backend server (Port 8443)
│   ├── frontend_connector.py       # Frontend integration
│   └── requirements.txt            # Backend dependencies
│
├── 🖥️ control_node/                # Control & orchestration
│   ├── main.py                     # Control node manager
│   └── desktop_app/                # Desktop interface
│       ├── omega-new.html          # Main UI Interface
│       ├── omega-style.css         # Styling and layout
│       ├── omega-renderer.js       # Core rendering engine
│       └── frontend/               # Modular Frontend System
│           ├── main.js             # Bootstrap entry point
│           ├── core/               # Core system modules
│           │   ├── StateStore.js   # State management
│           │   ├── apiClient.js    # API communication
│           │   ├── widgetManager.js # Widget management
│           │   └── sidebarManager.js # Navigation management
│           └── tabs/               # Tab renderers (dashboard, nodes, etc.)
│
├──  storage_node/                # Storage management
├──  compute_node/                # Compute resources
├──  ai_engine/                   # AI optimization
├──  network/                     # Network management
├──  security/                    # Security & authentication
├──  session-daemon/              # Session management
├──  omega-orchestrator/          # System orchestration
├──  memory-fabric/               # Memory management
├──  predictor-service/           # Predictive analytics
├──  render-router/               # Rendering router
├──  middleware/                  # Middleware services
├──  edge_node/                   # Edge computing
├──  network_node/                # Network node
├──  data/                        # Data storage (PostgreSQL, Redis, Object storage)
├──  logs/                        # System logs
├──  tests/                       # Test suites
├──  docs/                        # Documentation
└──  omega_env/                   # Python virtual environment
```

---

## 🛠️ **Development Workflow**

### **1. Initial Setup**

```bash
# One-time setup
git clone https://github.com/Chandu00756/Superdesktop.git
cd Superdesktop
chmod +x *.sh
```

### **2. Development Cycle**

```bash
# Start development environment
./start-omega.sh

# Code changes...
# (System auto-restarts services on changes)

# Stop when done
./stop-omega.sh
```

### **3. Testing**

```bash
# Start system
./start-omega.sh

# Test endpoints
curl http://127.0.0.1:8443/api/dashboard/metrics
curl http://127.0.0.1:7777/health

# Check desktop app
open http://localhost:8081/omega-new.html
```

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
export OMEGA_LOG_LEVEL="DEBUG"          # Logging level
export OMEGA_CLUSTER_NAME="dev-cluster" # Cluster name
export OMEGA_TLS_ENABLED="false"        # TLS in development
```

### **Development Mode**

The system automatically detects development mode when:

- Running on `localhost` or `127.0.0.1`
- Python environment is in project directory
- Debug logging is enabled

---

##  **Network Ports**

| Service | Port | Description |
|---------|------|-------------|
| Backend API | 8443 | Main API server (encrypted) |
| Control Node | 7777 | Orchestration & management |
| WebSocket | 7778 | Real-time updates |
| Metrics | 8000 | Prometheus metrics |
| Storage | 8001 | Storage node API |
| Compute | 8002 | Compute node API |

---

##  **Security Features**

-  **AES-256 Encryption** for all API communication
-  **HMAC Authentication** for message integrity
-  **SSL/TLS Support** for production deployments
-  **Role-Based Access Control** (RBAC)
-  **Secure Token Management**

---

##  **Performance Monitoring**

### **Built-in Dashboards**

- **Real-time Metrics**: CPU, GPU, Memory, Network, Storage
- **Node Health**: Status, temperature, power consumption
- **Workload Analytics**: Task distribution, completion rates
- **AI Predictions**: Resource optimization suggestions

### **Metrics Export**

- **Prometheus**: `http://127.0.0.1:8000/metrics`
- **JSON API**: `http://127.0.0.1:8443/api/dashboard/metrics`
- **WebSocket**: Real-time updates every 2 seconds

---

## 🔧 **Troubleshooting**

### **Common Issues**

### Python Environment Issues

```bash
# Recreate environment
rm -rf omega_env
python3 -m venv omega_env
source omega_env/bin/activate
pip install -r requirements.txt
```

### Port Conflicts

```bash
# Check what's using ports
lsof -i :8443
lsof -i :7777

# Kill conflicting processes
./stop-omega.sh
```

### Missing Dependencies**

```bash
# Reinstall dependencies
source omega_env/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Browser Access Issues**

```bash
# Ensure services are running
curl http://127.0.0.1:8443/api/dashboard/metrics

# Open desktop app manually
open http://localhost:8081/omega-new.html
```

### **Log Files**

```bash
# Check service logs
tail -f logs/*.log

# Check startup logs
./start-omega.sh | tee startup.log
```

---

##  **Production Deployment**

### **Docker Deployment**

```bash
# Build and run with Docker
docker-compose up -d
```

### **Manual Production Setup**

```bash
# Production environment
export OMEGA_TLS_ENABLED="true"
export OMEGA_LOG_LEVEL="INFO"
export OMEGA_CLUSTER_NAME="production"

# Start with production settings
./start-omega.sh
```

---

## 📞 **Support & Contact**

- ** Email**: <chandu@portalvii.com>
- ** Repository**: <https://github.com/Chandu00756/Superdesktop>
- ** Issues**: <https://github.com/Chandu00756/Superdesktop/issues>
- ** Documentation**: `/docs` directory

---

## 📄 **License**

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

##  **Acknowledgments**

- **Core Team**: Advanced distributed systems architecture
- **AI Integration**: Machine learning optimization engines
- **Security**: Enterprise-grade encryption implementation
- **UI/UX**: Professional dashboard design

---

**SuperDesktop v2.0** - *Revolutionizing distributed computing with AI-powered optimization*

**Contact**: <chandu@portalvii.com> | **Version**: 2.0 | **Status**: Production Ready

## [PACKAGE] Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 13+
- Redis 6+
- GPU support (optional, for AI workloads)

## [TOOLS] Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Database

```bash
# Install PostgreSQL and Redis
brew install postgresql redis  # macOS
# OR
sudo apt install postgresql redis-server  # Ubuntu

# Start services
brew services start postgresql redis  # macOS
# OR
sudo systemctl start postgresql redis  # Ubuntu

# Create database
createdb omega_desktop
```

### 3. Install Desktop Application Dependencies

```bash
cd control_node/desktop_app
npm install
```

## [LAUNCH] Quick Start

### 1. Start Control Node (Backend API)

```bash
cd control_node
python main.py
```

The control node will start on `http://localhost:8443` with:

- REST API endpoints
- WebSocket real-time updates
- Prometheus metrics on `/metrics`
- Interactive API docs on `/docs`

### 2. Start Compute Nodes

```bash
# Terminal 1 - CPU Node
cd compute_node
python advanced_compute.py --node-type cpu --port 8001

# Terminal 2 - GPU Node
cd compute_node
python advanced_compute.py --node-type gpu --port 8002

# Terminal 3 - Storage Node
cd storage_node
python advanced_storage.py --port 8003
```

### 3. Launch Desktop Application

```bash
cd control_node/desktop_app
npm start
```

## [DESKTOP] Desktop Application Usage

### Login

- Default credentials: `admin` / `password123`
- The application will authenticate with the control node backend

### Dashboard

- **System Overview**: Real-time CPU, memory, and network metrics
- **Performance Charts**: Historical performance data with trend analysis
- **Activity Feed**: Live system events and notifications
- **Resource Utilization**: Visual breakdown of compute resource usage

### Session API Reference

1. Click "Create Session" to launch new compute sessions
2. Configure resource requirements (CPU cores, GPU units, RAM)
3. Select applications or custom workloads
4. Monitor session status and performance

### Node Overview

- View all connected compute and storage nodes
- Monitor node health and resource availability
- Track node performance metrics

### AI Optimization

- View prediction model accuracy
- Monitor AI-driven resource optimization
- Analyze latency prediction performance

## [CONFIG] Configuration

### Environment Variables

```bash
# Control Node
export OMEGA_DB_URL="postgresql://user:pass@localhost/omega_desktop"
export OMEGA_REDIS_URL="redis://localhost:6379"
export OMEGA_SECRET_KEY="your-secret-key"

# Optional GPU Support
export CUDA_VISIBLE_DEVICES="0,1"
export OMEGA_ENABLE_GPU="true"
```

### Database Migration

```bash
cd control_node
alembic upgrade head
```

## [DASHBOARD] Monitoring & Metrics

### Prometheus Metrics

Access metrics at `http://localhost:8443/metrics`

### Key Metrics

- `omega_active_sessions_total`: Number of active compute sessions
- `omega_node_cpu_usage`: Per-node CPU utilization
- `omega_prediction_accuracy`: AI model prediction accuracy
- `omega_storage_tier_usage`: Storage utilization by tier

### Health Checks

- Control Node: `GET /health`
- Compute Nodes: `GET /node/health`
- Storage Nodes: `GET /storage/health`

## [ENCRYPTED] Security

### Authentication

- JWT tokens with configurable expiration
- Role-based access control (admin, user, viewer)
- Session management with automatic cleanup

### Encryption

- TLS 1.3 for all inter-node communication
- Fernet encryption for data at rest
- AES-256 for sensitive configuration

### Network Security

- Configurable firewall rules
- VPN support for node communication
- Rate limiting and DDoS protection

## [TEST] Testing

### Unit Tests

```bash
cd tests
python -m pytest unit/
```

### Integration Tests

```bash
python -m pytest integration/
```

### Load Testing

```bash
cd tests/load
python load_test.py --sessions 100 --duration 300
```

## [METRICS] Performance Tuning

### Database Optimization

- Connection pooling configured for high concurrency
- Query optimization with proper indexing
- Read replicas for analytics workloads

### Compute Optimization

- NUMA-aware scheduling
- CPU affinity for latency-sensitive workloads
- GPU memory management

### Storage Optimization

- Intelligent tiering (hot/warm/cold)
- Compression and deduplication
- Predictive prefetching

## [REFRESH] High Availability

### Control Node

- Multi-master setup with leader election
- Automatic failover and recovery
- State synchronization via Redis

### Compute Nodes

- Health monitoring with automatic restart
- Graceful task migration on node failure
- Resource rebalancing

### Storage Nodes

- Multi-replica storage with consistency guarantees
- Automatic data recovery
- Cross-node redundancy

## [DOCUMENT] API Reference

### Session Management

```bash
# Create Session
POST /session/create
{
  "user_id": "admin",
  "app_uri": "game://mygame.exe",
  "cpu_cores": 4,
  "gpu_units": 1,
  "ram_bytes": 8589934592
}

# List Sessions
GET /sessions

# Terminate Session
DELETE /session/{session_id}
```

### Node Management

```bash
# Register Node
POST /node/register
{
  "node_type": "compute",
  "resources": {...}
}

# Get Nodes
GET /nodes

# Node Health
GET /node/{node_id}/health
```

### Metrics & Monitoring

```bash
# System Metrics
GET /metrics/system

# Performance Data
GET /metrics/performance

# Prediction Analytics
GET /ai/predictions
```

## [COLLABORATE] Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## [NOTES] License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Documentation

- API Documentation: `http://localhost:8443/docs`
- Architecture Guide: `docs/architecture.md`
- Deployment Guide: `docs/deployment.md`

### Community

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Wiki: Project Wiki

## [TARGET] Roadmap

### v2.0 Planned Features

- [ ] Kubernetes integration
- [ ] Multi-cloud support
- [ ] Advanced ML workflows
- [ ] Container orchestration
- [ ] Service mesh integration
- [ ] GraphQL API
- [ ] Mobile application
- [ ] Edge computing support

### Performance Goals

- [ ] Sub-millisecond task scheduling
- [ ] 99.99% uptime SLA
- [ ] Linear scalability to 1000+ nodes
- [ ] AI prediction accuracy >95%

---

**Omega Super Desktop Console** - Powering the future of distributed computing with intelligence and elegance.
