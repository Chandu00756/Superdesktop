# Omega Super Desktop Console Documentation

## Overview

Omega Super Desktop Console is a distributed computing system that aggregates multiple PC resources into a unified super desktop experience. It features a hub-and-spoke architecture with mesh fallback, advanced latency mitigation, and modular resource management.

## Components

- **Control Node**: Master coordinator, user interface hub
- **Compute Nodes**: CPU/GPU/Memory agents
- **Storage Nodes**: Distributed storage management
- **Network**: RDMA, custom UDP, latency mitigation
- **Middleware**: Resource manager, memory manager, latency compensator
- **Security**: TLS, encryption, RBAC
- **Common**: Shared models/utilities
- **Desktop App**: Electron-based GUI

## Setup

1. Install Python 3.13+ and Node.js 18+
2. Create and activate Python virtual environment
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install Node dependencies for desktop app:

   ```bash
   cd control_node/desktop_app
   npm install
   ```

5. Generate TLS certificates for all nodes (see `security/certs/README.md`)
6. Start control node backend:

   ```bash
   python control_node/main.py
   ```

7. Start compute/storage nodes:

   ```bash
   python compute_node/main.py
   python storage_node/main.py
   ```

8. Launch desktop app:

   ```bash
   npm start
   ```

## Initial Prototype Deployment

- Use managed switches and hardware timestamping for low latency
- Configure RBAC and JWT secrets in environment variables
- Monitor system health via desktop app and backend APIs

## Roadmap

See `README.md` for architecture and roadmap details.
