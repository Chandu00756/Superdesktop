# SuperDesktop v2.0 - Quick Start Guide

## 🚀 **30-Second Setup**

```bash
# Clone and start
git clone https://github.com/Chandu00756/Superdesktop.git
cd Superdesktop
chmod +x start_core_services_v2.sh
./start_core_services_v2.sh
```

**Done!** System will auto-launch browser with desktop interface.

---

## 📋 **Development Checklist**

- [ ] Python 3.11+ installed (`python3 --version`)
- [ ] Git installed (`git --version`)
- [ ] Repository cloned locally
- [ ] Startup script has execute permissions (`chmod +x *.sh`)

---

## 🎯 **First-Time Testing**

### **1. Start System**

```bash
./start_core_services_v2.sh
```

**Expected Output:**
```
✅ Python environment: omega_env
✅ Dependencies installed successfully
✅ Backend API Server started (8443)
✅ Control Node started (7777)
✅ Storage Node started (8001)
✅ Compute Node started (8002)
✅ Desktop interface available
✅ Opening browser automatically...
```

### **2. Verify Services**

```bash
# Check all services running
curl http://127.0.0.1:8443/api/dashboard/metrics
curl http://127.0.0.1:7777/health

# Expected: JSON responses with status "healthy"
```

### **3. Desktop Interface**

- Browser opens automatically to `control_node/desktop_app/omega-control-center.html`
- Dashboard shows real-time metrics
- Node status displays "Online"

### **4. Stop System**

```bash
./stop-omega.sh
```

**Expected Output:**
```
🛑 Stopping SuperDesktop v2.0...
✅ Backend API Server stopped
✅ Control Node stopped
✅ All services terminated
✅ Cleanup completed
```

---

## 🔧 **Development Workflow**

### **Daily Development**

```bash
# Start work session
./start_core_services_v2.sh

# Code changes...
# (Services auto-restart on file changes)

# End work session
./stop-omega.sh
```

### **Testing Changes**

```bash
# After code modifications
curl http://127.0.0.1:8443/api/dashboard/metrics
curl http://127.0.0.1:7777/nodes

# Check desktop interface
open control_node/desktop_app/omega-control-center.html
```

---

## 🌐 **Key URLs**

| Service | URL | Purpose |
|---------|-----|---------|
| **Desktop Interface** | `control_node/desktop_app/omega-control-center.html` | Main UI |
| **Backend API** | `http://127.0.0.1:8443` | REST endpoints |
| **Control Node** | `http://127.0.0.1:7777` | Orchestration |
| **Metrics** | `http://127.0.0.1:8000/metrics` | Monitoring |

---

## 🐛 **Common Issues & Solutions**

### **Port Already in Use**

```bash
# Check what's using ports
lsof -i :8443
lsof -i :7777

# Solution
./stop-omega.sh
./start_core_services_v2.sh
```

### **Python Environment Issues**

```bash
# Recreate environment
rm -rf omega_env
./start_core_services_v2.sh
```

### **Permission Denied**

```bash
# Fix script permissions
chmod +x *.sh
```

### **Dependencies Failed**

```bash
# Manual dependency install
source omega_env/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

---

## 📊 **System Status Check**

### **All Services Running?**

```bash
ps aux | grep python | grep -E "(api_server|main\.py|advanced_)"
```

### **Ports Listening?**

```bash
netstat -an | grep -E ":8443|:7777|:8001|:8002"
```

### **Logs Available?**

```bash
tail -f logs/*.log  # If logs directory exists
```

---

## 🚀 **Ready for Production?**

### **Production Deployment**

```bash
# Set production environment
export OMEGA_TLS_ENABLED="true"
export OMEGA_LOG_LEVEL="INFO"
export OMEGA_CLUSTER_NAME="production"

# Start with production settings
./start_core_services_v2.sh
```

### **Docker Deployment**

```bash
docker-compose up -d
```

---

## 📞 **Need Help?**

- **📖 Full Documentation**: `README.md`
- **🏗️ Architecture**: `docs/architecture/`
- **📧 Support**: chandu@portalvii.com
- **🐙 Issues**: GitHub Issues

---

**SuperDesktop v2.0** - *Professional distributed computing made simple*

**Quick Start Time**: ~30 seconds | **Contact**: chandu@portalvii.com
