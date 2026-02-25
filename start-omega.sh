#!/bin/bash

# =============================================================================
# SUPERDESKTOP v2.0 - UNIFIED SYSTEM STARTUP SCRIPT
# =============================================================================
# Single standalone file to start the complete SuperDesktop ecosystem
# No external dependencies except Python 3.11+ and Node.js 18+
# Contact: chandu@portalvii.com
# =============================================================================

echo "SUPERDESKTOP v2.0 - UNIFIED SYSTEM STARTUP"
echo "=============================================="
echo "Starting complete distributed desktop environment"
echo "Features:"
echo "  ✓ Backend API Server with encrypted communication"
echo "  ✓ Control Node with AI-powered orchestration"
echo "  ✓ Desktop App with real-time data integration"
echo "  ✓ Storage/Compute/Network node management"
echo "  ✓ Fault-tolerant multi-master architecture"
echo "  ✓ Real-time performance monitoring"
echo "=============================================="
echo "Contact: chandu@portalvii.com"
echo ""

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# System requirements check
echo "🔍 Checking system requirements..."

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    echo "  ✓ Python $PYTHON_VERSION detected"
    
    # Check if version is 3.11 or higher
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        echo "  ⚠️  Python 3.11+ recommended for optimal performance"
    fi
else
    echo "  ❌ Python 3 not found. Please install Python 3.11+ and try again."
    exit 1
fi

# Check Node.js version
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    echo "  ✓ Node.js $NODE_VERSION detected"
else
    echo "  ⚠️  Node.js not found - Desktop app will use basic features only"
fi

# Check npm
if command -v npm >/dev/null 2>&1; then
    NPM_VERSION=$(npm --version)
    echo "  ✓ npm $NPM_VERSION detected"
else
    echo "  ⚠️  npm not found - Node.js dependencies will be skipped"
fi

echo ""

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================
echo "🐍 Setting up Python virtual environment..."

if [ ! -d "omega_env" ]; then
    echo "  📦 Creating virtual environment..."
    python3 -m venv omega_env
    if [ $? -ne 0 ]; then
        echo "  ❌ Failed to create virtual environment"
        exit 1
    fi
fi

echo "  🔄 Activating virtual environment..."
source omega_env/bin/activate

# Upgrade pip
echo "  📦 Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1

# Merge all requirements from different modules
echo "  📦 Installing Python dependencies..."

# Core requirements
pip install \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.5.0" \
    "aiohttp>=3.9.0" \
    "websockets>=11.0.0" \
    "cryptography>=41.0.0" \
    "psutil>=5.9.0" \
    "numpy>=1.25.0" \
    "requests>=2.31.0" \
    "pyjwt>=2.8.0" \
    "sqlalchemy>=2.0.0" \
    "python-multipart>=0.0.6" \
    "prometheus-client>=0.19.0" \
    "pyyaml>=6.0.1" \
    "rich>=13.7.0" \
    "asyncpg>=0.28.0" \
    "torch>=2.0.0" \
    "docker>=6.1.0" \
    "click>=8.1.7" >/dev/null 2>&1

# Optional ML dependencies
pip install "scikit-learn>=1.3.0" >/dev/null 2>&1 || echo "  ⚠️  ML libraries not installed (optional)"

# Optional monitoring dependencies  
pip install "redis>=5.0.0" >/dev/null 2>&1 || echo "  ⚠️  Redis not installed (will use fallback)"

echo "  ✅ Python environment ready"
echo ""

# =============================================================================
# NODE.JS DEPENDENCIES
# =============================================================================
if command -v npm >/dev/null 2>&1; then
    echo "� Setting up Node.js dependencies..."
    
    cd control_node/desktop_app
    
    if [ ! -d "node_modules" ]; then
        echo "  📦 Installing Node.js dependencies..."
        npm install >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✅ Node.js dependencies installed"
        else
            echo "  ⚠️  Some Node.js dependencies failed to install"
        fi
    else
        echo "  ✅ Node.js dependencies already installed"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
fi

# =============================================================================
# SYSTEM INITIALIZATION
# =============================================================================
echo "🔧 Initializing system components..."

# Create necessary directories
mkdir -p logs data/{postgresql,redis,object_storage} security/certs

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR"
export OMEGA_NODE_ID="omega-control-$(hostname)-$(date +%s)"
export OMEGA_LOG_LEVEL="INFO"
export OMEGA_CORE_SERVICES_ENABLED="true"
export OMEGA_CLUSTER_NAME="superdesktop-cluster-v2"
# Default to HTTP for local development to avoid browser self-signed TLS issues.
export OMEGA_ENABLE_TLS="0"
# Virtual Desktop image selection and pull policy
# Set OMEGA_DEFAULT_VD_IMAGE to override the default container image, e.g. "dorowu/ubuntu-desktop-lxde-vnc"
# OMEGA_VD_PULL_POLICY: Always|IfNotPresent|Never (default IfNotPresent)
export OMEGA_VD_PULL_POLICY="IfNotPresent"

# Generate temporary SSL certificates if not present
if [ ! -f "security/certs/control_node.crt" ]; then
    echo "  🔐 Generating temporary SSL certificates..."
    mkdir -p security/certs
    
    # Create self-signed certificate for development
    openssl req -x509 -newkey rsa:2048 -keyout security/certs/control_node.key \
        -out security/certs/control_node.crt -days 365 -nodes \
        -subj "/C=US/ST=CA/L=San Francisco/O=SuperDesktop/CN=localhost" \
        >/dev/null 2>&1 || echo "  ⚠️  SSL certificate generation failed (optional)"
fi

echo "  ✅ System initialization complete"
echo ""

# =============================================================================
# SERVICE STARTUP
# =============================================================================
echo " Starting SuperDesktop services..."

# Array to track service PIDs
declare -a SERVICE_PIDS

# Function to start service and track PID
start_service() {
    local service_name="$1"
    local service_command="$2"
    local service_dir="$3"
    
    echo "  🔄 Starting $service_name..."
    
    if [ -n "$service_dir" ]; then
        cd "$service_dir"
    fi
    
    eval "$service_command" &
    local pid=$!
    SERVICE_PIDS+=($pid)
    
    echo "    ↳ PID: $pid"
    
    # Give service time to start
    sleep 2
    
    # Check if service is still running
    if ps -p $pid > /dev/null 2>&1; then
        echo "    ✅ $service_name started successfully"
    else
        echo "    ⚠️  $service_name may have failed to start"
    fi
    
    cd "$SCRIPT_DIR"
}

# 1. Start Backend API Server (TLS-aware, proper cwd)
start_service "Backend API Server" \
    "python start_backend.py" \
    "backend"

# 2. Start Frontend HTTP Server (serve from desktop_app dir, bind to 127.0.0.1)
start_service "Frontend HTTP Server" \
    "cd control_node/desktop_app && python -m http.server 8081 --bind 127.0.0.1" \
    ""

# 3. Start Control Node
start_service "Control Node Manager" \
    "python control_node/main.py" \
    ""

# 4. Start Storage Node
start_service "Storage Node" \
    "python storage_node/main.py" \
    ""

# 5. Start Compute Node
start_service "Compute Node" \
    "python compute_node/main.py" \
    ""

# 6. Start Additional Services
start_service "Session Daemon" \
    "python session-daemon/main.py" \
    ""

start_service "Omega Orchestrator" \
    "python omega-orchestrator/main.py" \
    ""

start_service "Memory Fabric" \
    "python memory-fabric/main.py" \
    ""

start_service "Predictor Service" \
    "python predictor-service/main.py" \
    ""

start_service "Render Router" \
    "python render-router/main.py" \
    ""

# Wait for services to fully initialize
echo ""
echo "⏳ Waiting for services to initialize..."
sleep 5

# =============================================================================
# HEALTH CHECKS
# =============================================================================
echo ""
echo "🩺 Performing health checks..."

# Check Backend API (prefer HTTP in dev unless TLS enabled)
if curl -s "http://127.0.0.1:8443/docs" >/dev/null 2>&1 || curl -sk "https://127.0.0.1:8443/docs" >/dev/null 2>&1; then
    echo "  ✅ Backend API Server: 127.0.0.1:8443"
else
    echo "  ⚠️  Backend API Server: Starting up..."
fi

# Check Frontend HTTP Server
if curl -s "http://127.0.0.1:8081/omega-new.html" >/dev/null 2>&1; then
    echo "  ✅ Frontend HTTP Server: http://127.0.0.1:8081"
else
    echo "  ⚠️  Frontend HTTP Server: Starting up..."
fi

# Check Control Node
if curl -s "http://127.0.0.1:7777/health" >/dev/null 2>&1; then
    echo "  ✅ Control Node: http://127.0.0.1:7777"
else
    echo "  ⚠️  Control Node: Starting up..."
fi

# Check desktop app files
if [ -f "control_node/desktop_app/omega-new.html" ]; then
    echo "  ✅ Desktop App: control_node/desktop_app/omega-new.html"
else
    echo "  ❌ Desktop App: Missing"
fi

echo ""

# =============================================================================
# SYSTEM STATUS AND INFORMATION
# =============================================================================
echo "=============================================="
echo "🎯 SUPERDESKTOP v2.0 SYSTEM STATUS"
echo "=============================================="
echo ""
echo "📊 Service Status:"
echo "  • Backend API Server:    http://127.0.0.1:8443"
echo "  • Frontend HTTP Server:  http://127.0.0.1:8081"
echo "  • Control Node:          http://127.0.0.1:7777"
echo "  • Desktop App:           http://127.0.0.1:8081/omega-new.html"
echo "  • Metrics Endpoint:      http://127.0.0.1:8000/metrics"
echo ""
echo "🔧 Service PIDs:"
for pid in "${SERVICE_PIDS[@]}"; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "  • PID $pid: Running"
    else
        echo "  • PID $pid: Stopped"
    fi
done
echo ""
echo "📁 Key Files:"
echo "  • Main Interface:        control_node/desktop_app/omega-new.html"
echo "  • Backend API:           backend/api_server.py"
echo "  • Control Node:          control_node/main.py"
echo "  • Configuration:         omega_v2_config.ini"
echo "  • Logs Directory:        logs/"
echo ""
echo "� Contact Information:"
echo "  • Email:                 chandu@portalvii.com"
echo "  • Project:               SuperDesktop v2.0"
echo "  • Repository:            https://github.com/Chandu00756/Superdesktop"
echo ""
echo "🎮 Usage Instructions:"
echo "  1. Open desktop app in browser: http://127.0.0.1:8081/omega-new.html"
echo "  2. Access API documentation: http://127.0.0.1:8443/docs"
echo "  3. View system metrics: http://127.0.0.1:8000/metrics"
echo "  4. Monitor logs: tail -f logs/*.log"
echo ""
echo "🛑 To stop all services:"
echo "   ./stop-omega.sh"
echo ""
echo "=============================================="

# =============================================================================
# KEEP SERVICES RUNNING
# =============================================================================

# Create PID file for stop script
echo "${SERVICE_PIDS[@]}" > omega.pid

# Function to handle cleanup on script exit
cleanup() {
    echo ""
    echo "🛑 Shutting down SuperDesktop services..."
    
    for pid in "${SERVICE_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  🔄 Stopping PID $pid..."
            kill $pid 2>/dev/null
        fi
    done
    
    # Wait for graceful shutdown
    sleep 3
    
    # Force kill if necessary
    for pid in "${SERVICE_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  � Force stopping PID $pid..."
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # Clean up PID file
    rm -f omega.pid
    
    echo "  ✅ All services stopped"
    echo "=============================================="
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM

# Auto-open desktop app in browser (optional)
if command -v open >/dev/null 2>&1; then
    # macOS
    echo "🌐 Opening desktop app in browser..."
    open "http://127.0.0.1:8081/omega-new.html" 2>/dev/null &
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    echo "🌐 Opening desktop app in browser..."
    xdg-open "http://127.0.0.1:8081/omega-new.html" 2>/dev/null &
elif command -v start >/dev/null 2>&1; then
    # Windows
    echo "🌐 Opening desktop app in browser..."
    start "http://127.0.0.1:8081/omega-new.html" 2>/dev/null &
fi

echo "✅ SuperDesktop v2.0 is running successfully!"
echo "   Press Ctrl+C to stop all services"
echo ""

# Keep script running and monitor services
while true; do
    sleep 30
    
    # Check if any service has died
    for i in "${!SERVICE_PIDS[@]}"; do
        pid="${SERVICE_PIDS[$i]}"
        if ! ps -p $pid > /dev/null 2>&1; then
            echo "⚠️  Service PID $pid has stopped unexpectedly"
            unset SERVICE_PIDS[$i]
        fi
    done
    
    # Update PID file
    echo "${SERVICE_PIDS[@]}" > omega.pid
done
