#!/bin/bash

# Omega Control Center - Initial Prototype Startup Script
# ================================================

echo "[LAUNCH] Starting Omega Super Desktop Console..."
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[OMEGA]${NC} $1"
}

# Check if Python 3.13+ is available
print_header "Checking Python version..."
if command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION >= 3.10" | bc -l) )); then
        PYTHON_CMD="python3"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.10+ not found"
    exit 1
fi

print_status "Using Python: $($PYTHON_CMD --version)"

# Activate virtual environment FIRST
print_header "Activating virtual environment..."
if [ -d "omega_env" ]; then
    source omega_env/bin/activate
    print_status "Activated Python virtual environment (omega_env)"
else
    print_error "Virtual environment omega_env not found. Run: python3 -m venv omega_env && source omega_env/bin/activate"
    exit 1
fi

# Check if Node.js is available for Electron
print_header "Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Using Node.js: $NODE_VERSION"
else
    print_warning "Node.js not found - Electron desktop app will not be available"
fi

# Install Python dependencies in venv
print_header "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    print_status "Python dependencies installed"
else
    print_warning "requirements.txt not found"
fi

# Note: Database schema will be auto-created by SQLAlchemy
print_header "Database will be auto-initialized by the application..."

# Check if PostgreSQL is running
print_header "Checking PostgreSQL..."
if pg_isready -h localhost -p 5432 &> /dev/null; then
    print_status "PostgreSQL is running"
else
    print_warning "PostgreSQL not running - using SQLite fallback"
fi

# Check if Redis is running
print_header "Checking Redis..."
if redis-cli ping &> /dev/null; then
    print_status "Redis is running"
else
    print_warning "Redis not running - using in-memory cache"
fi


# Check/generate CA certificate for SSL
if [ ! -f "security/certs/ca.crt" ]; then
    print_warning "CA certificate not found, generating new CA cert..."
    mkdir -p security/certs
    openssl req -x509 -newkey rsa:4096 -keyout security/certs/ca.key -out security/certs/ca.crt -days 365 -nodes -subj "/CN=OmegaCA"
    print_status "CA certificate generated at security/certs/ca.crt"
else
    print_status "CA certificate found."
fi

# Start control node (main server) FIRST
print_header "Starting Omega Control Center..."
cd control_node
export OMEGA_ENV=prototype
export OMEGA_LOG_LEVEL=INFO
export OMEGA_HOST=0.0.0.0
export OMEGA_PORT=8443
export OMEGA_SESSIONS_ENABLED=true
export OMEGA_WEBSOCKET_ENABLED=true
print_status "Control Center starting on https://localhost:8443"
print_status "Sessions API enabled with real-time updates"
python main.py &
CONTROL_PID=$!
cd ..

# Wait for control node to start and listen on port 8443
print_status "Waiting for control node to be ready..."
for i in {1..10}; do
    sleep 1
    if curl -k -s https://localhost:8443/health >/dev/null 2>&1; then
        print_status "Control node is ready!"
        break
    fi
done

# Start desktop application if Node.js is available
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    print_header "Starting Omega Desktop Application..."
    cd control_node/desktop_app
    if [ ! -d "node_modules" ]; then
        print_status "Installing Electron dependencies..."
        npm install --silent
    fi
    print_status "Launching desktop application..."
    npm start &
    ELECTRON_PID=$!
    cd ../..
else
    print_warning "Electron desktop app not available - access via web browser at https://localhost:8443"
fi


# Create PID file for cleanup
echo "$CONTROL_PID ${ELECTRON_PID:-}" > omega.pid

print_header "[CELEBRATE] Omega Super Desktop Console started successfully!"
echo ""
echo "[DASHBOARD] Web Interface:    https://localhost:8443"
echo "[DESKTOP] Desktop App:      Electron Sessions Tab Ready"
echo "[API] API Documentation:    https://localhost:8443/docs"
echo "[SESSIONS] Sessions API:    https://localhost:8443/api/test/sessions"
echo "[WEBSOCKET] Real-time:      ws://localhost:8443/ws/sessions"
echo "[HEALTH] Health Check:      https://localhost:8443/health"
echo ""
echo "[CHECKLIST] Active Services:"
echo "   • Control Node:   PID $CONTROL_PID"
if [ ! -z "${ELECTRON_PID:-}" ]; then
    echo "   • Desktop App:    PID $ELECTRON_PID"
fi
echo ""
echo "[STOP] To stop all services: ./stop-omega.sh"
echo "[NOTES] Logs: tail -f logs/omega-*.log"
echo ""

# Wait for user input or signal
print_status "Press Ctrl+C to stop all services"
trap 'echo ""; print_header "Stopping Omega Control Center..."; kill $CONTROL_PID ${ELECTRON_PID:-} 2>/dev/null; rm -f omega.pid; print_status "All services stopped"; exit 0' INT

# Keep script running
wait
