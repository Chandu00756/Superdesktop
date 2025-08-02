#!/bin/bash

# Omega Super Desktop Console - Simple Startup Script
# ================================================

echo "[LAUNCH] Starting Omega Super Desktop Console (Simple Mode)..."
echo "=========================================================="

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

# Check if Python 3 is available
print_header "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION >= 3.8" | bc -l 2>/dev/null || echo "1") )); then
        print_status "Using Python: $($PYTHON_CMD --version)"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi

# Install Python dependencies if requirements.txt exists
print_header "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install --upgrade pip --quiet
    pip3 install -r requirements.txt --quiet
    print_status "Python dependencies installed"
else
    print_warning "requirements.txt not found"
fi

# Start control node (main server)
print_header "Starting Omega Control Center..."
cd control_node
export OMEGA_ENV=simple
export OMEGA_LOG_LEVEL=INFO
export OMEGA_HOST=0.0.0.0
export OMEGA_PORT=8443
print_status "Control Center starting on http://localhost:8443"
python3 main.py &
CONTROL_PID=$!
cd ..

# Wait for control node to start
print_status "Waiting for control node to be ready..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:8443/health >/dev/null 2>&1; then
        print_status "Control node is ready!"
        break
    fi
done

# Create PID file for cleanup
echo "$CONTROL_PID" > omega-simple.pid

print_header "[CELEBRATE] Omega Super Desktop Console started successfully!"
echo ""
echo "[DASHBOARD] Web Interface:    http://localhost:8443"
echo "[API] API Documentation:    http://localhost:8443/docs"
echo "[HEALTH] Health Check:      http://localhost:8443/health"
echo ""
echo "[CHECKLIST] Active Services:"
echo "   • Control Node:   PID $CONTROL_PID"
echo ""
echo "[STOP] To stop all services: kill $CONTROL_PID && rm -f omega-simple.pid"
echo "[NOTES] Logs: tail -f logs/omega-*.log"
echo ""

# Wait for user input or signal
print_status "Press Ctrl+C to stop all services"
trap 'echo ""; print_header "Stopping Omega Control Center..."; kill $CONTROL_PID 2>/dev/null; rm -f omega-simple.pid; print_status "All services stopped"; exit 0' INT

# Keep script running
wait