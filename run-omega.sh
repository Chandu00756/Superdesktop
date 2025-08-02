#!/bin/bash

# Omega Super Desktop Console - Simplified Startup Script
# =======================================================

echo "Starting Omega Super Desktop Console..."
echo "======================================="

# Activate virtual environment
source omega_env/bin/activate

# Start the Control Node (Backend)
echo "Starting Control Node (Backend API)..."
cd control_node
python main.py &
CONTROL_PID=$!
cd ..

# Wait for control node to start
sleep 3

# Start Desktop Application
echo "Starting Desktop Application..."
cd control_node/desktop_app
npm start &
DESKTOP_PID=$!
cd ../..

# Create PID file for cleanup
echo "$CONTROL_PID $DESKTOP_PID" > omega.pid

echo ""
echo "Omega Super Desktop Console Started Successfully!"
echo "================================================"
echo ""
echo "Backend API:    http://localhost:8443"
echo "Desktop App:    Launching in Electron..."
echo "API Docs:       http://localhost:8443/docs"
echo "Health Check:   http://localhost:8443/health"
echo ""
echo "To stop: ./stop-omega.sh"
echo "Press Ctrl+C to stop all services"

# Keep script running and handle cleanup
trap 'echo ""; echo "Stopping services..."; kill $CONTROL_PID $DESKTOP_PID 2>/dev/null; rm -f omega.pid; echo "All services stopped"; exit 0' INT

# Wait for processes
wait
