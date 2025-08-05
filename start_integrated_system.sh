#!/bin/bash

# SuperDesktop v2.0 - Integrated System Startup Script
# Starts backend API server and opens frontend with real data integration

echo "🚀 Starting SuperDesktop v2.0 with Real Backend Integration..."

# Check if Python environment exists
if [ ! -d "omega_env" ]; then
    echo "❌ Python environment not found. Creating virtual environment..."
    python3 -m venv omega_env
    source omega_env/bin/activate
    pip install -r backend/requirements.txt
else
    echo "✅ Activating Python environment..."
    source omega_env/bin/activate
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Start the backend API server in background
echo "🔧 Starting backend API server..."
cd backend
python api_server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Check if backend is running
if curl -s "http://127.0.0.1:8443/api/dashboard/metrics" > /dev/null; then
    echo "✅ Backend API server running successfully"
else
    echo "⚠️  Backend may still be starting up..."
fi

echo "🌐 Opening SuperDesktop Control Center..."
echo ""
echo "=== SYSTEM STATUS ==="
echo "Backend API: http://127.0.0.1:8443"
echo "Frontend: control_node/desktop_app/omega-control-center.html"
echo "Email Contact: chandu@portalvii.com"
echo "Data Source: REAL BACKEND (NO SIMULATION)"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "To stop backend: kill $BACKEND_PID"
echo ""
echo "🎯 SuperDesktop v2.0 is now running with REAL data integration!"

# Open the frontend in default browser
open "control_node/desktop_app/omega-control-center.html"

# Keep script running
echo "Press Ctrl+C to stop the system"
wait $BACKEND_PID
