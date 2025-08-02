#!/bin/bash

# Omega Control Center - Stop Script
# ==================================

echo "Stopping Omega Super Desktop Console..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[OMEGA]${NC} $1"
}

# Read PIDs from file
if [ -f "omega.pid" ]; then
    PIDS=$(cat omega.pid)
    print_status "Stopping services: $PIDS"
    
    for PID in $PIDS; do
        if [ ! -z "$PID" ] && kill -0 $PID 2>/dev/null; then
            print_status "Stopping process $PID"
            kill $PID
            sleep 1
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID 2>/dev/null
            fi
        fi
    done
    
    rm -f omega.pid
    print_status "PID file removed"
else
    print_status "No PID file found, attempting to stop by process name"
    
    # Kill by process name as fallback
    pkill -f "python.*main.py" 2>/dev/null
    pkill -f "electron" 2>/dev/null
fi

# Clean up any remaining processes
print_status "Cleaning up remaining processes..."
sleep 1

print_header "[COMPLETE] Omega Control Center stopped"
