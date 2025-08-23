#!/usr/bin/env bash

# Robust SUPERDESKTOP stop script
# - Searches for PIDs in omega.pid
# - Kills any lingering processes matching known patterns
# - Releases known ports
# - Works on macOS and Linux

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

FORCE=${1:-}

echo "� SUPERDESKTOP - stopping services"

# Helper to attempt gentle stop then force
_terminate_pid() {
    local pid=$1
    if [ -z "$pid" ]; then return; fi
    if ! ps -p "$pid" > /dev/null 2>&1; then return; fi
    echo "  • Terminating PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        if [ "$FORCE" = "--force" ]; then
            echo "    ↳ Killing PID $pid (force)"
            kill -9 "$pid" 2>/dev/null || true
        else
            echo "    ↳ Still running: PID $pid (use --force to kill)"
        fi
    fi
}

# 1) Kill PIDs listed in omega.pid (if present)
if [ -f "omega.pid" ]; then
    echo "Reading omega.pid"
    mapfile -t PIDS < <(awk '{print $1}' omega.pid || true)
    for p in "${PIDS[@]:-}"; do
        _terminate_pid "$p"
    done
    rm -f omega.pid || true
fi

# 2) Kill processes by known patterns
PATTERNS=(
    "backend/api_server.py"
    "api_server.py"
    "control_node/main.py"
    "storage_node/main.py"
    "compute_node/main.py"
    "session-daemon/main.py"
    "omega-orchestrator/main.py"
    "memory-fabric/main.py"
    "predictor-service/main.py"
    "render-router/main.py"
    "uvicorn:"
)

echo "Searching processes by pattern..."
for pat in "${PATTERNS[@]}"; do
    while IFS= read -r line; do
        pid=$(echo "$line" | awk '{print $1}')
        cmd=$(echo "$line" | cut -d' ' -f2-)
        if [ -n "$pid" ]; then
            echo "  Found $pid -> $cmd"
            _terminate_pid "$pid"
        fi
    done < <(pgrep -af "$pat" 2>/dev/null || true)
done

# 3) Free known ports (if they look like our processes)
PORTS=(8443 8081 7777 7778 8000 8001)
echo "Releasing ports if used by SuperDesktop..."
for port in "${PORTS[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        for pid in $pids; do
            if [ -z "$pid" ]; then continue; fi
            cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
            echo "  Port $port used by PID $pid -> $cmd"
            if echo "$cmd" | grep -E "(python|uvicorn|node|omega|superdesktop)" >/dev/null 2>&1; then
                _terminate_pid "$pid"
            else
                echo "    ↳ skipping unrelated process"
            fi
        done
    fi
done

# 4) Final sweep: any lingering processes under workspace path
echo "Final sweep for any processes under $SCRIPT_DIR"
while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $1}')
    cmd=$(echo "$line" | cut -d' ' -f2-)
    if [ -n "$pid" ]; then
        echo "  Sweep: $pid -> $cmd"
        _terminate_pid "$pid"
    fi
done < <(pgrep -af "$SCRIPT_DIR" 2>/dev/null || true)

echo "Cleanup temp files"
rm -f omega.pid || true
find logs -type f -name "*.tmp" -delete 2>/dev/null || true

echo "Shutdown complete"
exit 0
                process_info=$(ps -p "$pid" -o pid,command --no-headers 2>/dev/null)
