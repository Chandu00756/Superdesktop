#!/bin/bash

# =============================================================================
# SUPERDESKTOP v2.0 - UNIFIED SYSTEM STOP SCRIPT
# =============================================================================
# Gracefully stops all SuperDesktop services
# Contact: chandu@portalvii.com
# =============================================================================

echo "🛑 SUPERDESKTOP v2.0 - SYSTEM SHUTDOWN"
echo "=============================================="
echo "Gracefully stopping all services..."
echo "Contact: chandu@portalvii.com"
echo ""

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# STOP SERVICES BY PID FILE
# =============================================================================

if [ -f "omega.pid" ]; then
    echo "📋 Reading service PIDs from omega.pid..."
    
    # Read PIDs from file
    SERVICE_PIDS=($(cat omega.pid))
    
    if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
        echo "  ⚠️  No PIDs found in omega.pid"
    else
        echo "  📍 Found ${#SERVICE_PIDS[@]} service PID(s)"
        
        # Step 1: Graceful shutdown (SIGTERM)
        echo ""
        echo "🔄 Attempting graceful shutdown..."
        
        for pid in "${SERVICE_PIDS[@]}"; do
            if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
                echo "  🔄 Stopping PID $pid (graceful)..."
                kill -TERM "$pid" 2>/dev/null
            fi
        done
        
        # Wait for graceful shutdown
        echo "  ⏳ Waiting 10 seconds for graceful shutdown..."
        sleep 10
        
        # Step 2: Force shutdown (SIGKILL) for remaining processes
        echo ""
        echo "💀 Force stopping remaining processes..."
        
        remaining_pids=()
        for pid in "${SERVICE_PIDS[@]}"; do
            if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
                echo "  💀 Force stopping PID $pid..."
                kill -9 "$pid" 2>/dev/null
                remaining_pids+=($pid)
            fi
        done
        
        if [ ${#remaining_pids[@]} -eq 0 ]; then
            echo "  ✅ All services stopped gracefully"
        else
            echo "  ⚠️  ${#remaining_pids[@]} process(es) required force termination"
        fi
    fi
    
    # Remove PID file
    rm -f omega.pid
    echo "  🗑️  Removed omega.pid file"
else
    echo "📋 No omega.pid file found, searching for processes..."
fi

echo ""

# =============================================================================
# STOP SERVICES BY PROCESS NAME
# =============================================================================

echo "🔍 Searching for SuperDesktop processes by name..."

# List of process patterns to search for
PROCESS_PATTERNS=(
    "api_server.py"
    "control_node.*main.py"
    "storage_node.*main.py"
    "compute_node.*main.py"
    "session-daemon.*main.py"
    "omega-orchestrator.*main.py"
    "memory-fabric.*main.py"
    "predictor-service.*main.py"
    "render-router.*main.py"
    "uvicorn.*backend/api_server"
)

stopped_processes=0

for pattern in "${PROCESS_PATTERNS[@]}"; do
    # Find processes matching pattern
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo "  🎯 Found processes matching '$pattern':"
        for pid in $pids; do
            if ps -p "$pid" > /dev/null 2>&1; then
                process_info=$(ps -p "$pid" -o pid,command --no-headers 2>/dev/null)
                echo "    • PID $pid: $(echo "$process_info" | cut -c 1-80)..."
                
                # Graceful stop first
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                
                # Force stop if still running
                if ps -p "$pid" > /dev/null 2>&1; then
                    kill -9 "$pid" 2>/dev/null
                    echo "    ↳ Force stopped PID $pid"
                else
                    echo "    ↳ Gracefully stopped PID $pid"
                fi
                
                ((stopped_processes++))
            fi
        done
    fi
done

if [ $stopped_processes -eq 0 ]; then
    echo "  ✅ No SuperDesktop processes found running"
else
    echo "  ✅ Stopped $stopped_processes process(es)"
fi

echo ""

# =============================================================================
# STOP SERVICES BY PORT
# =============================================================================

echo "🔌 Checking for services on known ports..."

# List of ports used by SuperDesktop
PORTS=(8443 8081 7777 7778 8000 8001 8002 8003 8004 8005)

killed_by_port=0

for port in "${PORTS[@]}"; do
    # Find process using the port
    if command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti:"$port" 2>/dev/null)
        
        if [ -n "$pid" ]; then
            process_info=$(ps -p "$pid" -o command --no-headers 2>/dev/null)
            echo "  🔌 Port $port in use by PID $pid: $(echo "$process_info" | cut -c 1-60)..."
            
            # Only kill if it looks like a SuperDesktop process
            if echo "$process_info" | grep -q -E "(python|uvicorn|fastapi|omega|superdesktop)" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
                sleep 1
                
                if ps -p "$pid" > /dev/null 2>&1; then
                    kill -9 "$pid" 2>/dev/null
                    echo "    ↳ Force stopped process on port $port"
                else
                    echo "    ↳ Gracefully stopped process on port $port"
                fi
                
                ((killed_by_port++))
            else
                echo "    ↳ Skipping non-SuperDesktop process"
            fi
        fi
    elif command -v netstat >/dev/null 2>&1; then
        # Fallback using netstat
        pid=$(netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
        if [ -n "$pid" ] && [ "$pid" != "-" ]; then
            echo "  🔌 Port $port in use by PID $pid"
            kill -TERM "$pid" 2>/dev/null
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null
            fi
            ((killed_by_port++))
        fi
    fi
done

if [ $killed_by_port -eq 0 ]; then
    echo "  ✅ No services found on SuperDesktop ports"
else
    echo "  ✅ Stopped $killed_by_port service(s) by port"
fi

echo ""

# =============================================================================
# CLEANUP
# =============================================================================

echo "🧹 Performing cleanup..."

# Remove temporary files
temp_files=(
    "omega.pid"
    "logs/*.tmp"
    "data/*.lock"
    "*.log.lock"
)

cleaned_files=0
for pattern in "${temp_files[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "  🗑️  Removed $file"
            ((cleaned_files++))
        fi
    done
done

# Clean up empty log files
if [ -d "logs" ]; then
    find logs -name "*.log" -size 0 -delete 2>/dev/null
fi

if [ $cleaned_files -eq 0 ]; then
    echo "  ✅ No temporary files to clean"
else
    echo "  ✅ Cleaned $cleaned_files temporary file(s)"
fi

echo ""

# =============================================================================
# VERIFICATION
# =============================================================================

echo "🔍 Verifying shutdown..."

# Check if any SuperDesktop processes are still running
remaining_processes=$(pgrep -f "omega|superdesktop|api_server|control_node|storage_node|compute_node" 2>/dev/null | wc -l)

if [ "$remaining_processes" -eq 0 ]; then
    echo "  ✅ All SuperDesktop processes stopped"
else
    echo "  ⚠️  $remaining_processes SuperDesktop process(es) may still be running"
    echo "  📋 Remaining processes:"
    pgrep -f "omega|superdesktop|api_server|control_node|storage_node|compute_node" -l 2>/dev/null | head -5
fi

# Check ports
active_ports=0
for port in "${PORTS[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
        if lsof -i:"$port" >/dev/null 2>&1; then
            ((active_ports++))
        fi
    fi
done

if [ $active_ports -eq 0 ]; then
    echo "  ✅ All SuperDesktop ports released"
else
    echo "  ⚠️  $active_ports port(s) may still be in use"
fi

echo ""

# =============================================================================
# FINAL STATUS
# =============================================================================

echo "=============================================="
echo "🏁 SUPERDESKTOP v2.0 SHUTDOWN COMPLETE"
echo "=============================================="
echo ""

if [ "$remaining_processes" -eq 0 ] && [ $active_ports -eq 0 ]; then
    echo "✅ SYSTEM FULLY STOPPED"
    echo "   All services and ports have been released"
else
    echo "⚠️  PARTIAL SHUTDOWN"
    echo "   Some processes or ports may still be active"
    echo "   You may need to manually stop remaining processes"
fi

echo ""
echo "📊 Shutdown Summary:"
echo "  • Processes stopped:     $((stopped_processes + killed_by_port))"
echo "  • Files cleaned:         $cleaned_files"
echo "  • Remaining processes:   $remaining_processes"
echo "  • Active ports:          $active_ports"
echo ""
echo "📧 Contact: chandu@portalvii.com"
echo "🔄 To restart: ./start_core_services_v2.sh"
echo ""
echo "=============================================="
