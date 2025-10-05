#!/bin/bash
# Enterprise Startup Script

echo "🚀 Starting Superdesktop Enterprise Edition..."

# Load environment variables
if [ -f ".env.enterprise" ]; then
    export $(cat .env.enterprise | grep -v '^#' | xargs)
fi

# Activate virtual environment
source omega_env/bin/activate

# Start Prometheus (if enabled)
if [ "$OMEGA_PROMETHEUS_ENABLED" = "true" ]; then
    echo "📊 Starting Prometheus..."
    prometheus --config.file=config/prometheus/prometheus.yml --storage.tsdb.path=data/prometheus &
fi

# Start main application
echo "🖥️  Starting Superdesktop services..."
./start-omega.sh

echo "✅ Superdesktop Enterprise Edition is running!"
echo "🌐 Backend API: http://localhost:8443"
echo "🎛️  Control Panel: http://localhost:7777"
echo "🖥️  Desktop App: http://localhost:8081"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 API Docs: http://localhost:8443/docs"
echo ""
echo "Enterprise endpoints available at: /api/enterprise/*"
