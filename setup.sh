#!/bin/bash
# Enterprise Setup Script for Superdesktop v2.0
# Installs all enterprise dependencies and configures advanced features

set -e

echo "🚀 Setting up Superdesktop Enterprise Edition..."

# Check if running in the project directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the Superdesktop project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "omega_env" ]; then
    echo "📦 Activating virtual environment..."
    source omega_env/bin/activate
else
    echo "🔧 Creating virtual environment..."
    python3 -m venv omega_env
    source omega_env/bin/activate
fi

# Install core dependencies
echo "📋 Installing core dependencies..."
pip install -r requirements.txt

# Install enterprise dependencies
echo "🏢 Installing enterprise dependencies..."
pip install -r requirements-enterprise.txt

# Install ML dependencies with GPU support if available
echo "🤖 Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing GPU-accelerated packages..."
    pip install tensorflow[gpu] torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No GPU detected, installing CPU-only packages..."
    pip install tensorflow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Create enterprise configuration directory
echo "⚙️  Setting up enterprise configuration..."
mkdir -p config/enterprise
mkdir -p data/ml_models
mkdir -p data/analytics
mkdir -p logs/enterprise

# Generate default enterprise configuration
cat > config/enterprise/config.yaml << EOF
# Superdesktop Enterprise Configuration
enterprise:
  enabled: true
  license_key: "TRIAL_LICENSE_KEY"
  
virtual_desktop:
  max_sessions_per_user: 10
  default_protocol: "vnc"
  gpu_acceleration: true
  snapshot_retention_days: 30
  
cloud:
  providers:
    aws:
      enabled: true
      regions: ["us-east-1", "us-west-2", "eu-west-1"]
    azure:
      enabled: true
      regions: ["eastus", "westus2", "westeurope"]
    gcp:
      enabled: true
      regions: ["us-central1", "us-east1", "europe-west1"]
  
machine_learning:
  auto_training: true
  training_interval_hours: 6
  min_samples_for_training: 1000
  anomaly_detection_threshold: 0.1
  
monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
  log_level: "INFO"
  
security:
  encryption: "AES-256-GCM"
  session_timeout_minutes: 480
  max_login_attempts: 5
  audit_logging: true
EOF

# Set up environment variables
cat > .env.enterprise << EOF
# Superdesktop Enterprise Environment Variables

# Enterprise Features
OMEGA_ENTERPRISE_ENABLED=true
OMEGA_ENTERPRISE_CONFIG=config/enterprise/config.yaml

# Virtual Desktop
OMEGA_VD_MAX_SESSIONS=100
OMEGA_VD_GPU_ENABLED=true
OMEGA_VD_SNAPSHOT_STORAGE=data/vd_snapshots

# Machine Learning
OMEGA_ML_ENABLED=true
OMEGA_ML_DATA_PATH=data/ml_data.db
OMEGA_ML_MODELS_PATH=data/ml_models
OMEGA_ML_AUTO_TRAINING=true

# Cloud Providers (set your credentials)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AZURE_SUBSCRIPTION_ID=your_azure_subscription_id
# GOOGLE_CLOUD_PROJECT=your_gcp_project_id

# Monitoring
OMEGA_PROMETHEUS_ENABLED=true
OMEGA_GRAFANA_ENABLED=true
OMEGA_METRICS_RETENTION_DAYS=30

# Security
OMEGA_AUDIT_LOGGING=true
OMEGA_SESSION_ENCRYPTION=true
OMEGA_BACKUP_ENCRYPTION=true
EOF

# Download and setup Prometheus (if not already installed)
if ! command -v prometheus &> /dev/null; then
    echo "📊 Installing Prometheus..."
    PROM_VERSION="2.47.0"
    wget https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz
    tar xzf prometheus-${PROM_VERSION}.linux-amd64.tar.gz
    sudo mv prometheus-${PROM_VERSION}.linux-amd64/prometheus /usr/local/bin/
    sudo mv prometheus-${PROM_VERSION}.linux-amd64/promtool /usr/local/bin/
    rm -rf prometheus-${PROM_VERSION}*
fi

# Create Prometheus configuration
mkdir -p config/prometheus
cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'superdesktop-backend'
    static_configs:
      - targets: ['localhost:8443']
    metrics_path: '/api/secure/metrics'
    
  - job_name: 'superdesktop-control'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Setup ML model directories
echo "🧠 Setting up ML model storage..."
mkdir -p data/ml_models/{resource_predictor,anomaly_detector,load_forecaster,performance_optimizer,failure_predictor}

# Create enterprise startup script
cat > start-enterprise.sh << 'EOF'
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
EOF

chmod +x start-enterprise.sh

# Run tests to verify installation
echo "🧪 Running enterprise feature tests..."
python -c "
try:
    import tensorflow as tf
    import sklearn
    import pandas as pd
    import docker
    print('✅ All enterprise dependencies installed successfully!')
    print(f'TensorFlow version: {tf.__version__}')
    print(f'Scikit-learn version: {sklearn.__version__}')
    print(f'Pandas version: {pd.__version__}')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

echo ""
echo "🎉 Enterprise setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "   1. Configure cloud provider credentials in .env.enterprise"
echo "   2. Review enterprise configuration in config/enterprise/config.yaml"
echo "   3. Start the enterprise edition with: ./start-enterprise.sh"
echo "   4. Access enterprise endpoints at /api/enterprise/*"
echo ""
echo "📖 Documentation: See README.md for complete API reference"
echo "🔒 License: MIT License (see LICENSE file)"
echo ""
echo "Happy coding! 🚀"