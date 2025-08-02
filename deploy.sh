#!/bin/bash

#################################################################
# Omega Super Desktop Console - Initial Prototype Deployment Script
# 
# This script automates the deployment of the complete Omega
# distributed computing platform with all required services.
#################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="omega-super-desktop"
ENVIRONMENT=${ENVIRONMENT:-prototype}

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "################################################################"
    echo "#                                                              #"
    echo "#           Omega Super Desktop Console                       #"
    echo "#           Initial Prototype Deployment Script                 #"
    echo "#                                                              #"
    echo "#           Version: 1.0.0                                    #"
    echo "#           Environment: $ENVIRONMENT                              #"
    echo "#                                                              #"
    echo "################################################################"
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check minimum Docker version
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    REQUIRED_VERSION="20.10"
    
    if [[ $(echo -e "$REQUIRED_VERSION\n$DOCKER_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
        print_error "Docker version $DOCKER_VERSION is too old. Minimum required: $REQUIRED_VERSION"
        exit 1
    fi
    
    # Check available memory
    AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$AVAILABLE_MEMORY" -lt 8 ]; then
        print_warning "Available memory ($AVAILABLE_MEMORY GB) may be insufficient for optimal performance. Recommended: 16GB+"
    fi
    
    # Check available disk space
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_DISK" -lt 50 ]; then
        print_warning "Available disk space ($AVAILABLE_DISK GB) may be insufficient. Recommended: 100GB+"
    fi
    
    print_success "System requirements check completed"
}

# Create required directories
create_directories() {
    print_status "Creating required directories..."
    
    mkdir -p {logs,data,config,monitoring,nginx,scripts}
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    mkdir -p nginx/ssl
    mkdir -p elk/{elasticsearch/data,logstash/{config,pipeline}}
    mkdir -p pgadmin
    mkdir -p tests/performance
    
    # Set permissions
    chmod 755 logs data config monitoring
    chmod 700 nginx/ssl
    
    print_success "Directories created successfully"
}

# Generate SSL certificates
generate_ssl_certificates() {
    print_status "Generating SSL certificates..."
    
    if [ ! -f "nginx/ssl/omega.crt" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/omega.key \
            -out nginx/ssl/omega.crt \
            -subj "/C=US/ST=CA/L=San Francisco/O=Omega/OU=Engineering/CN=omega-desktop.local"
        
        print_success "SSL certificates generated"
    else
        print_status "SSL certificates already exist"
    fi
}

# Create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Nginx configuration
    cat > nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;
    
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    # Upstream servers
    upstream control_node {
        server control-node:8000;
    }
    
    upstream orchestrator {
        server orchestrator:8001;
    }
    
    upstream session_daemon {
        server session-daemon:8003;
    }
    
    upstream predictor_service {
        server predictor-service:8004;
    }
    
    upstream render_router {
        server render-router:8005;
    }
    
    upstream memory_fabric {
        server memory-fabric:8006;
    }
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }
    
    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name omega-desktop.local localhost;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/omega.crt;
        ssl_certificate_key /etc/nginx/ssl/omega.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
        
        # API routes
        location /api/control/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://control_node/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/orchestrator/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://orchestrator/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/sessions/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://session_daemon/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/predictor/ {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://predictor_service/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/render/ {
            limit_req zone=api burst=50 nodelay;
            proxy_pass http://render_router/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/memory/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://memory_fabric/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket support
        location /ws/ {
            proxy_pass http://control_node;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Static files and desktop app
        location / {
            proxy_pass http://control_node/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check endpoint
        location /health {
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF

    # Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files: []

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'omega-control-node'
    static_configs:
      - targets: ['control-node:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'omega-orchestrator'
    static_configs:
      - targets: ['orchestrator:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'omega-session-daemon'
    static_configs:
      - targets: ['session-daemon:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'omega-predictor-service'
    static_configs:
      - targets: ['predictor-service:8002']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'omega-render-router'
    static_configs:
      - targets: ['render-router:8003']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'omega-memory-fabric'
    static_configs:
      - targets: ['memory-fabric:8004']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

    # Grafana datasources
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # PgAdmin servers configuration
    cat > pgadmin/servers.json << 'EOF'
{
    "Servers": {
        "1": {
            "Name": "Omega PostgreSQL",
            "Group": "Servers",
            "Host": "postgres",
            "Port": 5432,
            "MaintenanceDB": "omega_sessions",
            "Username": "omega",
            "SSLMode": "prefer",
            "SSLCompression": 0,
            "Timeout": 10,
            "UseSSHTunnel": 0,
            "TunnelPort": "22",
            "TunnelAuthentication": 0
        }
    }
}
EOF

    print_success "Configuration files created"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    # Wait for PostgreSQL to be ready
    print_status "Waiting for PostgreSQL to be ready..."
    sleep 30
    
    # Run database initialization
    docker-compose exec -T postgres python /docker-entrypoint-initdb.d/01-init.py || {
        print_warning "Database initialization script failed, but continuing..."
    }
    
    print_success "Database initialization completed"
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    local services=("control-node" "orchestrator" "session-daemon" "predictor-service" "render-router" "memory-fabric")
    local healthy_services=0
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up (healthy)"; then
            print_success "$service is healthy"
            ((healthy_services++))
        else
            print_warning "$service is not healthy"
        fi
    done
    
    print_status "Health check completed: $healthy_services/${#services[@]} services healthy"
}

# Deploy the system
deploy() {
    print_status "Starting Omega Super Desktop Console deployment..."
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Build custom images
    print_status "Building custom images..."
    docker-compose build
    
    # Start infrastructure services first
    print_status "Starting infrastructure services..."
    docker-compose up -d postgres redis etcd
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure services..."
    sleep 45
    
    # Initialize database
    init_database
    
    # Start core services
    print_status "Starting core Omega services..."
    docker-compose up -d control-node orchestrator session-daemon predictor-service render-router memory-fabric
    
    # Wait for core services
    print_status "Waiting for core services to start..."
    sleep 30
    
    # Start compute and storage nodes
    print_status "Starting compute and storage nodes..."
    docker-compose up -d compute-node-1 compute-node-2 storage-node-1
    
    # Start monitoring services
    print_status "Starting monitoring services..."
    docker-compose up -d prometheus grafana
    
    # Start load balancer
    print_status "Starting load balancer..."
    docker-compose up -d nginx
    
    # Start log management (optional)
    if [ "$ENVIRONMENT" = "production" ]; then
        print_status "Starting log management services..."
        docker-compose up -d elasticsearch logstash kibana
    fi
    
    print_success "Deployment completed!"
}

# Show service URLs
show_urls() {
    echo -e "\n${GREEN}=== Omega Super Desktop Console URLs ===${NC}"
    echo -e "${BLUE}Main Application:${NC}     https://localhost"
    echo -e "${BLUE}API Documentation:${NC}    https://localhost/docs"
    echo -e "${BLUE}Grafana Monitoring:${NC}   http://localhost:3000 (admin/omega_admin_2025)"
    echo -e "${BLUE}Prometheus Metrics:${NC}   http://localhost:9090"
    echo -e "${BLUE}PgAdmin Database:${NC}     http://localhost:5050 (admin@omega-desktop.io/omega_admin_2025)"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "${BLUE}Kibana Logs:${NC}          http://localhost:5601"
        echo -e "${BLUE}Elasticsearch:${NC}        http://localhost:9200"
    fi
    
    echo -e "\n${YELLOW}Note: It may take a few minutes for all services to be fully ready.${NC}"
}

# Main deployment function
main() {
    print_banner
    
    case "${1:-deploy}" in
        "deploy")
            check_requirements
            create_directories
            generate_ssl_certificates
            create_configs
            deploy
            echo ""
            print_status "Waiting for services to stabilize..."
            sleep 60
            check_health
            show_urls
            ;;
        "start")
            print_status "Starting existing deployment..."
            docker-compose start
            sleep 30
            check_health
            show_urls
            ;;
        "stop")
            print_status "Stopping all services..."
            docker-compose stop
            print_success "All services stopped"
            ;;
        "restart")
            print_status "Restarting all services..."
            docker-compose restart
            sleep 30
            check_health
            ;;
        "status")
            print_status "Service status:"
            docker-compose ps
            echo ""
            check_health
            ;;
        "logs")
            service=${2:-}
            if [ -n "$service" ]; then
                docker-compose logs -f "$service"
            else
                docker-compose logs -f
            fi
            ;;
        "clean")
            print_warning "This will remove all containers, volumes, and data. Are you sure? (y/N)"
            read -r confirmation
            if [[ $confirmation =~ ^[Yy]$ ]]; then
                print_status "Cleaning up deployment..."
                docker-compose down -v --remove-orphans
                docker system prune -f
                print_success "Cleanup completed"
            else
                print_status "Cleanup cancelled"
            fi
            ;;
        "help"|*)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Full deployment (default)"
            echo "  start     - Start stopped services"
            echo "  stop      - Stop all services"
            echo "  restart   - Restart all services"
            echo "  status    - Show service status"
            echo "  logs      - Show logs (optionally for specific service)"
            echo "  clean     - Remove all containers and data"
            echo "  help      - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 deploy"
            echo "  $0 logs control-node"
            echo "  $0 status"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
