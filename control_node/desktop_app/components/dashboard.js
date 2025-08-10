// Dashboard Component
class DashboardComponent {
    constructor() {
        this.systemMetrics = {
            totalNodes: 8,
            healthyNodes: 7,
            activeSessions: 12,
            totalSessions: 15,
            cpuUsage: 68,
            memoryUsage: 74,
            networkThroughput: '1.2 GB/s',
            networkLatency: '2.1ms'
        };
        
        this.activityItems = [
            {
                type: 'success',
                icon: 'fas fa-desktop',
                title: 'Session Started',
                description: 'User john.doe started session on node-03',
                time: '2 minutes ago'
            },
            {
                type: 'warning',
                icon: 'fas fa-exclamation-triangle',
                title: 'High CPU Usage',
                description: 'Node-07 CPU usage exceeded 90% threshold',
                time: '5 minutes ago'
            },
            {
                type: 'info',
                icon: 'fas fa-server',
                title: 'Node Added',
                description: 'New compute node node-08 joined the cluster',
                time: '12 minutes ago'
            }
        ];
        
        this.init();
    }

    init() {
        this.updateMetrics();
        this.updateActivity();
        this.startPeriodicUpdates();
    }

    render() {
        return `
            <div class="dashboard-grid">
                <!-- System Overview Cards -->
                <div class="overview-cards">
                    ${this.renderOverviewCards()}
                </div>
                
                <!-- Performance Charts -->
                <div class="dashboard-charts">
                    ${this.renderCharts()}
                </div>
                
                <!-- Recent Activity -->
                <div class="activity-feed">
                    ${this.renderActivityFeed()}
                </div>
            </div>
        `;
    }

    renderOverviewCards() {
        return `
            <div class="overview-card">
                <div class="card-header">
                    <i class="fas fa-server"></i>
                    <span>Cluster Status</span>
                </div>
                <div class="card-content">
                    <div class="metric">
                        <span class="metric-value" id="total-nodes">${this.systemMetrics.totalNodes}</span>
                        <span class="metric-label">Active Nodes</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value" id="healthy-nodes">${this.systemMetrics.healthyNodes}</span>
                        <span class="metric-label">Healthy</span>
                    </div>
                </div>
            </div>
            
            <div class="overview-card">
                <div class="card-header">
                    <i class="fas fa-desktop"></i>
                    <span>Active Sessions</span>
                </div>
                <div class="card-content">
                    <div class="metric">
                        <span class="metric-value" id="active-sessions">${this.systemMetrics.activeSessions}</span>
                        <span class="metric-label">Running</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value" id="total-sessions">${this.systemMetrics.totalSessions}</span>
                        <span class="metric-label">Total</span>
                    </div>
                </div>
            </div>
            
            <div class="overview-card">
                <div class="card-header">
                    <i class="fas fa-microchip"></i>
                    <span>Resource Usage</span>
                </div>
                <div class="card-content">
                    <div class="metric">
                        <span class="metric-value" id="cpu-usage-overview">${this.systemMetrics.cpuUsage}%</span>
                        <span class="metric-label">CPU</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value" id="memory-usage-overview">${this.systemMetrics.memoryUsage}%</span>
                        <span class="metric-label">Memory</span>
                    </div>
                </div>
            </div>
            
            <div class="overview-card">
                <div class="card-header">
                    <i class="fas fa-network-wired"></i>
                    <span>Network Status</span>
                </div>
                <div class="card-content">
                    <div class="metric">
                        <span class="metric-value" id="network-throughput">${this.systemMetrics.networkThroughput}</span>
                        <span class="metric-label">Throughput</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value" id="network-latency">${this.systemMetrics.networkLatency}</span>
                        <span class="metric-label">Latency</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderCharts() {
        return `
            <div class="chart-container">
                <div class="chart-header">
                    <h3>System Performance</h3>
                    <div class="chart-controls">
                        <button class="chart-btn active" data-timeframe="1h" onclick="dashboardComponent.updateTimeframe('1h')">1H</button>
                        <button class="chart-btn" data-timeframe="6h" onclick="dashboardComponent.updateTimeframe('6h')">6H</button>
                        <button class="chart-btn" data-timeframe="24h" onclick="dashboardComponent.updateTimeframe('24h')">24H</button>
                        <button class="chart-btn" data-timeframe="7d" onclick="dashboardComponent.updateTimeframe('7d')">7D</button>
                    </div>
                </div>
                <canvas id="performance-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-header">
                    <h3>Resource Distribution</h3>
                </div>
                <canvas id="resource-chart" width="300" height="200"></canvas>
            </div>
        `;
    }

    renderActivityFeed() {
        return `
            <div class="activity-header">
                <h3>Recent Activity</h3>
                <button class="refresh-btn" onclick="dashboardComponent.refreshActivity()" title="Refresh activity feed">
                    <i class="fas fa-sync-alt"></i>
                </button>
            </div>
            <div class="activity-list" id="activity-list">
                ${this.activityItems.map(item => `
                    <div class="activity-item">
                        <div class="activity-icon ${item.type}">
                            <i class="${item.icon}"></i>
                        </div>
                        <div class="activity-content">
                            <div class="activity-title">${item.title}</div>
                            <div class="activity-description">${item.description}</div>
                            <div class="activity-time">${item.time}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    updateMetrics() {
        // Simulate metric updates
        this.systemMetrics.cpuUsage = Math.floor(Math.random() * 40) + 40;
        this.systemMetrics.memoryUsage = Math.floor(Math.random() * 30) + 50;
        
        // Update DOM if elements exist
        const cpuElement = document.getElementById('cpu-usage-overview');
        const memoryElement = document.getElementById('memory-usage-overview');
        
        if (cpuElement) cpuElement.textContent = this.systemMetrics.cpuUsage + '%';
        if (memoryElement) memoryElement.textContent = this.systemMetrics.memoryUsage + '%';
    }

    updateActivity() {
        // Add new activity item occasionally
        if (Math.random() < 0.3) {
            const newActivity = {
                type: ['success', 'info', 'warning'][Math.floor(Math.random() * 3)],
                icon: 'fas fa-info-circle',
                title: 'System Update',
                description: 'System status updated',
                time: 'Just now'
            };
            
            this.activityItems.unshift(newActivity);
            if (this.activityItems.length > 10) {
                this.activityItems.pop();
            }
        }
    }

    updateTimeframe(timeframe) {
        // Update chart timeframe
        document.querySelectorAll('.chart-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-timeframe="${timeframe}"]`).classList.add('active');
        
        console.log(`Updated chart timeframe to: ${timeframe}`);
        // Here you would update the actual chart data
    }

    refreshActivity() {
        console.log('Refreshing activity feed...');
        this.updateActivity();
        
        // Re-render activity feed
        const activityList = document.getElementById('activity-list');
        if (activityList) {
            activityList.innerHTML = this.activityItems.map(item => `
                <div class="activity-item">
                    <div class="activity-icon ${item.type}">
                        <i class="${item.icon}"></i>
                    </div>
                    <div class="activity-content">
                        <div class="activity-title">${item.title}</div>
                        <div class="activity-description">${item.description}</div>
                        <div class="activity-time">${item.time}</div>
                    </div>
                </div>
            `).join('');
        }
    }

    startPeriodicUpdates() {
        setInterval(() => {
            this.updateMetrics();
            this.updateActivity();
        }, 30000); // Update every 30 seconds
    }
}

// Global instance
window.dashboardComponent = new DashboardComponent();
