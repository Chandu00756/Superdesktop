// Performance Monitoring Component
class PerformanceComponent {
    constructor() {
        this.metrics = {
            system: {
                cpu: { usage: 45, cores: 8, temperature: 65, frequency: 3.2 },
                memory: { total: 32, used: 18, cached: 8, swap: 2 },
                disk: { read: 125.6, write: 89.3, iops: 1240, latency: 2.3 },
                network: { rx: 2.1, tx: 1.8, packets: 45620, errors: 0 }
            },
            processes: [
                { pid: 1234, name: 'omega-core', cpu: 12.5, memory: 512, status: 'running' },
                { pid: 5678, name: 'ai-engine', cpu: 8.3, memory: 1024, status: 'running' },
                { pid: 9012, name: 'network-daemon', cpu: 3.2, memory: 256, status: 'running' },
                { pid: 3456, name: 'storage-manager', cpu: 5.1, memory: 384, status: 'running' }
            ],
            historical: {
                cpu: [45, 42, 48, 44, 46, 43, 45, 47],
                memory: [18, 17, 19, 18, 18, 17, 18, 19],
                network: [2.1, 1.9, 2.3, 2.0, 2.1, 1.8, 2.1, 2.2]
            }
        };
        
        this.activeView = 'overview';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="performance-container">
                <div class="performance-header">
                    <h2><i class="fas fa-chart-line"></i> Performance Monitor</h2>
                    <div class="performance-controls">
                        <select onchange="performanceComponent.setTimeRange(this.value)">
                            <option value="1h">Last Hour</option>
                            <option value="6h">Last 6 Hours</option>
                            <option value="24h">Last 24 Hours</option>
                            <option value="7d">Last Week</option>
                        </select>
                        <button class="btn-primary" onclick="performanceComponent.exportMetrics()">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <button class="btn-secondary" onclick="performanceComponent.refreshMetrics()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="performance-tabs">
                    <button class="tab-btn ${this.activeView === 'overview' ? 'active' : ''}" onclick="performanceComponent.switchView('overview')">
                        <i class="fas fa-chart-area"></i> Overview
                    </button>
                    <button class="tab-btn ${this.activeView === 'processes' ? 'active' : ''}" onclick="performanceComponent.switchView('processes')">
                        <i class="fas fa-tasks"></i> Processes
                    </button>
                    <button class="tab-btn ${this.activeView === 'historical' ? 'active' : ''}" onclick="performanceComponent.switchView('historical')">
                        <i class="fas fa-history"></i> Historical
                    </button>
                    <button class="tab-btn ${this.activeView === 'alerts' ? 'active' : ''}" onclick="performanceComponent.switchView('alerts')">
                        <i class="fas fa-bell"></i> Alerts
                    </button>
                </div>
                
                <div class="performance-content" id="performance-content">
                    ${this.renderViewContent()}
                </div>
            </div>
        `;
    }

    renderViewContent() {
        switch (this.activeView) {
            case 'overview': return this.renderOverview();
            case 'processes': return this.renderProcesses();
            case 'historical': return this.renderHistorical();
            case 'alerts': return this.renderAlerts();
            default: return this.renderOverview();
        }
    }

    renderOverview() {
        const system = this.metrics.system;
        
        return `
            <div class="performance-overview">
                <div class="metrics-grid">
                    <div class="metric-card cpu">
                        <div class="metric-header">
                            <h3><i class="fas fa-microchip"></i> CPU Performance</h3>
                            <span class="metric-value">${system.cpu.usage}%</span>
                        </div>
                        <div class="metric-details">
                            <div class="detail-row">
                                <span>Cores:</span>
                                <span>${system.cpu.cores}</span>
                            </div>
                            <div class="detail-row">
                                <span>Temperature:</span>
                                <span>${system.cpu.temperature}°C</span>
                            </div>
                            <div class="detail-row">
                                <span>Frequency:</span>
                                <span>${system.cpu.frequency}GHz</span>
                            </div>
                        </div>
                        <div class="metric-chart">
                            <div class="progress-ring">
                                <div class="progress-circle" style="--progress: ${system.cpu.usage}"></div>
                                <span class="progress-text">${system.cpu.usage}%</span>
                            </div>
                        </div>
                    </div>

                    <div class="metric-card memory">
                        <div class="metric-header">
                            <h3><i class="fas fa-memory"></i> Memory Usage</h3>
                            <span class="metric-value">${system.memory.used}GB</span>
                        </div>
                        <div class="metric-details">
                            <div class="detail-row">
                                <span>Total:</span>
                                <span>${system.memory.total}GB</span>
                            </div>
                            <div class="detail-row">
                                <span>Cached:</span>
                                <span>${system.memory.cached}GB</span>
                            </div>
                            <div class="detail-row">
                                <span>Swap:</span>
                                <span>${system.memory.swap}GB</span>
                            </div>
                        </div>
                        <div class="metric-chart">
                            <div class="memory-breakdown">
                                <div class="memory-bar used" style="width: ${(system.memory.used / system.memory.total) * 100}%"></div>
                                <div class="memory-bar cached" style="width: ${(system.memory.cached / system.memory.total) * 100}%"></div>
                            </div>
                        </div>
                    </div>

                    <div class="metric-card disk">
                        <div class="metric-header">
                            <h3><i class="fas fa-hdd"></i> Disk I/O</h3>
                            <span class="metric-value">${system.disk.iops}</span>
                        </div>
                        <div class="metric-details">
                            <div class="detail-row">
                                <span>Read:</span>
                                <span>${system.disk.read}MB/s</span>
                            </div>
                            <div class="detail-row">
                                <span>Write:</span>
                                <span>${system.disk.write}MB/s</span>
                            </div>
                            <div class="detail-row">
                                <span>Latency:</span>
                                <span>${system.disk.latency}ms</span>
                            </div>
                        </div>
                        <div class="metric-chart">
                            <div class="disk-activity">
                                <div class="activity-indicator read"></div>
                                <div class="activity-indicator write"></div>
                            </div>
                        </div>
                    </div>

                    <div class="metric-card network">
                        <div class="metric-header">
                            <h3><i class="fas fa-network-wired"></i> Network I/O</h3>
                            <span class="metric-value">${system.network.packets}/s</span>
                        </div>
                        <div class="metric-details">
                            <div class="detail-row">
                                <span>RX:</span>
                                <span>${system.network.rx}MB/s</span>
                            </div>
                            <div class="detail-row">
                                <span>TX:</span>
                                <span>${system.network.tx}MB/s</span>
                            </div>
                            <div class="detail-row">
                                <span>Errors:</span>
                                <span>${system.network.errors}</span>
                            </div>
                        </div>
                        <div class="metric-chart">
                            <div class="network-activity">
                                <div class="traffic-bar rx" style="height: ${(system.network.rx / 10) * 100}%"></div>
                                <div class="traffic-bar tx" style="height: ${(system.network.tx / 10) * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="real-time-charts">
                    <div class="chart-container">
                        <h3>Real-time System Metrics</h3>
                        <canvas id="realtime-chart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    renderProcesses() {
        return `
            <div class="processes-section">
                <div class="processes-toolbar">
                    <div class="search-controls">
                        <input type="text" placeholder="Search processes..." onkeyup="performanceComponent.searchProcesses(this.value)">
                    </div>
                    <div class="sort-controls">
                        <select onchange="performanceComponent.sortProcesses(this.value)">
                            <option value="cpu">Sort by CPU</option>
                            <option value="memory">Sort by Memory</option>
                            <option value="name">Sort by Name</option>
                            <option value="pid">Sort by PID</option>
                        </select>
                    </div>
                </div>
                
                <div class="processes-table">
                    <div class="table-header">
                        <div class="col-pid">PID</div>
                        <div class="col-name">Process Name</div>
                        <div class="col-cpu">CPU %</div>
                        <div class="col-memory">Memory</div>
                        <div class="col-status">Status</div>
                        <div class="col-actions">Actions</div>
                    </div>
                    <div class="table-body">
                        ${this.metrics.processes.map(proc => `
                            <div class="table-row">
                                <div class="col-pid">${proc.pid}</div>
                                <div class="col-name">
                                    <div class="process-info">
                                        <i class="fas fa-cog"></i>
                                        <span>${proc.name}</span>
                                    </div>
                                </div>
                                <div class="col-cpu">
                                    <div class="cpu-usage">
                                        <span>${proc.cpu}%</span>
                                        <div class="cpu-bar">
                                            <div class="cpu-fill" style="width: ${proc.cpu}%"></div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-memory">
                                    <span>${proc.memory}MB</span>
                                </div>
                                <div class="col-status">
                                    <span class="status ${proc.status}">${proc.status}</span>
                                </div>
                                <div class="col-actions">
                                    <button class="btn-sm" onclick="performanceComponent.killProcess(${proc.pid})" title="Kill Process">
                                        <i class="fas fa-times"></i>
                                    </button>
                                    <button class="btn-sm" onclick="performanceComponent.viewProcessDetails(${proc.pid})" title="Details">
                                        <i class="fas fa-info"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderHistorical() {
        return `
            <div class="historical-section">
                <div class="historical-charts">
                    <div class="chart-card">
                        <h3>CPU Usage Trend</h3>
                        <canvas id="cpu-history-chart" width="400" height="200"></canvas>
                    </div>
                    <div class="chart-card">
                        <h3>Memory Usage Trend</h3>
                        <canvas id="memory-history-chart" width="400" height="200"></canvas>
                    </div>
                    <div class="chart-card">
                        <h3>Network Traffic Trend</h3>
                        <canvas id="network-history-chart" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="historical-stats">
                    <h3>Performance Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <label>Average CPU:</label>
                            <span>${Math.round(this.metrics.historical.cpu.reduce((a, b) => a + b) / this.metrics.historical.cpu.length)}%</span>
                        </div>
                        <div class="stat-item">
                            <label>Peak CPU:</label>
                            <span>${Math.max(...this.metrics.historical.cpu)}%</span>
                        </div>
                        <div class="stat-item">
                            <label>Average Memory:</label>
                            <span>${Math.round(this.metrics.historical.memory.reduce((a, b) => a + b) / this.metrics.historical.memory.length)}GB</span>
                        </div>
                        <div class="stat-item">
                            <label>Peak Memory:</label>
                            <span>${Math.max(...this.metrics.historical.memory)}GB</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderAlerts() {
        return `
            <div class="alerts-section">
                <div class="alerts-config">
                    <h3>Performance Alert Configuration</h3>
                    <div class="alert-settings">
                        <div class="alert-item">
                            <label>CPU Usage Threshold:</label>
                            <input type="range" min="50" max="100" value="80" onchange="performanceComponent.updateCpuThreshold(this.value)">
                            <span>80%</span>
                        </div>
                        <div class="alert-item">
                            <label>Memory Usage Threshold:</label>
                            <input type="range" min="50" max="100" value="85" onchange="performanceComponent.updateMemoryThreshold(this.value)">
                            <span>85%</span>
                        </div>
                        <div class="alert-item">
                            <label>Disk I/O Threshold:</label>
                            <input type="range" min="100" max="1000" value="500" onchange="performanceComponent.updateDiskThreshold(this.value)">
                            <span>500 IOPS</span>
                        </div>
                    </div>
                </div>
                
                <div class="active-alerts">
                    <h3>Active Performance Alerts</h3>
                    <div class="alert-list">
                        <div class="alert-item warning">
                            <i class="fas fa-exclamation-triangle"></i>
                            <div class="alert-content">
                                <span class="alert-title">High CPU Usage</span>
                                <span class="alert-description">CPU usage has exceeded 75% for the last 10 minutes</span>
                                <span class="alert-time">5 minutes ago</span>
                            </div>
                        </div>
                        <div class="alert-item info">
                            <i class="fas fa-info-circle"></i>
                            <div class="alert-content">
                                <span class="alert-title">Memory Usage Spike</span>
                                <span class="alert-description">Memory usage increased by 15% in the last hour</span>
                                <span class="alert-time">15 minutes ago</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    switchView(view) {
        this.activeView = view;
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.tab-btn').classList.add('active');
        document.getElementById('performance-content').innerHTML = this.renderViewContent();
    }

    // Action handlers
    setTimeRange(range) { console.log(`Setting time range to: ${range}`); }
    exportMetrics() { console.log('Exporting performance metrics...'); }
    refreshMetrics() { console.log('Refreshing performance metrics...'); }
    searchProcesses(query) { console.log(`Searching processes: ${query}`); }
    sortProcesses(criteria) { console.log(`Sorting processes by: ${criteria}`); }
    killProcess(pid) { console.log(`Killing process: ${pid}`); }
    viewProcessDetails(pid) { console.log(`Viewing details for process: ${pid}`); }
    updateCpuThreshold(value) { console.log(`CPU threshold updated to: ${value}%`); }
    updateMemoryThreshold(value) { console.log(`Memory threshold updated to: ${value}%`); }
    updateDiskThreshold(value) { console.log(`Disk threshold updated to: ${value} IOPS`); }

    startMonitoring() {
        setInterval(() => {
            // Update metrics with simulated data
            this.metrics.system.cpu.usage = Math.floor(Math.random() * 30) + 30;
            this.metrics.system.memory.used = Math.floor(Math.random() * 10) + 15;
            this.metrics.system.disk.iops = Math.floor(Math.random() * 500) + 1000;
            
            if (document.getElementById('performance-content')) {
                document.getElementById('performance-content').innerHTML = this.renderViewContent();
            }
        }, 30000);
    }
}

window.performanceComponent = new PerformanceComponent();
