// Resources Management Component
class ResourcesComponent {
    constructor() {
        this.resources = {
            cpu: {
                total: 32,
                used: 18,
                available: 14,
                cores: [
                    { id: 0, usage: 45 }, { id: 1, usage: 32 }, { id: 2, usage: 67 }, { id: 3, usage: 23 },
                    { id: 4, usage: 78 }, { id: 5, usage: 34 }, { id: 6, usage: 56 }, { id: 7, usage: 41 }
                ]
            },
            memory: {
                total: 128, // GB
                used: 78,
                available: 50,
                breakdown: {
                    system: 12,
                    applications: 45,
                    cache: 21,
                    free: 50
                }
            },
            storage: {
                total: 2048, // GB
                used: 1234,
                available: 814,
                drives: [
                    { name: '/dev/sda1', mount: '/', size: 512, used: 298, type: 'SSD' },
                    { name: '/dev/sdb1', mount: '/data', size: 1024, used: 567, type: 'HDD' },
                    { name: '/dev/sdc1', mount: '/backup', size: 512, used: 369, type: 'SSD' }
                ]
            },
            network: {
                interfaces: [
                    { name: 'eth0', ip: '192.168.1.10', speed: '1Gbps', status: 'up', tx: 2.3, rx: 1.8 },
                    { name: 'eth1', ip: '10.0.0.5', speed: '10Gbps', status: 'up', tx: 8.7, rx: 6.2 },
                    { name: 'wlan0', ip: '192.168.0.100', speed: '867Mbps', status: 'up', tx: 0.5, rx: 0.8 }
                ],
                totalTraffic: {
                    inbound: 125.6, // GB
                    outbound: 98.4
                }
            }
        };
        
        this.activeTab = 'overview';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="resources-container">
                <div class="resources-header">
                    <h2><i class="fas fa-chart-pie"></i> Resource Management</h2>
                    <div class="resources-controls">
                        <button class="btn-primary" onclick="resourcesComponent.exportReport()">
                            <i class="fas fa-download"></i> Export Report
                        </button>
                        <button class="btn-secondary" onclick="resourcesComponent.refreshData()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="resources-tabs">
                    <button class="tab-btn ${this.activeTab === 'overview' ? 'active' : ''}" onclick="resourcesComponent.switchTab('overview')">
                        <i class="fas fa-chart-area"></i> Overview
                    </button>
                    <button class="tab-btn ${this.activeTab === 'cpu' ? 'active' : ''}" onclick="resourcesComponent.switchTab('cpu')">
                        <i class="fas fa-microchip"></i> CPU
                    </button>
                    <button class="tab-btn ${this.activeTab === 'memory' ? 'active' : ''}" onclick="resourcesComponent.switchTab('memory')">
                        <i class="fas fa-memory"></i> Memory
                    </button>
                    <button class="tab-btn ${this.activeTab === 'storage' ? 'active' : ''}" onclick="resourcesComponent.switchTab('storage')">
                        <i class="fas fa-hdd"></i> Storage
                    </button>
                    <button class="tab-btn ${this.activeTab === 'network' ? 'active' : ''}" onclick="resourcesComponent.switchTab('network')">
                        <i class="fas fa-network-wired"></i> Network
                    </button>
                </div>
                
                <div class="resources-content" id="resources-content">
                    ${this.renderTabContent()}
                </div>
            </div>
        `;
    }

    renderTabContent() {
        switch (this.activeTab) {
            case 'overview':
                return this.renderOverview();
            case 'cpu':
                return this.renderCPUDetails();
            case 'memory':
                return this.renderMemoryDetails();
            case 'storage':
                return this.renderStorageDetails();
            case 'network':
                return this.renderNetworkDetails();
            default:
                return this.renderOverview();
        }
    }

    renderOverview() {
        const cpuUsage = Math.round((this.resources.cpu.used / this.resources.cpu.total) * 100);
        const memoryUsage = Math.round((this.resources.memory.used / this.resources.memory.total) * 100);
        const storageUsage = Math.round((this.resources.storage.used / this.resources.storage.total) * 100);

        return `
            <div class="overview-grid">
                <div class="resource-card">
                    <div class="card-header">
                        <h3><i class="fas fa-microchip"></i> CPU Usage</h3>
                        <span class="usage-percent">${cpuUsage}%</span>
                    </div>
                    <div class="resource-chart">
                        <div class="circular-progress" data-percentage="${cpuUsage}">
                            <div class="progress-circle">
                                <div class="progress-fill" style="background: conic-gradient(#4CAF50 ${cpuUsage * 3.6}deg, #e0e0e0 0)"></div>
                                <div class="progress-text">${cpuUsage}%</div>
                            </div>
                        </div>
                    </div>
                    <div class="resource-details">
                        <span>${this.resources.cpu.used} of ${this.resources.cpu.total} cores in use</span>
                    </div>
                </div>

                <div class="resource-card">
                    <div class="card-header">
                        <h3><i class="fas fa-memory"></i> Memory Usage</h3>
                        <span class="usage-percent">${memoryUsage}%</span>
                    </div>
                    <div class="resource-chart">
                        <div class="circular-progress" data-percentage="${memoryUsage}">
                            <div class="progress-circle">
                                <div class="progress-fill" style="background: conic-gradient(#2196F3 ${memoryUsage * 3.6}deg, #e0e0e0 0)"></div>
                                <div class="progress-text">${memoryUsage}%</div>
                            </div>
                        </div>
                    </div>
                    <div class="resource-details">
                        <span>${this.resources.memory.used}GB of ${this.resources.memory.total}GB used</span>
                    </div>
                </div>

                <div class="resource-card">
                    <div class="card-header">
                        <h3><i class="fas fa-hdd"></i> Storage Usage</h3>
                        <span class="usage-percent">${storageUsage}%</span>
                    </div>
                    <div class="resource-chart">
                        <div class="circular-progress" data-percentage="${storageUsage}">
                            <div class="progress-circle">
                                <div class="progress-fill" style="background: conic-gradient(#FF9800 ${storageUsage * 3.6}deg, #e0e0e0 0)"></div>
                                <div class="progress-text">${storageUsage}%</div>
                            </div>
                        </div>
                    </div>
                    <div class="resource-details">
                        <span>${this.resources.storage.used}GB of ${this.resources.storage.total}GB used</span>
                    </div>
                </div>

                <div class="resource-card">
                    <div class="card-header">
                        <h3><i class="fas fa-network-wired"></i> Network Activity</h3>
                        <span class="network-status">Active</span>
                    </div>
                    <div class="network-stats">
                        <div class="stat-item">
                            <label>Inbound</label>
                            <span>${this.resources.network.totalTraffic.inbound}GB</span>
                        </div>
                        <div class="stat-item">
                            <label>Outbound</label>
                            <span>${this.resources.network.totalTraffic.outbound}GB</span>
                        </div>
                        <div class="stat-item">
                            <label>Interfaces</label>
                            <span>${this.resources.network.interfaces.filter(i => i.status === 'up').length}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="recent-activity">
                <h3>Resource Alerts & Notifications</h3>
                <div class="activity-list">
                    <div class="activity-item warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        <div class="activity-content">
                            <span class="activity-text">High memory usage detected on Storage Node 03</span>
                            <span class="activity-time">2 minutes ago</span>
                        </div>
                    </div>
                    <div class="activity-item info">
                        <i class="fas fa-info-circle"></i>
                        <div class="activity-content">
                            <span class="activity-text">CPU load balanced across available cores</span>
                            <span class="activity-time">15 minutes ago</span>
                        </div>
                    </div>
                    <div class="activity-item success">
                        <i class="fas fa-check-circle"></i>
                        <div class="activity-content">
                            <span class="activity-text">Network interface eth1 performance optimized</span>
                            <span class="activity-time">1 hour ago</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderCPUDetails() {
        return `
            <div class="cpu-details">
                <div class="cpu-overview">
                    <h3>CPU Core Usage</h3>
                    <div class="cores-grid">
                        ${this.resources.cpu.cores.map(core => `
                            <div class="core-item">
                                <div class="core-header">
                                    <label>Core ${core.id}</label>
                                    <span>${core.usage}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${core.usage}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="cpu-stats">
                    <h3>CPU Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <label>Total Cores</label>
                            <span class="stat-value">${this.resources.cpu.total}</span>
                        </div>
                        <div class="stat-card">
                            <label>Cores in Use</label>
                            <span class="stat-value">${this.resources.cpu.used}</span>
                        </div>
                        <div class="stat-card">
                            <label>Available</label>
                            <span class="stat-value">${this.resources.cpu.available}</span>
                        </div>
                        <div class="stat-card">
                            <label>Average Load</label>
                            <span class="stat-value">${Math.round(this.resources.cpu.cores.reduce((sum, core) => sum + core.usage, 0) / this.resources.cpu.cores.length)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderMemoryDetails() {
        const breakdown = this.resources.memory.breakdown;
        
        return `
            <div class="memory-details">
                <div class="memory-breakdown">
                    <h3>Memory Breakdown</h3>
                    <div class="breakdown-chart">
                        <div class="breakdown-item">
                            <div class="breakdown-bar system" style="width: ${(breakdown.system / this.resources.memory.total) * 100}%"></div>
                            <label>System (${breakdown.system}GB)</label>
                        </div>
                        <div class="breakdown-item">
                            <div class="breakdown-bar applications" style="width: ${(breakdown.applications / this.resources.memory.total) * 100}%"></div>
                            <label>Applications (${breakdown.applications}GB)</label>
                        </div>
                        <div class="breakdown-item">
                            <div class="breakdown-bar cache" style="width: ${(breakdown.cache / this.resources.memory.total) * 100}%"></div>
                            <label>Cache (${breakdown.cache}GB)</label>
                        </div>
                        <div class="breakdown-item">
                            <div class="breakdown-bar free" style="width: ${(breakdown.free / this.resources.memory.total) * 100}%"></div>
                            <label>Free (${breakdown.free}GB)</label>
                        </div>
                    </div>
                </div>
                
                <div class="memory-stats">
                    <h3>Memory Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <label>Total Memory</label>
                            <span class="stat-value">${this.resources.memory.total}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Used Memory</label>
                            <span class="stat-value">${this.resources.memory.used}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Available</label>
                            <span class="stat-value">${this.resources.memory.available}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Usage %</label>
                            <span class="stat-value">${Math.round((this.resources.memory.used / this.resources.memory.total) * 100)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderStorageDetails() {
        return `
            <div class="storage-details">
                <div class="drives-list">
                    <h3>Storage Drives</h3>
                    <div class="drives-grid">
                        ${this.resources.storage.drives.map(drive => {
                            const usage = Math.round((drive.used / drive.size) * 100);
                            return `
                                <div class="drive-card">
                                    <div class="drive-header">
                                        <div class="drive-info">
                                            <h4>${drive.name}</h4>
                                            <span class="drive-mount">${drive.mount}</span>
                                        </div>
                                        <span class="drive-type ${drive.type.toLowerCase()}">${drive.type}</span>
                                    </div>
                                    <div class="drive-usage">
                                        <div class="usage-bar">
                                            <div class="usage-fill" style="width: ${usage}%"></div>
                                        </div>
                                        <div class="usage-text">
                                            <span>${drive.used}GB / ${drive.size}GB</span>
                                            <span class="usage-percent">${usage}%</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
                
                <div class="storage-stats">
                    <h3>Storage Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <label>Total Storage</label>
                            <span class="stat-value">${this.resources.storage.total}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Used Storage</label>
                            <span class="stat-value">${this.resources.storage.used}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Available</label>
                            <span class="stat-value">${this.resources.storage.available}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Drive Count</label>
                            <span class="stat-value">${this.resources.storage.drives.length}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderNetworkDetails() {
        return `
            <div class="network-details">
                <div class="interfaces-list">
                    <h3>Network Interfaces</h3>
                    <div class="interfaces-grid">
                        ${this.resources.network.interfaces.map(iface => `
                            <div class="interface-card">
                                <div class="interface-header">
                                    <div class="interface-info">
                                        <h4>${iface.name}</h4>
                                        <span class="interface-ip">${iface.ip}</span>
                                    </div>
                                    <span class="interface-status ${iface.status}">${iface.status}</span>
                                </div>
                                <div class="interface-stats">
                                    <div class="stat-row">
                                        <label>Speed</label>
                                        <span>${iface.speed}</span>
                                    </div>
                                    <div class="stat-row">
                                        <label>TX</label>
                                        <span>${iface.tx} GB/s</span>
                                    </div>
                                    <div class="stat-row">
                                        <label>RX</label>
                                        <span>${iface.rx} GB/s</span>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="network-stats">
                    <h3>Network Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <label>Active Interfaces</label>
                            <span class="stat-value">${this.resources.network.interfaces.filter(i => i.status === 'up').length}</span>
                        </div>
                        <div class="stat-card">
                            <label>Total Inbound</label>
                            <span class="stat-value">${this.resources.network.totalTraffic.inbound}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Total Outbound</label>
                            <span class="stat-value">${this.resources.network.totalTraffic.outbound}GB</span>
                        </div>
                        <div class="stat-card">
                            <label>Total Traffic</label>
                            <span class="stat-value">${this.resources.network.totalTraffic.inbound + this.resources.network.totalTraffic.outbound}GB</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    switchTab(tab) {
        this.activeTab = tab;
        
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.tab-btn').classList.add('active');
        
        // Update content
        document.getElementById('resources-content').innerHTML = this.renderTabContent();
    }

    exportReport() {
        console.log('Exporting resource report...');
        // Implementation for exporting resource report
    }

    refreshData() {
        console.log('Refreshing resource data...');
        this.updateResourceMetrics();
        document.getElementById('resources-content').innerHTML = this.renderTabContent();
    }

    updateResourceMetrics() {
        // Simulate dynamic resource updates
        this.resources.cpu.cores.forEach(core => {
            core.usage = Math.floor(Math.random() * 80) + 10;
        });
        
        this.resources.memory.used = Math.floor(Math.random() * 50) + 50;
        this.resources.memory.available = this.resources.memory.total - this.resources.memory.used;
        
        this.resources.network.interfaces.forEach(iface => {
            iface.tx = (Math.random() * 10).toFixed(1);
            iface.rx = (Math.random() * 8).toFixed(1);
        });
    }

    startMonitoring() {
        setInterval(() => {
            this.updateResourceMetrics();
            if (document.getElementById('resources-content')) {
                document.getElementById('resources-content').innerHTML = this.renderTabContent();
            }
        }, 15000);
    }
}

// Global instance
window.resourcesComponent = new ResourcesComponent();
