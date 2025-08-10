// Network Management Component
class NetworkComponent {
    constructor() {
        this.networkData = {
            overview: {
                totalBandwidth: 10000, // Mbps
                usedBandwidth: 3500,
                activeConnections: 1247,
                packetsPerSecond: 45620,
                latency: 12.5,
                uptime: 99.97
            },
            interfaces: [
                {
                    id: 'eth0',
                    name: 'Primary Ethernet',
                    type: 'ethernet',
                    status: 'up',
                    speed: '10Gbps',
                    duplex: 'full',
                    mtu: 1500,
                    ip: '192.168.1.10',
                    mask: '255.255.255.0',
                    gateway: '192.168.1.1',
                    rx: { bytes: 2.5e9, packets: 1.8e6, errors: 0, dropped: 0 },
                    tx: { bytes: 1.8e9, packets: 1.2e6, errors: 0, dropped: 0 }
                },
                {
                    id: 'eth1',
                    name: 'Secondary Ethernet',
                    type: 'ethernet', 
                    status: 'up',
                    speed: '1Gbps',
                    duplex: 'full',
                    mtu: 1500,
                    ip: '10.0.0.5',
                    mask: '255.255.0.0',
                    gateway: '10.0.0.1',
                    rx: { bytes: 1.2e9, packets: 850000, errors: 0, dropped: 0 },
                    tx: { bytes: 980e6, packets: 720000, errors: 0, dropped: 0 }
                }
            ],
            routes: [
                { destination: '0.0.0.0/0', gateway: '192.168.1.1', interface: 'eth0', metric: 100 },
                { destination: '192.168.1.0/24', gateway: '0.0.0.0', interface: 'eth0', metric: 0 },
                { destination: '10.0.0.0/16', gateway: '0.0.0.0', interface: 'eth1', metric: 0 }
            ],
            connections: [
                { local: '192.168.1.10:22', remote: '192.168.1.50:54321', state: 'ESTABLISHED', protocol: 'TCP' },
                { local: '192.168.1.10:80', remote: '203.0.113.1:45678', state: 'ESTABLISHED', protocol: 'TCP' },
                { local: '192.168.1.10:443', remote: '198.51.100.5:39876', state: 'TIME_WAIT', protocol: 'TCP' }
            ]
        };
        
        this.activeTab = 'overview';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="network-container">
                <div class="network-header">
                    <h2><i class="fas fa-network-wired"></i> Network Management</h2>
                    <div class="network-controls">
                        <button class="btn-primary" onclick="networkComponent.runDiagnostics()">
                            <i class="fas fa-stethoscope"></i> Diagnostics
                        </button>
                        <button class="btn-secondary" onclick="networkComponent.refreshData()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="network-tabs">
                    <button class="tab-btn ${this.activeTab === 'overview' ? 'active' : ''}" onclick="networkComponent.switchTab('overview')">
                        <i class="fas fa-chart-area"></i> Overview
                    </button>
                    <button class="tab-btn ${this.activeTab === 'interfaces' ? 'active' : ''}" onclick="networkComponent.switchTab('interfaces')">
                        <i class="fas fa-ethernet"></i> Interfaces
                    </button>
                    <button class="tab-btn ${this.activeTab === 'routing' ? 'active' : ''}" onclick="networkComponent.switchTab('routing')">
                        <i class="fas fa-route"></i> Routing
                    </button>
                    <button class="tab-btn ${this.activeTab === 'connections' ? 'active' : ''}" onclick="networkComponent.switchTab('connections')">
                        <i class="fas fa-plug"></i> Connections
                    </button>
                </div>
                
                <div class="network-content" id="network-content">
                    ${this.renderTabContent()}
                </div>
            </div>
        `;
    }

    renderTabContent() {
        switch (this.activeTab) {
            case 'overview': return this.renderOverview();
            case 'interfaces': return this.renderInterfaces();
            case 'routing': return this.renderRouting();
            case 'connections': return this.renderConnections();
            default: return this.renderOverview();
        }
    }

    renderOverview() {
        const bandwidth = this.networkData.overview;
        const usage = Math.round((bandwidth.usedBandwidth / bandwidth.totalBandwidth) * 100);

        return `
            <div class="network-overview">
                <div class="overview-stats">
                    <div class="stat-card">
                        <i class="fas fa-tachometer-alt"></i>
                        <div class="stat-info">
                            <span class="stat-value">${bandwidth.usedBandwidth}/${bandwidth.totalBandwidth} Mbps</span>
                            <span class="stat-label">Bandwidth Usage</span>
                        </div>
                        <div class="stat-progress">
                            <div class="progress-fill" style="width: ${usage}%"></div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <i class="fas fa-link"></i>
                        <div class="stat-info">
                            <span class="stat-value">${bandwidth.activeConnections.toLocaleString()}</span>
                            <span class="stat-label">Active Connections</span>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <i class="fas fa-exchange-alt"></i>
                        <div class="stat-info">
                            <span class="stat-value">${bandwidth.packetsPerSecond.toLocaleString()}/s</span>
                            <span class="stat-label">Packets</span>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <i class="fas fa-clock"></i>
                        <div class="stat-info">
                            <span class="stat-value">${bandwidth.latency}ms</span>
                            <span class="stat-label">Average Latency</span>
                        </div>
                    </div>
                </div>
                
                <div class="traffic-chart">
                    <h3>Network Traffic</h3>
                    <div id="traffic-chart-container">
                        <canvas id="traffic-chart" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="interface-summary">
                    <h3>Interface Status</h3>
                    <div class="interface-grid">
                        ${this.networkData.interfaces.map(iface => `
                            <div class="interface-card">
                                <div class="interface-header">
                                    <h4>${iface.name}</h4>
                                    <span class="status ${iface.status}">${iface.status}</span>
                                </div>
                                <div class="interface-stats">
                                    <div class="stat-row">
                                        <label>Speed:</label>
                                        <span>${iface.speed}</span>
                                    </div>
                                    <div class="stat-row">
                                        <label>RX:</label>
                                        <span>${this.formatBytes(iface.rx.bytes)}</span>
                                    </div>
                                    <div class="stat-row">
                                        <label>TX:</label>
                                        <span>${this.formatBytes(iface.tx.bytes)}</span>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderInterfaces() {
        return `
            <div class="interfaces-section">
                <div class="interfaces-toolbar">
                    <button class="btn-primary" onclick="networkComponent.addInterface()">
                        <i class="fas fa-plus"></i> Add Interface
                    </button>
                    <button class="btn-secondary" onclick="networkComponent.configureInterfaces()">
                        <i class="fas fa-cog"></i> Configure
                    </button>
                </div>
                
                <div class="interfaces-table">
                    <div class="table-header">
                        <div class="col-name">Interface</div>
                        <div class="col-status">Status</div>
                        <div class="col-config">Configuration</div>
                        <div class="col-traffic">Traffic</div>
                        <div class="col-errors">Errors</div>
                        <div class="col-actions">Actions</div>
                    </div>
                    <div class="table-body">
                        ${this.networkData.interfaces.map(iface => `
                            <div class="table-row">
                                <div class="col-name">
                                    <div class="interface-info">
                                        <strong>${iface.name}</strong>
                                        <span class="interface-id">${iface.id}</span>
                                    </div>
                                </div>
                                <div class="col-status">
                                    <span class="status-badge ${iface.status}">${iface.status}</span>
                                    <span class="speed">${iface.speed}</span>
                                </div>
                                <div class="col-config">
                                    <div class="config-info">
                                        <div>IP: ${iface.ip}</div>
                                        <div>Gateway: ${iface.gateway}</div>
                                        <div>MTU: ${iface.mtu}</div>
                                    </div>
                                </div>
                                <div class="col-traffic">
                                    <div class="traffic-info">
                                        <div class="traffic-item">
                                            <label>RX:</label>
                                            <span>${this.formatBytes(iface.rx.bytes)}</span>
                                        </div>
                                        <div class="traffic-item">
                                            <label>TX:</label>
                                            <span>${this.formatBytes(iface.tx.bytes)}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-errors">
                                    <div class="error-info">
                                        <div>RX Errors: ${iface.rx.errors}</div>
                                        <div>TX Errors: ${iface.tx.errors}</div>
                                        <div>Dropped: ${iface.rx.dropped + iface.tx.dropped}</div>
                                    </div>
                                </div>
                                <div class="col-actions">
                                    <button class="btn-sm" onclick="networkComponent.configureInterface('${iface.id}')" title="Configure">
                                        <i class="fas fa-cog"></i>
                                    </button>
                                    <button class="btn-sm" onclick="networkComponent.restartInterface('${iface.id}')" title="Restart">
                                        <i class="fas fa-redo"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderRouting() {
        return `
            <div class="routing-section">
                <div class="routing-toolbar">
                    <button class="btn-primary" onclick="networkComponent.addRoute()">
                        <i class="fas fa-plus"></i> Add Route
                    </button>
                    <button class="btn-secondary" onclick="networkComponent.refreshRoutes()">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
                
                <div class="routing-table">
                    <div class="table-header">
                        <div class="col-destination">Destination</div>
                        <div class="col-gateway">Gateway</div>
                        <div class="col-interface">Interface</div>
                        <div class="col-metric">Metric</div>
                        <div class="col-actions">Actions</div>
                    </div>
                    <div class="table-body">
                        ${this.networkData.routes.map((route, index) => `
                            <div class="table-row">
                                <div class="col-destination">
                                    <span class="destination">${route.destination}</span>
                                </div>
                                <div class="col-gateway">
                                    <span class="gateway">${route.gateway === '0.0.0.0' ? 'Direct' : route.gateway}</span>
                                </div>
                                <div class="col-interface">
                                    <span class="interface">${route.interface}</span>
                                </div>
                                <div class="col-metric">
                                    <span class="metric">${route.metric}</span>
                                </div>
                                <div class="col-actions">
                                    <button class="btn-sm" onclick="networkComponent.editRoute(${index})" title="Edit">
                                        <i class="fas fa-edit"></i>
                                    </button>
                                    <button class="btn-sm btn-danger" onclick="networkComponent.deleteRoute(${index})" title="Delete">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderConnections() {
        return `
            <div class="connections-section">
                <div class="connections-toolbar">
                    <button class="btn-secondary" onclick="networkComponent.refreshConnections()">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                    <button class="btn-secondary" onclick="networkComponent.exportConnections()">
                        <i class="fas fa-download"></i> Export
                    </button>
                </div>
                
                <div class="connections-table">
                    <div class="table-header">
                        <div class="col-local">Local Address</div>
                        <div class="col-remote">Remote Address</div>
                        <div class="col-state">State</div>
                        <div class="col-protocol">Protocol</div>
                        <div class="col-actions">Actions</div>
                    </div>
                    <div class="table-body">
                        ${this.networkData.connections.map((conn, index) => `
                            <div class="table-row">
                                <div class="col-local">
                                    <span class="address">${conn.local}</span>
                                </div>
                                <div class="col-remote">
                                    <span class="address">${conn.remote}</span>
                                </div>
                                <div class="col-state">
                                    <span class="state ${conn.state.toLowerCase()}">${conn.state}</span>
                                </div>
                                <div class="col-protocol">
                                    <span class="protocol">${conn.protocol}</span>
                                </div>
                                <div class="col-actions">
                                    <button class="btn-sm" onclick="networkComponent.viewConnectionDetails(${index})" title="Details">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                    <button class="btn-sm btn-danger" onclick="networkComponent.closeConnection(${index})" title="Close">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    switchTab(tab) {
        this.activeTab = tab;
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.tab-btn').classList.add('active');
        document.getElementById('network-content').innerHTML = this.renderTabContent();
    }

    // Action handlers
    runDiagnostics() { console.log('Running network diagnostics...'); }
    refreshData() { console.log('Refreshing network data...'); }
    addInterface() { console.log('Adding network interface...'); }
    configureInterfaces() { console.log('Configuring interfaces...'); }
    configureInterface(id) { console.log(`Configuring interface: ${id}`); }
    restartInterface(id) { console.log(`Restarting interface: ${id}`); }
    addRoute() { console.log('Adding route...'); }
    refreshRoutes() { console.log('Refreshing routes...'); }
    editRoute(index) { console.log(`Editing route: ${index}`); }
    deleteRoute(index) { console.log(`Deleting route: ${index}`); }
    refreshConnections() { console.log('Refreshing connections...'); }
    exportConnections() { console.log('Exporting connections...'); }
    viewConnectionDetails(index) { console.log(`Viewing connection details: ${index}`); }
    closeConnection(index) { console.log(`Closing connection: ${index}`); }

    startMonitoring() {
        setInterval(() => {
            // Update network metrics
            this.networkData.overview.usedBandwidth = Math.floor(Math.random() * 2000) + 2000;
            this.networkData.overview.activeConnections = Math.floor(Math.random() * 500) + 1000;
        }, 30000);
    }
}

window.networkComponent = new NetworkComponent();
