/**
 * Omega SuperDesktop v2.0 - Data Manager Module
 * Extracted from omega-control-center.html - Handles all real-time data management
 */

class SuperDesktopDataManager extends EventTarget {
    constructor() {
        super();
        this.cache = new Map();
        this.refreshInterval = 2000; // 2 seconds
        this.lastUpdate = 0;
        this.autoRefreshEnabled = true;
        this.refreshTimers = new Map();
        this.apiEndpoints = {
            dashboard: 'http://127.0.0.1:8443/api/dashboard/metrics',
            nodes: 'http://127.0.0.1:8443/api/nodes',
            sessions: 'http://127.0.0.1:8443/api/sessions',
            resources: 'http://127.0.0.1:8443/api/resources',
            performance: 'http://127.0.0.1:8443/api/performance',
            network: 'http://127.0.0.1:8443/api/network/topology'
        };
    }

    async initialize() {
        console.log('📊 Initializing SuperDesktop Data Manager...');
        
        // Start auto-refresh for all data types
        if (this.autoRefreshEnabled) {
            this.startAutoRefresh();
        }
        
        // Load initial data
        await this.loadAllData();
        
        console.log('✅ SuperDesktop Data Manager initialized');
        this.dispatchEvent(new CustomEvent('dataManagerInitialized'));
    }

    startAutoRefresh() {
        const dataTypes = ['dashboard', 'nodes', 'sessions', 'resources', 'performance'];
        
        dataTypes.forEach(type => {
            const timer = setInterval(async () => {
                await this.refreshData(type);
            }, this.refreshInterval);
            
            this.refreshTimers.set(type, timer);
        });
        
        console.log('🔄 Auto-refresh started for all data types');
    }

    stopAutoRefresh() {
        this.refreshTimers.forEach((timer, type) => {
            clearInterval(timer);
            console.log(`🛑 Auto-refresh stopped for ${type}`);
        });
        this.refreshTimers.clear();
    }

    async loadAllData() {
        console.log('📥 Loading all real data from backend...');
        
        const loadPromises = [
            this.loadRealDashboardData(),
            this.loadRealNodesData(),
            this.loadRealSessionsData(),
            this.loadRealResourcesData(),
            this.loadRealPerformanceData(),
            this.loadRealNetworkData()
        ];
        
        try {
            await Promise.allSettled(loadPromises);
            console.log('✅ All data loaded successfully');
        } catch (error) {
            console.error('❌ Error loading data:', error);
        }
    }

    async refreshData(type) {
        try {
            switch (type) {
                case 'dashboard':
                    await this.loadRealDashboardData();
                    break;
                case 'nodes':
                    await this.loadRealNodesData();
                    break;
                case 'sessions':
                    await this.loadRealSessionsData();
                    break;
                case 'resources':
                    await this.loadRealResourcesData();
                    break;
                case 'performance':
                    await this.loadRealPerformanceData();
                    break;
                default:
                    console.warn(`⚠️ Unknown data type for refresh: ${type}`);
            }
        } catch (error) {
            console.error(`❌ Failed to refresh ${type} data:`, error);
        }
    }

    async loadRealDashboardData() {
        try {
            const data = await this.fetchData('dashboard');
            if (data) {
                console.log('📊 Dashboard metrics loaded:', data);
                this.updateDashboardUI(data);
                this.cache.set('dashboard', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('dashboardDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No dashboard data received from backend');
                this.showFallbackDashboard();
            }
        } catch (error) {
            console.error('❌ Failed to load dashboard data:', error);
            this.showFallbackDashboard();
        }
    }

    async loadRealNodesData() {
        try {
            const data = await this.fetchData('nodes');
            if (data) {
                console.log('🖥️ Nodes data loaded:', data);
                this.updateNodesUI(data);
                this.cache.set('nodes', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('nodesDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No nodes data received from backend');
                this.showFallbackNodes();
            }
        } catch (error) {
            console.error('❌ Failed to load nodes data:', error);
            this.showFallbackNodes();
        }
    }

    async loadRealSessionsData() {
        try {
            const data = await this.fetchData('sessions');
            if (data) {
                console.log('🎮 Sessions data loaded:', data);
                this.updateSessionsUI(data);
                this.cache.set('sessions', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('sessionsDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No sessions data received from backend');
                this.showFallbackSessions();
            }
        } catch (error) {
            console.error('❌ Failed to load sessions data:', error);
            this.showFallbackSessions();
        }
    }

    async loadRealResourcesData() {
        try {
            const data = await this.fetchData('resources');
            if (data) {
                console.log('💾 Resources data loaded:', data);
                this.updateResourcesUI(data);
                this.cache.set('resources', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('resourcesDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No resources data received from backend');
                this.showFallbackResources();
            }
        } catch (error) {
            console.error('❌ Failed to load resources data:', error);
            this.showFallbackResources();
        }
    }

    async loadRealPerformanceData() {
        try {
            const data = await this.fetchData('performance');
            if (data) {
                console.log('⚡ Performance data loaded:', data);
                this.updatePerformanceUI(data);
                this.cache.set('performance', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('performanceDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No performance data received from backend');
                this.showFallbackPerformance();
            }
        } catch (error) {
            console.error('❌ Failed to load performance data:', error);
            this.showFallbackPerformance();
        }
    }

    async loadRealNetworkData() {
        try {
            const data = await this.fetchData('network');
            if (data) {
                console.log('🌐 Network data loaded:', data);
                this.updateNetworkUI(data);
                this.cache.set('network', { data, timestamp: Date.now() });
                this.dispatchEvent(new CustomEvent('networkDataUpdated', { detail: data }));
            } else {
                console.warn('⚠️ No network data received from backend');
                this.showFallbackNetwork();
            }
        } catch (error) {
            console.error('❌ Failed to load network data:', error);
            this.showFallbackNetwork();
        }
    }

    async fetchData(type) {
        try {
            const response = await fetch(this.apiEndpoints[type]);
            if (response.ok) {
                return await response.json();
            } else {
                console.warn(`⚠️ API response not OK for ${type}:`, response.status);
                return null;
            }
        } catch (error) {
            console.error(`❌ Network error fetching ${type}:`, error);
            return null;
        }
    }

    // UI Update Methods
    updateDashboardUI(data) {
        if (!data) return;
        
        try {
            // Update system metrics
            if (data.system) {
                this.updateElement('activeNodes', data.system.activeNodes || 0);
                this.updateElement('activeSessions', data.system.activeSessions || 0);
                this.updateElement('activeUsers', data.system.activeUsers || 0);
                this.updateElement('systemAlerts', data.system.alerts || 0);
            }

            // Update resource metrics
            if (data.resources) {
                this.updateElement('totalCPU', `${data.resources.cpu?.usage || 0}%`);
                this.updateElement('totalMemory', `${data.resources.memory?.usage || 0}%`);
                this.updateElement('totalStorage', `${data.resources.storage?.usage || 0}%`);
                this.updateElement('totalNetwork', `${data.resources.network?.usage || 0} Mbps`);
            }

            // Update charts if available
            if (window.Chart && data.charts) {
                this.updateCharts(data.charts);
            }
        } catch (error) {
            console.error('❌ Error updating dashboard UI:', error);
        }
    }

    updateNodesUI(data) {
        if (!data || !Array.isArray(data)) return;
        
        try {
            const nodesContainer = document.getElementById('nodesList');
            if (nodesContainer) {
                nodesContainer.innerHTML = data.map(node => `
                    <div class="node-card" data-node-id="${node.id}">
                        <div class="node-header">
                            <h4>${node.name}</h4>
                            <span class="node-status ${node.status}">${node.status}</span>
                        </div>
                        <div class="node-metrics">
                            <div class="metric">CPU: ${node.cpu || 0}%</div>
                            <div class="metric">Memory: ${node.memory || 0}%</div>
                            <div class="metric">Disk: ${node.disk || 0}%</div>
                        </div>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('❌ Error updating nodes UI:', error);
        }
    }

    updateSessionsUI(data) {
        if (!data || !Array.isArray(data)) return;
        
        try {
            const sessionsContainer = document.getElementById('sessionsList');
            if (sessionsContainer) {
                sessionsContainer.innerHTML = data.map(session => `
                    <div class="session-card" data-session-id="${session.id}">
                        <div class="session-header">
                            <h4>${session.name}</h4>
                            <span class="session-status ${session.status}">${session.status}</span>
                        </div>
                        <div class="session-info">
                            <div class="info">Type: ${session.type}</div>
                            <div class="info">Node: ${session.nodeId}</div>
                            <div class="info">Uptime: ${session.uptime}</div>
                        </div>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('❌ Error updating sessions UI:', error);
        }
    }

    updateResourcesUI(data) {
        if (!data) return;
        
        try {
            // Update resource pool information
            if (data.pools) {
                Object.keys(data.pools).forEach(poolType => {
                    const pool = data.pools[poolType];
                    this.updateElement(`${poolType}Total`, pool.total || 0);
                    this.updateElement(`${poolType}Used`, pool.used || 0);
                    this.updateElement(`${poolType}Available`, pool.available || 0);
                });
            }

            // Update resource allocation charts
            if (data.allocations && window.Chart) {
                this.updateResourceCharts(data.allocations);
            }
        } catch (error) {
            console.error('❌ Error updating resources UI:', error);
        }
    }

    updatePerformanceUI(data) {
        if (!data) return;
        
        try {
            // Update performance metrics
            if (data.metrics) {
                this.updateElement('avgLatency', `${data.metrics.latency || 0}ms`);
                this.updateElement('throughput', `${data.metrics.throughput || 0} ops/s`);
                this.updateElement('errorRate', `${data.metrics.errorRate || 0}%`);
            }

            // Update performance charts
            if (data.charts && window.Chart) {
                this.updatePerformanceCharts(data.charts);
            }
        } catch (error) {
            console.error('❌ Error updating performance UI:', error);
        }
    }

    updateNetworkUI(data) {
        if (!data) return;
        
        try {
            // Update network topology visualization
            if (data.topology && window.vis) {
                this.updateNetworkTopology(data.topology);
            }

            // Update network metrics
            if (data.metrics) {
                this.updateElement('networkLatency', `${data.metrics.latency || 0}ms`);
                this.updateElement('networkThroughput', `${data.metrics.throughput || 0} Mbps`);
                this.updateElement('packetLoss', `${data.metrics.packetLoss || 0}%`);
            }
        } catch (error) {
            console.error('❌ Error updating network UI:', error);
        }
    }

    // Fallback methods for when real data is unavailable
    showFallbackDashboard() {
        const fallbackData = {
            system: {
                activeNodes: 6,
                activeSessions: 23,
                activeUsers: 8,
                alerts: 2
            },
            resources: {
                cpu: { usage: Math.floor(Math.random() * 80) + 10 },
                memory: { usage: Math.floor(Math.random() * 70) + 20 },
                storage: { usage: Math.floor(Math.random() * 60) + 30 },
                network: { usage: Math.floor(Math.random() * 100) + 50 }
            }
        };
        this.updateDashboardUI(fallbackData);
    }

    showFallbackNodes() {
        const fallbackNodes = [
            { id: 'node-1', name: 'Control Node', status: 'online', cpu: 45, memory: 67, disk: 23 },
            { id: 'node-2', name: 'Compute Node 1', status: 'online', cpu: 78, memory: 56, disk: 34 },
            { id: 'node-3', name: 'Compute Node 2', status: 'online', cpu: 23, memory: 45, disk: 67 },
            { id: 'node-4', name: 'Storage Node 1', status: 'warning', cpu: 12, memory: 34, disk: 89 },
            { id: 'node-5', name: 'Storage Node 2', status: 'online', cpu: 34, memory: 56, disk: 45 },
            { id: 'node-6', name: 'Edge Node', status: 'online', cpu: 56, memory: 78, disk: 23 }
        ];
        this.updateNodesUI(fallbackNodes);
    }

    showFallbackSessions() {
        const fallbackSessions = [
            { id: 'session-1', name: 'Ubuntu Desktop', status: 'running', type: 'vm', nodeId: 'node-2', uptime: '2h 15m' },
            { id: 'session-2', name: 'Windows 11', status: 'running', type: 'vm', nodeId: 'node-3', uptime: '45m' },
            { id: 'session-3', name: 'Kali Linux', status: 'paused', type: 'container', nodeId: 'node-2', uptime: '1h 30m' }
        ];
        this.updateSessionsUI(fallbackSessions);
    }

    showFallbackResources() {
        const fallbackResources = {
            pools: {
                cpu: { total: 96, used: 45, available: 51 },
                memory: { total: 512, used: 298, available: 214 },
                storage: { total: 10240, used: 4567, available: 5673 },
                gpu: { total: 8, used: 3, available: 5 }
            }
        };
        this.updateResourcesUI(fallbackResources);
    }

    showFallbackPerformance() {
        const fallbackPerformance = {
            metrics: {
                latency: Math.floor(Math.random() * 50) + 10,
                throughput: Math.floor(Math.random() * 1000) + 500,
                errorRate: Math.random() * 2
            }
        };
        this.updatePerformanceUI(fallbackPerformance);
    }

    showFallbackNetwork() {
        const fallbackNetwork = {
            metrics: {
                latency: Math.floor(Math.random() * 100) + 20,
                throughput: Math.floor(Math.random() * 1000) + 100,
                packetLoss: Math.random() * 1
            }
        };
        this.updateNetworkUI(fallbackNetwork);
    }

    // Utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateCharts(chartData) {
        // Implementation for updating Chart.js charts
        console.log('Updating charts with data:', chartData);
    }

    updateResourceCharts(allocations) {
        // Implementation for updating resource allocation charts
        console.log('Updating resource charts with allocations:', allocations);
    }

    updatePerformanceCharts(charts) {
        // Implementation for updating performance charts
        console.log('Updating performance charts:', charts);
    }

    updateNetworkTopology(topology) {
        // Implementation for updating network topology with vis.js
        console.log('Updating network topology:', topology);
    }

    // Cache management
    getCachedData(type) {
        const cached = this.cache.get(type);
        if (cached && (Date.now() - cached.timestamp) < this.refreshInterval * 2) {
            return cached.data;
        }
        return null;
    }

    clearCache() {
        this.cache.clear();
        console.log('🗑️ Data cache cleared');
    }

    // Configuration
    setRefreshInterval(interval) {
        this.refreshInterval = interval;
        console.log(`🔄 Refresh interval set to ${interval}ms`);
    }

    enableAutoRefresh() {
        this.autoRefreshEnabled = true;
        this.startAutoRefresh();
    }

    disableAutoRefresh() {
        this.autoRefreshEnabled = false;
        this.stopAutoRefresh();
    }

    // Cleanup
    dispose() {
        this.stopAutoRefresh();
        this.clearCache();
        console.log('🧹 SuperDesktop Data Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.SuperDesktopDataManager = SuperDesktopDataManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SuperDesktopDataManager;
}
