/**
 * Omega SuperDesktop v2.0 - Superdesktop Manager Module
 * Extracted from omega-control-center.html - Main orchestration and system management
 */

class SuperdesktopManager extends EventTarget {
    constructor() {
        super();
        this.virtualDesktops = new Map();
        this.currentTab = 'dashboard';
        this.isInitialized = false;
        this.updateIntervals = new Map();
        
        // Core subsystem managers
        this.aiAssistant = null;
        this.sessionManager = null;
        this.menuBarManager = null;
        this.performanceWidgetManager = null;
        this.nodeControlManager = null;
        this.nlpSearchEngine = null;
        this.dataManager = null;
        
        // System metrics
        this.systemMetrics = {
            memory: { used: 720, total: 1600, cached: 120 },
            network: { upload: 2.4, download: 8.7, latency: 0.8, capacity: 100 },
            storage: { read: 1.2, write: 0.8, iops: 45000, queue: 12 },
            cpu: { usage: 62, cores: 384, frequency: 3.2 },
            performance: { score: 94.7, trend: 2.3 },
            power: { draw: 12.8, thermal: 68 }
        };
        
        this.systemState = {
            uptime: Date.now(),
            totalNodes: 6,
            connectedNodes: 5,
            alerts: [],
            operations: []
        };
    }

    async initialize() {
        console.log('🚀 Initializing Omega SuperDesktop Manager...');
        
        try {
            // Initialize core subsystems
            await this.initializeSubsystems();
            
            // Initialize UI components
            this.initializeUI();
            
            // Start real-time updates
            this.startSystemMonitoring();
            
            this.isInitialized = true;
            console.log('✅ Omega SuperDesktop Manager fully initialized');
            
            this.dispatchEvent(new CustomEvent('superdesktopInitialized', {
                detail: { timestamp: new Date(), version: '2.0' }
            }));
            
        } catch (error) {
            console.error('❌ Failed to initialize SuperDesktop Manager:', error);
            this.handleInitializationError(error);
        }
    }

    async initializeSubsystems() {
        console.log('🔧 Initializing subsystems...');
        
        // Initialize data manager first (other systems depend on it)
        if (window.SuperDesktopDataManager) {
            this.dataManager = new window.SuperDesktopDataManager();
            await this.dataManager.initialize();
            window.superDesktopDataManager = this.dataManager;
        }

        // Initialize menu bar manager
        if (window.MenuBarManager) {
            this.menuBarManager = new window.MenuBarManager();
            this.menuBarManager.initialize();
            window.menuBarManager = this.menuBarManager;
        }

        // Initialize session manager
        if (window.SessionManager) {
            this.sessionManager = new window.SessionManager();
            this.sessionManager.initialize();
            window.sessionManager = this.sessionManager;
        }

        // Initialize performance widgets
        if (window.PerformanceWidgetManager) {
            this.performanceWidgetManager = new window.PerformanceWidgetManager();
            this.performanceWidgetManager.initialize();
            window.performanceWidgetManager = this.performanceWidgetManager;
        }

        // Initialize node control manager
        if (window.NodeControlManager) {
            this.nodeControlManager = new window.NodeControlManager();
            this.nodeControlManager.initialize();
            window.nodeControlManager = this.nodeControlManager;
        }

        // Initialize NLP search engine
        if (window.NLPSearchEngine) {
            this.nlpSearchEngine = new window.NLPSearchEngine();
            this.nlpSearchEngine.initialize();
            window.nlpSearchEngine = this.nlpSearchEngine;
        }

        // Initialize AI assistant last (depends on other systems)
        if (window.AIAssistant) {
            this.aiAssistant = new window.AIAssistant();
            this.aiAssistant.initialize();
            window.aiAssistant = this.aiAssistant;
        }

        // Set up inter-system communication
        this.setupInterSystemCommunication();
    }

    setupInterSystemCommunication() {
        // Listen for events from subsystems
        if (this.sessionManager) {
            this.sessionManager.addEventListener('sessionCreated', (e) => {
                this.handleSessionEvent('created', e.detail);
            });
            
            this.sessionManager.addEventListener('sessionStatusChanged', (e) => {
                this.handleSessionEvent('statusChanged', e.detail);
            });
        }

        if (this.nodeControlManager) {
            this.nodeControlManager.addEventListener('nodeStatusChanged', (e) => {
                this.handleNodeEvent('statusChanged', e.detail);
            });
        }

        if (this.menuBarManager) {
            this.menuBarManager.addEventListener('systemStatsUpdated', (e) => {
                this.updateSystemMetrics(e.detail);
            });
        }
    }

    initializeUI() {
        // Initialize time displays
        this.updateSystemTime();
        
        // Initialize status displays
        this.updateSystemStatus();
        
        // Initialize virtual desktop
        this.initializeVirtualDesktop();
        
        // Setup global search
        this.setupGlobalSearch();
        
        // Setup event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Tab switching
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-tab]')) {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            }
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleGlobalKeyboard(e);
        });

        // Window resize handling
        window.addEventListener('resize', () => {
            this.handleWindowResize();
        });
    }

    handleGlobalKeyboard(e) {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case '1':
                    e.preventDefault();
                    this.switchTab('dashboard');
                    break;
                case '2':
                    e.preventDefault();
                    this.switchTab('sessions');
                    break;
                case '3':
                    e.preventDefault();
                    this.switchTab('nodes');
                    break;
                case '4':
                    e.preventDefault();
                    this.switchTab('performance');
                    break;
                case 'f':
                    e.preventDefault();
                    this.focusGlobalSearch();
                    break;
                case 'h':
                    e.preventDefault();
                    this.showHelp();
                    break;
            }
        }
    }

    startSystemMonitoring() {
        // Update system time every second
        const timeInterval = setInterval(() => {
            this.updateSystemTime();
        }, 1000);
        this.updateIntervals.set('time', timeInterval);

        // Update system metrics every 2 seconds
        const metricsInterval = setInterval(() => {
            this.updateSystemMetrics();
        }, 2000);
        this.updateIntervals.set('metrics', metricsInterval);

        // Update session metrics every 5 seconds
        const sessionInterval = setInterval(() => {
            this.updateSessionMetrics();
        }, 5000);
        this.updateIntervals.set('sessions', sessionInterval);

        // Update node health every 10 seconds
        const nodeInterval = setInterval(() => {
            this.updateNodeHealth();
        }, 10000);
        this.updateIntervals.set('nodes', nodeInterval);

        // Update resource usage every 3 seconds
        const resourceInterval = setInterval(() => {
            this.updateResourceUsage();
        }, 3000);
        this.updateIntervals.set('resources', resourceInterval);
    }

    updateSystemTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit',
            hour12: false 
        });
        const dateStr = now.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            year: 'numeric'
        });
        
        // Update all time displays
        this.updateElementText('systemClock', timeStr.substring(0, 5));
        this.updateElementText('systemDate', dateStr.substring(0, 6));
        this.updateElementText('enhancedCurrentTime', timeStr);
        this.updateElementText('enhancedCurrentDate', dateStr);
        this.updateElementText('currentTime', timeStr);
        this.updateElementText('headerTime', timeStr);
        this.updateElementText('headerDate', dateStr);
    }

    updateSystemMetrics(externalMetrics = null) {
        if (externalMetrics) {
            // Update from external source (like menu bar manager)
            Object.assign(this.systemMetrics, externalMetrics);
        } else {
            // Get fresh data from data manager or simulate
            if (this.dataManager) {
                const dashboardData = this.dataManager.getCachedData('dashboard');
                if (dashboardData && dashboardData.resources) {
                    this.systemMetrics = { ...this.systemMetrics, ...dashboardData.resources };
                }
            } else {
                // Simulate metric updates
                this.simulateMetricUpdates();
            }
        }
        
        this.updateStatusBarMetrics();
        this.updateDashboardMetrics();
    }

    simulateMetricUpdates() {
        // Simulate gradual changes in system metrics
        this.systemMetrics.memory.used += (Math.random() - 0.5) * 20;
        this.systemMetrics.memory.used = Math.max(100, Math.min(1500, this.systemMetrics.memory.used));
        
        this.systemMetrics.cpu.usage += (Math.random() - 0.5) * 5;
        this.systemMetrics.cpu.usage = Math.max(10, Math.min(95, this.systemMetrics.cpu.usage));
        
        this.systemMetrics.network.download += (Math.random() - 0.5) * 2;
        this.systemMetrics.network.download = Math.max(0.1, Math.min(50, this.systemMetrics.network.download));
        
        this.systemMetrics.storage.read += (Math.random() - 0.5) * 0.3;
        this.systemMetrics.storage.read = Math.max(0.1, Math.min(5, this.systemMetrics.storage.read));
    }

    updateStatusBarMetrics() {
        // Update memory display
        this.updateElementText('memoryUsed', Math.round(this.systemMetrics.memory.used));
        this.updateElementText('memoryTotal', this.systemMetrics.memory.total);
        this.updateElementText('memoryFree', this.systemMetrics.memory.total - Math.round(this.systemMetrics.memory.used));
        this.updateElementText('memoryCached', this.systemMetrics.memory.cached);
        
        const memoryUsagePercent = (this.systemMetrics.memory.used / this.systemMetrics.memory.total) * 100;
        this.updateElementStyle('memoryUsageFill', 'width', memoryUsagePercent + '%');
        
        // Update network display
        this.updateElementText('uploadSpeed', this.systemMetrics.network.upload.toFixed(1));
        this.updateElementText('downloadSpeed', this.systemMetrics.network.download.toFixed(1));
        this.updateElementText('networkLatency', this.systemMetrics.network.latency + 'ms');
        this.updateElementText('networkCapacity', this.systemMetrics.network.capacity + 'Gbps');
        
        // Update storage I/O
        this.updateElementText('readSpeed', this.systemMetrics.storage.read.toFixed(1));
        this.updateElementText('writeSpeed', this.systemMetrics.storage.write.toFixed(1));
        this.updateElementText('storageIOPS', Math.round(this.systemMetrics.storage.iops / 1000) + 'K');
        this.updateElementText('storageQueue', this.systemMetrics.storage.queue);
        
        // Update CPU display
        this.updateElementText('cpuUsage', Math.round(this.systemMetrics.cpu.usage));
        this.updateElementText('totalCores', this.systemMetrics.cpu.cores);
        this.updateElementText('cpuFrequency', this.systemMetrics.cpu.frequency + 'GHz');
        this.updateElementStyle('cpuUsageFill', 'width', this.systemMetrics.cpu.usage + '%');
        
        // Update performance metrics
        this.updateElementText('performanceScore', this.systemMetrics.performance.score.toFixed(1));
        this.updateElementText('performanceTrend', '+' + this.systemMetrics.performance.trend.toFixed(1) + '%');
        
        // Update power and thermal
        this.updateElementText('powerDraw', this.systemMetrics.power.draw.toFixed(1));
        this.updateElementText('thermalStatus', this.systemMetrics.power.thermal + '°C');
    }

    updateDashboardMetrics() {
        // Update dashboard metric cards with large displays
        this.updateElementText('dashboardMemoryUsed', Math.round(this.systemMetrics.memory.used) + 'GB');
        this.updateElementText('dashboardNetworkDown', this.systemMetrics.network.download.toFixed(1) + 'GB/s');
        this.updateElementText('dashboardStorageRead', this.systemMetrics.storage.read.toFixed(1) + 'GB/s');
        this.updateElementText('dashboardCpuUsage', Math.round(this.systemMetrics.cpu.usage) + '%');
    }

    updateSystemStatus() {
        // Update session counts
        let activeSessions = 0;
        let totalSessions = 0;
        
        if (this.sessionManager) {
            const sessions = this.sessionManager.getAllSessions();
            totalSessions = sessions.length;
            activeSessions = sessions.filter(s => s.status === 'running').length;
        }
        
        this.updateElementText('activeSessionCount', activeSessions);
        this.updateElementText('activeSessions', activeSessions);
        this.updateElementText('totalSessions', totalSessions);
        this.updateElementText('activeUsers', Math.min(activeSessions, 5));
        
        // Update node health
        let healthyNodes = 0;
        let totalNodes = 0;
        
        if (this.nodeControlManager) {
            const nodeStates = this.nodeControlManager.getNodeStates();
            totalNodes = nodeStates.size;
            healthyNodes = Array.from(nodeStates.values()).filter(n => n.status === 'online').length;
        } else {
            totalNodes = this.systemState.totalNodes;
            healthyNodes = this.systemState.connectedNodes;
        }
        
        this.updateElementText('healthyNodes', healthyNodes);
        this.updateElementText('totalNodes', totalNodes);
        this.updateElementText('totalNodesStatus', totalNodes);
        this.updateElementText('connectedNodes', healthyNodes);
        
        // Update system uptime
        const uptimeMs = Date.now() - this.systemState.uptime;
        const days = Math.floor(uptimeMs / (1000 * 60 * 60 * 24));
        const hours = Math.floor((uptimeMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
        this.updateElementText('systemUptime', `${days}d ${hours}h ${minutes}m`);
        
        // Update load average (simulated)
        this.updateElementText('load1', (this.systemMetrics.cpu.usage / 50).toFixed(2));
        this.updateElementText('load5', ((this.systemMetrics.cpu.usage - 5) / 50).toFixed(2));
        this.updateElementText('load15', ((this.systemMetrics.cpu.usage - 10) / 50).toFixed(2));
        
        // Update alert counts
        const alertCount = this.systemState.alerts.length;
        this.updateElementText('alertCount', alertCount);
        this.updateElementText('alertBadge', alertCount);
        this.updateElementText('resourceAlertCount', alertCount);
        
        // Update operations
        const operationCount = this.systemState.operations.length;
        this.updateElementText('activeOperationsCount', operationCount);
        this.updateElementText('activeOperationsDetail', this.getOperationsDescription());
        
        // Update latest activity
        this.updateElementText('latestAlert', this.getLatestAlert());
        this.updateElementText('sessionActivity', this.getSessionActivity());
    }

    updateSessionMetrics() {
        if (!this.sessionManager) return;
        
        const sessions = this.sessionManager.getAllSessions();
        const running = sessions.filter(s => s.status === 'running').length;
        const total = sessions.length;
        
        this.updateElementText('runningSessionsCount', running);
        this.updateElementText('allSessionsCount', total);
        this.updateElementText('vmSessionsCount', sessions.filter(s => s.type === 'vm').length);
        this.updateElementText('containerSessionsCount', sessions.filter(s => s.type === 'container').length);
        this.updateElementText('aiSessionsCount', sessions.filter(s => s.category === 'ai').length);
        this.updateElementText('gamingSessionsCount', sessions.filter(s => s.category === 'gaming').length);
    }

    updateNodeHealth() {
        if (!this.nodeControlManager) return;
        
        const nodeStates = this.nodeControlManager.getNodeStates();
        this.systemState.totalNodes = nodeStates.size;
        this.systemState.connectedNodes = Array.from(nodeStates.values()).filter(n => 
            ['online', 'running'].includes(n.status)
        ).length;
    }

    updateResourceUsage() {
        // Update resource meters
        this.updateResourceMeters();
        
        // Update performance charts
        this.updatePerformanceCharts();
        
        // Dispatch resource update event
        this.dispatchEvent(new CustomEvent('resourcesUpdated', {
            detail: this.systemMetrics
        }));
    }

    updateResourceMeters() {
        const meters = document.querySelectorAll('.meter-fill, .progress-fill');
        meters.forEach(meter => {
            if (meter.id && meter.id.includes('cpu')) {
                meter.style.width = this.systemMetrics.cpu.usage + '%';
            } else if (meter.id && meter.id.includes('memory')) {
                const memPercent = (this.systemMetrics.memory.used / this.systemMetrics.memory.total) * 100;
                meter.style.width = memPercent + '%';
            }
        });
    }

    updatePerformanceCharts() {
        if (this.performanceWidgetManager) {
            // Charts are updated automatically by the performance widget manager
            return;
        }
        
        // Fallback chart updates if Chart.js is available
        if (typeof Chart !== 'undefined') {
            this.updateChartData();
        }
    }

    updateChartData() {
        // Update existing charts with new data
        // This is handled by individual chart instances
    }

    initializeVirtualDesktop() {
        this.virtualDesktop = document.getElementById('virtualDesktop');
        if (this.virtualDesktop) {
            this.virtualDesktop.style.display = 'none';
        }
    }

    setupGlobalSearch() {
        const searchInputs = document.querySelectorAll('#globalSearch, #nlpSearchInput');
        searchInputs.forEach(input => {
            if (input) {
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        this.performGlobalSearch(input.value);
                    }
                });
            }
        });
    }

    performGlobalSearch(query) {
        console.log('Global search:', query);
        
        if (this.nlpSearchEngine) {
            const results = this.nlpSearchEngine.search(query);
            this.displaySearchResults(results);
        }
        
        if (this.aiAssistant) {
            this.aiAssistant.processQuery(query);
        }
        
        this.dispatchEvent(new CustomEvent('globalSearchPerformed', {
            detail: { query, timestamp: new Date() }
        }));
    }

    displaySearchResults(results) {
        const resultsContainer = document.getElementById('searchResults') || 
                               document.getElementById('nlpSearchResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = results;
            resultsContainer.style.display = 'block';
        }
    }

    switchTab(tabName) {
        if (this.currentTab === tabName) return;
        
        // Update tab buttons
        document.querySelectorAll('.tab-button, .nav-item').forEach(tab => {
            tab.classList.remove('active');
        });
        
        document.querySelectorAll(`[data-tab="${tabName}"]`).forEach(tab => {
            tab.classList.add('active');
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content, .main-content-section').forEach(content => {
            content.classList.remove('active');
        });
        
        const activeContent = document.getElementById(`${tabName}-content`) || 
                             document.getElementById(tabName);
        if (activeContent) {
            activeContent.classList.add('active');
        }
        
        this.currentTab = tabName;
        
        // Trigger tab-specific initialization
        this.onTabChange(tabName);
        
        this.dispatchEvent(new CustomEvent('tabChanged', {
            detail: { tabName, previousTab: this.currentTab }
        }));
    }

    onTabChange(tabName) {
        switch (tabName) {
            case 'performance':
                if (this.performanceWidgetManager) {
                    this.performanceWidgetManager.resizeAllCharts();
                }
                break;
            case 'nodes':
                if (this.nodeControlManager) {
                    this.nodeControlManager.updateNodeDisplay();
                }
                break;
            case 'sessions':
                if (this.sessionManager) {
                    this.sessionManager.updateSessionDisplay();
                }
                break;
        }
    }

    handleSessionEvent(eventType, sessionData) {
        switch (eventType) {
            case 'created':
                this.systemState.operations.push({
                    type: 'session_creation',
                    target: sessionData.name,
                    timestamp: new Date()
                });
                break;
            case 'statusChanged':
                // Update system status when session status changes
                this.updateSystemStatus();
                break;
        }
    }

    handleNodeEvent(eventType, nodeData) {
        switch (eventType) {
            case 'statusChanged':
                this.updateNodeHealth();
                break;
        }
    }

    handleWindowResize() {
        if (this.performanceWidgetManager) {
            this.performanceWidgetManager.resizeAllCharts();
        }
    }

    focusGlobalSearch() {
        const searchInput = document.getElementById('globalSearch') || 
                           document.getElementById('nlpSearchInput');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }

    showHelp() {
        // Show help modal or navigate to help section
        console.log('Showing help...');
    }

    handleInitializationError(error) {
        // Show error notification
        if (this.menuBarManager) {
            this.menuBarManager.showNotification(
                'Initialization Error',
                'Some components failed to initialize. Check console for details.',
                'error'
            );
        }
        
        // Continue with partial initialization
        this.isInitialized = true;
    }

    // Utility methods
    getOperationsDescription() {
        if (this.systemState.operations.length === 0) return 'No active operations';
        
        const types = {};
        this.systemState.operations.forEach(op => {
            types[op.type] = (types[op.type] || 0) + 1;
        });
        
        return Object.entries(types)
            .map(([type, count]) => `${count} ${type.replace('_', ' ')}`)
            .join(', ');
    }

    getLatestAlert() {
        if (this.systemState.alerts.length === 0) return 'No recent alerts';
        
        const latest = this.systemState.alerts[this.systemState.alerts.length - 1];
        return latest.message || 'System alert';
    }

    getSessionActivity() {
        if (!this.sessionManager) return 'No session data';
        
        const sessions = this.sessionManager.getAllSessions();
        const running = sessions.filter(s => s.status === 'running').length;
        
        if (running === 0) return 'No active sessions';
        return `${running} active session${running !== 1 ? 's' : ''}`;
    }

    updateElementText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    updateElementStyle(id, property, value) {
        const el = document.getElementById(id);
        if (el) el.style[property] = value;
    }

    // Public API methods
    getSystemMetrics() {
        return { ...this.systemMetrics };
    }

    getSystemState() {
        return { ...this.systemState };
    }

    getCurrentTab() {
        return this.currentTab;
    }

    isSystemInitialized() {
        return this.isInitialized;
    }

    getConfiguration() {
        return {
            version: '2.0',
            currentTab: this.currentTab,
            systemMetrics: this.systemMetrics,
            systemState: this.systemState,
            timestamp: new Date().toISOString()
        };
    }

    loadConfiguration(config) {
        if (config.currentTab) {
            this.switchTab(config.currentTab);
        }
        
        if (config.systemMetrics) {
            this.systemMetrics = { ...this.systemMetrics, ...config.systemMetrics };
        }
    }

    resetToDefaults() {
        this.currentTab = 'dashboard';
        this.systemState.alerts = [];
        this.systemState.operations = [];
        this.switchTab('dashboard');
    }

    dispose() {
        // Clear all intervals
        this.updateIntervals.forEach((interval, key) => {
            clearInterval(interval);
        });
        this.updateIntervals.clear();
        
        // Dispose subsystems
        if (this.performanceWidgetManager) {
            this.performanceWidgetManager.dispose();
        }
        
        if (this.nodeControlManager) {
            this.nodeControlManager.dispose();
        }
        
        if (this.sessionManager) {
            this.sessionManager.dispose();
        }
        
        if (this.nlpSearchEngine) {
            this.nlpSearchEngine.dispose();
        }
        
        console.log('🧹 SuperDesktop Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.SuperdesktopManager = SuperdesktopManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SuperdesktopManager;
}
