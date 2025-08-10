/**
 * StateStore - Reactive state management with subscriptions
 */
class StateStore {
    constructor(initialState = {}) {
        this.state = this.deepClone(initialState);
        this.subscribers = new Set();
        this.middleware = [];
        this.history = [];
        this.maxHistorySize = 50;
    }

    /**
     * Get current state or specific path
     */
    getState(path = null) {
        if (!path) return this.deepClone(this.state);
        
        const keys = path.split('.');
        let value = this.state;
        
        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                return undefined;
            }
        }
        
        return this.deepClone(value);
    }

    /**
     * Get data property for direct access (backward compatibility)
     */
    get data() {
        return this.state;
    }

    /**
     * Set state and notify subscribers
     */
    setState(updates, actionType = 'setState') {
        const prevState = this.deepClone(this.state);
        const newState = { ...this.state, ...updates };
        
        // Run middleware
        let finalState = newState;
        for (const middleware of this.middleware) {
            finalState = middleware(finalState, prevState, actionType) || finalState;
        }
        
        this.state = finalState;
        
        // Add to history
        this.history.push({
            action: actionType,
            state: this.deepClone(prevState),
            timestamp: Date.now()
        });
        
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
        
        // Notify subscribers
        this.notifySubscribers(prevState, finalState, actionType);
    }

    /**
     * Update nested state path
     */
    updatePath(path, value, actionType = 'updatePath') {
        const keys = path.split('.');
        const updates = {};
        let current = updates;
        
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            // Prevent prototype pollution
            if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
                throw new Error('Invalid property name for security reasons');
            }
            current[key] = {};
            current = current[key];
        }
        
        const finalKey = keys[keys.length - 1];
        // Prevent prototype pollution
        if (finalKey === '__proto__' || finalKey === 'constructor' || finalKey === 'prototype') {
            throw new Error('Invalid property name for security reasons');
        }
        current[finalKey] = value;
        this.setState(this.deepMerge(this.state, updates), actionType);
    }

    /**
     * Subscribe to state changes
     */
    subscribe(callback, selector = null) {
        const subscription = {
            callback,
            selector,
            id: Date.now() + Math.random(),
            lastValue: selector ? selector(this.state) : this.state
        };
        
        this.subscribers.add(subscription);
        
        return () => {
            this.subscribers.delete(subscription);
        };
    }

    /**
     * Subscribe to specific path changes
     */
    subscribeTo(path, callback) {
        return this.subscribe(callback, state => this.getValueByPath(state, path));
    }

    /**
     * Add middleware
     */
    use(middleware) {
        this.middleware.push(middleware);
        return () => {
            const index = this.middleware.indexOf(middleware);
            if (index >= 0) {
                this.middleware.splice(index, 1);
            }
        };
    }

    /**
     * Dispatch action (Redux-style)
     */
    dispatch(action) {
        if (typeof action === 'function') {
            return action(this.dispatch.bind(this), this.getState.bind(this));
        }
        
        const { type, payload } = action;
        this.setState(payload || {}, type);
        
        // Emit event for external listeners
        if (window.EventBus) {
            window.EventBus.emit('stateChange', { type, payload, state: this.state });
        }
    }

    /**
     * Reset state
     */
    reset(newState = {}) {
        this.setState(newState, 'reset');
    }

    /**
     * Get action history
     */
    getHistory() {
        return [...this.history];
    }

    /**
     * Time travel (for debugging)
     */
    timeTravel(index) {
        if (index >= 0 && index < this.history.length) {
            this.state = this.deepClone(this.history[index].state);
            this.notifySubscribers({}, this.state, 'timeTravel');
        }
    }

    // Private methods
    notifySubscribers(prevState, newState, actionType) {
        this.subscribers.forEach(subscription => {
            try {
                let shouldNotify = true;
                let currentValue = newState;
                
                if (subscription.selector) {
                    currentValue = subscription.selector(newState);
                    const prevValue = subscription.lastValue;
                    shouldNotify = !this.deepEqual(currentValue, prevValue);
                    subscription.lastValue = currentValue;
                }
                
                if (shouldNotify) {
                    subscription.callback(currentValue, prevState, actionType);
                }
            } catch (error) {
                console.error('StateStore subscriber error:', error);
            }
        });
    }

    getValueByPath(obj, path) {
        const keys = path.split('.');
        let value = obj;
        
        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                return undefined;
            }
        }
        
        return value;
    }

    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => this.deepClone(item));
        if (typeof obj === 'object') {
            const cloned = {};
            for (const key in obj) {
                cloned[key] = this.deepClone(obj[key]);
            }
            return cloned;
        }
        return obj;
    }

    deepEqual(a, b) {
        if (a === b) return true;
        if (a === null || b === null) return false;
        if (typeof a !== typeof b) return false;
        
        if (typeof a === 'object') {
            const keysA = Object.keys(a);
            const keysB = Object.keys(b);
            
            if (keysA.length !== keysB.length) return false;
            
            for (const key of keysA) {
                if (!keysB.includes(key) || !this.deepEqual(a[key], b[key])) {
                    return false;
                }
            }
            
            return true;
        }
        
        return false;
    }

    /**
     * Get system metrics for UI display
     */
    getSystemMetrics() {
        return this.getState('metrics.system') || {
            cpu: Math.floor(Math.random() * 100),
            memory: Math.floor(Math.random() * 100),
            network: Math.floor(Math.random() * 50) + 10,
            sessions: this.getState('sessions')?.size || 0,
            activeNodes: this.getState('network.topology.nodes')?.length || 3
        };
    }

    /**
     * Get AI recommendations
     */
    getAIRecommendations() {
        const recs = this.getState('ai.recommendations') || [];
        if (recs.length === 0) {
            // Return mock recommendations for demo
            return [
                {
                    id: 'rec-1',
                    type: 'performance',
                    title: 'Optimize memory usage',
                    impact: '+15% performance',
                    timestamp: Date.now() - 300000
                },
                {
                    id: 'rec-2', 
                    type: 'security',
                    title: 'Update encryption keys',
                    impact: 'Enhanced security',
                    timestamp: Date.now() - 600000
                }
            ];
        }
        return recs;
    }

    /**
     * Get system alerts
     */
    getAlerts() {
        const alerts = this.getState('ui.alerts') || [];
        if (alerts.length === 0) {
            // Return mock alerts for demo
            const metrics = this.getSystemMetrics();
            const mockAlerts = [];
            
            if (metrics.cpu > 90) {
                mockAlerts.push({
                    id: 'alert-cpu',
                    type: 'performance',
                    severity: 'warning',
                    title: 'High CPU Usage',
                    message: `CPU usage at ${metrics.cpu}%`,
                    timestamp: Date.now() - 120000
                });
            }
            
            if (metrics.memory > 85) {
                mockAlerts.push({
                    id: 'alert-memory',
                    type: 'performance', 
                    severity: 'warning',
                    title: 'Memory Usage High',
                    message: `Memory usage at ${metrics.memory}%`,
                    timestamp: Date.now() - 180000
                });
            }
            
            return mockAlerts;
        }
        return alerts;
    }

    /**
     * Get resource usage details
     */
    getResourceUsage() {
        return this.getState('system.resources') || {
            cpuCores: 8,
            memoryTotal: 16,
            storageTotal: 1,
            gpu: 'RTX 4090'
        };
    }

    /**
     * Get recent activity
     */
    getRecentActivity() {
        const activity = this.getState('system.activity') || [];
        if (activity.length === 0) {
            // Return mock activity for demo
            return [
                {
                    id: 'act-1',
                    type: 'session',
                    message: 'Session started',
                    timestamp: Date.now() - 120000
                },
                {
                    id: 'act-2',
                    type: 'plugin',
                    message: 'Plugin updated',
                    timestamp: Date.now() - 300000
                },
                {
                    id: 'act-3',
                    type: 'network',
                    message: 'Network optimized',
                    timestamp: Date.now() - 480000
                }
            ];
        }
        return activity;
    }

    /**
     * Get network status
     */
    getNetworkStatus() {
        return this.getState('network.status') || {
            status: 'Connected',
            latency: Math.floor(Math.random() * 30) + 5,
            bandwidth: Math.floor(Math.random() * 100) + 50,
            connections: Math.floor(Math.random() * 10) + 2
        };
    }

    /**
     * Get security status
     */
    getSecurityStatus() {
        return this.getState('security.status') || {
            firewall: 'Active',
            encryption: 'AES-256',
            threatsBlocked: Math.floor(Math.random() * 5),
            lastScan: new Date(Date.now() - 86400000).toLocaleDateString()
        };
    }

    /**
     * Refresh all data
     */
    refreshAll() {
        this.refreshData();
        console.log('[StateStore] Refreshing all data...');
        
        // Trigger refresh of various subsystems
        this.dispatch({ type: 'REFRESH_SYSTEM_METRICS' });
        this.dispatch({ type: 'REFRESH_NETWORK_STATUS' });
        this.dispatch({ type: 'REFRESH_SECURITY_STATUS' });
    }

    /**
     * Refresh system data
     */
    refreshData() {
        // Update metrics with fresh data
        const newMetrics = {
            cpu: Math.floor(Math.random() * 100),
            memory: Math.floor(Math.random() * 100),
            network: Math.floor(Math.random() * 50) + 10,
            timestamp: Date.now()
        };
        
        this.updatePath('metrics.system', newMetrics, 'REFRESH_METRICS');
        
        // Update network status
        const newNetwork = {
            latency: Math.floor(Math.random() * 30) + 5,
            bandwidth: Math.floor(Math.random() * 100) + 50,
            connections: Math.floor(Math.random() * 10) + 2,
            timestamp: Date.now()
        };
        
        this.updatePath('network.status', newNetwork, 'REFRESH_NETWORK');
    }

    deepMerge(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this.deepMerge(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }
}

// Create global store with initial state
const initialAppState = {
    user: {
        id: null,
        username: '',
        role: 'user',
        permissions: [],
        preferences: {
            theme: 'dark',
            language: 'en',
            accessibility: {
                reducedMotion: false,
                highContrast: false,
                screenReader: false
            }
        }
    },
    sessions: new Map(),
    activeSessionId: null,
    ui: {
        sidebarCollapsed: false,
        activeTab: 'dashboard',
        activeSidebarTab: 'sessions',
        modals: {
            settings: false,
            plugins: false,
            security: false,
            sessionInspector: false
        },
        notifications: []
    },
    metrics: {
        system: {
            cpu: 0,
            memory: 0,
            network: 0,
            storage: 0,
            gpu: {
                util: 0,
                memory: 0,
                temp: 0,
                powerW: 0
            }
        },
        sessions: new Map()
    },
    network: {
        topology: {
            nodes: [],
            links: []
        },
        quality: 'auto'
    },
    plugins: {
        installed: new Map(),
        running: new Set(),
        marketplace: []
    },
    ai: {
        recommendations: [],
        insights: [],
        models: [],
        activeJobs: []
    },
    settings: {
        profiles: {
            active: 'auto',
            configs: {
                auto: { cpu: 'auto', gpu: 'auto', network: 'auto' },
                performance: { cpu: 'high', gpu: 'high', network: 'high' },
                efficiency: { cpu: 'low', gpu: 'low', network: 'low' }
            }
        },
        virtualDesktop: {
            defaultCodec: 'h264',
            defaultBitrate: 5000,
            defaultFPS: 60,
            clipboardMode: 'bidirectional',
            defaultMonitors: 1,
            watermark: false,
            recording: false
        },
        network: {
            turnServers: [],
            stunServers: ['stun:stun.l.google.com:19302'],
            qosClasses: ['realtime', 'interactive', 'bulk'],
            fallbackOrder: ['webrtc', 'rdp', 'vnc']
        },
        security: {
            clipboardPolicy: 'bidirectional',
            fileTransferPolicy: 'enabled',
            maxFileSize: 100 * 1024 * 1024, // 100MB
            allowedFileTypes: ['*'],
            devicePermissions: {
                microphone: false,
                camera: false,
                usb: false
            },
            auditLogging: true,
            zeroTrust: false
        }
    }
};

window.AppState = new StateStore(initialAppState);

export default StateStore;
