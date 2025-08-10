/**
 * Omega SuperDesktop v2.0 - State Manager Module
 * Extracted from omega-control-center.html - Application state management
 */

class StateManager extends EventTarget {
    constructor() {
        super();
        this.state = {
            application: {
                version: '2.0',
                initialized: false,
                activeTab: 'dashboard',
                theme: 'dark',
                language: 'en'
            },
            user: {
                authenticated: false,
                username: null,
                permissions: []
            },
            system: {
                connected: false,
                nodes: new Map(),
                sessions: new Map(),
                performance: {}
            },
            ui: {
                notifications: [],
                modals: new Set(),
                sidebarCollapsed: false,
                fullscreen: false
            }
        };
        this.stateHistory = [];
        this.maxHistorySize = 50;
        this.listeners = new Map();
    }

    initialize() {
        console.log('🏛️ Initializing State Manager...');
        this.loadPersistedState();
        this.setupEventListeners();
        this.state.application.initialized = true;
        this.notifyStateChange('application.initialized', true);
        console.log('✅ State Manager initialized');
        this.dispatchEvent(new CustomEvent('stateManagerInitialized'));
    }

    setupEventListeners() {
        // Global state management functions
        window.setState = (path, value) => {
            this.setState(path, value);
        };

        window.getState = (path) => {
            return this.getState(path);
        };

        window.subscribeToState = (path, callback) => {
            return this.subscribe(path, callback);
        };

        window.saveApplicationState = () => {
            this.saveState();
        };

        window.resetApplicationState = () => {
            this.resetState();
        };

        // Auto-save state periodically
        setInterval(() => {
            this.saveState();
        }, 30000); // Save every 30 seconds

        // Save state before page unload
        window.addEventListener('beforeunload', () => {
            this.saveState();
        });
    }

    setState(path, value) {
        const pathArray = path.split('.');
        const oldValue = this.getNestedValue(this.state, pathArray);
        
        // Add to history before changing
        this.addToHistory();
        
        // Set the new value
        this.setNestedValue(this.state, pathArray, value);
        
        // Notify listeners
        this.notifyStateChange(path, value, oldValue);
        
        console.log(`State updated: ${path} = ${JSON.stringify(value)}`);
        
        // Emit global state change event
        this.dispatchEvent(new CustomEvent('stateChanged', {
            detail: { path, value, oldValue }
        }));
    }

    getState(path = null) {
        if (!path) return this.state;
        
        const pathArray = path.split('.');
        return this.getNestedValue(this.state, pathArray);
    }

    subscribe(path, callback) {
        if (!this.listeners.has(path)) {
            this.listeners.set(path, new Set());
        }
        this.listeners.get(path).add(callback);
        
        // Return unsubscribe function
        return () => {
            const pathListeners = this.listeners.get(path);
            if (pathListeners) {
                pathListeners.delete(callback);
                if (pathListeners.size === 0) {
                    this.listeners.delete(path);
                }
            }
        };
    }

    notifyStateChange(path, newValue, oldValue) {
        // Notify exact path listeners
        const exactListeners = this.listeners.get(path);
        if (exactListeners) {
            exactListeners.forEach(callback => {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    console.error('State listener error:', error);
                }
            });
        }

        // Notify wildcard listeners (parent paths)
        const pathParts = path.split('.');
        for (let i = 0; i < pathParts.length; i++) {
            const parentPath = pathParts.slice(0, i + 1).join('.');
            const wildcardPath = parentPath + '.*';
            const wildcardListeners = this.listeners.get(wildcardPath);
            if (wildcardListeners) {
                wildcardListeners.forEach(callback => {
                    try {
                        callback(newValue, oldValue, path);
                    } catch (error) {
                        console.error('Wildcard state listener error:', error);
                    }
                });
            }
        }
    }

    getNestedValue(obj, pathArray) {
        return pathArray.reduce((current, key) => {
            return current && typeof current === 'object' ? current[key] : undefined;
        }, obj);
    }

    setNestedValue(obj, pathArray, value) {
        const lastKey = pathArray.pop();
        const target = pathArray.reduce((current, key) => {
            if (!current[key] || typeof current[key] !== 'object') {
                current[key] = {};
            }
            return current[key];
        }, obj);
        target[lastKey] = value;
    }

    addToHistory() {
        const stateCopy = JSON.parse(JSON.stringify(this.state));
        this.stateHistory.push({
            state: stateCopy,
            timestamp: Date.now()
        });
        
        // Keep history size manageable
        if (this.stateHistory.length > this.maxHistorySize) {
            this.stateHistory = this.stateHistory.slice(-this.maxHistorySize);
        }
    }

    undo() {
        if (this.stateHistory.length > 1) {
            this.stateHistory.pop(); // Remove current state
            const previousState = this.stateHistory[this.stateHistory.length - 1];
            this.state = JSON.parse(JSON.stringify(previousState.state));
            this.notifyStateChange('*', this.state);
            console.log('State reverted to previous version');
            return true;
        }
        return false;
    }

    saveState() {
        try {
            const stateToSave = {
                ...this.state,
                _timestamp: Date.now()
            };
            localStorage.setItem('omega_app_state', JSON.stringify(stateToSave));
            console.log('Application state saved to localStorage');
        } catch (error) {
            console.error('Failed to save state:', error);
        }
    }

    loadPersistedState() {
        try {
            const savedState = localStorage.getItem('omega_app_state');
            if (savedState) {
                const parsedState = JSON.parse(savedState);
                
                // Merge saved state with default state
                this.state = this.mergeStates(this.state, parsedState);
                
                console.log('Application state loaded from localStorage');
                this.dispatchEvent(new CustomEvent('stateLoaded', {
                    detail: { state: this.state }
                }));
            }
        } catch (error) {
            console.error('Failed to load persisted state:', error);
        }
    }

    mergeStates(defaultState, savedState) {
        const merged = JSON.parse(JSON.stringify(defaultState));
        
        const mergeRecursive = (target, source) => {
            for (const key in source) {
                if (source.hasOwnProperty(key)) {
                    if (typeof source[key] === 'object' && !Array.isArray(source[key]) && source[key] !== null) {
                        if (!target[key]) target[key] = {};
                        mergeRecursive(target[key], source[key]);
                    } else {
                        target[key] = source[key];
                    }
                }
            }
        };
        
        mergeRecursive(merged, savedState);
        return merged;
    }

    resetState() {
        this.state = {
            application: {
                version: '2.0',
                initialized: true,
                activeTab: 'dashboard',
                theme: 'dark',
                language: 'en'
            },
            user: {
                authenticated: false,
                username: null,
                permissions: []
            },
            system: {
                connected: false,
                nodes: new Map(),
                sessions: new Map(),
                performance: {}
            },
            ui: {
                notifications: [],
                modals: new Set(),
                sidebarCollapsed: false,
                fullscreen: false
            }
        };
        
        this.stateHistory = [];
        localStorage.removeItem('omega_app_state');
        
        this.notifyStateChange('*', this.state);
        console.log('Application state reset to defaults');
        
        this.dispatchEvent(new CustomEvent('stateReset'));
    }

    // Convenience methods for common state operations
    setActiveTab(tabName) {
        this.setState('application.activeTab', tabName);
    }

    getActiveTab() {
        return this.getState('application.activeTab');
    }

    setTheme(theme) {
        this.setState('application.theme', theme);
        document.body.setAttribute('data-theme', theme);
    }

    getTheme() {
        return this.getState('application.theme');
    }

    addNotification(notification) {
        const notifications = this.getState('ui.notifications') || [];
        notifications.push({
            id: Date.now(),
            timestamp: new Date().toISOString(),
            ...notification
        });
        this.setState('ui.notifications', notifications);
    }

    removeNotification(notificationId) {
        const notifications = this.getState('ui.notifications') || [];
        const filtered = notifications.filter(n => n.id !== notificationId);
        this.setState('ui.notifications', filtered);
    }

    addNode(nodeId, nodeData) {
        const nodes = this.getState('system.nodes') || new Map();
        nodes.set(nodeId, {
            id: nodeId,
            lastUpdated: Date.now(),
            ...nodeData
        });
        this.setState('system.nodes', nodes);
    }

    removeNode(nodeId) {
        const nodes = this.getState('system.nodes') || new Map();
        nodes.delete(nodeId);
        this.setState('system.nodes', nodes);
    }

    addSession(sessionId, sessionData) {
        const sessions = this.getState('system.sessions') || new Map();
        sessions.set(sessionId, {
            id: sessionId,
            createdAt: Date.now(),
            ...sessionData
        });
        this.setState('system.sessions', sessions);
    }

    removeSession(sessionId) {
        const sessions = this.getState('system.sessions') || new Map();
        sessions.delete(sessionId);
        this.setState('system.sessions', sessions);
    }

    updatePerformanceMetrics(metrics) {
        this.setState('system.performance', {
            ...this.getState('system.performance'),
            ...metrics,
            lastUpdated: Date.now()
        });
    }

    // Debug methods
    debugState() {
        console.log('Current Application State:', this.state);
        console.log('State History Length:', this.stateHistory.length);
        console.log('Active Listeners:', Array.from(this.listeners.keys()));
    }

    exportState() {
        const exportData = {
            state: this.state,
            history: this.stateHistory,
            timestamp: Date.now(),
            version: this.state.application.version
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', `omega_state_export_${Date.now()}.json`);
        linkElement.click();
    }

    dispose() {
        this.saveState();
        this.listeners.clear();
        this.stateHistory = [];
        console.log('🧹 State Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.StateManager = StateManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = StateManager;
}
