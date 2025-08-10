// Application State Management
class AppState {
    constructor() {
        this.state = {
            theme: 'dark',
            activeTab: 'dashboard',
            activeSidebarTab: 'sessions',
            activeDesktop: 'desktop1',
            sessions: new Map(),
            notifications: [],
            settings: {
                autoSave: true,
                notifications: true,
                animationsEnabled: true,
                theme: 'dark'
            },
            systemMetrics: {
                cpu: 0,
                memory: 0,
                network: 0,
                temperature: 0
            }
        };
        
        this.subscribers = new Map();
        this.init();
    }
    
    init() {
        // Load saved state from storage
        this.loadState();
        
        // Set up auto-save
        setInterval(() => {
            if (this.state.settings.autoSave) {
                this.saveState();
            }
        }, 30000); // Save every 30 seconds
    }
    
    // State subscription
    subscribe(key, callback) {
        if (!this.subscribers.has(key)) {
            this.subscribers.set(key, []);
        }
        this.subscribers.get(key).push(callback);
        
        // Return unsubscribe function
        return () => {
            const callbacks = this.subscribers.get(key);
            if (callbacks) {
                const index = callbacks.indexOf(callback);
                if (index > -1) {
                    callbacks.splice(index, 1);
                }
            }
        };
    }
    
    // State updates
    setState(key, value) {
        const oldValue = this.getState(key);
        this.setNestedProperty(this.state, key, value);
        
        // Notify subscribers
        this.notifySubscribers(key, value, oldValue);
        
        // Save to storage if auto-save is enabled
        if (this.state.settings.autoSave) {
            this.saveState();
        }
    }
    
    getState(key = null) {
        if (!key) return this.state;
        return this.getNestedProperty(this.state, key);
    }
    
    // Utility methods
    setNestedProperty(obj, path, value) {
        const keys = path.split('.');
        let current = obj;
        
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            if (!(key in current) || typeof current[key] !== 'object') {
                current[key] = {};
            }
            current = current[key];
        }
        
        current[keys[keys.length - 1]] = value;
    }
    
    getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => {
            return current && current[key] !== undefined ? current[key] : undefined;
        }, obj);
    }
    
    notifySubscribers(key, newValue, oldValue) {
        // Notify exact key subscribers
        const callbacks = this.subscribers.get(key);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(newValue, oldValue, key);
                } catch (error) {
                    console.error('Error in state subscriber:', error);
                }
            });
        }
        
        // Notify wildcard subscribers
        const wildcardCallbacks = this.subscribers.get('*');
        if (wildcardCallbacks) {
            wildcardCallbacks.forEach(callback => {
                try {
                    callback(newValue, oldValue, key);
                } catch (error) {
                    console.error('Error in wildcard subscriber:', error);
                }
            });
        }
    }
    
    // Persistence
    saveState() {
        try {
            const stateToSave = {
                ...this.state,
                sessions: Array.from(this.state.sessions.entries())
            };
            Utils.Storage.set('omega-app-state', stateToSave);
        } catch (error) {
            console.error('Failed to save app state:', error);
        }
    }
    
    loadState() {
        try {
            const savedState = Utils.Storage.get('omega-app-state');
            if (savedState) {
                this.state = {
                    ...this.state,
                    ...savedState,
                    sessions: new Map(savedState.sessions || [])
                };
            }
        } catch (error) {
            console.error('Failed to load app state:', error);
        }
    }
    
    // Action methods
    setTheme(theme) {
        this.setState('theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
    }
    
    switchTab(tabName) {
        this.setState('activeTab', tabName);
    }
    
    switchSidebarTab(tabName) {
        this.setState('activeSidebarTab', tabName);
    }
    
    switchDesktop(desktopId) {
        this.setState('activeDesktop', desktopId);
    }
    
    addSession(session) {
        this.state.sessions.set(session.id, session);
        this.setState('sessions', this.state.sessions);
    }
    
    removeSession(sessionId) {
        this.state.sessions.delete(sessionId);
        this.setState('sessions', this.state.sessions);
    }
    
    updateSession(sessionId, updates) {
        const session = this.state.sessions.get(sessionId);
        if (session) {
            Object.assign(session, updates);
            this.setState('sessions', this.state.sessions);
        }
    }
    
    addNotification(notification) {
        const notifications = [...this.state.notifications];
        notifications.unshift({
            id: `notif_${Date.now()}`,
            timestamp: new Date(),
            ...notification
        });
        
        // Keep only last 50 notifications
        if (notifications.length > 50) {
            notifications.splice(50);
        }
        
        this.setState('notifications', notifications);
    }
    
    removeNotification(notificationId) {
        const notifications = this.state.notifications.filter(n => n.id !== notificationId);
        this.setState('notifications', notifications);
    }
    
    clearNotifications() {
        this.setState('notifications', []);
    }
    
    updateSystemMetrics(metrics) {
        this.setState('systemMetrics', { ...this.state.systemMetrics, ...metrics });
    }
    
    updateSettings(settings) {
        this.setState('settings', { ...this.state.settings, ...settings });
    }
}

// Notification System
class NotificationManager {
    constructor(appState) {
        this.appState = appState;
        this.container = null;
        this.init();
    }
    
    init() {
        this.createContainer();
        this.setupSubscriptions();
    }
    
    createContainer() {
        this.container = Utils.DOM.create('div', {
            className: 'notification-container',
            style: `
                position: fixed;
                top: 80px;
                right: 20px;
                z-index: 10000;
                max-width: 400px;
            `
        });
        document.body.appendChild(this.container);
    }
    
    setupSubscriptions() {
        this.appState.subscribe('notifications', (notifications) => {
            this.renderNotifications(notifications);
        });
    }
    
    renderNotifications(notifications) {
        // Clear existing notifications
        this.container.innerHTML = '';
        
        // Render new notifications (only show last 5)
        notifications.slice(0, 5).forEach(notification => {
            this.renderNotification(notification);
        });
    }
    
    renderNotification(notification) {
        const element = Utils.DOM.create('div', {
            className: `notification notification-${notification.type || 'info'}`,
            style: `
                background: var(--bg-surface);
                border: 1px solid var(--bg-elevated);
                border-radius: var(--border-radius-lg);
                padding: var(--space-md);
                margin-bottom: var(--space-sm);
                box-shadow: var(--shadow-lg);
                animation: slideInRight 0.3s ease;
                position: relative;
            `
        });
        
        const content = Utils.DOM.create('div', {
            style: 'padding-right: 30px;'
        });
        
        if (notification.title) {
            const title = Utils.DOM.create('div', {
                style: 'font-weight: 600; margin-bottom: 4px; color: var(--text-primary);'
            });
            title.textContent = notification.title;
            content.appendChild(title);
        }
        
        if (notification.message) {
            const message = Utils.DOM.create('div', {
                style: 'color: var(--text-secondary); font-size: var(--font-size-sm);'
            });
            message.textContent = notification.message;
            content.appendChild(message);
        }
        
        const closeBtn = Utils.DOM.create('button', {
            style: `
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                color: var(--text-muted);
                cursor: pointer;
                padding: 4px;
                border-radius: 2px;
            `,
            innerHTML: '×',
            onclick: () => {
                this.appState.removeNotification(notification.id);
            }
        });
        
        element.appendChild(content);
        element.appendChild(closeBtn);
        this.container.appendChild(element);
        
        // Auto-remove after delay
        if (notification.autoRemove !== false) {
            setTimeout(() => {
                this.appState.removeNotification(notification.id);
            }, notification.duration || 5000);
        }
    }
    
    // Convenience methods
    info(title, message, options = {}) {
        this.appState.addNotification({
            type: 'info',
            title,
            message,
            ...options
        });
    }
    
    success(title, message, options = {}) {
        this.appState.addNotification({
            type: 'success',
            title,
            message,
            ...options
        });
    }
    
    warning(title, message, options = {}) {
        this.appState.addNotification({
            type: 'warning',
            title,
            message,
            ...options
        });
    }
    
    error(title, message, options = {}) {
        this.appState.addNotification({
            type: 'error',
            title,
            message,
            autoRemove: false,
            ...options
        });
    }
}

// Export
window.AppState = AppState;
window.NotificationManager = NotificationManager;
