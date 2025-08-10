/**
 * Omega SuperDesktop v2.0 - Menu Bar Manager Module
 * Extracted from omega-control-center.html - Handles menu bar, status, and global actions
 */

class MenuBarManager extends EventTarget {
    constructor() {
        super();
        this.notificationPanel = null;
        this.userMenu = null;
        this.systemMenu = null;
        this.currentTime = null;
        this.systemStats = {
            activeSessions: 0,
            alerts: 0,
            memory: 0,
            network: 0,
            connectivity: 'Connected'
        };
        this.notifications = [];
        this.searchResults = [];
    }

    initializeMenuBar() {
        console.log('📋 Initializing Menu Bar Manager...');
        this.setupTimeDisplay();
        this.setupStatusUpdates();
        this.setupEventListeners();
        this.setupNotificationSystem();
        this.startPerformanceMonitoring();
        console.log('✅ Menu Bar Manager initialized');
        this.dispatchEvent(new CustomEvent('menuBarInitialized'));
    }

    setupTimeDisplay() {
        this.updateTime();
        setInterval(() => this.updateTime(), 1000);
    }

    updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour12: false });
        const dateString = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const timezone = now.toLocaleTimeString('en-US', { timeZoneName: 'short' }).split(' ')[2];

        this.updateElementText('current-time', timeString);
        this.updateElementText('current-date', dateString);
        this.updateElementText('timezone', timezone);
        this.updateElementText('systemClock', timeString);
        this.updateElementText('systemDate', dateString);
    }

    setupStatusUpdates() {
        this.updateSystemStats();
        setInterval(() => this.updateSystemStats(), 2000);
    }

    updateSystemStats() {
        // Simulate real-time system stats or get from data manager
        if (window.superDesktopDataManager) {
            const dashboardData = window.superDesktopDataManager.getCachedData('dashboard');
            if (dashboardData) {
                this.systemStats.activeSessions = dashboardData.system?.activeSessions || 0;
                this.systemStats.alerts = dashboardData.system?.alerts || 0;
                this.systemStats.memory = dashboardData.resources?.memory?.usage || 0;
                this.systemStats.network = dashboardData.resources?.network?.usage || 0;
            }
        } else {
            // Fallback simulation
            this.systemStats.activeSessions = Math.floor(Math.random() * 20) + 5;
            this.systemStats.alerts = Math.floor(Math.random() * 5);
            this.systemStats.memory = Math.floor(Math.random() * 80) + 10;
            this.systemStats.network = Math.floor(Math.random() * 100) + 10;
        }

        this.updateElementText('sessions-count', this.systemStats.activeSessions);
        this.updateElementText('alerts-count', this.systemStats.alerts);
        this.updateElementText('memory-usage', `${this.systemStats.memory}%`);
        this.updateElementText('network-usage', `${this.systemStats.network}%`);
        this.updateElementText('activeNodes', 6);
        this.updateElementText('activeSessions', this.systemStats.activeSessions);
        this.updateElementText('activeUsers', Math.floor(this.systemStats.activeSessions / 3));
        this.updateElementText('systemAlerts', this.systemStats.alerts);

        this.updateMeter('memory-meter', this.systemStats.memory);
        this.updateMeter('network-meter', this.systemStats.network);
        this.updateStatusIndicators();

        this.dispatchEvent(new CustomEvent('systemStatsUpdated', {
            detail: this.systemStats
        }));
    }

    updateMeter(meterId, percentage) {
        const meter = document.getElementById(meterId);
        if (meter) {
            meter.style.width = `${percentage}%`;
        }
    }

    updateStatusIndicators() {
        const sessionStatus = document.getElementById('session-status');
        if (sessionStatus) {
            sessionStatus.className = 'indicator-status ' + 
                (this.systemStats.activeSessions > 0 ? 'running' : 'warning');
        }

        const alertsBadge = document.getElementById('alerts-badge');
        if (alertsBadge) {
            alertsBadge.textContent = this.systemStats.alerts;
            alertsBadge.style.display = this.systemStats.alerts > 0 ? 'flex' : 'none';
        }

        this.updateSignalBars();
    }

    updateSignalBars() {
        const signalBars = document.querySelectorAll('.signal-bar');
        const strength = Math.floor(this.systemStats.network / 20);
        
        signalBars.forEach((bar, index) => {
            if (index < strength) {
                bar.classList.add('active');
            } else {
                bar.classList.remove('active');
            }
        });
    }

    setupEventListeners() {
        this.setupQuickActions();
        this.setupGlobalSearch();
        this.setupNotificationPanel();
        this.setupUserMenu();
        this.setupSystemMenu();
    }

    setupQuickActions() {
        // New Session Button
        document.getElementById('new-session-btn')?.addEventListener('click', () => {
            this.createNewSession();
        });

        // Configuration Button
        document.getElementById('config-btn')?.addEventListener('click', () => {
            this.openConfiguration();
        });

        // Discovery Button
        document.getElementById('discovery-btn')?.addEventListener('click', () => {
            this.startNetworkDiscovery();
        });

        // Emergency Button
        document.getElementById('emergency-btn')?.addEventListener('click', () => {
            this.emergencyStop();
        });

        // Settings Button
        const systemSettingsBtn = document.querySelectorAll('[onclick="systemSettings()"]');
        systemSettingsBtn.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.openSystemSettings();
            });
        });
    }

    createNewSession() {
        const sessionTypes = ['Ubuntu Desktop', 'Windows 11', 'Kali Linux', 'CentOS', 'Docker Container'];
        const randomType = sessionTypes[Math.floor(Math.random() * sessionTypes.length)];
        
        this.showNotification('New Session', `Creating new ${randomType} session...`, 'info');
        
        // Dispatch event for virtual desktop manager
        this.dispatchEvent(new CustomEvent('createSessionRequested', {
            detail: { type: randomType }
        }));
    }

    openConfiguration() {
        this.showNotification('Configuration', 'Opening system configuration panel...', 'info');
        this.dispatchEvent(new CustomEvent('configurationRequested'));
    }

    startNetworkDiscovery() {
        this.showNotification('Network Discovery', 'Scanning for available nodes and resources...', 'info');
        this.dispatchEvent(new CustomEvent('networkDiscoveryRequested'));
    }

    emergencyStop() {
        this.showNotification('Emergency Stop', 'Initiating emergency shutdown procedures...', 'critical');
        this.dispatchEvent(new CustomEvent('emergencyStopRequested'));
    }

    openSystemSettings() {
        this.showNotification('System Settings', 'Opening system settings...', 'info');
        this.dispatchEvent(new CustomEvent('systemSettingsRequested'));
    }

    setupGlobalSearch() {
        const searchInput = document.getElementById('global-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value;
                if (query.length > 2) {
                    this.performGlobalSearch(query);
                } else {
                    this.clearSearchResults();
                }
            });

            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const query = e.target.value;
                    this.executeSearch(query);
                }
            });
        }

        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const filter = btn.dataset.filter;
                this.applySearchFilter(filter);
            });
        });
    }

    performGlobalSearch(query) {
        console.log(`Performing global search for: ${query}`);
        
        // Simulate search results
        this.searchResults = [
            { type: 'node', title: `Node containing "${query}"`, description: 'Compute node with matching name' },
            { type: 'session', title: `Session "${query}"`, description: 'Active desktop session' },
            { type: 'resource', title: `Resource pool "${query}"`, description: 'Available resource allocation' },
            { type: 'plugin', title: `Plugin "${query}"`, description: 'Available plugin in marketplace' }
        ].filter(() => Math.random() > 0.5); // Simulate partial matches

        this.displaySearchResults();
        
        this.dispatchEvent(new CustomEvent('globalSearchPerformed', {
            detail: { query, results: this.searchResults }
        }));
    }

    displaySearchResults() {
        const resultsContainer = document.getElementById('search-results');
        if (resultsContainer && this.searchResults.length > 0) {
            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = this.searchResults.map(result => `
                <div class="search-result-item" data-type="${result.type}">
                    <div class="result-title">${result.title}</div>
                    <div class="result-description">${result.description}</div>
                </div>
            `).join('');

            // Add click handlers
            resultsContainer.querySelectorAll('.search-result-item').forEach((item, index) => {
                item.addEventListener('click', () => {
                    this.selectSearchResult(this.searchResults[index]);
                });
            });
        }
    }

    clearSearchResults() {
        const resultsContainer = document.getElementById('search-results');
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
            resultsContainer.innerHTML = '';
        }
        this.searchResults = [];
    }

    selectSearchResult(result) {
        console.log('Selected search result:', result);
        this.clearSearchResults();
        
        this.dispatchEvent(new CustomEvent('searchResultSelected', {
            detail: result
        }));
    }

    executeSearch(query) {
        this.showNotification('Search', `Executing search for "${query}"...`, 'info');
        this.clearSearchResults();
    }

    applySearchFilter(filter) {
        console.log(`Applying search filter: ${filter}`);
        this.dispatchEvent(new CustomEvent('searchFilterApplied', {
            detail: { filter }
        }));
    }

    setupNotificationSystem() {
        // Create notification panel if it doesn't exist
        this.createNotificationPanel();
        
        // Setup notification toggle button
        const notificationBtn = document.getElementById('notification-btn');
        if (notificationBtn) {
            notificationBtn.addEventListener('click', () => {
                this.toggleNotificationPanel();
            });
        }

        // Setup notifications button in various places
        document.querySelectorAll('[onclick="toggleNotifications()"]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleNotificationPanel();
            });
        });
    }

    createNotificationPanel() {
        if (document.getElementById('notification-panel')) return;

        const panel = document.createElement('div');
        panel.id = 'notification-panel';
        panel.className = 'notification-panel';
        panel.innerHTML = `
            <div class="notification-header">
                <h3><i class="fas fa-bell"></i> Notifications</h3>
                <button class="clear-all-btn" onclick="menuBarManager.clearAllNotifications()">Clear All</button>
            </div>
            <div class="notification-list" id="notification-list">
                <div class="no-notifications">No notifications</div>
            </div>
        `;
        document.body.appendChild(panel);
        this.notificationPanel = panel;
    }

    toggleNotificationPanel() {
        if (this.notificationPanel) {
            this.notificationPanel.classList.toggle('visible');
        }
    }

    showNotification(title, message, type = 'info', duration = 5000) {
        const notification = {
            id: Date.now(),
            title,
            message,
            type,
            timestamp: new Date(),
            read: false
        };

        this.notifications.unshift(notification);
        this.updateNotificationList();
        this.updateNotificationBadge();

        // Create toast notification
        this.createToastNotification(notification, duration);

        this.dispatchEvent(new CustomEvent('notificationCreated', {
            detail: notification
        }));
    }

    createToastNotification(notification, duration) {
        const toast = document.createElement('div');
        toast.className = `toast-notification ${notification.type}`;
        toast.innerHTML = `
            <div class="toast-header">
                <i class="fas fa-${this.getNotificationIcon(notification.type)}"></i>
                <span class="toast-title">${notification.title}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="toast-message">${notification.message}</div>
        `;

        document.body.appendChild(toast);

        // Show animation
        setTimeout(() => toast.classList.add('show'), 100);

        // Auto-remove
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            info: 'info-circle',
            success: 'check-circle',
            warning: 'exclamation-triangle',
            error: 'times-circle',
            critical: 'exclamation-circle'
        };
        return icons[type] || 'bell';
    }

    updateNotificationList() {
        const list = document.getElementById('notification-list');
        if (!list) return;

        if (this.notifications.length === 0) {
            list.innerHTML = '<div class="no-notifications">No notifications</div>';
            return;
        }

        list.innerHTML = this.notifications.map(notification => `
            <div class="notification-item ${notification.read ? 'read' : 'unread'}" data-id="${notification.id}">
                <div class="notification-header">
                    <span class="notification-title">${notification.title}</span>
                    <span class="notification-time">${this.formatTime(notification.timestamp)}</span>
                </div>
                <div class="notification-message">${notification.message}</div>
                <div class="notification-type ${notification.type}">${notification.type}</div>
            </div>
        `).join('');

        // Add click handlers
        list.querySelectorAll('.notification-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = parseInt(item.dataset.id);
                this.markNotificationAsRead(id);
            });
        });
    }

    updateNotificationBadge() {
        const unreadCount = this.notifications.filter(n => !n.read).length;
        const badge = document.getElementById('notification-badge');
        
        if (badge) {
            badge.textContent = unreadCount;
            badge.style.display = unreadCount > 0 ? 'flex' : 'none';
        }

        // Update alerts count in system stats
        this.systemStats.alerts = unreadCount;
    }

    markNotificationAsRead(id) {
        const notification = this.notifications.find(n => n.id === id);
        if (notification && !notification.read) {
            notification.read = true;
            this.updateNotificationList();
            this.updateNotificationBadge();
        }
    }

    clearAllNotifications() {
        this.notifications = [];
        this.updateNotificationList();
        this.updateNotificationBadge();
    }

    formatTime(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
        return timestamp.toLocaleDateString();
    }

    setupUserMenu() {
        const userMenuBtn = document.getElementById('user-menu-btn');
        if (userMenuBtn) {
            userMenuBtn.addEventListener('click', () => {
                this.toggleUserMenu();
            });
        }
    }

    setupSystemMenu() {
        const systemMenuBtn = document.querySelectorAll('[onclick="toggleSystemMenu()"]');
        systemMenuBtn.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleSystemMenu();
            });
        });
    }

    toggleUserMenu() {
        // Implementation for user menu toggle
        console.log('Toggling user menu');
        this.dispatchEvent(new CustomEvent('userMenuToggled'));
    }

    toggleSystemMenu() {
        const systemMenu = document.getElementById('systemMenu');
        if (systemMenu) {
            systemMenu.classList.toggle('visible');
        }
        this.dispatchEvent(new CustomEvent('systemMenuToggled'));
    }

    startPerformanceMonitoring() {
        // Start monitoring performance metrics for menu bar display
        setInterval(() => {
            this.updatePerformanceMetrics();
        }, 5000);
    }

    updatePerformanceMetrics() {
        // Update performance-related UI elements
        const memoryBar = document.querySelector('.memory-bar');
        const networkBar = document.querySelector('.network-bar');
        
        if (memoryBar) {
            memoryBar.style.width = `${this.systemStats.memory}%`;
        }
        
        if (networkBar) {
            networkBar.style.width = `${this.systemStats.network}%`;
        }
    }

    // Utility methods
    updateElementText(id, text) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = text;
        }
    }

    // Public API methods
    getSystemStats() {
        return { ...this.systemStats };
    }

    getNotifications() {
        return [...this.notifications];
    }

    getUnreadNotificationCount() {
        return this.notifications.filter(n => !n.read).length;
    }

    // Cleanup
    dispose() {
        // Clear any intervals or event listeners
        console.log('🧹 Menu Bar Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.MenuBarManager = MenuBarManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = MenuBarManager;
}
