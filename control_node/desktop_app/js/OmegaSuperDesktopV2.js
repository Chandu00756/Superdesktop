/**
 * Main Omega SuperDesktop V2 Application
 * Single controller with modular architecture
 */
class OmegaSuperDesktopV2 {
    constructor() {
        this.initialized = false;
        this.activeTab = 'dashboard';
        this.activeSidebarTab = 'sessions';
        this.updateTimers = new Map();
        this.charts = new Map();
        
        // Component managers
        this.desktopManager = null;
        this.sessionManager = null;
        this.pluginManager = null;
        this.aiManager = null;
        this.renderer = null;
    }

    /**
     * Initialize the application
     */
    async init() {
        if (this.initialized) return;

        try {
            console.log('Initializing Omega SuperDesktop V2...');

            // Initialize core systems
            await this.initializeCore();
            
            // Initialize UI components
            this.initializeUI();
            
            // Initialize managers
            await this.initializeManagers();
            
            // Bind events
            this.bindEvents();
            
            // Start background processes
            this.startBackgroundProcesses();
            
            // Set initial state
            this.setInitialState();
            
            this.initialized = true;
            console.log('Omega SuperDesktop V2 initialized successfully');

            if (window.EventBus) {
                window.EventBus.emit('applicationReady');
            }

        } catch (error) {
            console.error('Failed to initialize Omega SuperDesktop V2:', error);
            this.handleInitializationError(error);
        }
    }

    /**
     * Switch to a tab
     */
    switchTab(tab, event = null) {
        if (event) {
            event.preventDefault();
        }

        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Add active class to selected tab
        const tabBtn = document.querySelector(`[data-tab="${tab}"]`);
        const tabContent = document.getElementById(`${tab}-tab`);

        if (tabBtn) tabBtn.classList.add('active');
        if (tabContent) tabContent.classList.add('active');

        this.activeTab = tab;

        // Update state
        if (window.AppState) {
            window.AppState.updatePath('ui.activeTab', tab);
        }

        // Load tab content
        this.loadTabContent(tab);

        // Initialize charts if needed
        if (tab === 'dashboard') {
            this.initializeDashboardCharts();
        }

        if (window.EventBus) {
            window.EventBus.emit('tabSwitched', { tab, previousTab: this.activeTab });
        }
    }

    /**
     * Switch sidebar tab
     */
    switchSidebarTab(tab, event = null) {
        if (event) {
            event.preventDefault();
        }

        // Remove active class from all sidebar tabs
        document.querySelectorAll('.sidebar-tab').forEach(btn => {
            btn.classList.remove('active');
        });

        document.querySelectorAll('.sidebar-panel').forEach(panel => {
            panel.classList.remove('active');
        });

        // Add active class to selected tab
        const tabBtn = document.querySelector(`[data-sidebar-tab="${tab}"]`);
        const tabPanel = document.getElementById(`${tab}-panel`);

        if (tabBtn) tabBtn.classList.add('active');
        if (tabPanel) tabPanel.classList.add('active');

        this.activeSidebarTab = tab;

        // Update state
        if (window.AppState) {
            window.AppState.updatePath('ui.activeSidebarTab', tab);
        }

        // Render sidebar content
        this.renderSidebarContent(tab);

        if (window.EventBus) {
            window.EventBus.emit('sidebarTabSwitched', { tab });
        }
    }

    /**
     * Create new session with wizard
     */
    async createNewSession() {
        try {
            const sessionConfig = await this.showSessionWizard();
            if (!sessionConfig) return; // User cancelled

            const session = await window.VirtualDesktopManager.createSession(sessionConfig);
            
            // Switch to sessions tab and activate new session
            this.switchSidebarTab('sessions');
            this.switchTab('virtual-desktop');
            
            // Render updated session list
            this.renderSessionList();
            this.renderDesktopTabs();

            return session;
        } catch (error) {
            console.error('Failed to create session:', error);
            if (window.NotificationManager) {
                window.NotificationManager.error('Session Creation Failed', error.message);
            }
        }
    }

    /**
     * Switch to a session
     */
    switchToSession(sessionId) {
        try {
            const session = window.VirtualDesktopManager.switchToSession(sessionId);
            
            // Update desktop tabs
            this.renderDesktopTabs();
            
            // Update virtual desktop canvas
            this.renderDesktopCanvas(session);

            if (window.AppState) {
                window.AppState.updatePath('activeSessionId', sessionId);
            }

        } catch (error) {
            console.error('Failed to switch session:', error);
            if (window.NotificationManager) {
                window.NotificationManager.error('Session Switch Failed', error.message);
            }
        }
    }

    /**
     * Close active session
     */
    async closeActiveSession() {
        const activeSessionId = window.AppState ? window.AppState.getState('activeSessionId') : null;
        if (!activeSessionId) return;

        try {
            const confirmed = await this.showConfirmDialog(
                'Close Session',
                'Are you sure you want to close this session? Any unsaved work will be lost.'
            );

            if (confirmed) {
                await window.VirtualDesktopManager.terminateSession(activeSessionId);
                this.renderSessionList();
                this.renderDesktopTabs();
            }
        } catch (error) {
            console.error('Failed to close session:', error);
            if (window.NotificationManager) {
                window.NotificationManager.error('Failed to Close Session', error.message);
            }
        }
    }

    /**
     * Toggle sidebar collapsed state
     */
    toggleSidebar() {
        const sidebar = document.querySelector('.omega-sidebar');
        const desktop = document.querySelector('.omega-desktop');
        
        if (sidebar && desktop) {
            const isCollapsed = sidebar.classList.contains('collapsed');
            
            if (isCollapsed) {
                sidebar.classList.remove('collapsed');
                desktop.classList.remove('sidebar-collapsed');
            } else {
                sidebar.classList.add('collapsed');
                desktop.classList.add('sidebar-collapsed');
            }

            if (window.AppState) {
                window.AppState.updatePath('ui.sidebarCollapsed', !isCollapsed);
            }

            if (window.EventBus) {
                window.EventBus.emit('sidebarToggled', { collapsed: !isCollapsed });
            }
        }
    }

    /**
     * Open settings modal
     */
    openSettings() {
        this.showModal('settings');
    }

    /**
     * Open plugins modal
     */
    openPlugins() {
        this.showModal('plugins');
        this.renderPluginMarketplace();
    }

    /**
     * Show modal
     */
    showModal(modalId) {
        const overlay = document.getElementById(`${modalId}-modal`);
        if (overlay) {
            overlay.classList.add('active');
            
            if (window.AppState) {
                window.AppState.updatePath(`ui.modals.${modalId}`, true);
            }

            // Set focus to modal for accessibility
            const modal = overlay.querySelector('.modal');
            if (modal) {
                modal.focus();
            }
        }
    }

    /**
     * Close modal
     */
    closeModal(modalId = null) {
        const overlays = modalId ? 
            [document.getElementById(`${modalId}-modal`)] : 
            document.querySelectorAll('.modal-overlay.active');

        overlays.forEach(overlay => {
            if (overlay && overlay.classList.contains('active')) {
                overlay.classList.remove('active');
                
                if (window.AppState && modalId) {
                    window.AppState.updatePath(`ui.modals.${modalId}`, false);
                }
            }
        });
    }

    /**
     * Handle escape key
     */
    handleEscape() {
        // Close any open modals first
        const activeModal = document.querySelector('.modal-overlay.active');
        if (activeModal) {
            this.closeModal();
            return;
        }

        // Hide any open panels
        const activePanel = document.querySelector('.desktop-controls.visible');
        if (activePanel) {
            activePanel.classList.remove('visible');
            return;
        }
    }

    /**
     * Toggle fullscreen
     */
    async toggleFullscreen() {
        try {
            if (!document.fullscreenElement) {
                await document.documentElement.requestFullscreen();
            } else {
                await document.exitFullscreen();
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    }

    /**
     * Next tab navigation
     */
    nextTab() {
        const tabs = ['dashboard', 'virtual-desktop', 'ai-hub', 'network', 'plugins', 'settings'];
        const currentIndex = tabs.indexOf(this.activeTab);
        const nextIndex = (currentIndex + 1) % tabs.length;
        this.switchTab(tabs[nextIndex]);
    }

    /**
     * Previous tab navigation
     */
    prevTab() {
        const tabs = ['dashboard', 'virtual-desktop', 'ai-hub', 'network', 'plugins', 'settings'];
        const currentIndex = tabs.indexOf(this.activeTab);
        const prevIndex = currentIndex === 0 ? tabs.length - 1 : currentIndex - 1;
        this.switchTab(tabs[prevIndex]);
    }

    // Private methods
    async initializeCore() {
        // Core systems should already be initialized
        // Verify they exist
        if (!window.EventBus) {
            throw new Error('EventBus not initialized');
        }
        if (!window.AppState) {
            throw new Error('AppState not initialized');
        }
        if (!window.NotificationManager) {
            throw new Error('NotificationManager not initialized');
        }
        if (!window.ShortcutsManager) {
            throw new Error('ShortcutsManager not initialized');
        }
    }

    initializeUI() {
        // Set up time display
        this.updateTime();
        
        // Set up system metrics
        this.updateSystemMetrics();
        
        // Initialize responsive handlers
        this.initializeResponsive();
    }

    async initializeManagers() {
        // Initialize component managers
        this.desktopManager = window.VirtualDesktopManager;
        this.sessionManager = window.VirtualDesktopManager;
        this.pluginManager = window.PluginMarketplace;
        this.aiManager = window.AIHub;
        
        // Initialize renderer
        this.renderer = new UIRenderer();
    }

    bindEvents() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.getAttribute('data-tab');
                if (tab) this.switchTab(tab, e);
            });
        });

        // Sidebar navigation
        document.querySelectorAll('.sidebar-tab').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.getAttribute('data-sidebar-tab');
                if (tab) this.switchSidebarTab(tab, e);
            });
        });

        // Sidebar toggle
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }

        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal-overlay');
                if (modal) {
                    modal.classList.remove('active');
                }
            });
        });

        // Modal overlay clicks
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    overlay.classList.remove('active');
                }
            });
        });

        // Menu items
        document.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.handleMenuClick(e.target);
            });
        });

        // Quick action buttons
        this.bindQuickActions();

        // Keyboard shortcuts context
        if (window.ShortcutsManager) {
            window.ShortcutsManager.setContext('global');
        }

        // Event bus listeners
        this.bindEventBusListeners();
    }

    bindQuickActions() {
        // New session button
        const newSessionBtn = document.querySelector('[data-action="new-session"]');
        if (newSessionBtn) {
            newSessionBtn.addEventListener('click', () => this.createNewSession());
        }

        // Settings button
        const settingsBtn = document.querySelector('[data-action="settings"]');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.openSettings());
        }

        // Plugins button
        const pluginsBtn = document.querySelector('[data-action="plugins"]');
        if (pluginsBtn) {
            pluginsBtn.addEventListener('click', () => this.openPlugins());
        }
    }

    bindEventBusListeners() {
        if (!window.EventBus) return;

        // Session events
        window.EventBus.on('sessionCreated', () => {
            this.renderSessionList();
            this.renderDesktopTabs();
        });

        window.EventBus.on('sessionTerminated', () => {
            this.renderSessionList();
            this.renderDesktopTabs();
        });

        window.EventBus.on('sessionActivated', (session) => {
            this.renderDesktopCanvas(session);
        });

        // Plugin events
        window.EventBus.on('pluginInstalled', () => {
            this.renderPluginList();
        });

        window.EventBus.on('pluginStarted', () => {
            this.renderPluginList();
        });

        window.EventBus.on('pluginStopped', () => {
            this.renderPluginList();
        });

        // AI events
        window.EventBus.on('recommendationsUpdated', () => {
            this.renderAIRecommendations();
        });

        // Window events
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    startBackgroundProcesses() {
        // Time update
        this.updateTimers.set('time', setInterval(() => {
            this.updateTime();
        }, 1000));

        // System metrics update
        this.updateTimers.set('metrics', setInterval(() => {
            this.updateSystemMetrics();
        }, 2000));

        // Auto-save state
        this.updateTimers.set('autosave', setInterval(() => {
            this.saveApplicationState();
        }, 30000));
    }

    setInitialState() {
        // Set initial tab
        this.switchTab('dashboard');
        this.switchSidebarTab('sessions');

        // Render initial content
        this.renderSessionList();
        this.renderPluginList();
        this.renderNetworkTopology();
        this.renderAIRecommendations();
    }

    loadTabContent(tab) {
        switch (tab) {
            case 'dashboard':
                this.renderDashboard();
                break;
            case 'virtual-desktop':
                this.renderVirtualDesktop();
                break;
            case 'ai-hub':
                this.renderAIHub();
                break;
            case 'network':
                this.renderNetworkTab();
                break;
            case 'plugins':
                this.renderPluginsTab();
                break;
            case 'settings':
                this.renderSettingsTab();
                break;
        }
    }

    renderSidebarContent(tab) {
        switch (tab) {
            case 'sessions':
                this.renderSessionList();
                break;
            case 'plugins':
                this.renderPluginList();
                break;
            case 'files':
                this.renderFilePanel();
                break;
        }
    }

    // Render methods (delegated to UIRenderer)
    renderSessionList() {
        if (this.renderer) {
            this.renderer.renderSessionList();
        }
    }

    renderDesktopTabs() {
        if (this.renderer) {
            this.renderer.renderDesktopTabs();
        }
    }

    renderDesktopCanvas(session) {
        if (this.renderer) {
            this.renderer.renderDesktopCanvas(session);
        }
    }

    renderPluginList() {
        if (this.renderer) {
            this.renderer.renderPluginList();
        }
    }

    renderPluginMarketplace() {
        if (this.renderer) {
            this.renderer.renderPluginMarketplace();
        }
    }

    renderDashboard() {
        if (this.renderer) {
            this.renderer.renderDashboard();
        }
    }

    renderAIRecommendations() {
        if (this.renderer) {
            this.renderer.renderAIRecommendations();
        }
    }

    renderNetworkTopology() {
        if (this.renderer) {
            this.renderer.renderNetworkTopology();
        }
    }

    renderVirtualDesktop() {
        if (this.renderer) {
            this.renderer.renderVirtualDesktop();
        }
    }

    renderAIHub() {
        if (this.renderer) {
            this.renderer.renderAIHub();
        }
    }

    renderNetworkTab() {
        if (this.renderer) {
            this.renderer.renderNetworkTab();
        }
    }

    renderPluginsTab() {
        if (this.renderer) {
            this.renderer.renderPluginsTab();
        }
    }

    renderSettingsTab() {
        if (this.renderer) {
            this.renderer.renderSettingsTab();
        }
    }

    renderFilePanel() {
        if (this.renderer) {
            this.renderer.renderFilePanel();
        }
    }

    // Utility methods
    updateTime() {
        const timeElement = document.querySelector('.current-time');
        if (timeElement) {
            timeElement.textContent = new Date().toLocaleTimeString();
        }
    }

    updateSystemMetrics() {
        // Simulate system metrics
        const metrics = {
            cpu: Math.random() * 100,
            memory: Math.random() * 100,
            network: Math.random() * 100,
            storage: Math.random() * 100,
            gpu: {
                util: Math.random() * 100,
                memory: Math.random() * 100,
                temp: 40 + Math.random() * 40,
                powerW: 50 + Math.random() * 200
            }
        };

        if (window.AppState) {
            window.AppState.updatePath('metrics.system', metrics);
        }

        if (window.EventBus) {
            window.EventBus.emit('systemMetricsUpdated', metrics);
        }
    }

    initializeResponsive() {
        // Set up responsive behavior
        const mediaQuery = window.matchMedia('(max-width: 768px)');
        
        const handleMediaChange = (e) => {
            if (e.matches) {
                // Mobile layout
                document.body.classList.add('mobile-layout');
            } else {
                // Desktop layout
                document.body.classList.remove('mobile-layout');
            }
        };

        mediaQuery.addListener(handleMediaChange);
        handleMediaChange(mediaQuery);
    }

    initializeDashboardCharts() {
        // Initialize charts with deferred loading
        requestAnimationFrame(() => {
            this.createSystemChart();
            this.createPerformanceChart();
        });
    }

    createSystemChart() {
        const canvas = document.getElementById('system-chart');
        if (!canvas || this.charts.has('system')) return;

        const chart = new SimpleChart(canvas, {
            type: 'line',
            data: {
                labels: Array.from({ length: 10 }, (_, i) => `${i}s`),
                datasets: [{
                    label: 'CPU',
                    data: Array.from({ length: 10 }, () => Math.random() * 100),
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { max: 100 }
                }
            }
        });

        this.charts.set('system', chart);
    }

    createPerformanceChart() {
        const canvas = document.getElementById('performance-chart');
        if (!canvas || this.charts.has('performance')) return;

        const chart = new SimpleChart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'Storage', 'Network'],
                datasets: [{
                    data: [30, 45, 20, 35],
                    backgroundColor: ['#00d4ff', '#00ff7f', '#ff6b6b', '#ffd93d']
                }]
            },
            options: {
                responsive: true
            }
        });

        this.charts.set('performance', chart);
    }

    handleMenuClick(menuItem) {
        const action = menuItem.getAttribute('data-action');
        
        switch (action) {
            case 'new-session':
                this.createNewSession();
                break;
            case 'settings':
                this.openSettings();
                break;
            case 'help':
                this.showHelp();
                break;
            case 'about':
                this.showAbout();
                break;
        }
    }

    handleResize() {
        // Update charts on resize
        this.charts.forEach(chart => {
            if (chart.resize) {
                chart.resize();
            }
        });
    }

    async showSessionWizard() {
        return new Promise((resolve) => {
            // Simplified session wizard - in a real app this would be a proper modal
            const name = prompt('Session Name:', `Session ${Date.now()}`);
            if (!name) {
                resolve(null);
                return;
            }

            const type = prompt('Session Type (local/remote/rdp/vnc):', 'local');
            
            resolve({
                name,
                type: type || 'local',
                resolution: '1920x1080',
                codec: 'h264',
                bitrate: 5000,
                fps: 60
            });
        });
    }

    async showConfirmDialog(title, message) {
        // Simplified confirm dialog - in a real app this would be a proper modal
        return confirm(`${title}\n\n${message}`);
    }

    showHelp() {
        if (window.NotificationManager) {
            window.NotificationManager.info('Help', 'Help documentation is available at /docs');
        }
    }

    showAbout() {
        if (window.NotificationManager) {
            window.NotificationManager.info('About', 'Omega SuperDesktop v2.0 - Ultra Advanced Modular');
        }
    }

    saveApplicationState() {
        try {
            const state = {
                activeTab: this.activeTab,
                activeSidebarTab: this.activeSidebarTab,
                timestamp: Date.now()
            };
            
            localStorage.setItem('omega-app-state', JSON.stringify(state));
        } catch (error) {
            console.warn('Failed to save application state:', error);
        }
    }

    handleInitializationError(error) {
        document.body.innerHTML = `
            <div class="error-screen">
                <h1>Initialization Failed</h1>
                <p>Failed to initialize Omega SuperDesktop V2</p>
                <pre>${error.message}</pre>
                <button onclick="location.reload()">Reload</button>
            </div>
        `;
    }

    cleanup() {
        // Clear timers
        this.updateTimers.forEach(timer => clearInterval(timer));
        this.updateTimers.clear();

        // Destroy charts
        this.charts.forEach(chart => {
            if (chart.destroy) chart.destroy();
        });
        this.charts.clear();

        // Clean up event listeners
        if (window.EventBus) {
            window.EventBus.clear();
        }
    }
}

/**
 * UI Renderer - Handles all UI rendering operations
 */
class UIRenderer {
    constructor() {
        this.templates = new Map();
        this.loadTemplates();
    }

    loadTemplates() {
        // Session list item template
        this.templates.set('sessionItem', `
            <div class="session-item" data-session-id="{{id}}">
                <div class="session-status {{status}}"></div>
                <div class="session-info">
                    <div class="session-name">{{name}}</div>
                    <div class="session-type">{{type}} - {{resolution}}</div>
                    <div class="session-metrics">
                        Latency: {{latency}}ms | FPS: {{fps}}
                    </div>
                </div>
                <div class="session-actions">
                    <button class="btn-secondary" onclick="omegaDesktop.switchToSession('{{id}}')">
                        Open
                    </button>
                    <button class="btn-secondary" onclick="omegaDesktop.pauseSession('{{id}}')">
                        Pause
                    </button>
                </div>
            </div>
        `);

        // Plugin item template
        this.templates.set('pluginItem', `
            <div class="plugin-item" data-plugin-id="{{id}}">
                <div class="plugin-icon">
                    <i class="{{icon}}"></i>
                </div>
                <div class="plugin-info">
                    <div class="plugin-name">{{name}}</div>
                    <div class="plugin-status {{status}}">{{statusText}}</div>
                </div>
                <div class="plugin-actions">
                    {{#if running}}
                        <button class="btn-secondary" onclick="pluginManager.stopPlugin('{{id}}')">
                            Stop
                        </button>
                    {{else}}
                        <button class="btn-primary" onclick="pluginManager.startPlugin('{{id}}')">
                            Start
                        </button>
                    {{/if}}
                </div>
            </div>
        `);
    }

    renderSessionList() {
        const container = document.getElementById('session-list');
        if (!container || !window.VirtualDesktopManager) return;

        const sessions = window.VirtualDesktopManager.getActiveSessions();
        
        if (sessions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-desktop"></i>
                    <p>No active sessions</p>
                    <button class="btn-primary" onclick="omegaDesktop.createNewSession()">
                        Create New Session
                    </button>
                </div>
            `;
            return;
        }

        const html = sessions.map(session => {
            return this.renderTemplate('sessionItem', {
                id: session.id,
                name: session.name,
                type: session.type,
                status: session.status,
                resolution: session.config.resolution,
                latency: Math.round(session.metrics.latency || 0),
                fps: Math.round(session.metrics.fps || 0)
            });
        }).join('');

        container.innerHTML = html;
    }

    renderDesktopTabs() {
        const container = document.querySelector('.desktop-tabs');
        if (!container || !window.VirtualDesktopManager) return;

        const sessions = window.VirtualDesktopManager.getActiveSessions();
        const activeSessionId = window.AppState ? window.AppState.getState('activeSessionId') : null;

        const html = sessions.map(session => {
            const isActive = session.id === activeSessionId;
            return `
                <div class="desktop-tab ${isActive ? 'active' : ''}" 
                     data-session-id="${session.id}"
                     onclick="omegaDesktop.switchToSession('${session.id}')">
                    <i class="fas fa-desktop"></i>
                    <span>${session.name}</span>
                    <button class="tab-close" onclick="event.stopPropagation(); omegaDesktop.closeSession('${session.id}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    renderDesktopCanvas(session) {
        const container = document.querySelector('.desktop-content');
        if (!container) return;

        // Find or create virtual desktop for this session
        let desktop = container.querySelector(`[data-session-id="${session.id}"]`);
        
        if (!desktop) {
            desktop = document.createElement('div');
            desktop.className = 'virtual-desktop';
            desktop.setAttribute('data-session-id', session.id);
            container.appendChild(desktop);

            // Initialize desktop canvas for this session
            if (session.desktop && !session.desktop.canvas) {
                session.desktop.canvas = new DesktopCanvas(desktop, session);
            }
        }

        // Hide all desktops and show the active one
        container.querySelectorAll('.virtual-desktop').forEach(d => {
            d.classList.remove('active');
        });
        desktop.classList.add('active');

        // Set stream if available
        if (session.desktop.stream && session.desktop.canvas) {
            session.desktop.canvas.setStream(session.desktop.stream);
        }
    }

    renderPluginList() {
        const container = document.getElementById('plugin-list');
        if (!container || !window.PluginMarketplace) return;

        const plugins = window.PluginMarketplace.getInstalledPlugins();
        const runningPlugins = new Set(
            window.PluginMarketplace.getRunningPlugins().map(p => p.id)
        );

        if (plugins.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-puzzle-piece"></i>
                    <p>No plugins installed</p>
                    <button class="btn-primary" onclick="omegaDesktop.openPlugins()">
                        Browse Marketplace
                    </button>
                </div>
            `;
            return;
        }

        const html = plugins.map(plugin => {
            const running = runningPlugins.has(plugin.id);
            return this.renderTemplate('pluginItem', {
                id: plugin.id,
                name: plugin.name,
                icon: plugin.icon || 'fas fa-puzzle-piece',
                status: running ? 'running' : 'stopped',
                statusText: running ? 'Running' : 'Stopped',
                running
            });
        }).join('');

        container.innerHTML = html;
    }

    renderPluginMarketplace() {
        const container = document.querySelector('#plugins-modal .modal-content');
        if (!container || !window.PluginMarketplace) return;

        const plugins = window.PluginMarketplace.getMarketplacePlugins();
        
        const html = `
            <div class="marketplace-header">
                <h3>Plugin Marketplace</h3>
                <div class="marketplace-search">
                    <input type="text" placeholder="Search plugins..." id="plugin-search">
                    <select id="plugin-category">
                        <option value="All">All Categories</option>
                        <option value="Productivity">Productivity</option>
                        <option value="Development">Development</option>
                        <option value="Media">Media</option>
                        <option value="Games">Games</option>
                        <option value="Utilities">Utilities</option>
                    </select>
                </div>
            </div>
            <div class="plugin-grid">
                ${plugins.map(plugin => `
                    <div class="plugin-card" data-plugin-id="${plugin.id}">
                        <div class="plugin-icon">
                            <i class="${plugin.icon || 'fas fa-puzzle-piece'}"></i>
                        </div>
                        <h4>${plugin.name}</h4>
                        <p class="plugin-description">${plugin.description}</p>
                        <div class="plugin-meta">
                            <span class="plugin-price ${plugin.price === 0 ? 'free' : ''}">
                                ${plugin.price === 0 ? 'Free' : `$${plugin.price}`}
                            </span>
                            <span class="plugin-rating">
                                ★ ${plugin.rating}
                            </span>
                            <span class="plugin-downloads">
                                ${plugin.downloads} downloads
                            </span>
                        </div>
                        <div class="plugin-actions">
                            ${plugin.installed ? 
                                (plugin.running ? 
                                    '<button class="btn-secondary" disabled>Running</button>' :
                                    '<button class="btn-primary" onclick="pluginManager.startPlugin(\'' + plugin.id + '\')">Open</button>'
                                ) :
                                '<button class="btn-primary" onclick="pluginManager.installPlugin(\'' + plugin.id + '\')">Install</button>'
                            }
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        container.innerHTML = html;

        // Bind search functionality
        this.bindMarketplaceSearch();
    }

    renderDashboard() {
        // Dashboard is mostly static HTML, just update metrics
        this.updateDashboardMetrics();
    }

    renderAIRecommendations() {
        const container = document.getElementById('ai-recommendations');
        if (!container || !window.AIHub) return;

        const insights = window.AIHub.getInsights();
        const recommendations = insights.recommendations || [];

        if (recommendations.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-lightbulb"></i>
                    <p>No recommendations available</p>
                </div>
            `;
            return;
        }

        const html = recommendations.slice(0, 3).map(rec => `
            <div class="recommendation-item">
                <div class="recommendation-header">
                    <span class="recommendation-priority ${rec.priority}">${rec.priority}</span>
                    <h5>${rec.title}</h5>
                </div>
                <p>${rec.description}</p>
                <div class="recommendation-actions">
                    ${rec.actions.map(action => `
                        <button class="btn-secondary" onclick="aiHub.applyRecommendation('${rec.id}')">
                            ${action.label}
                        </button>
                    `).join('')}
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    renderNetworkTopology() {
        const container = document.getElementById('network-topology');
        if (!container) return;

        // Simple network topology visualization
        container.innerHTML = `
            <div class="topology-placeholder">
                <i class="fas fa-network-wired"></i>
                <p>Network topology visualization will be implemented here</p>
            </div>
        `;
    }

    // Additional render methods for other tabs...
    renderVirtualDesktop() {
        // Virtual desktop tab is handled by desktop canvas rendering
    }

    renderAIHub() {
        // AI Hub tab rendering
        this.renderAIRecommendations();
    }

    renderNetworkTab() {
        // Network tab rendering
        this.renderNetworkTopology();
    }

    renderPluginsTab() {
        // Plugins tab is handled by plugin list rendering
    }

    renderSettingsTab() {
        // Settings tab rendering
        const container = document.getElementById('settings-tab');
        if (!container) return;

        // Settings content is mostly static HTML
    }

    renderFilePanel() {
        const container = document.getElementById('files-panel');
        if (!container) return;

        container.innerHTML = `
            <div class="file-panel-placeholder">
                <i class="fas fa-folder"></i>
                <p>File transfer and clipboard panel</p>
            </div>
        `;
    }

    // Utility methods
    renderTemplate(templateName, data) {
        let template = this.templates.get(templateName);
        if (!template) return '';

        // Simple template replacement
        Object.keys(data).forEach(key => {
            const value = data[key];
            template = template.replace(new RegExp(`{{${key}}}`, 'g'), value);
        });

        return template;
    }

    updateDashboardMetrics() {
        const systemMetrics = window.AppState ? window.AppState.getState('metrics.system') : {};
        
        // Update metric values
        const updateMetric = (id, value, suffix = '%') => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = `${Math.round(value || 0)}${suffix}`;
            }
        };

        updateMetric('cpu-usage', systemMetrics.cpu);
        updateMetric('memory-usage', systemMetrics.memory);
        updateMetric('network-usage', systemMetrics.network);
        updateMetric('storage-usage', systemMetrics.storage);

        if (systemMetrics.gpu) {
            updateMetric('gpu-usage', systemMetrics.gpu.util);
            updateMetric('gpu-temp', systemMetrics.gpu.temp, '°C');
        }
    }

    bindMarketplaceSearch() {
        const searchInput = document.getElementById('plugin-search');
        const categorySelect = document.getElementById('plugin-category');
        
        if (searchInput && categorySelect) {
            const handleSearch = () => {
                const search = searchInput.value;
                const category = categorySelect.value;
                
                // Re-render with filtered results
                const plugins = window.PluginMarketplace.getMarketplacePlugins(category, search);
                this.updatePluginGrid(plugins);
            };

            searchInput.addEventListener('input', handleSearch);
            categorySelect.addEventListener('change', handleSearch);
        }
    }

    updatePluginGrid(plugins) {
        const grid = document.querySelector('.plugin-grid');
        if (!grid) return;

        const html = plugins.map(plugin => `
            <div class="plugin-card" data-plugin-id="${plugin.id}">
                <div class="plugin-icon">
                    <i class="${plugin.icon || 'fas fa-puzzle-piece'}"></i>
                </div>
                <h4>${plugin.name}</h4>
                <p class="plugin-description">${plugin.description}</p>
                <div class="plugin-meta">
                    <span class="plugin-price ${plugin.price === 0 ? 'free' : ''}">
                        ${plugin.price === 0 ? 'Free' : `$${plugin.price}`}
                    </span>
                    <span class="plugin-rating">★ ${plugin.rating}</span>
                    <span class="plugin-downloads">${plugin.downloads} downloads</span>
                </div>
                <div class="plugin-actions">
                    ${plugin.installed ? 
                        (plugin.running ? 
                            '<button class="btn-secondary" disabled>Running</button>' :
                            '<button class="btn-primary" onclick="pluginManager.startPlugin(\'' + plugin.id + '\')">Open</button>'
                        ) :
                        '<button class="btn-primary" onclick="pluginManager.installPlugin(\'' + plugin.id + '\')">Install</button>'
                    }
                </div>
            </div>
        `).join('');

        grid.innerHTML = html;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.omegaDesktop = new OmegaSuperDesktopV2();
    window.omegaDesktop.init();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OmegaSuperDesktopV2, UIRenderer };
}
