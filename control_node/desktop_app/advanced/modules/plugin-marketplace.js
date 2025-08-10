/**
 * Omega SuperDesktop v2.0 - Complete Plugin Marketplace System
 * Extracted and enhanced from omega-control-center.html reference implementation
 */

class PluginMarketplace extends EventTarget {
    constructor() {
        super();
        this.installedPlugins = [];
        this.availablePlugins = [];
        this.activeTab = 'marketplace';
        this.pluginPerformanceData = {};
        this.sandboxEnabled = false;
        this.devProjects = [];
        this.plugins = new Map();
        this.runningPlugins = new Map();
        this.sandboxes = new Map();
        this.marketplace = [];
        this.eventBus = new EventTarget();
        this.sdk = new PluginSDK();
        this.security = new PluginSecurity();
        this.init();
    }

    init() {
        console.log('🔌 Plugin Marketplace initializing...');
        this.loadAvailablePlugins();
        this.loadInstalledPlugins();
        this.loadDevProjects();
        this.initializePerformanceMonitoring();
        this.setupHotReload();
        this.initializeSandbox();
        console.log('🔌 Plugin Marketplace ready');
    }

    // Tab Management
    switchTab(tabName) {
        this.activeTab = tabName;
        
        this.dispatchEvent(new CustomEvent('tabSwitched', {
            detail: { tabName }
        }));
        
        // Load tab-specific content
        switch(tabName) {
            case 'marketplace':
                this.renderAvailablePlugins();
                break;
            case 'installed':
                this.renderInstalledPlugins();
                this.updatePerformanceCharts();
                break;
            case 'development':
                this.renderDevProjects();
                break;
            case 'sandbox':
                this.initializeSandbox();
                break;
        }
    }

    // Marketplace Functions
    loadAvailablePlugins() {
        this.availablePlugins = [
            {
                id: 'cluster-analytics-pro',
                name: 'Cluster Analytics Pro',
                author: 'SuperDesktop Labs',
                category: 'analytics',
                description: 'Advanced cluster performance analytics with ML-powered insights, predictive analysis, and comprehensive reporting dashboards.',
                version: '2.1.4',
                rating: 4.8,
                downloads: 15420,
                price: 'Free',
                icon: 'fas fa-chart-line',
                features: ['Real-time analytics', 'ML predictions', 'Custom dashboards', 'Export reports'],
                screenshots: ['analytics1.png', 'analytics2.png'],
                lastUpdated: '2025-08-01'
            },
            {
                id: 'security-shield',
                name: 'Security Shield',
                author: 'CyberSec Solutions',
                category: 'security',
                description: 'Comprehensive security monitoring and threat detection with real-time alerts, vulnerability scanning, and automated responses.',
                version: '1.8.2',
                rating: 4.9,
                downloads: 23150,
                price: '$29.99',
                icon: 'fas fa-shield-alt',
                features: ['Threat detection', 'Vulnerability scanning', 'Auto-response', 'Compliance reports'],
                screenshots: ['security1.png', 'security2.png'],
                lastUpdated: '2025-07-28'
            },
            {
                id: 'slack-integration',
                name: 'Slack Integration Hub',
                author: 'Integration Partners',
                category: 'integration',
                description: 'Seamless Slack integration for notifications, alerts, and team collaboration with customizable channels and workflows.',
                version: '3.0.1',
                rating: 4.6,
                downloads: 8930,
                price: 'Free',
                icon: 'fab fa-slack',
                features: ['Real-time notifications', 'Custom workflows', 'Team channels', 'Alert routing'],
                screenshots: ['slack1.png', 'slack2.png'],
                lastUpdated: '2025-08-03'
            },
            {
                id: 'gpu-monitor-advanced',
                name: 'GPU Monitor Advanced',
                author: 'Hardware Labs',
                category: 'monitoring',
                description: 'Advanced GPU monitoring with temperature tracking, memory analysis, and performance optimization recommendations.',
                version: '1.5.7',
                rating: 4.7,
                downloads: 12680,
                price: '$19.99',
                icon: 'fas fa-microchip',
                features: ['Temperature monitoring', 'Memory analysis', 'Performance tips', 'Usage history'],
                screenshots: ['gpu1.png', 'gpu2.png'],
                lastUpdated: '2025-07-30'
            },
            {
                id: 'data-visualizer-3d',
                name: '3D Data Visualizer',
                author: 'Viz Studio',
                category: 'visualization',
                description: 'Create stunning 3D visualizations of your cluster data with interactive charts, animations, and VR support.',
                version: '2.3.0',
                rating: 4.5,
                downloads: 6420,
                price: '$49.99',
                icon: 'fas fa-cube',
                features: ['3D charts', 'VR support', 'Animations', 'Interactive exploration'],
                screenshots: ['viz1.png', 'viz2.png'],
                lastUpdated: '2025-08-02'
            },
            {
                id: 'workflow-automator',
                name: 'Workflow Automator',
                author: 'AutoFlow Inc',
                category: 'automation',
                description: 'Automate complex workflows with drag-and-drop builder, conditional logic, and integration with external services.',
                version: '1.9.3',
                rating: 4.8,
                downloads: 19240,
                price: '$39.99',
                icon: 'fas fa-cogs',
                features: ['Drag-drop builder', 'Conditional logic', 'External integrations', 'Scheduled tasks'],
                screenshots: ['workflow1.png', 'workflow2.png'],
                lastUpdated: '2025-07-25'
            },
            {
                id: 'terminal-enhanced',
                name: 'Enhanced Terminal',
                author: 'DevTools Pro',
                category: 'development',
                description: 'Advanced terminal with syntax highlighting, auto-completion, and multi-session management.',
                version: '3.2.1',
                rating: 4.9,
                downloads: 34520,
                price: 'Free',
                icon: 'fas fa-terminal',
                features: ['Syntax highlighting', 'Auto-completion', 'Multi-session', 'Custom themes'],
                screenshots: ['terminal1.png', 'terminal2.png'],
                lastUpdated: '2025-08-01'
            },
            {
                id: 'ai-assistant-pro',
                name: 'AI Assistant Pro',
                author: 'AI Labs',
                category: 'ai',
                description: 'Intelligent assistant with natural language processing, code analysis, and automated task execution.',
                version: '2.0.0',
                rating: 4.8,
                downloads: 12890,
                price: '$59.99',
                icon: 'fas fa-robot',
                features: ['NLP processing', 'Code analysis', 'Task automation', 'Learning capabilities'],
                screenshots: ['ai1.png', 'ai2.png'],
                lastUpdated: '2025-07-29'
            }
        ];

        this.marketplace = [...this.availablePlugins];
    }

    searchPlugins(query, category = null) {
        const results = this.availablePlugins.filter(plugin => {
            const matchesQuery = !query || 
                plugin.name.toLowerCase().includes(query.toLowerCase()) ||
                plugin.description.toLowerCase().includes(query.toLowerCase()) ||
                plugin.features.some(feature => feature.toLowerCase().includes(query.toLowerCase()));
            
            const matchesCategory = !category || plugin.category === category;
            
            return matchesQuery && matchesCategory;
        });

        return results.sort((a, b) => b.rating - a.rating);
    }

    getPluginsByCategory(category) {
        return this.availablePlugins
            .filter(plugin => plugin.category === category)
            .sort((a, b) => b.downloads - a.downloads);
    }

    getFeaturedPlugins() {
        return this.availablePlugins
            .filter(plugin => plugin.rating >= 4.5 && plugin.downloads > 10000)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 6);
    }

    getRecentlyUpdated() {
        return this.availablePlugins
            .slice(0, 3)
            .map(plugin => ({...plugin, lastUpdated: new Date()}));
    }

    async installPlugin(pluginId) {
        const plugin = this.availablePlugins.find(p => p.id === pluginId);
        if (!plugin) {
            throw new Error(`Plugin ${pluginId} not found`);
        }

        if (this.installedPlugins.find(p => p.id === pluginId)) {
            throw new Error(`Plugin ${pluginId} is already installed`);
        }

        // Security check
        const securityCheck = await this.security.validatePlugin(plugin);
        if (!securityCheck.safe) {
            throw new Error(`Security validation failed: ${securityCheck.reason}`);
        }

        // Dispatch installation started event
        this.dispatchEvent(new CustomEvent('installationStarted', {
            detail: { plugin }
        }));

        return new Promise((resolve, reject) => {
            // Simulate installation process
            setTimeout(() => {
                try {
                    const installedPlugin = {
                        ...plugin,
                        enabled: true,
                        installDate: new Date().toISOString(),
                        cpuUsage: Math.random() * 5, // 0-5% CPU usage
                        memoryUsage: Math.random() * 50 + 10, // 10-60MB memory usage
                        status: 'installed'
                    };

                    this.installedPlugins.push(installedPlugin);
                    this.plugins.set(pluginId, installedPlugin);

                    this.dispatchEvent(new CustomEvent('installationCompleted', {
                        detail: { plugin: installedPlugin }
                    }));

                    this.saveInstalledPlugins();
                    resolve(installedPlugin);
                } catch (error) {
                    this.dispatchEvent(new CustomEvent('installationFailed', {
                        detail: { plugin, error }
                    }));
                    reject(error);
                }
            }, 2000 + Math.random() * 3000);
        });
    }

    async uninstallPlugin(pluginId) {
        const plugin = this.installedPlugins.find(p => p.id === pluginId);
        if (!plugin) {
            throw new Error(`Plugin ${pluginId} is not installed`);
        }

        // Stop plugin if running
        if (this.runningPlugins.has(pluginId)) {
            await this.stopPlugin(pluginId);
        }

        return new Promise((resolve) => {
            setTimeout(() => {
                this.installedPlugins = this.installedPlugins.filter(p => p.id !== pluginId);
                this.plugins.delete(pluginId);
                
                this.dispatchEvent(new CustomEvent('pluginUninstalled', {
                    detail: { pluginId, plugin }
                }));

                this.saveInstalledPlugins();
                resolve();
            }, 1000);
        });
    }

    activatePlugin(pluginId) {
        const plugin = this.installedPlugins.find(p => p.id === pluginId);
        if (plugin && !plugin.enabled) {
            plugin.enabled = true;
            plugin.cpuUsage = Math.random() * 5;
            plugin.memoryUsage = Math.random() * 50 + 10;
            
            this.dispatchEvent(new CustomEvent('pluginActivated', {
                detail: { plugin }
            }));
        }
    }

    deactivatePlugin(pluginId) {
        const plugin = this.installedPlugins.find(p => p.id === pluginId);
        if (plugin && plugin.enabled) {
            plugin.enabled = false;
            plugin.cpuUsage = 0;
            plugin.memoryUsage = 0;
            
            this.dispatchEvent(new CustomEvent('pluginDeactivated', {
                detail: { plugin }
            }));
        }
    }

    // Installed Plugins Management
    loadInstalledPlugins() {
        const stored = localStorage.getItem('omega-installed-plugins');
        if (stored) {
            try {
                this.installedPlugins = JSON.parse(stored);
            } catch (e) {
                console.warn('Failed to load installed plugins from storage');
                this.installedPlugins = [];
            }
        }

        // Add default core plugins if not present
        const corePlugins = [
            {
                id: 'system-monitor',
                name: 'System Monitor',
                author: 'SuperDesktop Core',
                category: 'monitoring',
                version: '1.0.0',
                enabled: true,
                installDate: '2025-07-15T10:00:00Z',
                cpuUsage: 1.2,
                memoryUsage: 25.5,
                icon: 'fas fa-desktop',
                status: 'active'
            },
            {
                id: 'network-analyzer',
                name: 'Network Analyzer',
                author: 'SuperDesktop Core',
                category: 'monitoring',
                version: '1.0.0',
                enabled: true,
                installDate: '2025-07-15T10:00:00Z',
                cpuUsage: 0.8,
                memoryUsage: 18.2,
                icon: 'fas fa-network-wired',
                status: 'active'
            },
            {
                id: 'performance-tracker',
                name: 'Performance Tracker',
                author: 'SuperDesktop Core',
                category: 'analytics',
                version: '1.0.0',
                enabled: false,
                installDate: '2025-07-15T10:00:00Z',
                cpuUsage: 0,
                memoryUsage: 0,
                icon: 'fas fa-tachometer-alt',
                status: 'inactive'
            }
        ];

        corePlugins.forEach(plugin => {
            if (!this.installedPlugins.find(p => p.id === plugin.id)) {
                this.installedPlugins.push(plugin);
            }
        });
    }

    saveInstalledPlugins() {
        try {
            localStorage.setItem('omega-installed-plugins', JSON.stringify(this.installedPlugins));
        } catch (e) {
            console.warn('Failed to save installed plugins to storage');
        }
    }

    togglePlugin(pluginId) {
        const plugin = this.installedPlugins.find(p => p.id === pluginId);
        if (!plugin) return;

        if (plugin.enabled) {
            this.deactivatePlugin(pluginId);
        } else {
            this.activatePlugin(pluginId);
        }
    }

    updateAllPlugins() {
        return new Promise((resolve) => {
            let updatesFound = 0;
            
            this.installedPlugins.forEach(plugin => {
                // Simulate random updates
                if (Math.random() < 0.3) {
                    const versionParts = plugin.version.split('.');
                    versionParts[2] = (parseInt(versionParts[2]) + 1).toString();
                    plugin.version = versionParts.join('.');
                    updatesFound++;
                }
            });
            
            setTimeout(() => {
                this.dispatchEvent(new CustomEvent('pluginsUpdated', {
                    detail: { updatesFound, plugins: this.installedPlugins }
                }));
                this.saveInstalledPlugins();
                resolve(updatesFound);
            }, 2000);
        });
    }

    exportPluginConfig() {
        const config = {
            timestamp: new Date().toISOString(),
            plugins: this.installedPlugins.map(plugin => ({
                id: plugin.id,
                name: plugin.name,
                version: plugin.version,
                enabled: plugin.enabled,
                settings: plugin.settings || {}
            }))
        };
        
        return config;
    }

    // Development Features
    loadDevProjects() {
        this.devProjects = [
            {
                id: 'custom-dashboard',
                name: 'Custom Dashboard Plugin',
                type: 'analytics',
                status: 'in-development',
                lastModified: new Date(),
                files: ['main.js', 'style.css', 'config.json'],
                description: 'Customizable dashboard with drag-and-drop widgets'
            },
            {
                id: 'notification-center',
                name: 'Enhanced Notifications',
                type: 'ui',
                status: 'testing',
                lastModified: new Date(),
                files: ['notifications.js', 'templates.html', 'settings.json'],
                description: 'Advanced notification system with custom templates'
            },
            {
                id: 'monitoring-plugin-v2',
                name: 'Advanced Monitoring Plugin',
                type: 'monitoring',
                status: 'in-development',
                lastModified: new Date(),
                files: ['plugin.js', 'manifest.json', 'styles.css'],
                description: 'Enhanced monitoring with ML predictions'
            }
        ];
    }

    createNewPlugin() {
        const pluginName = prompt('Enter plugin name:');
        if (!pluginName) return;

        const newProject = {
            id: `custom-${Date.now()}`,
            name: pluginName,
            type: 'custom',
            status: 'in-development',
            lastModified: new Date(),
            files: ['main.js', 'style.css', 'manifest.json'],
            description: 'New custom plugin'
        };

        this.devProjects.push(newProject);
        this.renderDevProjects();
    }

    createFromTemplate(templateType) {
        const templates = {
            analytics: 'Analytics Plugin Template',
            security: 'Security Plugin Template',
            integration: 'Integration Plugin Template',
            monitoring: 'Monitoring Plugin Template'
        };

        const templateName = templates[templateType];
        if (!templateName) return;

        const newProject = {
            id: `${templateType}-${Date.now()}`,
            name: templateName,
            type: templateType,
            status: 'template',
            lastModified: new Date(),
            files: ['main.js', 'style.css', 'config.json', 'README.md'],
            description: `${templateName} generated from template`
        };

        this.devProjects.push(newProject);
        this.renderDevProjects();
    }

    openProject(projectId) {
        console.log(`Opening project: ${projectId}`);
        this.dispatchEvent(new CustomEvent('projectOpened', {
            detail: { projectId }
        }));
    }

    testProject(projectId) {
        console.log(`Testing project: ${projectId}`);
        this.dispatchEvent(new CustomEvent('projectTested', {
            detail: { projectId }
        }));
    }

    deployProject(projectId) {
        console.log(`Deploying project: ${projectId}`);
        this.dispatchEvent(new CustomEvent('projectDeployed', {
            detail: { projectId }
        }));
    }

    deleteProject(projectId) {
        if (confirm('Are you sure you want to delete this project?')) {
            this.devProjects = this.devProjects.filter(p => p.id !== projectId);
            this.renderDevProjects();
        }
    }

    // Sandbox Features
    async initializeSandbox() {
        this.sandboxEnabled = true;
        
        // Create sandbox container if it doesn't exist
        if (!this.sandboxContainer) {
            this.sandboxContainer = document.createElement('div');
            this.sandboxContainer.style.cssText = `
                position: absolute;
                top: -9999px;
                left: -9999px;
                width: 1px;
                height: 1px;
                overflow: hidden;
            `;
            document.body.appendChild(this.sandboxContainer);
        }
        
        this.dispatchEvent(new CustomEvent('sandboxInitialized'));
    }

    startSandbox() {
        this.dispatchEvent(new CustomEvent('sandboxStarted'));
    }

    resetSandbox() {
        // Clear all sandboxes
        this.sandboxes.forEach((sandbox, pluginId) => {
            this.cleanupPluginSandbox(sandbox);
        });
        this.sandboxes.clear();
        
        this.dispatchEvent(new CustomEvent('sandboxReset'));
    }

    async createPluginSandbox(plugin) {
        // Create isolated iframe sandbox
        const iframe = document.createElement('iframe');
        iframe.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            background: transparent;
        `;
        
        // Sandbox attributes for security
        iframe.sandbox = 'allow-scripts allow-same-origin allow-forms';
        
        // Create sandbox HTML
        const sandboxHTML = `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>${plugin.name} - Sandbox</title>
                <style>
                    body { 
                        margin: 0; 
                        padding: 8px; 
                        background: #1a1a1a; 
                        color: #ffffff; 
                        font-family: monospace; 
                    }
                    .plugin-container {
                        width: 100%;
                        height: 100%;
                        overflow: auto;
                    }
                </style>
            </head>
            <body>
                <div class="plugin-container" id="plugin-root"></div>
                <script>
                    // Plugin SDK injection
                    window.OmegaSDK = ${JSON.stringify(this.createSDKInterface())};
                    
                    // Plugin communication
                    window.addEventListener('message', function(event) {
                        if (event.data.type === 'plugin-command') {
                            handlePluginCommand(event.data);
                        }
                    });
                    
                    function handlePluginCommand(command) {
                        console.log('Plugin command:', command);
                    }
                    
                    // Ready signal
                    window.parent.postMessage({
                        type: 'sandbox-ready',
                        pluginId: '${plugin.id}'
                    }, '*');
                </script>
            </body>
            </html>
        `;
        
        iframe.srcdoc = sandboxHTML;
        
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Sandbox initialization timeout'));
            }, 5000);
            
            const messageHandler = (event) => {
                if (event.data.type === 'sandbox-ready' && 
                    event.data.pluginId === plugin.id) {
                    clearTimeout(timeout);
                    window.removeEventListener('message', messageHandler);
                    resolve({
                        iframe: iframe,
                        window: iframe.contentWindow,
                        document: iframe.contentDocument
                    });
                }
            };
            
            window.addEventListener('message', messageHandler);
            this.sandboxContainer.appendChild(iframe);
        });
    }

    async cleanupPluginSandbox(sandbox) {
        if (sandbox.iframe && sandbox.iframe.parentNode) {
            sandbox.iframe.parentNode.removeChild(sandbox.iframe);
        }
    }

    createSDKInterface() {
        return {
            version: '2.0.0',
            
            // UI creation methods
            createElement: (type, properties = {}) => {
                return { type, properties, id: Math.random().toString(36) };
            },
            
            // Event handling
            addEventListener: (event, callback) => {
                // Proxied event handling
            },
            
            // Storage
            storage: {
                get: (key) => localStorage.getItem(`plugin-${key}`),
                set: (key, value) => localStorage.setItem(`plugin-${key}`, value),
                remove: (key) => localStorage.removeItem(`plugin-${key}`)
            },
            
            // Network (restricted)
            fetch: (url, options = {}) => {
                // Proxied and filtered fetch
                return Promise.resolve({ json: () => ({}) });
            },
            
            // File system (restricted)
            fs: {
                readText: (path) => Promise.resolve(''),
                writeText: (path, content) => Promise.resolve(),
                list: (path) => Promise.resolve([])
            },
            
            // Notifications
            notify: (message, type = 'info') => {
                // Show notification in parent
            }
        };
    }

    // Plugin execution methods
    async startPlugin(pluginId, options = {}) {
        const plugin = this.installedPlugins.find(p => p.id === pluginId);
        if (!plugin) throw new Error('Plugin not installed');
        
        if (this.runningPlugins.has(pluginId)) {
            throw new Error('Plugin already running');
        }
        
        try {
            // Create sandbox
            const sandbox = await this.createPluginSandbox(plugin);
            
            // Initialize plugin in sandbox
            await this.initializePluginInSandbox(sandbox, plugin, options);
            
            plugin.sandbox = sandbox;
            plugin.status = 'running';
            plugin.startedAt = Date.now();
            
            this.runningPlugins.set(pluginId, plugin);
            this.sandboxes.set(pluginId, sandbox);
            
            this.dispatchEvent(new CustomEvent('pluginStarted', { 
                detail: { plugin } 
            }));
            
            return plugin;
            
        } catch (error) {
            console.error('Failed to start plugin:', error);
            throw error;
        }
    }

    async stopPlugin(pluginId) {
        const plugin = this.runningPlugins.get(pluginId);
        if (!plugin) throw new Error('Plugin not running');
        
        try {
            // Cleanup plugin in sandbox
            if (plugin.sandbox) {
                await this.cleanupPluginSandbox(plugin.sandbox);
            }
            
            // Remove from running plugins
            this.runningPlugins.delete(pluginId);
            this.sandboxes.delete(pluginId);
            
            plugin.status = 'stopped';
            plugin.sandbox = null;
            
            this.dispatchEvent(new CustomEvent('pluginStopped', { 
                detail: { pluginId } 
            }));
            
        } catch (error) {
            console.error('Failed to stop plugin:', error);
            throw error;
        }
    }

    async initializePluginInSandbox(sandbox, plugin, options) {
        // Get plugin code
        const pluginCode = await this.getPluginCode(plugin);
        
        // Inject plugin code
        const script = sandbox.document.createElement('script');
        script.textContent = pluginCode;
        sandbox.document.head.appendChild(script);
        
        // Initialize plugin
        sandbox.window.postMessage({
            type: 'plugin-init',
            plugin: {
                id: plugin.id,
                name: plugin.name,
                version: plugin.version,
                options: options
            }
        }, '*');
    }

    async getPluginCode(plugin) {
        // For demo purposes, return sample plugin code
        return `
            // Plugin: ${plugin.name}
            class ${plugin.id.replace(/-/g, '')}Plugin {
                constructor() {
                    this.name = '${plugin.name}';
                    this.version = '${plugin.version}';
                }
                
                init(container) {
                    container.innerHTML = \`
                        <div style="padding: 16px; background: #2a2a2a; border-radius: 8px; margin: 8px;">
                            <h3 style="color: #00ff88; margin: 0 0 8px 0;">${plugin.name}</h3>
                            <p style="color: #cccccc; margin: 0 0 12px 0;">${plugin.description}</p>
                            <div style="display: flex; gap: 8px;">
                                <button onclick="this.parentNode.parentNode.querySelector('p').style.display=this.parentNode.parentNode.querySelector('p').style.display==='none'?'block':'none'" 
                                        style="background: #0088ff; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer;">
                                    Toggle Description
                                </button>
                                <button onclick="console.log('Plugin action executed')" 
                                        style="background: #00ff88; color: black; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer;">
                                    Execute Action
                                </button>
                            </div>
                        </div>
                    \`;
                }
            }
            
            // Initialize plugin
            window.addEventListener('message', function(event) {
                if (event.data.type === 'plugin-init') {
                    const plugin = new ${plugin.id.replace(/-/g, '')}Plugin();
                    const container = document.getElementById('plugin-root');
                    plugin.init(container);
                }
            });
        `;
    }

    // Performance Monitoring
    initializePerformanceMonitoring() {
        this.updatePerformanceMetrics();
        
        // Update performance data every 30 seconds
        setInterval(() => {
            this.updatePerformanceMetrics();
        }, 30000);
    }

    updatePerformanceMetrics() {
        this.installedPlugins.forEach(plugin => {
            if (plugin.enabled) {
                // Simulate small fluctuations in performance
                const cpuVariation = (Math.random() - 0.5) * 0.5;
                const memVariation = (Math.random() - 0.5) * 2;
                
                plugin.cpuUsage = Math.max(0, plugin.cpuUsage + cpuVariation);
                plugin.memoryUsage = Math.max(5, plugin.memoryUsage + memVariation);
            }
        });

        this.dispatchEvent(new CustomEvent('performanceUpdated', {
            detail: { plugins: this.installedPlugins }
        }));
    }

    updatePerformanceCharts() {
        // CPU Chart
        this.updateCpuChart();
        // Memory Chart
        this.updateMemoryChart();
    }

    updateCpuChart() {
        const enabledPlugins = this.installedPlugins.filter(p => p.enabled);
        const labels = enabledPlugins.map(p => p.name);
        const data = enabledPlugins.map(p => p.cpuUsage);

        this.dispatchEvent(new CustomEvent('cpuChartUpdateRequested', {
            detail: { labels, data }
        }));
    }

    updateMemoryChart() {
        const enabledPlugins = this.installedPlugins.filter(p => p.enabled);
        const labels = enabledPlugins.map(p => p.name);
        const data = enabledPlugins.map(p => p.memoryUsage);

        this.dispatchEvent(new CustomEvent('memoryChartUpdateRequested', {
            detail: { labels, data }
        }));
    }

    // Hot Reload for Development
    setupHotReload() {
        if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
            console.log('🔥 Hot reload enabled for plugin development');
        }
    }

    hotReload() {
        this.dispatchEvent(new CustomEvent('hotReload'));
    }

    // Utility methods
    renderAvailablePlugins() {
        this.dispatchEvent(new CustomEvent('renderRequested', {
            detail: { type: 'available', plugins: this.availablePlugins }
        }));
    }

    renderInstalledPlugins() {
        this.dispatchEvent(new CustomEvent('renderRequested', {
            detail: { type: 'installed', plugins: this.installedPlugins }
        }));
    }

    renderDevProjects() {
        this.dispatchEvent(new CustomEvent('renderRequested', {
            detail: { type: 'development', projects: this.devProjects }
        }));
    }

    // Notification utility
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--superdesktop-bg-secondary, #2a2a2a);
            border: 1px solid var(--superdesktop-border, #333);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--superdesktop-text-primary, #ffffff);
            z-index: 10000;
            transform: translateX(400px);
            transition: transform 0.3s ease;
            max-width: 300px;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        setTimeout(() => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Create modal utility
    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 800px;">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button type="button" class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        return modal;
    }

    // Getters
    getInstalledPlugins() {
        return this.installedPlugins;
    }

    getActivePlugins() {
        return this.installedPlugins.filter(plugin => plugin.enabled);
    }

    getRunningPlugins() {
        return Array.from(this.runningPlugins.values());
    }

    getAllPlugins() {
        return this.availablePlugins;
    }

    getCategories() {
        const categories = [...new Set(this.availablePlugins.map(plugin => plugin.category))];
        return categories.sort();
    }

    getPluginCategories() {
        return this.getCategories();
    }

    getPluginById(pluginId) {
        return this.installedPlugins.find(p => p.id === pluginId) || 
               this.availablePlugins.find(p => p.id === pluginId);
    }

    getInstalledCount() {
        return this.installedPlugins.length;
    }

    getActiveCount() {
        return this.getActivePlugins().length;
    }

    getTotalCPUUsage() {
        return this.installedPlugins
            .filter(p => p.enabled)
            .reduce((total, plugin) => total + plugin.cpuUsage, 0);
    }

    getTotalMemoryUsage() {
        return this.installedPlugins
            .filter(p => p.enabled)
            .reduce((total, plugin) => total + plugin.memoryUsage, 0);
    }

    getMarketplacePlugins() {
        return this.marketplace;
    }

    // Event handling
    addEventListener(type, callback) {
        this.eventBus.addEventListener(type, callback);
    }

    removeEventListener(type, callback) {
        this.eventBus.removeEventListener(type, callback);
    }

    emitEvent(type, data) {
        this.eventBus.dispatchEvent(new CustomEvent(type, { detail: data }));
    }
}

// Plugin SDK class
class PluginSDK {
    constructor() {
        this.version = '2.0.0';
        this.capabilities = {
            ui: true,
            storage: true,
            network: true,
            filesystem: false,
            system: false
        };
    }

    getVersion() {
        return this.version;
    }

    getCapabilities() {
        return this.capabilities;
    }
}

// Plugin Security class
class PluginSecurity {
    async validatePlugin(plugin) {
        // Basic security validation
        const dangerousPatterns = [
            /eval\(/,
            /Function\(/,
            /document\.write/,
            /innerHTML.*<script/,
            /location\.href\s*=/,
            /document\.cookie/,
            /localStorage\.clear/,
            /sessionStorage\.clear/
        ];
        
        if (plugin.code) {
            for (const pattern of dangerousPatterns) {
                if (pattern.test(plugin.code)) {
                    return {
                        safe: false,
                        reason: `Potentially dangerous code pattern: ${pattern}`
                    };
                }
            }
        }
        
        // Check permissions
        if (plugin.permissions && plugin.permissions.includes('system')) {
            return {
                safe: false,
                reason: 'System-level permissions not allowed'
            };
        }
        
        return { safe: true };
    }

    sanitizeInput(input) {
        return input.replace(/<script[^>]*>.*?<\/script>/gi, '')
                   .replace(/javascript:/gi, '')
                   .replace(/on\w+\s*=/gi, '');
    }
}

// Export for use in main application
if (typeof window !== 'undefined') {
    window.PluginMarketplace = PluginMarketplace;
    window.PluginSDK = PluginSDK;
    window.PluginSecurity = PluginSecurity;
}

// Also support CommonJS/ES modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PluginMarketplace, PluginSDK, PluginSecurity };
}
