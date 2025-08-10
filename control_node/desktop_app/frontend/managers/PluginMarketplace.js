/**
 * PluginMarketplace - Plugin management and marketplace
 */
class PluginMarketplace {
    constructor() {
        this.installedPlugins = new Map();
        this.runningPlugins = new Set();
        this.marketplacePlugins = [];
        this.categories = ['All', 'Productivity', 'Development', 'Media', 'Games', 'Utilities'];
        
        this.init();
    }

    async init() {
        await this.loadInstalledPlugins();
        await this.loadMarketplacePlugins();
        this.bindEvents();
    }

    /**
     * Install a plugin
     */
    async installPlugin(pluginId) {
        try {
            const plugin = this.marketplacePlugins.find(p => p.id === pluginId);
            if (!plugin) {
                throw new Error('Plugin not found in marketplace');
            }

            if (this.installedPlugins.has(pluginId)) {
                throw new Error('Plugin is already installed');
            }

            // Start progress notification
            const progress = window.NotificationManager.showProgress(
                'Installing Plugin',
                `Installing ${plugin.name}...`
            );

            // Simulate installation process
            await this.simulateInstallation(plugin, progress);

            // Add to installed plugins
            const installedPlugin = {
                ...plugin,
                installedDate: new Date(),
                version: plugin.version || '1.0.0',
                enabled: true,
                settings: {}
            };

            this.installedPlugins.set(pluginId, installedPlugin);
            
            // Save to storage
            this.saveInstalledPlugins();

            // Complete progress
            progress.complete(`${plugin.name} installed successfully`);

            // Emit events
            if (window.EventBus) {
                window.EventBus.emit('pluginInstalled', installedPlugin);
            }

            if (window.AppState) {
                window.AppState.dispatch({
                    type: 'PLUGIN_INSTALLED',
                    payload: { pluginId, plugin: installedPlugin }
                });
            }

            return installedPlugin;

        } catch (error) {
            if (window.NotificationManager) {
                window.NotificationManager.error('Installation Failed', error.message);
            }
            throw error;
        }
    }

    /**
     * Start (activate) a plugin
     */
    async startPlugin(pluginId) {
        const plugin = this.installedPlugins.get(pluginId);
        if (!plugin) {
            throw new Error('Plugin not installed');
        }

        if (this.runningPlugins.has(pluginId)) {
            throw new Error('Plugin is already running');
        }

        try {
            // Initialize plugin
            await this.initializePlugin(plugin);
            
            this.runningPlugins.add(pluginId);
            plugin.lastStarted = new Date();

            if (window.EventBus) {
                window.EventBus.emit('pluginStarted', plugin);
            }

            if (window.NotificationManager) {
                window.NotificationManager.success('Plugin Started', `${plugin.name} is now running`);
            }

            return plugin;

        } catch (error) {
            if (window.NotificationManager) {
                window.NotificationManager.error('Plugin Start Failed', error.message);
            }
            throw error;
        }
    }

    /**
     * Stop (deactivate) a plugin
     */
    async stopPlugin(pluginId) {
        const plugin = this.installedPlugins.get(pluginId);
        if (!plugin) {
            throw new Error('Plugin not installed');
        }

        if (!this.runningPlugins.has(pluginId)) {
            throw new Error('Plugin is not running');
        }

        try {
            // Cleanup plugin
            await this.cleanupPlugin(plugin);
            
            this.runningPlugins.delete(pluginId);

            if (window.EventBus) {
                window.EventBus.emit('pluginStopped', plugin);
            }

            if (window.NotificationManager) {
                window.NotificationManager.info('Plugin Stopped', `${plugin.name} has been stopped`);
            }

            return plugin;

        } catch (error) {
            if (window.NotificationManager) {
                window.NotificationManager.error('Plugin Stop Failed', error.message);
            }
            throw error;
        }
    }

    /**
     * Remove (uninstall) a plugin
     */
    async removePlugin(pluginId) {
        const plugin = this.installedPlugins.get(pluginId);
        if (!plugin) {
            throw new Error('Plugin not installed');
        }

        try {
            // Stop plugin if running
            if (this.runningPlugins.has(pluginId)) {
                await this.stopPlugin(pluginId);
            }

            // Remove plugin files and data
            await this.uninstallPlugin(plugin);

            this.installedPlugins.delete(pluginId);
            this.saveInstalledPlugins();

            if (window.EventBus) {
                window.EventBus.emit('pluginRemoved', plugin);
            }

            if (window.NotificationManager) {
                window.NotificationManager.success('Plugin Removed', `${plugin.name} has been uninstalled`);
            }

            return plugin;

        } catch (error) {
            if (window.NotificationManager) {
                window.NotificationManager.error('Plugin Removal Failed', error.message);
            }
            throw error;
        }
    }

    /**
     * Get installed plugins
     */
    getInstalledPlugins() {
        return Array.from(this.installedPlugins.values());
    }

    /**
     * Get running plugins
     */
    getRunningPlugins() {
        return Array.from(this.installedPlugins.values()).filter(p => 
            this.runningPlugins.has(p.id)
        );
    }

    /**
     * Get marketplace plugins
     */
    getMarketplacePlugins(category = 'All', search = '') {
        let plugins = this.marketplacePlugins;

        if (category !== 'All') {
            plugins = plugins.filter(p => p.category === category);
        }

        if (search) {
            const searchLower = search.toLowerCase();
            plugins = plugins.filter(p => 
                p.name.toLowerCase().includes(searchLower) ||
                p.description.toLowerCase().includes(searchLower) ||
                p.tags.some(tag => tag.toLowerCase().includes(searchLower))
            );
        }

        return plugins.map(plugin => ({
            ...plugin,
            installed: this.installedPlugins.has(plugin.id),
            running: this.runningPlugins.has(plugin.id)
        }));
    }

    /**
     * Get installed count
     */
    getInstalledCount() {
        return this.installedPlugins.size;
    }

    /**
     * Get active (running) count
     */
    getActiveCount() {
        return this.runningPlugins.size;
    }

    /**
     * Update plugin settings
     */
    updatePluginSettings(pluginId, settings) {
        const plugin = this.installedPlugins.get(pluginId);
        if (!plugin) {
            throw new Error('Plugin not installed');
        }

        plugin.settings = { ...plugin.settings, ...settings };
        this.saveInstalledPlugins();

        if (window.EventBus) {
            window.EventBus.emit('pluginSettingsUpdated', { pluginId, settings });
        }
    }

    /**
     * Get plugin by ID
     */
    getPlugin(pluginId) {
        return this.installedPlugins.get(pluginId);
    }

    // Private methods
    async loadInstalledPlugins() {
        try {
            const saved = localStorage.getItem('omega-installed-plugins');
            if (saved) {
                const plugins = JSON.parse(saved);
                plugins.forEach(plugin => {
                    this.installedPlugins.set(plugin.id, plugin);
                });
            }
        } catch (error) {
            console.warn('Failed to load installed plugins:', error);
        }
    }

    async loadMarketplacePlugins() {
        try {
            // In a real app, this would fetch from a backend API
            const response = await fetch('/api/plugins/marketplace');
            if (response.ok) {
                this.marketplacePlugins = await response.json();
            } else {
                throw new Error('Failed to load marketplace');
            }
        } catch (error) {
            console.warn('Failed to load marketplace, using mock data:', error);
            this.marketplacePlugins = this.getMockMarketplacePlugins();
        }
    }

    getMockMarketplacePlugins() {
        return [
            {
                id: 'code-editor-pro',
                name: 'Code Editor Pro',
                description: 'Advanced code editor with syntax highlighting and IntelliSense',
                version: '2.1.0',
                author: 'DevTools Inc.',
                category: 'Development',
                price: 29.99,
                rating: 4.8,
                downloads: 15420,
                tags: ['editor', 'coding', 'intellisense'],
                icon: 'fas fa-code',
                screenshots: [],
                features: [
                    'Syntax highlighting for 100+ languages',
                    'IntelliSense and autocomplete',
                    'Git integration',
                    'Plugin system'
                ]
            },
            {
                id: 'media-player-ultra',
                name: 'Media Player Ultra',
                description: 'High-performance media player with streaming support',
                version: '1.5.2',
                author: 'MediaSoft',
                category: 'Media',
                price: 0,
                rating: 4.6,
                downloads: 8930,
                tags: ['media', 'video', 'streaming'],
                icon: 'fas fa-play-circle',
                screenshots: [],
                features: [
                    'Support for all video formats',
                    'Hardware acceleration',
                    'Streaming protocols',
                    'Subtitle support'
                ]
            },
            {
                id: 'task-manager-plus',
                name: 'Task Manager Plus',
                description: 'Advanced task and project management tool',
                version: '3.0.1',
                author: 'ProductivityTools',
                category: 'Productivity',
                price: 19.99,
                rating: 4.9,
                downloads: 12150,
                tags: ['tasks', 'productivity', 'projects'],
                icon: 'fas fa-tasks',
                screenshots: [],
                features: [
                    'Project planning and tracking',
                    'Team collaboration',
                    'Time tracking',
                    'Reporting and analytics'
                ]
            },
            {
                id: 'file-explorer-enhanced',
                name: 'File Explorer Enhanced',
                description: 'Powerful file manager with advanced features',
                version: '1.8.4',
                author: 'FileUtils Corp',
                category: 'Utilities',
                price: 14.99,
                rating: 4.7,
                downloads: 20340,
                tags: ['files', 'explorer', 'management'],
                icon: 'fas fa-folder-open',
                screenshots: [],
                features: [
                    'Dual-pane interface',
                    'Advanced search',
                    'Batch operations',
                    'Cloud storage integration'
                ]
            },
            {
                id: 'game-launcher',
                name: 'Game Launcher',
                description: 'Unified game library and launcher',
                version: '2.3.0',
                author: 'GameHub',
                category: 'Games',
                price: 0,
                rating: 4.4,
                downloads: 45670,
                tags: ['games', 'launcher', 'library'],
                icon: 'fas fa-gamepad',
                screenshots: [],
                features: [
                    'Game library management',
                    'Auto-updates',
                    'Achievement tracking',
                    'Social features'
                ]
            },
            {
                id: 'terminal-advanced',
                name: 'Terminal Advanced',
                description: 'Feature-rich terminal emulator',
                version: '1.2.8',
                author: 'TerminalPro',
                category: 'Development',
                price: 9.99,
                rating: 4.5,
                downloads: 7890,
                tags: ['terminal', 'console', 'shell'],
                icon: 'fas fa-terminal',
                screenshots: [],
                features: [
                    'Multiple tabs and panes',
                    'Customizable themes',
                    'SSH integration',
                    'Script automation'
                ]
            }
        ];
    }

    saveInstalledPlugins() {
        try {
            const plugins = Array.from(this.installedPlugins.values());
            localStorage.setItem('omega-installed-plugins', JSON.stringify(plugins));
        } catch (error) {
            console.warn('Failed to save installed plugins:', error);
        }
    }

    async simulateInstallation(plugin, progress) {
        const steps = [
            { message: 'Downloading plugin...', progress: 20 },
            { message: 'Verifying package...', progress: 40 },
            { message: 'Installing dependencies...', progress: 60 },
            { message: 'Configuring plugin...', progress: 80 },
            { message: 'Finalizing installation...', progress: 95 }
        ];

        for (const step of steps) {
            await new Promise(resolve => setTimeout(resolve, 500));
            progress.updateProgress(step.progress, step.message);
        }

        await new Promise(resolve => setTimeout(resolve, 300));
        progress.updateProgress(100, 'Installation complete');
    }

    async initializePlugin(plugin) {
        // Simulate plugin initialization
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // In a real implementation, this would:
        // 1. Load plugin code
        // 2. Create plugin context
        // 3. Initialize plugin APIs
        // 4. Start plugin lifecycle
        
        console.log(`Initialized plugin: ${plugin.name}`);
    }

    async cleanupPlugin(plugin) {
        // Simulate plugin cleanup
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // In a real implementation, this would:
        // 1. Stop plugin processes
        // 2. Clean up resources
        // 3. Remove event listeners
        // 4. Save plugin state
        
        console.log(`Cleaned up plugin: ${plugin.name}`);
    }

    async uninstallPlugin(plugin) {
        // Simulate plugin uninstallation
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // In a real implementation, this would:
        // 1. Remove plugin files
        // 2. Clean up plugin data
        // 3. Remove registry entries
        // 4. Clean up dependencies
        
        console.log(`Uninstalled plugin: ${plugin.name}`);
    }

    bindEvents() {
        // Listen for application events
        if (window.EventBus) {
            window.EventBus.on('applicationShutdown', () => {
                // Stop all running plugins
                this.runningPlugins.forEach(pluginId => {
                    this.stopPlugin(pluginId).catch(console.error);
                });
            });
        }
    }
}

// Global instance
window.PluginMarketplace = new PluginMarketplace();

export default PluginMarketplace;
