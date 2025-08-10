// Plugins Management Component
class PluginsComponent {
    constructor() {
        this.plugins = {
            installed: [
                {
                    id: 'ai-assistant',
                    name: 'AI Assistant Pro',
                    version: '2.1.4',
                    author: 'Omega Team',
                    description: 'Advanced AI-powered virtual assistant with natural language processing',
                    status: 'active',
                    category: 'ai',
                    size: '45.2 MB',
                    lastUpdated: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
                    permissions: ['system', 'network', 'storage'],
                    rating: 4.8
                },
                {
                    id: 'security-monitor',
                    name: 'Security Monitor Suite',
                    version: '1.9.2',
                    author: 'SecureTech',
                    description: 'Real-time security monitoring and threat detection system',
                    status: 'active',
                    category: 'security',
                    size: '23.7 MB',
                    lastUpdated: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
                    permissions: ['system', 'network'],
                    rating: 4.6
                },
                {
                    id: 'performance-analyzer',
                    name: 'Performance Analyzer',
                    version: '3.0.1',
                    author: 'OptimizeTech',
                    description: 'Advanced system performance analysis and optimization tools',
                    status: 'inactive',
                    category: 'utilities',
                    size: '31.4 MB',
                    lastUpdated: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000),
                    permissions: ['system'],
                    rating: 4.2
                }
            ],
            marketplace: [
                {
                    id: 'ml-accelerator',
                    name: 'ML Accelerator Engine',
                    version: '1.5.0',
                    author: 'DeepTech AI',
                    description: 'Hardware-accelerated machine learning inference engine for edge computing',
                    category: 'ai',
                    size: '78.9 MB',
                    price: 'Free',
                    downloads: 15420,
                    rating: 4.9,
                    featured: true
                },
                {
                    id: 'cloud-sync',
                    name: 'Cloud Sync Manager',
                    version: '2.3.1',
                    author: 'CloudTech Solutions',
                    description: 'Seamless multi-cloud storage synchronization and backup solution',
                    category: 'storage',
                    size: '19.8 MB',
                    price: '$9.99',
                    downloads: 8930,
                    rating: 4.5,
                    featured: false
                },
                {
                    id: 'network-optimizer',
                    name: 'Network Optimizer Pro',
                    version: '4.1.2',
                    author: 'NetBoost Inc',
                    description: 'Advanced network traffic optimization and bandwidth management',
                    category: 'network',
                    size: '27.3 MB',
                    price: '$14.99',
                    downloads: 12340,
                    rating: 4.7,
                    featured: true
                }
            ]
        };
        
        this.activeTab = 'installed';
        this.selectedCategory = 'all';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="plugins-container">
                <div class="plugins-header">
                    <h2><i class="fas fa-plug"></i> Plugin Management</h2>
                    <div class="plugins-controls">
                        <button class="btn-primary" onclick="pluginsComponent.openMarketplace()">
                            <i class="fas fa-store"></i> Marketplace
                        </button>
                        <button class="btn-secondary" onclick="pluginsComponent.installFromFile()">
                            <i class="fas fa-upload"></i> Install from File
                        </button>
                        <button class="btn-secondary" onclick="pluginsComponent.refreshPlugins()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="plugins-tabs">
                    <button class="tab-btn ${this.activeTab === 'installed' ? 'active' : ''}" onclick="pluginsComponent.switchTab('installed')">
                        <i class="fas fa-list"></i> Installed (${this.plugins.installed.length})
                    </button>
                    <button class="tab-btn ${this.activeTab === 'marketplace' ? 'active' : ''}" onclick="pluginsComponent.switchTab('marketplace')">
                        <i class="fas fa-store"></i> Marketplace
                    </button>
                    <button class="tab-btn ${this.activeTab === 'developer' ? 'active' : ''}" onclick="pluginsComponent.switchTab('developer')">
                        <i class="fas fa-code"></i> Developer Tools
                    </button>
                </div>
                
                <div class="plugins-filters">
                    <div class="filter-group">
                        <label>Category:</label>
                        <select onchange="pluginsComponent.filterByCategory(this.value)">
                            <option value="all">All Categories</option>
                            <option value="ai">AI & Machine Learning</option>
                            <option value="security">Security</option>
                            <option value="utilities">Utilities</option>
                            <option value="network">Network</option>
                            <option value="storage">Storage</option>
                        </select>
                    </div>
                    <div class="search-group">
                        <input type="text" placeholder="Search plugins..." onkeyup="pluginsComponent.searchPlugins(this.value)">
                    </div>
                </div>
                
                <div class="plugins-content" id="plugins-content">
                    ${this.renderTabContent()}
                </div>
            </div>
        `;
    }

    renderTabContent() {
        switch (this.activeTab) {
            case 'installed': return this.renderInstalled();
            case 'marketplace': return this.renderMarketplace();
            case 'developer': return this.renderDeveloperTools();
            default: return this.renderInstalled();
        }
    }

    renderInstalled() {
        const filteredPlugins = this.selectedCategory === 'all' 
            ? this.plugins.installed 
            : this.plugins.installed.filter(plugin => plugin.category === this.selectedCategory);

        return `
            <div class="installed-plugins">
                <div class="plugins-grid">
                    ${filteredPlugins.map(plugin => `
                        <div class="plugin-card ${plugin.status}">
                            <div class="plugin-header">
                                <div class="plugin-icon">
                                    <i class="fas fa-${this.getCategoryIcon(plugin.category)}"></i>
                                </div>
                                <div class="plugin-info">
                                    <h3>${plugin.name}</h3>
                                    <span class="plugin-version">v${plugin.version}</span>
                                    <span class="plugin-author">by ${plugin.author}</span>
                                </div>
                                <div class="plugin-status">
                                    <span class="status-badge ${plugin.status}">${plugin.status}</span>
                                </div>
                            </div>
                            
                            <div class="plugin-description">
                                <p>${plugin.description}</p>
                            </div>
                            
                            <div class="plugin-details">
                                <div class="detail-row">
                                    <span class="detail-label">Size:</span>
                                    <span class="detail-value">${plugin.size}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">Updated:</span>
                                    <span class="detail-value">${this.formatDate(plugin.lastUpdated)}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">Rating:</span>
                                    <span class="detail-value">
                                        ${this.renderStars(plugin.rating)} ${plugin.rating}
                                    </span>
                                </div>
                            </div>
                            
                            <div class="plugin-permissions">
                                <strong>Permissions:</strong>
                                <div class="permissions-list">
                                    ${plugin.permissions.map(perm => `
                                        <span class="permission-tag">${perm}</span>
                                    `).join('')}
                                </div>
                            </div>
                            
                            <div class="plugin-actions">
                                ${plugin.status === 'active' 
                                    ? `<button class="btn-warning" onclick="pluginsComponent.deactivatePlugin('${plugin.id}')">
                                         <i class="fas fa-pause"></i> Deactivate
                                       </button>`
                                    : `<button class="btn-success" onclick="pluginsComponent.activatePlugin('${plugin.id}')">
                                         <i class="fas fa-play"></i> Activate
                                       </button>`
                                }
                                <button class="btn-secondary" onclick="pluginsComponent.configurePlugin('${plugin.id}')">
                                    <i class="fas fa-cog"></i> Configure
                                </button>
                                <button class="btn-secondary" onclick="pluginsComponent.updatePlugin('${plugin.id}')">
                                    <i class="fas fa-download"></i> Update
                                </button>
                                <button class="btn-danger" onclick="pluginsComponent.uninstallPlugin('${plugin.id}')">
                                    <i class="fas fa-trash"></i> Uninstall
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderMarketplace() {
        const featuredPlugins = this.plugins.marketplace.filter(plugin => plugin.featured);
        const regularPlugins = this.plugins.marketplace.filter(plugin => !plugin.featured);
        
        return `
            <div class="marketplace-section">
                ${featuredPlugins.length > 0 ? `
                    <div class="featured-plugins">
                        <h3><i class="fas fa-star"></i> Featured Plugins</h3>
                        <div class="plugins-grid featured">
                            ${featuredPlugins.map(plugin => this.renderMarketplacePlugin(plugin)).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div class="all-plugins">
                    <h3><i class="fas fa-store"></i> All Plugins</h3>
                    <div class="plugins-grid">
                        ${regularPlugins.map(plugin => this.renderMarketplacePlugin(plugin)).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderMarketplacePlugin(plugin) {
        return `
            <div class="marketplace-plugin-card ${plugin.featured ? 'featured' : ''}">
                <div class="plugin-header">
                    <div class="plugin-icon">
                        <i class="fas fa-${this.getCategoryIcon(plugin.category)}"></i>
                    </div>
                    <div class="plugin-info">
                        <h3>${plugin.name}</h3>
                        <span class="plugin-version">v${plugin.version}</span>
                        <span class="plugin-author">by ${plugin.author}</span>
                    </div>
                    <div class="plugin-price">
                        <span class="price ${plugin.price === 'Free' ? 'free' : 'paid'}">${plugin.price}</span>
                    </div>
                </div>
                
                <div class="plugin-description">
                    <p>${plugin.description}</p>
                </div>
                
                <div class="plugin-stats">
                    <div class="stat-item">
                        <i class="fas fa-download"></i>
                        <span>${plugin.downloads.toLocaleString()} downloads</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-star"></i>
                        <span>${plugin.rating} rating</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-hdd"></i>
                        <span>${plugin.size}</span>
                    </div>
                </div>
                
                <div class="plugin-actions">
                    <button class="btn-primary" onclick="pluginsComponent.installPlugin('${plugin.id}')">
                        <i class="fas fa-download"></i> Install
                    </button>
                    <button class="btn-secondary" onclick="pluginsComponent.viewPluginDetails('${plugin.id}')">
                        <i class="fas fa-info"></i> Details
                    </button>
                </div>
            </div>
        `;
    }

    renderDeveloperTools() {
        return `
            <div class="developer-section">
                <div class="dev-tools-header">
                    <h3><i class="fas fa-code"></i> Plugin Development Tools</h3>
                    <p>Create, test, and deploy custom plugins for Omega SuperDesktop</p>
                </div>
                
                <div class="dev-actions">
                    <div class="action-card">
                        <i class="fas fa-plus-circle"></i>
                        <h4>Create New Plugin</h4>
                        <p>Start developing a new plugin with our template generator</p>
                        <button class="btn-primary" onclick="pluginsComponent.createNewPlugin()">
                            <i class="fas fa-rocket"></i> Start Development
                        </button>
                    </div>
                    
                    <div class="action-card">
                        <i class="fas fa-upload"></i>
                        <h4>Load Plugin Package</h4>
                        <p>Load and test a plugin package from your local development environment</p>
                        <button class="btn-secondary" onclick="pluginsComponent.loadDevPlugin()">
                            <i class="fas fa-folder-open"></i> Load Package
                        </button>
                    </div>
                    
                    <div class="action-card">
                        <i class="fas fa-bug"></i>
                        <h4>Debug Console</h4>
                        <p>Access plugin debugging tools and runtime logs</p>
                        <button class="btn-secondary" onclick="pluginsComponent.openDebugConsole()">
                            <i class="fas fa-terminal"></i> Open Console
                        </button>
                    </div>
                </div>
                
                <div class="dev-resources">
                    <h4>Development Resources</h4>
                    <ul class="resource-list">
                        <li><a href="#" onclick="pluginsComponent.openDocumentation()">
                            <i class="fas fa-book"></i> Plugin API Documentation
                        </a></li>
                        <li><a href="#" onclick="pluginsComponent.openSamples()">
                            <i class="fas fa-code"></i> Sample Plugins & Templates
                        </a></li>
                        <li><a href="#" onclick="pluginsComponent.openSDK()">
                            <i class="fas fa-tools"></i> Development SDK
                        </a></li>
                        <li><a href="#" onclick="pluginsComponent.openCommunity()">
                            <i class="fas fa-users"></i> Developer Community
                        </a></li>
                    </ul>
                </div>
            </div>
        `;
    }

    getCategoryIcon(category) {
        const icons = {
            ai: 'brain',
            security: 'shield-alt',
            utilities: 'tools',
            network: 'network-wired',
            storage: 'database'
        };
        return icons[category] || 'plug';
    }

    renderStars(rating) {
        const fullStars = Math.floor(rating);
        const halfStar = rating % 1 >= 0.5;
        let stars = '';
        
        for (let i = 0; i < fullStars; i++) {
            stars += '<i class="fas fa-star"></i>';
        }
        
        if (halfStar) {
            stars += '<i class="fas fa-star-half-alt"></i>';
        }
        
        const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
        for (let i = 0; i < emptyStars; i++) {
            stars += '<i class="far fa-star"></i>';
        }
        
        return stars;
    }

    formatDate(date) {
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) return 'Today';
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
        return `${Math.floor(diffDays / 30)} months ago`;
    }

    switchTab(tab) {
        this.activeTab = tab;
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.tab-btn').classList.add('active');
        document.getElementById('plugins-content').innerHTML = this.renderTabContent();
    }

    filterByCategory(category) {
        this.selectedCategory = category;
        document.getElementById('plugins-content').innerHTML = this.renderTabContent();
    }

    searchPlugins(query) {
        console.log(`Searching plugins: ${query}`);
        // Implementation for plugin search
    }

    // Action handlers
    openMarketplace() { this.switchTab('marketplace'); }
    installFromFile() { console.log('Installing plugin from file...'); }
    refreshPlugins() { console.log('Refreshing plugins...'); }
    activatePlugin(id) { console.log(`Activating plugin: ${id}`); }
    deactivatePlugin(id) { console.log(`Deactivating plugin: ${id}`); }
    configurePlugin(id) { console.log(`Configuring plugin: ${id}`); }
    updatePlugin(id) { console.log(`Updating plugin: ${id}`); }
    uninstallPlugin(id) { console.log(`Uninstalling plugin: ${id}`); }
    installPlugin(id) { console.log(`Installing plugin: ${id}`); }
    viewPluginDetails(id) { console.log(`Viewing plugin details: ${id}`); }
    createNewPlugin() { console.log('Creating new plugin...'); }
    loadDevPlugin() { console.log('Loading development plugin...'); }
    openDebugConsole() { console.log('Opening debug console...'); }
    openDocumentation() { console.log('Opening documentation...'); }
    openSamples() { console.log('Opening samples...'); }
    openSDK() { console.log('Opening SDK...'); }
    openCommunity() { console.log('Opening community...'); }

    startMonitoring() {
        // Plugin monitoring could be implemented here
        setInterval(() => {
            // Update plugin status, check for updates, etc.
        }, 60000);
    }
}

window.pluginsComponent = new PluginsComponent();
