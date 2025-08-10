export function renderPlugins(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.plugins) {
    console.log('[Plugins] State structure missing:', { state, data: state?.data, plugins: state?.data?.plugins });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  // Advanced plugin management system
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- Plugin Status Header -->
      <div class="plugins-status-header">
        <div class="plugins-overview">
          <div class="plugin-metric">
            <i class="fas fa-puzzle-piece"></i>
            <span>Installed: <strong id="installed-count">0</strong></span>
          </div>
          <div class="plugin-metric">
            <i class="fas fa-cloud-download-alt"></i>
            <span>Available: <strong id="available-count">0</strong></span>
          </div>
          <div class="plugin-metric">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Updates: <strong id="updates-count">3</strong></span>
          </div>
          <div class="plugin-metric">
            <i class="fas fa-code"></i>
            <span>In Development: <strong id="dev-count">1</strong></span>
          </div>
        </div>
        <div class="plugins-actions">
          <button onclick="refreshPluginMarketplace()" class="plugin-btn">
            <i class="fas fa-sync"></i>
            Refresh
          </button>
          <button onclick="uploadPlugin()" class="plugin-btn">
            <i class="fas fa-upload"></i>
            Upload
          </button>
          <button onclick="createPlugin()" class="plugin-btn primary">
            <i class="fas fa-plus"></i>
            Create Plugin
          </button>
        </div>
      </div>

      <!-- Plugin Categories -->
      <div class="plugin-categories">
        <button class="plugin-category active" onclick="switchPluginCategory('all')">
          <i class="fas fa-th"></i>
          <span>All Plugins</span>
        </button>
        <button class="plugin-category" onclick="switchPluginCategory('installed')">
          <i class="fas fa-check-circle"></i>
          <span>Installed</span>
        </button>
        <button class="plugin-category" onclick="switchPluginCategory('marketplace')">
          <i class="fas fa-store"></i>
          <span>Marketplace</span>
        </button>
        <button class="plugin-category" onclick="switchPluginCategory('updates')">
          <i class="fas fa-arrow-up"></i>
          <span>Updates</span>
        </button>
        <button class="plugin-category" onclick="switchPluginCategory('development')">
          <i class="fas fa-code"></i>
          <span>Development</span>
        </button>
      </div>

      <!-- Plugin Content -->
      <div class="plugins-content">
        <!-- All Plugins View -->
        <div id="plugins-all" class="plugin-view active">
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="plugins-grid-container">
              <div class="plugins-filter-bar">
                <input type="text" id="plugin-search" placeholder="Search plugins..." class="plugin-search">
                <select id="plugin-sort" class="plugin-sort">
                  <option value="name">Sort by Name</option>
                  <option value="rating">Sort by Rating</option>
                  <option value="downloads">Sort by Downloads</option>
                  <option value="updated">Sort by Updated</option>
                </select>
                <select id="plugin-filter" class="plugin-filter">
                  <option value="all">All Categories</option>
                  <option value="desktop">Desktop</option>
                  <option value="development">Development</option>
                  <option value="security">Security</option>
                  <option value="productivity">Productivity</option>
                  <option value="system">System</option>
                </select>
              </div>
              <div class="plugins-grid" id="plugins-grid">
                <!-- Populated by renderPluginsGrid() -->
              </div>
            </div>
            <div class="plugin-details-sidebar">
              <div id="plugin-details-panel" class="plugin-details">
                <div class="no-selection">
                  <i class="fas fa-puzzle-piece"></i>
                  <p>Select a plugin to view details</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Installed Plugins View -->
        <div id="plugins-installed" class="plugin-view">
          <div class="installed-plugins-header">
            <div class="installed-controls">
              <button onclick="updateAllPlugins()" class="installed-btn primary">
                <i class="fas fa-arrow-up"></i>
                Update All
              </button>
              <button onclick="exportPluginsList()" class="installed-btn">
                <i class="fas fa-download"></i>
                Export List
              </button>
            </div>
          </div>
          <div class="installed-plugins-list" id="installed-plugins-list">
            <!-- Populated by renderInstalledPlugins() -->
          </div>
        </div>

        <!-- Marketplace View -->
        <div id="plugins-marketplace" class="plugin-view">
          <div class="marketplace-featured">
            <h5>Featured Plugins</h5>
            <div class="featured-plugins" id="featured-plugins">
              <!-- Populated by renderFeaturedPlugins() -->
            </div>
          </div>
          <div class="marketplace-categories">
            <h5>Browse by Category</h5>
            <div class="category-grid">
              <div class="category-card" onclick="browseCategory('desktop')">
                <i class="fas fa-desktop"></i>
                <span>Desktop Enhancement</span>
                <div class="category-count">12 plugins</div>
              </div>
              <div class="category-card" onclick="browseCategory('development')">
                <i class="fas fa-code"></i>
                <span>Development Tools</span>
                <div class="category-count">8 plugins</div>
              </div>
              <div class="category-card" onclick="browseCategory('security')">
                <i class="fas fa-shield-alt"></i>
                <span>Security</span>
                <div class="category-count">5 plugins</div>
              </div>
              <div class="category-card" onclick="browseCategory('productivity')">
                <i class="fas fa-rocket"></i>
                <span>Productivity</span>
                <div class="category-count">15 plugins</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Updates View -->
        <div id="plugins-updates" class="plugin-view">
          <div class="updates-header">
            <h5>Available Updates</h5>
            <button onclick="updateAllAvailable()" class="update-all-btn primary">
              <i class="fas fa-arrow-up"></i>
              Update All (3)
            </button>
          </div>
          <div class="updates-list" id="updates-list">
            <!-- Populated by renderUpdatesAvailable() -->
          </div>
        </div>

        <!-- Development View -->
        <div id="plugins-development" class="plugin-view">
          <div class="development-tools">
            <div class="dev-header">
              <h5>Plugin Development</h5>
              <button onclick="openPluginSDK()" class="dev-btn primary">
                <i class="fas fa-code"></i>
                Open SDK
              </button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
              <div class="dev-projects">
                <h6>Active Projects</h6>
                <div id="dev-projects-list">
                  <!-- Populated by renderDevProjects() -->
                </div>
              </div>
              <div class="dev-templates">
                <h6>Plugin Templates</h6>
                <div class="template-grid">
                  <div class="template-card" onclick="createFromTemplate('basic')">
                    <i class="fas fa-file-code"></i>
                    <span>Basic Plugin</span>
                  </div>
                  <div class="template-card" onclick="createFromTemplate('desktop')">
                    <i class="fas fa-window-maximize"></i>
                    <span>Desktop Widget</span>
                  </div>
                  <div class="template-card" onclick="createFromTemplate('service')">
                    <i class="fas fa-cogs"></i>
                    <span>Background Service</span>
                  </div>
                  <div class="template-card" onclick="createFromTemplate('api')">
                    <i class="fas fa-plug"></i>
                    <span>API Extension</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize plugin data
  updatePluginCounts(state);
  renderPluginsGrid(state);
  renderInstalledPlugins(state);
  renderFeaturedPlugins(state);
  renderUpdatesAvailable(state);
  renderDevProjects(state);
}

function updatePluginCounts(state) {
  const plugins = state.data.plugins || {};
  const installed = plugins.installed || [];
  const available = plugins.available || [];
  
  document.getElementById('installed-count').textContent = installed.length;
  document.getElementById('available-count').textContent = available.length;
}

function renderPluginsGrid(state) {
  const container = document.getElementById('plugins-grid');
  const plugins = state.data.plugins || {};
  const installed = plugins.installed || [];
  const available = plugins.available || [];
  
  // Combine installed and available plugins
  const allPlugins = [
    ...installed.map(p => ({ ...p, status: 'installed' })),
    ...available.map(p => ({ ...p, status: 'available' }))
  ];
  
  // Add sample data if none exists
  const samplePlugins = allPlugins.length > 0 ? allPlugins : [
    { name: 'Desktop Widgets Pro', version: '2.1.0', description: 'Advanced desktop widgets with real-time data', category: 'desktop', rating: 4.8, downloads: 1250, status: 'installed' },
    { name: 'Code Assistant', version: '1.5.2', description: 'AI-powered code completion and analysis', category: 'development', rating: 4.9, downloads: 2100, status: 'installed' },
    { name: 'Security Scanner', version: '3.0.1', description: 'Real-time security threat detection', category: 'security', rating: 4.7, downloads: 890, status: 'available' },
    { name: 'Task Automation', version: '1.8.0', description: 'Automate repetitive system tasks', category: 'productivity', rating: 4.6, downloads: 1570, status: 'available' },
    { name: 'System Monitor Plus', version: '2.3.1', description: 'Enhanced system monitoring and alerts', category: 'system', rating: 4.5, downloads: 980, status: 'installed' }
  ];
  
  container.innerHTML = `
    <div class="plugins-grid-items">
      ${samplePlugins.map(plugin => `
        <div class="plugin-card ${plugin.status}" onclick="selectPlugin('${plugin.name}')">
          <div class="plugin-header">
            <div class="plugin-icon">
              <i class="fas fa-${getPluginIcon(plugin.category)}"></i>
            </div>
            <div class="plugin-status-badge ${plugin.status}">
              ${plugin.status === 'installed' ? 'INSTALLED' : 'AVAILABLE'}
            </div>
          </div>
          <div class="plugin-info">
            <div class="plugin-name">${plugin.name}</div>
            <div class="plugin-version">v${plugin.version}</div>
            <div class="plugin-description">${plugin.description}</div>
            <div class="plugin-meta">
              <div class="plugin-rating">
                <i class="fas fa-star"></i>
                <span>${plugin.rating}</span>
              </div>
              <div class="plugin-downloads">
                <i class="fas fa-download"></i>
                <span>${plugin.downloads}</span>
              </div>
            </div>
          </div>
          <div class="plugin-actions">
            ${plugin.status === 'installed' ? 
              `<button onclick="uninstallPlugin('${plugin.name}')" class="plugin-action-btn danger">
                <i class="fas fa-trash"></i>
                Uninstall
              </button>` :
              `<button onclick="installPlugin('${plugin.name}')" class="plugin-action-btn primary">
                <i class="fas fa-download"></i>
                Install
              </button>`
            }
            <button onclick="viewPluginDetails('${plugin.name}')" class="plugin-action-btn">
              <i class="fas fa-info"></i>
              Details
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function getPluginIcon(category) {
  const icons = {
    desktop: 'window-maximize',
    development: 'code',
    security: 'shield-alt',
    productivity: 'rocket',
    system: 'cogs'
  };
  return icons[category] || 'puzzle-piece';
}

function selectPlugin(pluginName) {
  const detailsPanel = document.getElementById('plugin-details-panel');
  
  // Remove selection from other cards
  document.querySelectorAll('.plugin-card').forEach(card => card.classList.remove('selected'));
  
  // Add selection to clicked card
  event.currentTarget.classList.add('selected');
  
  // Sample plugin details
  detailsPanel.innerHTML = `
    <div class="plugin-detail-header">
      <div class="detail-icon">
        <i class="fas fa-puzzle-piece"></i>
      </div>
      <div class="detail-title">
        <h6>${pluginName}</h6>
        <span class="detail-version">v2.1.0</span>
      </div>
    </div>
    <div class="plugin-detail-content">
      <div class="detail-section">
        <h7>Description</h7>
        <p>Advanced desktop widgets with real-time data visualization and customizable layouts.</p>
      </div>
      <div class="detail-section">
        <h7>Features</h7>
        <ul>
          <li>Real-time system monitoring</li>
          <li>Customizable widget layouts</li>
          <li>Multiple data sources</li>
          <li>Dark/light theme support</li>
        </ul>
      </div>
      <div class="detail-section">
        <h7>Requirements</h7>
        <div class="requirement-item">
          <span class="req-label">Min Version:</span>
          <span class="req-value">Omega 1.0</span>
        </div>
        <div class="requirement-item">
          <span class="req-label">Dependencies:</span>
          <span class="req-value">widget-framework</span>
        </div>
      </div>
      <div class="detail-actions">
        <button onclick="launchPlugin('${pluginName}')" class="detail-action-btn primary">
          <i class="fas fa-play"></i>
          Launch
        </button>
        <button onclick="configurePlugin('${pluginName}')" class="detail-action-btn">
          <i class="fas fa-cog"></i>
          Configure
        </button>
      </div>
    </div>
  `;
}

function renderInstalledPlugins(state) {
  const container = document.getElementById('installed-plugins-list');
  const installed = state.data.plugins?.installed || [];
  
  // Sample installed plugins
  const sampleInstalled = installed.length > 0 ? installed : [
    { name: 'Desktop Widgets Pro', version: '2.1.0', status: 'active', lastUsed: '2024-01-15', updateAvailable: true },
    { name: 'Code Assistant', version: '1.5.2', status: 'active', lastUsed: '2024-01-14', updateAvailable: false },
    { name: 'System Monitor Plus', version: '2.3.1', status: 'inactive', lastUsed: '2024-01-10', updateAvailable: true }
  ];
  
  container.innerHTML = `
    <div class="installed-plugins">
      ${sampleInstalled.map(plugin => `
        <div class="installed-plugin-item">
          <div class="installed-plugin-info">
            <div class="installed-plugin-header">
              <div class="installed-plugin-name">${plugin.name}</div>
              <div class="installed-plugin-version">v${plugin.version}</div>
              <div class="installed-plugin-status ${plugin.status}">${plugin.status.toUpperCase()}</div>
            </div>
            <div class="installed-plugin-meta">
              <span>Last used: ${plugin.lastUsed}</span>
              ${plugin.updateAvailable ? '<span class="update-available">Update available</span>' : ''}
            </div>
          </div>
          <div class="installed-plugin-actions">
            <button onclick="togglePlugin('${plugin.name}')" class="installed-action-btn">
              <i class="fas fa-${plugin.status === 'active' ? 'pause' : 'play'}"></i>
              ${plugin.status === 'active' ? 'Disable' : 'Enable'}
            </button>
            ${plugin.updateAvailable ? 
              `<button onclick="updatePlugin('${plugin.name}')" class="installed-action-btn primary">
                <i class="fas fa-arrow-up"></i>
                Update
              </button>` : ''
            }
            <button onclick="uninstallPlugin('${plugin.name}')" class="installed-action-btn danger">
              <i class="fas fa-trash"></i>
              Uninstall
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderFeaturedPlugins(state) {
  const container = document.getElementById('featured-plugins');
  
  const featured = [
    { name: 'AI Desktop Assistant', description: 'Intelligent desktop automation with natural language processing', rating: 5.0, featured: true },
    { name: 'Advanced Terminal', description: 'Enhanced terminal with syntax highlighting and auto-completion', rating: 4.9, featured: true },
    { name: 'Secure File Vault', description: 'Encrypted file storage with biometric authentication', rating: 4.8, featured: true }
  ];
  
  container.innerHTML = `
    <div class="featured-plugins-carousel">
      ${featured.map(plugin => `
        <div class="featured-plugin-card">
          <div class="featured-plugin-badge">FEATURED</div>
          <div class="featured-plugin-content">
            <h6>${plugin.name}</h6>
            <p>${plugin.description}</p>
            <div class="featured-plugin-rating">
              <i class="fas fa-star"></i>
              <span>${plugin.rating}</span>
            </div>
          </div>
          <div class="featured-plugin-actions">
            <button onclick="installPlugin('${plugin.name}')" class="featured-action-btn primary">
              Install Now
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderUpdatesAvailable(state) {
  const container = document.getElementById('updates-list');
  
  const updates = [
    { name: 'Desktop Widgets Pro', currentVersion: '2.1.0', newVersion: '2.2.0', changeLog: 'Added new weather widget, fixed memory leaks' },
    { name: 'System Monitor Plus', currentVersion: '2.3.1', newVersion: '2.4.0', changeLog: 'Enhanced performance metrics, new GPU monitoring' },
    { name: 'Security Scanner', currentVersion: '3.0.1', newVersion: '3.1.0', changeLog: 'Updated threat database, improved scan speed' }
  ];
  
  container.innerHTML = `
    <div class="updates-items">
      ${updates.map(update => `
        <div class="update-item">
          <div class="update-info">
            <div class="update-header">
              <div class="update-name">${update.name}</div>
              <div class="update-versions">
                <span class="current-version">v${update.currentVersion}</span>
                <i class="fas fa-arrow-right"></i>
                <span class="new-version">v${update.newVersion}</span>
              </div>
            </div>
            <div class="update-changelog">${update.changeLog}</div>
          </div>
          <div class="update-actions">
            <button onclick="viewUpdateDetails('${update.name}')" class="update-action-btn">
              <i class="fas fa-info"></i>
              Details
            </button>
            <button onclick="updatePlugin('${update.name}')" class="update-action-btn primary">
              <i class="fas fa-arrow-up"></i>
              Update
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderDevProjects(state) {
  const container = document.getElementById('dev-projects-list');
  
  const projects = [
    { name: 'Custom Dashboard', status: 'active', progress: 75, lastModified: '2024-01-15' },
    { name: 'Data Connector', status: 'testing', progress: 90, lastModified: '2024-01-14' }
  ];
  
  container.innerHTML = `
    <div class="dev-projects">
      ${projects.map(project => `
        <div class="dev-project-item">
          <div class="dev-project-header">
            <div class="dev-project-name">${project.name}</div>
            <div class="dev-project-status ${project.status}">${project.status.toUpperCase()}</div>
          </div>
          <div class="dev-project-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${project.progress}%;"></div>
            </div>
            <span class="progress-text">${project.progress}%</span>
          </div>
          <div class="dev-project-meta">
            <span>Modified: ${project.lastModified}</span>
          </div>
          <div class="dev-project-actions">
            <button onclick="editProject('${project.name}')" class="dev-action-btn">
              <i class="fas fa-edit"></i>
              Edit
            </button>
            <button onclick="buildProject('${project.name}')" class="dev-action-btn primary">
              <i class="fas fa-hammer"></i>
              Build
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// Global action functions
window.switchPluginCategory = (category) => {
  // Remove active class from all categories and views
  document.querySelectorAll('.plugin-category').forEach(cat => cat.classList.remove('active'));
  document.querySelectorAll('.plugin-view').forEach(view => view.classList.remove('active'));
  
  // Add active class to selected category
  document.querySelector(`[onclick="switchPluginCategory('${category}')"]`).classList.add('active');
  
  // Show appropriate view
  if (category === 'all') {
    document.getElementById('plugins-all').classList.add('active');
  } else {
    document.getElementById(`plugins-${category}`).classList.add('active');
  }
};

window.installPlugin = async (pluginName) => {
  try {
    window.notify('info', 'Plugins', `Installing ${pluginName}...`);
    // Plugin installation would go here
  } catch (e) {
    window.notify('error', 'Plugins', e.message);
  }
};

window.createPlugin = async () => {
  try {
    window.notify('info', 'Plugins', 'Opening plugin development environment...');
    // Plugin creation would go here
  } catch (e) {
    window.notify('error', 'Plugins', e.message);
  }
};

// Add plugins-specific CSS
if (!document.getElementById('plugins-styles')) {
  const style = document.createElement('style');
  style.id = 'plugins-styles';
  style.textContent = `
    .plugins-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .plugins-overview {
      display: flex;
      gap: 24px;
    }
    
    .plugin-metric {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .plugin-metric i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .plugins-actions {
      display: flex;
      gap: 8px;
    }
    
    .plugin-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .plugin-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .plugin-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .plugin-categories {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .plugin-category {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-bottom: none;
      color: var(--omega-light-1);
      padding: 8px 16px;
      cursor: pointer;
      transition: all 0.15s ease;
      font: 400 11px var(--font-mono);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .plugin-category:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .plugin-category.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .plugins-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .plugin-view {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: var(--omega-dark-2);
      padding: 16px;
      opacity: 0;
      transform: translateX(20px);
      transition: all 0.2s ease;
      pointer-events: none;
      overflow: auto;
    }
    
    .plugin-view.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .plugins-grid-items {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    
    .plugin-card {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 16px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    
    .plugin-card:hover {
      border-color: var(--omega-cyan);
      transform: translateY(-2px);
    }
    
    .plugin-card.selected {
      border-color: var(--omega-cyan);
      background: var(--omega-dark-2);
    }
    
    .plugin-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    
    .plugin-icon {
      width: 40px;
      height: 40px;
      background: var(--omega-dark-4);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--omega-cyan);
      font-size: 18px;
    }
    
    .plugin-status-badge {
      padding: 2px 6px;
      border-radius: 2px;
      font: 600 8px var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .plugin-status-badge.installed {
      background: var(--omega-green);
      color: var(--omega-black);
    }
    
    .plugin-status-badge.available {
      background: var(--omega-blue);
      color: var(--omega-white);
    }
    
    .plugin-info {
      flex: 1;
    }
    
    .plugin-name {
      font: 600 14px var(--font-mono);
      color: var(--omega-white);
      margin-bottom: 2px;
    }
    
    .plugin-version {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
      margin-bottom: 8px;
    }
    
    .plugin-description {
      font: 400 11px var(--font-mono);
      color: var(--omega-light-1);
      line-height: 1.4;
      margin-bottom: 8px;
    }
    
    .plugin-meta {
      display: flex;
      gap: 12px;
    }
    
    .plugin-rating,
    .plugin-downloads {
      display: flex;
      align-items: center;
      gap: 4px;
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .plugin-rating i {
      color: var(--omega-yellow);
    }
    
    .plugin-actions {
      display: flex;
      gap: 8px;
    }
    
    .plugin-action-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 9px var(--font-mono);
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .plugin-action-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .plugin-action-btn.danger {
      background: var(--omega-red);
      color: var(--omega-white);
    }
    
    .plugin-action-btn:hover {
      transform: translateY(-1px);
    }
  `;
  document.head.appendChild(style);
}
