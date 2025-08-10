export function renderSettings(root, state) {
  // Ensure state structure exists
  if (!state || !state.data) {
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized</div>';
    return;
  }
  
  // Advanced settings with comprehensive system configuration
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- Settings Status Header -->
      <div class="settings-status-header">
        <div class="settings-overview">
          <div class="settings-metric">
            <i class="fas fa-cog"></i>
            <span>Categories: <strong>6</strong></span>
          </div>
          <div class="settings-metric">
            <i class="fas fa-save"></i>
            <span>Auto-save: <strong>Enabled</strong></span>
          </div>
          <div class="settings-metric">
            <i class="fas fa-shield-alt"></i>
            <span>Security: <strong>High</strong></span>
          </div>
          <div class="settings-metric">
            <i class="fas fa-sync"></i>
            <span>Last Sync: <strong>2m ago</strong></span>
          </div>
        </div>
        <div class="settings-actions">
          <button onclick="exportSettings()" class="settings-btn">
            <i class="fas fa-download"></i>
            Export
          </button>
          <button onclick="importSettings()" class="settings-btn">
            <i class="fas fa-upload"></i>
            Import
          </button>
          <button onclick="resetToDefaults()" class="settings-btn danger">
            <i class="fas fa-undo"></i>
            Reset
          </button>
        </div>
      </div>

      <!-- Settings Categories -->
      <div class="settings-categories">
        <button class="settings-category active" onclick="switchSettingsCategory('system')">
          <i class="fas fa-desktop"></i>
          <span>System</span>
        </button>
        <button class="settings-category" onclick="switchSettingsCategory('preferences')">
          <i class="fas fa-user-cog"></i>
          <span>Preferences</span>
        </button>
        <button class="settings-category" onclick="switchSettingsCategory('security')">
          <i class="fas fa-shield-alt"></i>
          <span>Security</span>
        </button>
        <button class="settings-category" onclick="switchSettingsCategory('network')">
          <i class="fas fa-network-wired"></i>
          <span>Network</span>
        </button>
        <button class="settings-category" onclick="switchSettingsCategory('performance')">
          <i class="fas fa-tachometer-alt"></i>
          <span>Performance</span>
        </button>
        <button class="settings-category" onclick="switchSettingsCategory('backup')">
          <i class="fas fa-cloud-upload-alt"></i>
          <span>Backup</span>
        </button>
      </div>

      <!-- Settings Content -->
      <div class="settings-content">
        <!-- System Settings -->
        <div id="settings-system" class="settings-view active">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>System Configuration</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>System Theme</span>
                    <small>Choose system appearance</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('theme', this.value)">
                    <option value="dark">Dark Theme</option>
                    <option value="light">Light Theme</option>
                    <option value="auto">Auto (System)</option>
                  </select>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Language</span>
                    <small>System language preference</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('language', this.value)">
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                  </select>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Auto-start on Boot</span>
                    <small>Launch Omega at system startup</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('autoStart', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Minimize to Tray</span>
                    <small>Keep running in system tray when closed</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('minimizeToTray', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Hardware Settings</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>GPU Acceleration</span>
                    <small>Enable hardware GPU acceleration</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('gpuAcceleration', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Hardware Monitoring</span>
                    <small>Monitor CPU, GPU, and memory usage</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('hardwareMonitoring', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Power Management</span>
                    <small>Optimize for power efficiency</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('powerMode', this.value)">
                    <option value="performance">Performance</option>
                    <option value="balanced">Balanced</option>
                    <option value="power-saver">Power Saver</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- User Preferences -->
        <div id="settings-preferences" class="settings-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>Interface Preferences</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>UI Scale</span>
                    <small>Adjust interface scaling</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="75" max="150" value="100" class="range-slider" 
                           onchange="updateSetting('uiScale', this.value)">
                    <span class="range-value">100%</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Animation Speed</span>
                    <small>Control UI animation timing</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="0.5" max="2" step="0.1" value="1" class="range-slider" 
                           onchange="updateSetting('animationSpeed', this.value)">
                    <span class="range-value">1.0x</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Show Tooltips</span>
                    <small>Display helpful tooltips on hover</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('showTooltips', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Compact Mode</span>
                    <small>Use compact interface layout</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('compactMode', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Notification Preferences</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Desktop Notifications</span>
                    <small>Show system notifications on desktop</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('desktopNotifications', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Sound Notifications</span>
                    <small>Play sound for notifications</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('soundNotifications', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Notification Duration</span>
                    <small>How long notifications stay visible</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('notificationDuration', this.value)">
                    <option value="3000">3 seconds</option>
                    <option value="5000" selected>5 seconds</option>
                    <option value="8000">8 seconds</option>
                    <option value="0">Until dismissed</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Security Settings -->
        <div id="settings-security" class="settings-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>Authentication</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Two-Factor Authentication</span>
                    <small>Enable 2FA for enhanced security</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('twoFactorAuth', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Session Timeout</span>
                    <small>Auto-logout after inactivity</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('sessionTimeout', this.value)">
                    <option value="15">15 minutes</option>
                    <option value="30" selected>30 minutes</option>
                    <option value="60">1 hour</option>
                    <option value="0">Never</option>
                  </select>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Biometric Login</span>
                    <small>Use fingerprint or face recognition</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('biometricLogin', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Privacy & Data</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Data Encryption</span>
                    <small>Encrypt all stored data</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('dataEncryption', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Analytics Collection</span>
                    <small>Allow anonymous usage analytics</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('analyticsCollection', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Remote Access</span>
                    <small>Allow remote connections</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('remoteAccess', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Network Settings -->
        <div id="settings-network" class="settings-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>Network Configuration</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Network Mode</span>
                    <small>Configure network connection type</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('networkMode', this.value)">
                    <option value="auto" selected>Automatic</option>
                    <option value="ethernet">Ethernet Only</option>
                    <option value="wifi">WiFi Only</option>
                    <option value="cellular">Cellular</option>
                  </select>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Proxy Settings</span>
                    <small>Configure proxy server</small>
                  </div>
                  <div class="setting-input-group">
                    <input type="text" placeholder="proxy.example.com:8080" class="setting-input" 
                           onchange="updateSetting('proxyServer', this.value)">
                    <button onclick="testProxyConnection()" class="setting-test-btn">Test</button>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>DNS Servers</span>
                    <small>Custom DNS configuration</small>
                  </div>
                  <div class="setting-input-group">
                    <input type="text" placeholder="1.1.1.1, 8.8.8.8" class="setting-input" 
                           onchange="updateSetting('dnsServers', this.value)">
                    <button onclick="testDNSServers()" class="setting-test-btn">Test</button>
                  </div>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Bandwidth & Limits</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Bandwidth Limit</span>
                    <small>Maximum network usage (MB/s)</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="1" max="1000" value="100" class="range-slider" 
                           onchange="updateSetting('bandwidthLimit', this.value)">
                    <span class="range-value">100 MB/s</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Connection Timeout</span>
                    <small>Network request timeout (seconds)</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="5" max="60" value="30" class="range-slider" 
                           onchange="updateSetting('connectionTimeout', this.value)">
                    <span class="range-value">30s</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Auto-reconnect</span>
                    <small>Automatically reconnect on network failure</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('autoReconnect', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Performance Settings -->
        <div id="settings-performance" class="settings-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>Resource Management</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>CPU Usage Limit</span>
                    <small>Maximum CPU usage percentage</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="20" max="100" value="80" class="range-slider" 
                           onchange="updateSetting('cpuLimit', this.value)">
                    <span class="range-value">80%</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Memory Usage Limit</span>
                    <small>Maximum memory usage (GB)</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="1" max="32" value="8" class="range-slider" 
                           onchange="updateSetting('memoryLimit', this.value)">
                    <span class="range-value">8 GB</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Auto-optimization</span>
                    <small>Automatically optimize resource usage</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('autoOptimization', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Monitoring & Alerts</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Performance Monitoring</span>
                    <small>Enable real-time performance tracking</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('performanceMonitoring', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Alert Threshold</span>
                    <small>CPU/Memory threshold for alerts</small>
                  </div>
                  <div class="setting-range">
                    <input type="range" min="50" max="95" value="85" class="range-slider" 
                           onchange="updateSetting('alertThreshold', this.value)">
                    <span class="range-value">85%</span>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Performance Logging</span>
                    <small>Log performance metrics to file</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('performanceLogging', this.checked)">
                    <span class="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Backup Settings -->
        <div id="settings-backup" class="settings-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="settings-section">
              <h5>Backup Configuration</h5>
              <div class="settings-group">
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Auto Backup</span>
                    <small>Automatically backup system configuration</small>
                  </div>
                  <label class="toggle-switch">
                    <input type="checkbox" onchange="updateSetting('autoBackup', this.checked)" checked>
                    <span class="toggle-slider"></span>
                  </label>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Backup Frequency</span>
                    <small>How often to create backups</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('backupFrequency', this.value)">
                    <option value="daily" selected>Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                    <option value="manual">Manual Only</option>
                  </select>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Backup Location</span>
                    <small>Where to store backup files</small>
                  </div>
                  <div class="setting-input-group">
                    <input type="text" placeholder="/path/to/backup/location" class="setting-input" 
                           onchange="updateSetting('backupLocation', this.value)">
                    <button onclick="selectBackupLocation()" class="setting-test-btn">Browse</button>
                  </div>
                </div>
                <div class="setting-item">
                  <div class="setting-label">
                    <span>Retention Period</span>
                    <small>How long to keep old backups</small>
                  </div>
                  <select class="setting-control" onchange="updateSetting('retentionPeriod', this.value)">
                    <option value="7">7 days</option>
                    <option value="30" selected>30 days</option>
                    <option value="90">90 days</option>
                    <option value="365">1 year</option>
                  </select>
                </div>
              </div>
            </div>
            <div class="settings-section">
              <h5>Backup Status</h5>
              <div class="backup-status">
                <div class="backup-info">
                  <div class="backup-stat">
                    <span class="stat-label">Last Backup:</span>
                    <span class="stat-value">2024-01-15 14:30</span>
                  </div>
                  <div class="backup-stat">
                    <span class="stat-label">Backup Size:</span>
                    <span class="stat-value">2.4 GB</span>
                  </div>
                  <div class="backup-stat">
                    <span class="stat-label">Total Backups:</span>
                    <span class="stat-value">15</span>
                  </div>
                  <div class="backup-stat">
                    <span class="stat-label">Next Backup:</span>
                    <span class="stat-value">Tomorrow 14:30</span>
                  </div>
                </div>
                <div class="backup-actions">
                  <button onclick="createBackupNow()" class="backup-action-btn primary">
                    <i class="fas fa-cloud-upload-alt"></i>
                    Backup Now
                  </button>
                  <button onclick="restoreFromBackup()" class="backup-action-btn">
                    <i class="fas fa-cloud-download-alt"></i>
                    Restore
                  </button>
                  <button onclick="viewBackupHistory()" class="backup-action-btn">
                    <i class="fas fa-history"></i>
                    History
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize settings with current values
  initializeSettings(state);
}

function initializeSettings(state) {
  // Load current settings from state or defaults
  const settings = state.settings || {};
  
  // Update UI controls with current values
  updateSettingsUI(settings);
}

function updateSettingsUI(settings) {
  // Update all form controls with current settings values
  Object.keys(settings).forEach(key => {
    const control = document.querySelector(`[onchange*="${key}"]`);
    if (control) {
      if (control.type === 'checkbox') {
        control.checked = settings[key];
      } else {
        control.value = settings[key];
      }
    }
  });
}

// Global action functions
window.switchSettingsCategory = (category) => {
  // Remove active class from all categories and views
  document.querySelectorAll('.settings-category').forEach(cat => cat.classList.remove('active'));
  document.querySelectorAll('.settings-view').forEach(view => view.classList.remove('active'));
  
  // Add active class to selected category
  document.querySelector(`[onclick="switchSettingsCategory('${category}')"]`).classList.add('active');
  
  // Show appropriate view
  document.getElementById(`settings-${category}`).classList.add('active');
};

window.updateSetting = async (key, value) => {
  try {
    // Update setting value
    console.log(`Updating setting: ${key} = ${value}`);
    
    // Update range display values
    if (event.target.type === 'range') {
      const rangeValue = event.target.parentElement.querySelector('.range-value');
      if (rangeValue) {
        if (key === 'uiScale') {
          rangeValue.textContent = `${value}%`;
        } else if (key === 'animationSpeed') {
          rangeValue.textContent = `${value}x`;
        } else if (key === 'bandwidthLimit') {
          rangeValue.textContent = `${value} MB/s`;
        } else if (key === 'connectionTimeout') {
          rangeValue.textContent = `${value}s`;
        } else if (key === 'memoryLimit') {
          rangeValue.textContent = `${value} GB`;
        } else {
          rangeValue.textContent = `${value}%`;
        }
      }
    }
    
    window.notify('success', 'Settings', `${key} updated successfully`);
  } catch (e) {
    window.notify('error', 'Settings', e.message);
  }
};

window.exportSettings = async () => {
  try {
    window.notify('info', 'Settings', 'Exporting configuration...');
    // Settings export logic would go here
  } catch (e) {
    window.notify('error', 'Settings', e.message);
  }
};

window.importSettings = async () => {
  try {
    window.notify('info', 'Settings', 'Importing configuration...');
    // Settings import logic would go here
  } catch (e) {
    window.notify('error', 'Settings', e.message);
  }
};

window.resetToDefaults = async () => {
  try {
    const confirmed = confirm('Are you sure you want to reset all settings to defaults? This action cannot be undone.');
    if (confirmed) {
      window.notify('info', 'Settings', 'Resetting to default configuration...');
      // Reset logic would go here
    }
  } catch (e) {
    window.notify('error', 'Settings', e.message);
  }
};

window.createBackupNow = async () => {
  try {
    window.notify('info', 'Settings', 'Creating backup...');
    // Backup creation logic would go here
  } catch (e) {
    window.notify('error', 'Settings', e.message);
  }
};

// Add settings-specific CSS
if (!document.getElementById('settings-styles')) {
  const style = document.createElement('style');
  style.id = 'settings-styles';
  style.textContent = `
    .settings-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .settings-overview {
      display: flex;
      gap: 24px;
    }
    
    .settings-metric {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .settings-metric i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .settings-actions {
      display: flex;
      gap: 8px;
    }
    
    .settings-btn {
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
    
    .settings-btn.danger {
      background: var(--omega-red);
      color: var(--omega-white);
    }
    
    .settings-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .settings-categories {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .settings-category {
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
    
    .settings-category:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .settings-category.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .settings-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .settings-view {
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
    
    .settings-view.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .settings-section {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 16px;
    }
    
    .settings-section h5 {
      margin: 0 0 16px 0;
      font: 600 14px var(--font-mono);
      color: var(--omega-white);
      letter-spacing: 0.5px;
    }
    
    .settings-group {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .setting-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .setting-item:last-child {
      border-bottom: none;
    }
    
    .setting-label {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    
    .setting-label span {
      font: 600 12px var(--font-mono);
      color: var(--omega-white);
    }
    
    .setting-label small {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .setting-control {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 10px;
      border-radius: 3px;
      font: 400 11px var(--font-mono);
      min-width: 120px;
    }
    
    .setting-input {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 10px;
      border-radius: 3px;
      font: 400 11px var(--font-mono);
      flex: 1;
    }
    
    .setting-input-group {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    
    .setting-test-btn {
      background: var(--omega-cyan);
      color: var(--omega-black);
      border: none;
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 600 10px var(--font-mono);
    }
    
    .setting-range {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    .range-slider {
      flex: 1;
      height: 4px;
      background: var(--omega-dark-4);
      outline: none;
      border-radius: 2px;
    }
    
    .range-value {
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
      min-width: 60px;
      text-align: right;
    }
    
    .toggle-switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    
    .toggle-switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    
    .toggle-slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: var(--omega-dark-4);
      transition: .4s;
      border-radius: 24px;
      border: 1px solid var(--omega-gray-1);
    }
    
    .toggle-slider:before {
      position: absolute;
      content: "";
      height: 16px;
      width: 16px;
      left: 3px;
      bottom: 3px;
      background-color: var(--omega-white);
      transition: .4s;
      border-radius: 50%;
    }
    
    input:checked + .toggle-slider {
      background-color: var(--omega-cyan);
    }
    
    input:checked + .toggle-slider:before {
      transform: translateX(20px);
      background-color: var(--omega-black);
    }
  `;
  document.head.appendChild(style);
}
