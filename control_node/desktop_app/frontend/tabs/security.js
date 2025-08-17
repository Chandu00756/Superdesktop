export function renderSecurity(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.security) {
    console.log('[Security] State structure missing:', { state, data: state?.data, security: state?.data?.security });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  // Advanced security management with 6 sub-tabs
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- Security Status Header -->
      <div class="security-status-header">
        <div class="security-overview">
          <div class="security-metric">
            <i class="fas fa-user-shield"></i>
            <span>Signed in: <strong id="whoami-user">loading...</strong></span>
          </div>
          <div class="security-metric">
            <i class="fas fa-id-badge"></i>
            <span>Roles: <strong id="whoami-roles">-</strong></span>
          </div>
          <div class="security-metric">
            <i class="fas fa-shield-alt"></i>
            <span>Security Level: <strong class="security-level high">HIGH</strong></span>
          </div>
          <div class="security-metric">
            <i class="fas fa-users"></i>
            <span>Active Users: <strong id="active-users-count">3</strong></span>
          </div>
          <div class="security-metric">
            <i class="fas fa-certificate"></i>
            <span>Valid Certificates: <strong id="valid-certs-count">12</strong></span>
          </div>
          <div class="security-metric">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Security Alerts: <strong id="security-alerts-count" class="alert-warning">2</strong></span>
          </div>
        </div>
        <div class="security-actions">
          <button onclick="runSecurityScan()" class="security-btn primary">
            <i class="fas fa-search"></i>
            Security Scan
          </button>
          <button onclick="refreshWhoami()" class="security-btn">
            <i class="fas fa-sync"></i>
            Refresh Identity
          </button>
          <button onclick="generateSecurityReport()" class="security-btn">
            <i class="fas fa-file-shield"></i>
            Report
          </button>
          <button onclick="exportSecurityConfig()" class="security-btn">
            <i class="fas fa-download"></i>
            Export
          </button>
        </div>
      </div>

      <!-- Security Sub-tabs -->
      <div class="security-subtabs">
        <button class="security-subtab active" onclick="switchSecurityTab('overview')">
          <i class="fas fa-tachometer-alt"></i>
          <span>Overview</span>
        </button>
        <button class="security-subtab" onclick="switchSecurityTab('users')">
          <i class="fas fa-users"></i>
          <span>Users & Access</span>
        </button>
        <button class="security-subtab" onclick="switchSecurityTab('certificates')">
          <i class="fas fa-certificate"></i>
          <span>Certificates</span>
        </button>
        <button class="security-subtab" onclick="switchSecurityTab('firewall')">
          <i class="fas fa-shield-alt"></i>
          <span>Firewall</span>
        </button>
        <button class="security-subtab" onclick="switchSecurityTab('audit')">
          <i class="fas fa-clipboard-list"></i>
          <span>Audit Log</span>
        </button>
        <button class="security-subtab" onclick="switchSecurityTab('policies')">
          <i class="fas fa-gavel"></i>
          <span>Policies</span>
        </button>
      </div>

      <!-- Security Content Panels -->
      <div class="security-content">
        <!-- Overview Panel -->
        <div id="security-overview" class="security-panel active">
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="security-dashboard">
              <div class="security-widgets">
                <div class="security-widget">
                  <div class="widget-header">
                    <h5>Threat Detection</h5>
                    <div class="widget-status safe">
                      <i class="fas fa-shield-alt"></i>
                      <span>PROTECTED</span>
                    </div>
                  </div>
                  <div class="widget-content">
                    <div class="threat-stats">
                      <div class="threat-stat">
                        <span class="threat-label">Blocked Attacks</span>
                        <span class="threat-value">0</span>
                      </div>
                      <div class="threat-stat">
                        <span class="threat-label">Quarantined Files</span>
                        <span class="threat-value">0</span>
                      </div>
                      <div class="threat-stat">
                        <span class="threat-label">Suspicious Activity</span>
                        <span class="threat-value">0</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div class="security-widget">
                  <div class="widget-header">
                    <h5>Access Control</h5>
                    <div class="widget-status active">
                      <i class="fas fa-lock"></i>
                      <span>ACTIVE</span>
                    </div>
                  </div>
                  <div class="widget-content">
                    <div class="access-stats">
                      <div class="access-stat">
                        <span class="access-label">Failed Logins</span>
                        <span class="access-value warning">3</span>
                      </div>
                      <div class="access-stat">
                        <span class="access-label">Active Sessions</span>
                        <span class="access-value">12</span>
                      </div>
                      <div class="access-stat">
                        <span class="access-label">Permission Changes</span>
                        <span class="access-value">1</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div class="security-widget">
                  <div class="widget-header">
                    <h5>Encryption Status</h5>
                    <div class="widget-status active">
                      <i class="fas fa-key"></i>
                      <span>ENCRYPTED</span>
                    </div>
                  </div>
                  <div class="widget-content">
                    <div class="encryption-info">
                      <div class="encryption-item">
                        <span class="encryption-label">Data at Rest</span>
                        <span class="encryption-status enabled">AES-256</span>
                      </div>
                      <div class="encryption-item">
                        <span class="encryption-label">Data in Transit</span>
                        <span class="encryption-status enabled">TLS 1.3</span>
                      </div>
                      <div class="encryption-item">
                        <span class="encryption-label">Key Rotation</span>
                        <span class="encryption-status enabled">24h</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div class="security-sidebar">
              <div class="security-alerts-panel">
                <h5>Recent Security Events</h5>
                <div id="security-events-list">
                  <!-- Populated by renderSecurityEvents() -->
                </div>
              </div>
              
              <div class="security-compliance">
                <h5>Compliance Status</h5>
                <div class="compliance-items">
                  <div class="compliance-item">
                    <span class="compliance-label">SOC 2</span>
                    <span class="compliance-status compliant">✓</span>
                  </div>
                  <div class="compliance-item">
                    <span class="compliance-label">ISO 27001</span>
                    <span class="compliance-status compliant">✓</span>
                  </div>
                  <div class="compliance-item">
                    <span class="compliance-label">GDPR</span>
                    <span class="compliance-status compliant">✓</span>
                  </div>
                  <div class="compliance-item">
                    <span class="compliance-label">HIPAA</span>
                    <span class="compliance-status non-compliant">✗</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Users & Access Panel -->
        <div id="security-users" class="security-panel">
          <div class="users-header">
            <div class="users-controls">
              <input type="text" id="user-search" placeholder="Search users..." class="user-search">
              <select id="user-filter" class="user-filter">
                <option value="all">All Users</option>
                <option value="admin">Administrators</option>
                <option value="user">Standard Users</option>
                <option value="active">Active Only</option>
              </select>
              <button onclick="addUser()" class="user-btn primary">
                <i class="fas fa-plus"></i>
                Add User
              </button>
              <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
                <input id="rbac-username" placeholder="username" class="user-search" style="width:140px;">
                <input id="rbac-role" placeholder="role (e.g., admin)" class="user-search" style="width:140px;">
                <button onclick="assignRoleUI()" class="user-btn">
                  <i class="fas fa-user-plus"></i>
                  Assign Role
                </button>
                <button onclick="removeRoleUI()" class="user-btn">
                  <i class="fas fa-user-minus"></i>
                  Remove Role
                </button>
              </div>
            </div>
          </div>
          <div id="rbac-banner" style="display:none; margin: 8px 0; padding: 8px; border-radius: 6px;"></div>
          <div class="users-table-container">
            <table class="users-table">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Last Login</th>
                  <th>Permissions</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="users-table-body">
                <!-- Populated by renderUsersTable() -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Certificates Panel -->
        <div id="security-certificates" class="security-panel">
          <div class="certificates-header">
            <div class="certificates-controls">
              <select id="cert-filter" class="cert-filter">
                <option value="all">All Certificates</option>
                <option value="valid">Valid Only</option>
                <option value="expiring">Expiring Soon</option>
                <option value="expired">Expired</option>
              </select>
              <button onclick="generateCertificate()" class="cert-btn primary">
                <i class="fas fa-plus"></i>
                Generate Certificate
              </button>
              <button onclick="importCertificate()" class="cert-btn">
                <i class="fas fa-upload"></i>
                Import
              </button>
            </div>
          </div>
          <div class="certificates-grid">
            <div id="certificates-list">
              <!-- Populated by renderCertificatesList() -->
            </div>
          </div>
        </div>

        <!-- Firewall Panel -->
        <div id="security-firewall" class="security-panel">
          <div class="firewall-header">
            <div class="firewall-status">
              <div class="firewall-indicator active">
                <i class="fas fa-shield-alt"></i>
                <span>Firewall Active</span>
              </div>
              <div class="firewall-stats">
                <span>Blocked: <strong>1,247</strong></span>
                <span>Allowed: <strong>89,432</strong></span>
              </div>
            </div>
            <div class="firewall-controls">
              <button onclick="addFirewallRule()" class="firewall-btn primary">
                <i class="fas fa-plus"></i>
                Add Rule
              </button>
              <button onclick="importFirewallRules()" class="firewall-btn">
                <i class="fas fa-upload"></i>
                Import Rules
              </button>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="firewall-rules-panel">
              <h5>Firewall Rules</h5>
              <div id="firewall-rules-list">
                <!-- Populated by renderFirewallRules() -->
              </div>
            </div>
            <div class="firewall-activity-panel">
              <h5>Recent Activity</h5>
              <div id="firewall-activity-list">
                <!-- Populated by renderFirewallActivity() -->
              </div>
            </div>
          </div>
        </div>

        <!-- Audit Log Panel -->
        <div id="security-audit" class="security-panel">
          <div class="audit-controls">
            <div class="audit-filters">
              <input type="text" id="audit-search" placeholder="Search audit logs..." class="audit-search">
              <select id="audit-category" class="audit-category">
                <option value="all">All Categories</option>
                <option value="authentication">Authentication</option>
                <option value="authorization">Authorization</option>
                <option value="data-access">Data Access</option>
                <option value="system">System Events</option>
              </select>
              <select id="audit-timeframe" class="audit-timeframe">
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="custom">Custom Range</option>
              </select>
            </div>
            <div class="audit-actions">
              <button onclick="exportAuditLog()" class="audit-btn">
                <i class="fas fa-download"></i>
                Export
              </button>
              <button onclick="archiveAuditLog()" class="audit-btn">
                <i class="fas fa-archive"></i>
                Archive
              </button>
            </div>
          </div>
          <div class="audit-log-container">
            <div id="audit-log-list">
              <!-- Populated by renderAuditLog() -->
            </div>
          </div>
        </div>

        <!-- Policies Panel -->
        <div id="security-policies" class="security-panel">
          <div class="policies-header">
            <div class="policies-controls">
              <select id="policy-category" class="policy-category">
                <option value="all">All Policies</option>
                <option value="access">Access Control</option>
                <option value="password">Password Policies</option>
                <option value="encryption">Encryption</option>
                <option value="compliance">Compliance</option>
              </select>
              <button onclick="addPolicy()" class="policy-btn primary">
                <i class="fas fa-plus"></i>
                Add Policy
              </button>
            </div>
          </div>
          <div class="policies-grid">
            <div id="policies-list">
              <!-- Populated by renderPoliciesList() -->
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize security components
  renderSecurityEvents(state);
  renderUsersTable(state);
  renderCertificatesList(state);
  renderFirewallRules(state);
  renderFirewallActivity(state);
  renderAuditLog(state);
  renderPoliciesList(state);

  // RBAC helpers bound to this render context
  function showBanner(msg, isError=false) {
    const el = root.querySelector('#rbac-banner');
    if (!el) return;
    el.style.display = 'block';
    el.textContent = msg;
    el.style.background = isError ? '#ffe6e6' : '#e6ffea';
    el.style.border = `1px solid ${isError ? '#ffb3b3' : '#b3ffcc'}`;
    el.style.color = '#333';
    setTimeout(() => {
      el.style.display = 'none';
    }, 3500);
  }

  async function refreshWhoami() {
    const userEl = root.querySelector('#whoami-user');
    const rolesEl = root.querySelector('#whoami-roles');
    try {
      userEl.textContent = 'loading...';
      rolesEl.textContent = '-';
  const info = await window.api?.whoami?.();
      userEl.textContent = info?.user || 'unknown';
      rolesEl.textContent = Array.isArray(info?.roles) ? (info.roles.join(', ') || 'none') : 'none';
    } catch (e) {
      userEl.textContent = 'error';
      rolesEl.textContent = '—';
      showBanner(`Failed to load identity: ${e?.message || e}`, true);
    }
  }

  async function assignRoleUI() {
    const u = root.querySelector('#rbac-username')?.value?.trim();
    const r = root.querySelector('#rbac-role')?.value?.trim();
    if (!u || !r) return showBanner('Username and role required.', true);
    try {
  await window.api?.assignRole?.(u, r);
      showBanner(`Assigned role '${r}' to ${u}.`);
      refreshWhoami();
    } catch (e) {
      showBanner(`Assign failed: ${e?.message || e}`, true);
    }
  }

  async function removeRoleUI() {
    const u = root.querySelector('#rbac-username')?.value?.trim();
    const r = root.querySelector('#rbac-role')?.value?.trim();
    if (!u || !r) return showBanner('Username and role required.', true);
    try {
  await window.api?.removeRole?.(u, r);
      showBanner(`Removed role '${r}' from ${u}.`);
      refreshWhoami();
    } catch (e) {
      showBanner(`Remove failed: ${e?.message || e}`, true);
    }
  }

  // expose for inline onclick handlers
  window.refreshWhoami = refreshWhoami;
  window.assignRoleUI = assignRoleUI;
  window.removeRoleUI = removeRoleUI;

  // initial identity fetch
  refreshWhoami();
}

function renderSecurityEvents(state) {
  const container = document.getElementById('security-events-list');
  
  const events = [
    { time: '14:32', type: 'warning', event: 'Failed login attempt from 192.168.1.50' },
    { time: '14:28', type: 'info', event: 'User admin logged in successfully' },
    { time: '14:25', type: 'error', event: 'Certificate expiration warning for web-cert' },
    { time: '14:20', type: 'info', event: 'Firewall rule updated' },
    { time: '14:15', type: 'success', event: 'Security scan completed successfully' }
  ];
  
  container.innerHTML = `
    <div class="security-events">
      ${events.map(event => `
        <div class="security-event ${event.type}">
          <div class="event-time">${event.time}</div>
          <div class="event-icon">
            <i class="fas fa-${event.type === 'success' ? 'check-circle' : event.type === 'warning' ? 'exclamation-triangle' : event.type === 'error' ? 'times-circle' : 'info-circle'}"></i>
          </div>
          <div class="event-message">${event.event}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderUsersTable(state) {
  const tbody = document.getElementById('users-table-body');
  const users = state.data.security?.users || [];
  
  // Add sample data if none exists
  const sampleUsers = users.length > 0 ? users : [
    { username: 'admin', role: 'Administrator', status: 'active', last_login: '2024-01-15 14:30', permissions: ['all'] },
    { username: 'user1', role: 'Standard User', status: 'active', last_login: '2024-01-15 09:15', permissions: ['read', 'write'] },
    { username: 'operator', role: 'Operator', status: 'active', last_login: '2024-01-14 16:45', permissions: ['read', 'execute'] },
    { username: 'guest', role: 'Guest', status: 'inactive', last_login: '2024-01-10 11:20', permissions: ['read'] }
  ];
  
  tbody.innerHTML = sampleUsers.map(user => `
    <tr class="user-row">
      <td>
        <div class="user-info">
          <div class="user-avatar">
            <i class="fas fa-user"></i>
          </div>
          <span>${user.username}</span>
        </div>
      </td>
      <td>
        <span class="user-role ${user.role.toLowerCase().replace(' ', '-')}">${user.role}</span>
      </td>
      <td>
        <span class="user-status ${user.status}">
          ${user.status.toUpperCase()}
        </span>
      </td>
      <td>${user.last_login}</td>
      <td>
        <div class="user-permissions">
          ${user.permissions.slice(0, 2).map(perm => `<span class="permission-badge">${perm}</span>`).join('')}
          ${user.permissions.length > 2 ? `<span class="permission-more">+${user.permissions.length - 2}</span>` : ''}
        </div>
      </td>
      <td>
        <div class="user-actions">
          <button onclick="editUser('${user.username}')" class="user-action-btn">
            <i class="fas fa-edit"></i>
          </button>
          <button onclick="resetPassword('${user.username}')" class="user-action-btn">
            <i class="fas fa-key"></i>
          </button>
          <button onclick="suspendUser('${user.username}')" class="user-action-btn ${user.status === 'active' ? 'warning' : ''}">
            <i class="fas fa-${user.status === 'active' ? 'pause' : 'play'}"></i>
          </button>
        </div>
      </td>
    </tr>
  `).join('');
}

function renderCertificatesList(state) {
  const container = document.getElementById('certificates-list');
  const certs = state.data.security?.certificates || [];
  
  // Add sample data if none exists
  const sampleCerts = certs.length > 0 ? certs : [
    { name: 'omega-control-cert', type: 'TLS Server', expires: '2024-12-15', status: 'valid', issuer: 'Omega CA' },
    { name: 'api-gateway-cert', type: 'TLS Server', expires: '2024-11-30', status: 'valid', issuer: 'Omega CA' },
    { name: 'client-auth-cert', type: 'Client Auth', expires: '2024-02-28', status: 'expiring', issuer: 'Omega CA' },
    { name: 'old-service-cert', type: 'TLS Server', expires: '2024-01-01', status: 'expired', issuer: 'Omega CA' }
  ];
  
  container.innerHTML = `
    <div class="certificates-items">
      ${sampleCerts.map(cert => `
        <div class="certificate-card ${cert.status}">
          <div class="cert-header">
            <div class="cert-name">${cert.name}</div>
            <div class="cert-status ${cert.status}">
              ${cert.status.toUpperCase()}
            </div>
          </div>
          <div class="cert-details">
            <div class="cert-detail">
              <span class="detail-label">Type:</span>
              <span class="detail-value">${cert.type}</span>
            </div>
            <div class="cert-detail">
              <span class="detail-label">Expires:</span>
              <span class="detail-value">${cert.expires}</span>
            </div>
            <div class="cert-detail">
              <span class="detail-label">Issuer:</span>
              <span class="detail-value">${cert.issuer}</span>
            </div>
          </div>
          <div class="cert-actions">
            <button onclick="viewCertificate('${cert.name}')" class="cert-action-btn">
              <i class="fas fa-eye"></i>
              View
            </button>
            <button onclick="renewCertificate('${cert.name}')" class="cert-action-btn">
              <i class="fas fa-redo"></i>
              Renew
            </button>
            <button onclick="revokeCertificate('${cert.name}')" class="cert-action-btn danger">
              <i class="fas fa-ban"></i>
              Revoke
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderFirewallRules(state) {
  const container = document.getElementById('firewall-rules-list');
  
  const rules = [
    { id: 1, action: 'ALLOW', protocol: 'TCP', source: 'any', destination: '22', description: 'SSH Access' },
    { id: 2, action: 'ALLOW', protocol: 'TCP', source: 'any', destination: '443', description: 'HTTPS Traffic' },
    { id: 3, action: 'DENY', protocol: 'TCP', source: 'any', destination: '23', description: 'Block Telnet' },
    { id: 4, action: 'ALLOW', protocol: 'UDP', source: '192.168.1.0/24', destination: '53', description: 'DNS Queries' }
  ];
  
  container.innerHTML = `
    <div class="firewall-rules">
      ${rules.map(rule => `
        <div class="firewall-rule">
          <div class="rule-header">
            <span class="rule-action ${rule.action.toLowerCase()}">${rule.action}</span>
            <span class="rule-protocol">${rule.protocol}</span>
            <div class="rule-controls">
              <button onclick="editFirewallRule(${rule.id})" class="rule-btn">
                <i class="fas fa-edit"></i>
              </button>
              <button onclick="deleteFirewallRule(${rule.id})" class="rule-btn danger">
                <i class="fas fa-trash"></i>
              </button>
            </div>
          </div>
          <div class="rule-details">
            <div class="rule-flow">
              <span class="rule-source">${rule.source}</span>
              <i class="fas fa-arrow-right"></i>
              <span class="rule-destination">${rule.destination}</span>
            </div>
          </div>
          <div class="rule-description">${rule.description}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderFirewallActivity(state) {
  const container = document.getElementById('firewall-activity-list');
  
  const activities = [
    { time: '14:32:15', action: 'BLOCKED', source: '192.168.1.100', destination: '23', protocol: 'TCP' },
    { time: '14:31:42', action: 'ALLOWED', source: '10.0.0.50', destination: '443', protocol: 'TCP' },
    { time: '14:30:28', action: 'BLOCKED', source: '172.16.0.200', destination: '135', protocol: 'TCP' },
    { time: '14:29:15', action: 'ALLOWED', source: '192.168.1.25', destination: '22', protocol: 'TCP' }
  ];
  
  container.innerHTML = `
    <div class="firewall-activities">
      ${activities.map(activity => `
        <div class="firewall-activity ${activity.action.toLowerCase()}">
          <div class="activity-time">${activity.time}</div>
          <div class="activity-action ${activity.action.toLowerCase()}">
            ${activity.action}
          </div>
          <div class="activity-details">
            <div class="activity-flow">
              <span class="activity-source">${activity.source}</span>
              <i class="fas fa-arrow-right"></i>
              <span class="activity-destination">${activity.destination}</span>
            </div>
            <div class="activity-protocol">${activity.protocol}</div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderAuditLog(state) {
  const container = document.getElementById('audit-log-list');
  
  const logs = [
    { timestamp: '2024-01-15 14:32:15', category: 'authentication', user: 'admin', action: 'login_success', details: 'Successful login from 192.168.1.100' },
    { timestamp: '2024-01-15 14:30:42', category: 'authorization', user: 'user1', action: 'permission_denied', details: 'Access denied to /admin/users' },
    { timestamp: '2024-01-15 14:28:30', category: 'data-access', user: 'operator', action: 'file_access', details: 'Accessed sensitive file: /data/config.json' },
    { timestamp: '2024-01-15 14:25:15', category: 'system', user: 'system', action: 'cert_expiry_warning', details: 'Certificate client-auth-cert expires in 30 days' }
  ];
  
  container.innerHTML = `
    <div class="audit-entries">
      ${logs.map(log => `
        <div class="audit-entry">
          <div class="audit-timestamp">${log.timestamp}</div>
          <div class="audit-category ${log.category}">${log.category.replace('-', ' ').toUpperCase()}</div>
          <div class="audit-user">${log.user}</div>
          <div class="audit-action">${log.action.replace('_', ' ').toUpperCase()}</div>
          <div class="audit-details">${log.details}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderPoliciesList(state) {
  const container = document.getElementById('policies-list');
  
  const policies = [
    { name: 'Password Policy', category: 'password', status: 'active', description: 'Minimum 12 characters, special characters required', updated: '2024-01-10' },
    { name: 'Access Control Policy', category: 'access', status: 'active', description: 'Role-based access control with least privilege principle', updated: '2024-01-08' },
    { name: 'Encryption Policy', category: 'encryption', status: 'active', description: 'AES-256 encryption for all sensitive data', updated: '2024-01-05' },
    { name: 'Data Retention Policy', category: 'compliance', status: 'review', description: 'Audit logs retained for 7 years', updated: '2023-12-15' }
  ];
  
  container.innerHTML = `
    <div class="policies-items">
      ${policies.map(policy => `
        <div class="policy-card ${policy.status}">
          <div class="policy-header">
            <div class="policy-name">${policy.name}</div>
            <div class="policy-status ${policy.status}">
              ${policy.status.toUpperCase()}
            </div>
          </div>
          <div class="policy-category">${policy.category.toUpperCase()}</div>
          <div class="policy-description">${policy.description}</div>
          <div class="policy-footer">
            <div class="policy-updated">Updated: ${policy.updated}</div>
            <div class="policy-actions">
              <button onclick="editPolicy('${policy.name}')" class="policy-action-btn">
                <i class="fas fa-edit"></i>
                Edit
              </button>
              <button onclick="reviewPolicy('${policy.name}')" class="policy-action-btn">
                <i class="fas fa-eye"></i>
                Review
              </button>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// Global action functions
window.switchSecurityTab = (tabName) => {
  // Remove active class from all tabs and panels
  document.querySelectorAll('.security-subtab').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.security-panel').forEach(panel => panel.classList.remove('active'));
  
  // Add active class to selected tab and panel
  document.querySelector(`[onclick="switchSecurityTab('${tabName}')"]`).classList.add('active');
  document.getElementById(`security-${tabName}`).classList.add('active');
};

window.runSecurityScan = async () => {
  try {
    window.notify('info', 'Security', 'Running comprehensive security scan...');
    // Security scan implementation would go here
  } catch (e) {
    window.notify('error', 'Security', e.message);
  }
};

// Add security-specific CSS
if (!document.getElementById('security-styles')) {
  const style = document.createElement('style');
  style.id = 'security-styles';
  style.textContent = `
    .security-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .security-overview {
      display: flex;
      gap: 24px;
    }
    
    .security-metric {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .security-metric i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .security-level.high {
      color: var(--omega-green);
    }
    
    .alert-warning {
      color: var(--omega-yellow);
    }
    
    .security-actions {
      display: flex;
      gap: 8px;
    }
    
    .security-btn {
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
    
    .security-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .security-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .security-subtabs {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .security-subtab {
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
    
    .security-subtab:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .security-subtab.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .security-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .security-panel {
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
    
    .security-panel.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .security-widgets {
      display: grid;
      gap: 16px;
    }
    
    .security-widget {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      overflow: hidden;
    }
    
    .widget-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: var(--omega-dark-4);
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .widget-header h5 {
      margin: 0;
      font: 600 12px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .widget-status {
      display: flex;
      align-items: center;
      gap: 6px;
      font: 600 9px var(--font-mono);
      padding: 3px 8px;
      border-radius: 2px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .widget-status.safe {
      background: var(--omega-green);
      color: var(--omega-black);
    }
    
    .widget-status.active {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .widget-content {
      padding: 16px;
    }
    
    .threat-stats,
    .access-stats {
      display: grid;
      gap: 8px;
    }
    
    .threat-stat,
    .access-stat {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .threat-label,
    .access-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .threat-value,
    .access-value {
      font: 600 12px var(--font-mono);
      color: var(--omega-white);
    }
    
    .access-value.warning {
      color: var(--omega-yellow);
    }
    
    .encryption-info {
      display: grid;
      gap: 6px;
    }
    
    .encryption-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .encryption-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .encryption-status.enabled {
      font: 600 10px var(--font-mono);
      color: var(--omega-green);
    }
    
    .security-sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .security-alerts-panel,
    .security-compliance {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px;
    }
    
    .security-alerts-panel h5,
    .security-compliance h5 {
      margin: 0 0 8px 0;
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
    }
  `;
  document.head.appendChild(style);
}
