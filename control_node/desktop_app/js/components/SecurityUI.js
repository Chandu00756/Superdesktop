/**
 * Security and RBAC (Role-Based Access Control) UI Manager
 * Handles user authentication, authorization, and security settings
 */

export class SecurityUI {
    constructor(eventBus, stateStore, notificationManager) {
        this.eventBus = eventBus;
        this.stateStore = stateStore;
        this.notificationManager = notificationManager;
        this.currentUser = null;
        this.permissions = new Set();
        this.securityPolicies = new Map();
        this.auditLog = [];
        this.loginAttempts = new Map();
        
        this.init();
    }

    /**
     * Initialize Security UI
     */
    init() {
        this.setupEventListeners();
        this.loadSecurityPolicies();
        this.startSecurityMonitoring();
        
        // Initialize default admin user for demo
        this.initializeDefaultUser();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        this.eventBus.on('security:login-attempt', (data) => {
            this.handleLoginAttempt(data);
        });

        this.eventBus.on('security:logout', () => {
            this.handleLogout();
        });

        this.eventBus.on('security:permission-check', (data) => {
            this.handlePermissionCheck(data);
        });

        this.eventBus.on('session:created', (data) => {
            this.auditLog.push({
                timestamp: Date.now(),
                user: this.currentUser?.username || 'system',
                action: 'session_created',
                details: { sessionId: data.sessionId, type: data.type }
            });
        });
    }

    /**
     * Initialize default user for demo
     */
    initializeDefaultUser() {
        const defaultUser = {
            id: 'admin-001',
            username: 'admin',
            email: 'admin@omega.local',
            role: 'administrator',
            permissions: ['*'], // All permissions
            lastLogin: null,
            mfaEnabled: false,
            sessionTimeout: 3600000 // 1 hour
        };

        this.stateStore.setState('security.users.admin', defaultUser);
        this.stateStore.setState('security.currentUser', null);
    }

    /**
     * Render Security UI
     */
    render(container) {
        container.innerHTML = `
            <div class="security-ui">
                <div class="security-header">
                    <h3><i class="fas fa-shield-alt"></i> Security & Access Control</h3>
                    <div class="security-status">
                        <span class="status-indicator ${this.currentUser ? 'authenticated' : 'unauthenticated'}">
                            <i class="fas fa-${this.currentUser ? 'lock' : 'unlock'}"></i>
                            ${this.currentUser ? 'Authenticated' : 'Not Authenticated'}
                        </span>
                    </div>
                </div>

                <div class="security-tabs">
                    <button class="security-tab active" data-tab="authentication">
                        <i class="fas fa-user-lock"></i> Authentication
                    </button>
                    <button class="security-tab" data-tab="permissions">
                        <i class="fas fa-key"></i> Permissions
                    </button>
                    <button class="security-tab" data-tab="policies">
                        <i class="fas fa-file-contract"></i> Policies
                    </button>
                    <button class="security-tab" data-tab="audit">
                        <i class="fas fa-clipboard-list"></i> Audit Log
                    </button>
                </div>

                <div class="security-content">
                    <div id="authentication-tab" class="security-tab-content active">
                        ${this.renderAuthenticationTab()}
                    </div>
                    <div id="permissions-tab" class="security-tab-content">
                        ${this.renderPermissionsTab()}
                    </div>
                    <div id="policies-tab" class="security-tab-content">
                        ${this.renderPoliciesTab()}
                    </div>
                    <div id="audit-tab" class="security-tab-content">
                        ${this.renderAuditTab()}
                    </div>
                </div>
            </div>
        `;

        this.attachEventListeners(container);
    }

    /**
     * Render Authentication Tab
     */
    renderAuthenticationTab() {
        if (this.currentUser) {
            return `
                <div class="auth-user-info">
                    <div class="user-avatar">
                        <i class="fas fa-user"></i>
                    </div>
                    <div class="user-details">
                        <h4>${this.currentUser.username}</h4>
                        <p>Role: <span class="user-role">${this.currentUser.role}</span></p>
                        <p>Last Login: ${this.currentUser.lastLogin ? new Date(this.currentUser.lastLogin).toLocaleString() : 'Never'}</p>
                        <p>Session Expires: ${new Date(Date.now() + this.currentUser.sessionTimeout).toLocaleString()}</p>
                    </div>
                </div>
                
                <div class="auth-controls">
                    <button class="btn-secondary" onclick="securityUI.showChangePassword()">
                        <i class="fas fa-key"></i> Change Password
                    </button>
                    <button class="btn-secondary" onclick="securityUI.toggleMFA()">
                        <i class="fas fa-mobile-alt"></i> ${this.currentUser.mfaEnabled ? 'Disable' : 'Enable'} MFA
                    </button>
                    <button class="btn-primary" onclick="securityUI.logout()">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </button>
                </div>
                
                <div class="session-management">
                    <h5>Active Sessions</h5>
                    <div class="session-list">
                        ${this.renderActiveSessions()}
                    </div>
                </div>
            `;
        } else {
            return `
                <div class="login-form">
                    <h4>Login Required</h4>
                    <form id="login-form">
                        <div class="form-group">
                            <label for="username">Username:</label>
                            <input type="text" id="username" name="username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password:</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="remember-me"> Remember me
                            </label>
                        </div>
                        <button type="submit" class="btn-primary">
                            <i class="fas fa-sign-in-alt"></i> Login
                        </button>
                    </form>
                </div>
                
                <div class="login-security">
                    <div class="security-notice">
                        <i class="fas fa-info-circle"></i>
                        <p>For demo purposes, use username "admin" with any password.</p>
                    </div>
                    
                    <div class="failed-attempts">
                        <h5>Recent Failed Attempts</h5>
                        <div id="failed-attempts-list">
                            ${this.renderFailedAttempts()}
                        </div>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Render Permissions Tab
     */
    renderPermissionsTab() {
        const userPermissions = this.currentUser ? this.getUserPermissions(this.currentUser) : [];
        
        return `
            <div class="permissions-overview">
                <h4>Permission Management</h4>
                <div class="current-permissions">
                    <h5>Current User Permissions</h5>
                    <div class="permissions-grid">
                        ${this.renderPermissionsList(userPermissions)}
                    </div>
                </div>
                
                <div class="role-management">
                    <h5>Role Management</h5>
                    <div class="roles-list">
                        ${this.renderRolesList()}
                    </div>
                    <button class="btn-primary" onclick="securityUI.showCreateRole()">
                        <i class="fas fa-plus"></i> Create Role
                    </button>
                </div>
                
                <div class="permission-requests">
                    <h5>Permission Requests</h5>
                    <div class="requests-list">
                        ${this.renderPermissionRequests()}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render Policies Tab
     */
    renderPoliciesTab() {
        return `
            <div class="security-policies">
                <h4>Security Policies</h4>
                
                <div class="policy-categories">
                    <div class="policy-category">
                        <h5><i class="fas fa-lock"></i> Password Policy</h5>
                        <div class="policy-settings">
                            <label>
                                Minimum Length: 
                                <input type="number" value="8" min="6" max="32">
                            </label>
                            <label>
                                <input type="checkbox" checked> Require uppercase letters
                            </label>
                            <label>
                                <input type="checkbox" checked> Require numbers
                            </label>
                            <label>
                                <input type="checkbox"> Require special characters
                            </label>
                            <label>
                                Password Expiry (days): 
                                <input type="number" value="90" min="0" max="365">
                            </label>
                        </div>
                    </div>
                    
                    <div class="policy-category">
                        <h5><i class="fas fa-clock"></i> Session Policy</h5>
                        <div class="policy-settings">
                            <label>
                                Session Timeout (minutes): 
                                <input type="number" value="60" min="5" max="480">
                            </label>
                            <label>
                                <input type="checkbox" checked> Auto-lock on idle
                            </label>
                            <label>
                                Idle Timeout (minutes): 
                                <input type="number" value="15" min="1" max="60">
                            </label>
                            <label>
                                Max Concurrent Sessions: 
                                <input type="number" value="3" min="1" max="10">
                            </label>
                        </div>
                    </div>
                    
                    <div class="policy-category">
                        <h5><i class="fas fa-network-wired"></i> Network Policy</h5>
                        <div class="policy-settings">
                            <label>
                                <input type="checkbox" checked> Restrict by IP range
                            </label>
                            <label>
                                Allowed IPs: 
                                <input type="text" placeholder="192.168.1.0/24">
                            </label>
                            <label>
                                <input type="checkbox" checked> Require VPN
                            </label>
                            <label>
                                <input type="checkbox"> Block suspicious IPs
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="policy-actions">
                    <button class="btn-primary" onclick="securityUI.savePolicies()">
                        <i class="fas fa-save"></i> Save Policies
                    </button>
                    <button class="btn-secondary" onclick="securityUI.resetPolicies()">
                        <i class="fas fa-undo"></i> Reset to Defaults
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render Audit Tab
     */
    renderAuditTab() {
        return `
            <div class="audit-log">
                <h4>Security Audit Log</h4>
                
                <div class="audit-filters">
                    <div class="filter-group">
                        <label>Date Range:</label>
                        <input type="date" id="audit-start-date">
                        <input type="date" id="audit-end-date">
                    </div>
                    <div class="filter-group">
                        <label>Action Type:</label>
                        <select id="audit-action-filter">
                            <option value="">All Actions</option>
                            <option value="login">Login</option>
                            <option value="logout">Logout</option>
                            <option value="session_created">Session Created</option>
                            <option value="permission_granted">Permission Granted</option>
                            <option value="policy_changed">Policy Changed</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>User:</label>
                        <input type="text" id="audit-user-filter" placeholder="Username">
                    </div>
                    <button class="btn-secondary" onclick="securityUI.filterAuditLog()">
                        <i class="fas fa-filter"></i> Filter
                    </button>
                </div>
                
                <div class="audit-entries">
                    ${this.renderAuditEntries()}
                </div>
                
                <div class="audit-export">
                    <button class="btn-secondary" onclick="securityUI.exportAuditLog()">
                        <i class="fas fa-download"></i> Export Log
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render active sessions
     */
    renderActiveSessions() {
        // Get sessions from state store
        const sessions = this.stateStore.getState('sessions') || {};
        
        return Object.entries(sessions).map(([sessionId, session]) => `
            <div class="session-item">
                <div class="session-info">
                    <span class="session-type">${session.type?.toUpperCase() || 'Unknown'}</span>
                    <span class="session-status status-${session.status}">${session.status}</span>
                </div>
                <div class="session-details">
                    <small>ID: ${sessionId}</small>
                    <small>Started: ${new Date(session.startTime || Date.now()).toLocaleString()}</small>
                </div>
                <button class="btn-small btn-secondary" onclick="securityUI.terminateSession('${sessionId}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    /**
     * Render failed login attempts
     */
    renderFailedAttempts() {
        const attempts = Array.from(this.loginAttempts.entries()).slice(-5);
        
        if (attempts.length === 0) {
            return '<p class="text-muted">No recent failed attempts</p>';
        }
        
        return attempts.map(([ip, data]) => `
            <div class="failed-attempt">
                <span class="attempt-ip">${ip}</span>
                <span class="attempt-count">${data.count} attempts</span>
                <span class="attempt-time">${new Date(data.lastAttempt).toLocaleString()}</span>
            </div>
        `).join('');
    }

    /**
     * Render permissions list
     */
    renderPermissionsList(permissions) {
        const allPermissions = [
            { id: 'sessions.create', name: 'Create Sessions', description: 'Create new virtual desktop sessions' },
            { id: 'sessions.manage', name: 'Manage Sessions', description: 'Start, stop, and configure sessions' },
            { id: 'plugins.install', name: 'Install Plugins', description: 'Install and remove plugins' },
            { id: 'plugins.configure', name: 'Configure Plugins', description: 'Modify plugin settings' },
            { id: 'network.monitor', name: 'Network Monitoring', description: 'View network topology and metrics' },
            { id: 'security.admin', name: 'Security Administration', description: 'Manage users, roles, and policies' },
            { id: 'system.admin', name: 'System Administration', description: 'Full system access' }
        ];
        
        return allPermissions.map(perm => {
            const hasPermission = permissions.includes('*') || permissions.includes(perm.id);
            return `
                <div class="permission-item ${hasPermission ? 'granted' : 'denied'}">
                    <div class="permission-icon">
                        <i class="fas fa-${hasPermission ? 'check' : 'times'}"></i>
                    </div>
                    <div class="permission-details">
                        <h6>${perm.name}</h6>
                        <p>${perm.description}</p>
                    </div>
                    <div class="permission-status">
                        ${hasPermission ? 'Granted' : 'Denied'}
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Render roles list
     */
    renderRolesList() {
        const roles = [
            { id: 'administrator', name: 'Administrator', permissions: ['*'], userCount: 1 },
            { id: 'power_user', name: 'Power User', permissions: ['sessions.*', 'plugins.*'], userCount: 0 },
            { id: 'user', name: 'Standard User', permissions: ['sessions.create'], userCount: 0 },
            { id: 'viewer', name: 'Viewer', permissions: ['sessions.view'], userCount: 0 }
        ];
        
        return roles.map(role => `
            <div class="role-item">
                <div class="role-info">
                    <h6>${role.name}</h6>
                    <p>${role.permissions.length} permissions • ${role.userCount} users</p>
                </div>
                <div class="role-actions">
                    <button class="btn-small btn-secondary" onclick="securityUI.editRole('${role.id}')">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn-small btn-secondary" onclick="securityUI.deleteRole('${role.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    /**
     * Render permission requests
     */
    renderPermissionRequests() {
        // Mock permission requests for demo
        const requests = [
            { id: 1, user: 'user001', permission: 'plugins.install', reason: 'Need to install development tools', status: 'pending' },
            { id: 2, user: 'user002', permission: 'network.monitor', reason: 'Troubleshooting network issues', status: 'pending' }
        ];
        
        if (requests.length === 0) {
            return '<p class="text-muted">No pending permission requests</p>';
        }
        
        return requests.map(req => `
            <div class="permission-request">
                <div class="request-info">
                    <h6>${req.user}</h6>
                    <p>Requests: <code>${req.permission}</code></p>
                    <p>Reason: ${req.reason}</p>
                </div>
                <div class="request-actions">
                    <button class="btn-small btn-primary" onclick="securityUI.approveRequest(${req.id})">
                        <i class="fas fa-check"></i> Approve
                    </button>
                    <button class="btn-small btn-secondary" onclick="securityUI.denyRequest(${req.id})">
                        <i class="fas fa-times"></i> Deny
                    </button>
                </div>
            </div>
        `).join('');
    }

    /**
     * Render audit entries
     */
    renderAuditEntries() {
        const recentEntries = this.auditLog.slice(-20).reverse();
        
        if (recentEntries.length === 0) {
            return '<p class="text-muted">No audit entries found</p>';
        }
        
        return recentEntries.map(entry => `
            <div class="audit-entry">
                <div class="audit-timestamp">
                    ${new Date(entry.timestamp).toLocaleString()}
                </div>
                <div class="audit-user">
                    ${entry.user}
                </div>
                <div class="audit-action">
                    ${entry.action.replace('_', ' ')}
                </div>
                <div class="audit-details">
                    ${JSON.stringify(entry.details)}
                </div>
            </div>
        `).join('');
    }

    /**
     * Attach event listeners
     */
    attachEventListeners(container) {
        // Tab switching
        container.querySelectorAll('.security-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const targetTab = e.target.dataset.tab;
                this.switchSecurityTab(targetTab);
            });
        });

        // Login form
        const loginForm = container.querySelector('#login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin(new FormData(loginForm));
            });
        }
    }

    /**
     * Switch security tab
     */
    switchSecurityTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.security-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.security-tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });
    }

    /**
     * Handle login
     */
    async handleLogin(formData) {
        const username = formData.get('username');
        const password = formData.get('password');
        
        try {
            // Simulate authentication
            await this.delay(500);
            
            if (username === 'admin') { // Demo login
                const user = this.stateStore.getState('security.users.admin');
                user.lastLogin = Date.now();
                
                this.currentUser = user;
                this.permissions = new Set(this.getUserPermissions(user));
                
                this.stateStore.setState('security.currentUser', user);
                
                this.auditLog.push({
                    timestamp: Date.now(),
                    user: username,
                    action: 'login',
                    details: { success: true }
                });
                
                this.notificationManager.success('Login Successful', `Welcome back, ${username}!`);
                this.eventBus.emit('security:authenticated', { user });
                
                // Re-render to show authenticated state
                const container = document.querySelector('.security-ui').parentElement;
                this.render(container);
                
            } else {
                throw new Error('Invalid credentials');
            }
            
        } catch (error) {
            this.handleLoginFailure(username, error.message);
        }
    }

    /**
     * Handle login failure
     */
    handleLoginFailure(username, error) {
        const clientIP = '127.0.0.1'; // Mock IP
        
        if (!this.loginAttempts.has(clientIP)) {
            this.loginAttempts.set(clientIP, { count: 0, lastAttempt: 0 });
        }
        
        const attempts = this.loginAttempts.get(clientIP);
        attempts.count++;
        attempts.lastAttempt = Date.now();
        
        this.auditLog.push({
            timestamp: Date.now(),
            user: username,
            action: 'login_failed',
            details: { error, attempts: attempts.count }
        });
        
        this.notificationManager.error('Login Failed', error);
        this.eventBus.emit('security:login-failed', { username, error });
    }

    /**
     * Handle logout
     */
    logout() {
        if (this.currentUser) {
            this.auditLog.push({
                timestamp: Date.now(),
                user: this.currentUser.username,
                action: 'logout',
                details: {}
            });
            
            this.currentUser = null;
            this.permissions.clear();
            this.stateStore.setState('security.currentUser', null);
            
            this.notificationManager.info('Logged Out', 'You have been logged out successfully.');
            this.eventBus.emit('security:logged-out');
            
            // Re-render to show login form
            const container = document.querySelector('.security-ui').parentElement;
            this.render(container);
        }
    }

    /**
     * Get user permissions
     */
    getUserPermissions(user) {
        if (user.permissions.includes('*')) {
            return ['*']; // All permissions
        }
        return user.permissions;
    }

    /**
     * Check if user has permission
     */
    hasPermission(permission) {
        if (!this.currentUser) return false;
        return this.permissions.has('*') || this.permissions.has(permission);
    }

    /**
     * Load security policies
     */
    loadSecurityPolicies() {
        // Load default policies
        this.securityPolicies.set('password', {
            minLength: 8,
            requireUppercase: true,
            requireNumbers: true,
            requireSpecialChars: false,
            expiryDays: 90
        });

        this.securityPolicies.set('session', {
            timeoutMinutes: 60,
            autoLockOnIdle: true,
            idleTimeoutMinutes: 15,
            maxConcurrentSessions: 3
        });

        this.securityPolicies.set('network', {
            restrictByIP: true,
            allowedIPs: ['192.168.1.0/24'],
            requireVPN: true,
            blockSuspiciousIPs: false
        });
    }

    /**
     * Start security monitoring
     */
    startSecurityMonitoring() {
        setInterval(() => {
            this.checkSessionTimeout();
            this.checkSuspiciousActivity();
        }, 30000); // Check every 30 seconds
    }

    /**
     * Check session timeout
     */
    checkSessionTimeout() {
        if (this.currentUser) {
            const sessionPolicy = this.securityPolicies.get('session');
            const timeoutMs = sessionPolicy.timeoutMinutes * 60 * 1000;
            
            if (Date.now() - this.currentUser.lastLogin > timeoutMs) {
                this.notificationManager.warning('Session Expired', 'Your session has expired. Please log in again.');
                this.logout();
            }
        }
    }

    /**
     * Check for suspicious activity
     */
    checkSuspiciousActivity() {
        // Check for excessive failed login attempts
        for (const [ip, data] of this.loginAttempts) {
            if (data.count > 5 && Date.now() - data.lastAttempt < 300000) { // 5 minutes
                this.auditLog.push({
                    timestamp: Date.now(),
                    user: 'system',
                    action: 'suspicious_activity',
                    details: { type: 'excessive_failed_logins', ip, count: data.count }
                });
            }
        }
    }

    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Placeholder methods for UI actions
     */
    showChangePassword() {
        this.notificationManager.info('Change Password', 'Password change dialog would be shown here.');
    }

    toggleMFA() {
        if (this.currentUser) {
            this.currentUser.mfaEnabled = !this.currentUser.mfaEnabled;
            this.stateStore.setState('security.currentUser.mfaEnabled', this.currentUser.mfaEnabled);
            this.notificationManager.info('MFA Updated', `MFA has been ${this.currentUser.mfaEnabled ? 'enabled' : 'disabled'}.`);
        }
    }

    terminateSession(sessionId) {
        this.eventBus.emit('session:terminate', { sessionId });
        this.notificationManager.info('Session Terminated', `Session ${sessionId} has been terminated.`);
    }

    savePolicies() {
        this.notificationManager.success('Policies Saved', 'Security policies have been updated.');
    }

    resetPolicies() {
        this.loadSecurityPolicies();
        this.notificationManager.info('Policies Reset', 'Security policies have been reset to defaults.');
    }

    exportAuditLog() {
        const logData = JSON.stringify(this.auditLog, null, 2);
        const blob = new Blob([logData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `omega-audit-log-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.notificationManager.success('Export Complete', 'Audit log has been exported.');
    }

    // Make methods globally available
    static createGlobalMethods(instance) {
        window.securityUI = {
            logout: () => instance.logout(),
            showChangePassword: () => instance.showChangePassword(),
            toggleMFA: () => instance.toggleMFA(),
            terminateSession: (sessionId) => instance.terminateSession(sessionId),
            savePolicies: () => instance.savePolicies(),
            resetPolicies: () => instance.resetPolicies(),
            exportAuditLog: () => instance.exportAuditLog(),
            filterAuditLog: () => instance.notificationManager.info('Filter', 'Audit log filtering would be implemented here.'),
            editRole: (roleId) => instance.notificationManager.info('Edit Role', `Edit role ${roleId} dialog would be shown here.`),
            deleteRole: (roleId) => instance.notificationManager.info('Delete Role', `Delete role ${roleId} confirmation would be shown here.`),
            showCreateRole: () => instance.notificationManager.info('Create Role', 'Create new role dialog would be shown here.'),
            approveRequest: (requestId) => instance.notificationManager.success('Request Approved', `Permission request ${requestId} has been approved.`),
            denyRequest: (requestId) => instance.notificationManager.info('Request Denied', `Permission request ${requestId} has been denied.`)
        };
    }
}
