// Sessions Management Component
class SessionsComponent {
    constructor() {
        this.sessions = [
            {
                id: 'session-001',
                user: 'admin',
                nodeId: 'node-01',
                nodeName: 'Control Node 01',
                type: 'desktop',
                status: 'active',
                startTime: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
                lastActivity: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
                cpuUsage: 23,
                memoryUsage: 512,
                bandwidth: 2.3,
                applications: ['Terminal', 'File Manager', 'Text Editor']
            },
            {
                id: 'session-002',
                user: 'developer',
                nodeId: 'node-02',
                nodeName: 'Compute Node 02',
                type: 'application',
                status: 'active',
                startTime: new Date(Date.now() - 4 * 60 * 60 * 1000), // 4 hours ago
                lastActivity: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
                cpuUsage: 78,
                memoryUsage: 2048,
                bandwidth: 5.7,
                applications: ['IDE', 'Docker', 'Database Client']
            },
            {
                id: 'session-003',
                user: 'analyst',
                nodeId: 'node-03',
                nodeName: 'Storage Node 03',
                type: 'desktop',
                status: 'idle',
                startTime: new Date(Date.now() - 6 * 60 * 60 * 1000), // 6 hours ago
                lastActivity: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
                cpuUsage: 5,
                memoryUsage: 256,
                bandwidth: 0.2,
                applications: ['Browser', 'Calculator']
            },
            {
                id: 'session-004',
                user: 'guest',
                nodeId: 'node-04',
                nodeName: 'Edge Node 04',
                type: 'terminal',
                status: 'active',
                startTime: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
                lastActivity: new Date(Date.now() - 1 * 60 * 1000), // 1 minute ago
                cpuUsage: 12,
                memoryUsage: 128,
                bandwidth: 0.8,
                applications: ['SSH Terminal']
            }
        ];
        
        this.selectedSession = null;
        this.activeFilter = 'all';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="sessions-container">
                <div class="sessions-header">
                    <h2><i class="fas fa-users"></i> Session Management</h2>
                    <div class="sessions-controls">
                        <button class="btn-primary" onclick="sessionsComponent.createSession()">
                            <i class="fas fa-plus"></i> New Session
                        </button>
                        <button class="btn-secondary" onclick="sessionsComponent.refreshSessions()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="sessions-stats">
                    <div class="stat-card">
                        <i class="fas fa-circle text-success"></i>
                        <div class="stat-info">
                            <span class="stat-value">${this.getActiveSessionsCount()}</span>
                            <span class="stat-label">Active Sessions</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-circle text-warning"></i>
                        <div class="stat-info">
                            <span class="stat-value">${this.getIdleSessionsCount()}</span>
                            <span class="stat-label">Idle Sessions</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-users"></i>
                        <div class="stat-info">
                            <span class="stat-value">${this.getUniqueUsersCount()}</span>
                            <span class="stat-label">Connected Users</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-chart-line"></i>
                        <div class="stat-info">
                            <span class="stat-value">${this.getTotalResourceUsage()}%</span>
                            <span class="stat-label">Resource Usage</span>
                        </div>
                    </div>
                </div>
                
                <div class="sessions-grid">
                    <div class="sessions-list">
                        <div class="sessions-toolbar">
                            <div class="filter-controls">
                                <label>Filter by status:</label>
                                <select onchange="sessionsComponent.filterByStatus(this.value)">
                                    <option value="all">All Sessions</option>
                                    <option value="active">Active</option>
                                    <option value="idle">Idle</option>
                                    <option value="disconnected">Disconnected</option>
                                </select>
                            </div>
                            <div class="search-controls">
                                <input type="text" placeholder="Search sessions..." onkeyup="sessionsComponent.searchSessions(this.value)">
                            </div>
                        </div>
                        
                        <div class="sessions-table">
                            <div class="table-header">
                                <div class="col-user">User</div>
                                <div class="col-node">Node</div>
                                <div class="col-type">Type</div>
                                <div class="col-status">Status</div>
                                <div class="col-duration">Duration</div>
                                <div class="col-resources">Resources</div>
                                <div class="col-actions">Actions</div>
                            </div>
                            <div class="table-body" id="sessions-table-body">
                                ${this.renderSessionsTable()}
                            </div>
                        </div>
                    </div>
                    
                    <div class="session-details" id="session-details">
                        ${this.renderSessionDetails()}
                    </div>
                </div>
            </div>
        `;
    }

    renderSessionsTable() {
        const filteredSessions = this.activeFilter === 'all' 
            ? this.sessions 
            : this.sessions.filter(session => session.status === this.activeFilter);

        return filteredSessions.map(session => `
            <div class="table-row ${session.status}" onclick="sessionsComponent.selectSession('${session.id}')">
                <div class="col-user">
                    <div class="user-info">
                        <i class="fas fa-user-circle"></i>
                        <span class="user-name">${session.user}</span>
                    </div>
                </div>
                <div class="col-node">
                    <div class="node-info">
                        <span class="node-name">${session.nodeName}</span>
                        <span class="node-id">${session.nodeId}</span>
                    </div>
                </div>
                <div class="col-type">
                    <span class="session-type ${session.type}">
                        <i class="fas fa-${this.getTypeIcon(session.type)}"></i>
                        ${session.type}
                    </span>
                </div>
                <div class="col-status">
                    <span class="status-indicator ${session.status}"></span>
                    ${session.status}
                </div>
                <div class="col-duration">
                    <span class="duration">${this.formatDuration(session.startTime)}</span>
                    <span class="last-activity">Last: ${this.formatLastActivity(session.lastActivity)}</span>
                </div>
                <div class="col-resources">
                    <div class="resource-mini">
                        <div class="resource-item">
                            <label>CPU</label>
                            <span>${session.cpuUsage}%</span>
                        </div>
                        <div class="resource-item">
                            <label>RAM</label>
                            <span>${session.memoryUsage}MB</span>
                        </div>
                        <div class="resource-item">
                            <label>Net</label>
                            <span>${session.bandwidth}MB/s</span>
                        </div>
                    </div>
                </div>
                <div class="col-actions">
                    <button class="btn-sm" onclick="sessionsComponent.connectToSession('${session.id}')" title="Connect">
                        <i class="fas fa-desktop"></i>
                    </button>
                    <button class="btn-sm" onclick="sessionsComponent.terminateSession('${session.id}')" title="Terminate">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    renderSessionDetails() {
        if (!this.selectedSession) {
            return `
                <div class="no-selection">
                    <i class="fas fa-desktop"></i>
                    <h3>Select a Session</h3>
                    <p>Choose a session from the list to view detailed information and management options.</p>
                </div>
            `;
        }

        const session = this.sessions.find(s => s.id === this.selectedSession);
        if (!session) return '<div class="no-selection">Session not found</div>';

        return `
            <div class="session-details-content">
                <div class="details-header">
                    <h3><i class="fas fa-user-circle"></i> ${session.user}'s Session</h3>
                    <span class="status-badge ${session.status}">${session.status}</span>
                </div>
                
                <div class="details-sections">
                    <div class="details-section">
                        <h4>Session Information</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <label>Session ID</label>
                                <span>${session.id}</span>
                            </div>
                            <div class="info-item">
                                <label>User</label>
                                <span>${session.user}</span>
                            </div>
                            <div class="info-item">
                                <label>Session Type</label>
                                <span class="session-type ${session.type}">
                                    <i class="fas fa-${this.getTypeIcon(session.type)}"></i>
                                    ${session.type}
                                </span>
                            </div>
                            <div class="info-item">
                                <label>Node</label>
                                <span>${session.nodeName} (${session.nodeId})</span>
                            </div>
                            <div class="info-item">
                                <label>Start Time</label>
                                <span>${session.startTime.toLocaleString()}</span>
                            </div>
                            <div class="info-item">
                                <label>Duration</label>
                                <span>${this.formatDuration(session.startTime)}</span>
                            </div>
                            <div class="info-item">
                                <label>Last Activity</label>
                                <span>${session.lastActivity.toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-section">
                        <h4>Resource Usage</h4>
                        <div class="resource-details">
                            <div class="resource-item">
                                <div class="resource-header">
                                    <label>CPU Usage</label>
                                    <span>${session.cpuUsage}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${session.cpuUsage}%"></div>
                                </div>
                            </div>
                            <div class="resource-item">
                                <div class="resource-header">
                                    <label>Memory Usage</label>
                                    <span>${session.memoryUsage}MB</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${(session.memoryUsage / 4096) * 100}%"></div>
                                </div>
                            </div>
                            <div class="resource-item">
                                <div class="resource-header">
                                    <label>Network Bandwidth</label>
                                    <span>${session.bandwidth}MB/s</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${(session.bandwidth / 10) * 100}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-section">
                        <h4>Running Applications</h4>
                        <div class="applications-list">
                            ${session.applications.map(app => `
                                <div class="application-item">
                                    <i class="fas fa-${this.getAppIcon(app)}"></i>
                                    <span>${app}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="details-section">
                        <h4>Session Actions</h4>
                        <div class="action-buttons">
                            <button class="btn-primary" onclick="sessionsComponent.connectToSession('${session.id}')">
                                <i class="fas fa-desktop"></i> Connect
                            </button>
                            <button class="btn-secondary" onclick="sessionsComponent.shareSession('${session.id}')">
                                <i class="fas fa-share"></i> Share
                            </button>
                            <button class="btn-secondary" onclick="sessionsComponent.takeScreenshot('${session.id}')">
                                <i class="fas fa-camera"></i> Screenshot
                            </button>
                            <button class="btn-warning" onclick="sessionsComponent.suspendSession('${session.id}')">
                                <i class="fas fa-pause"></i> Suspend
                            </button>
                            <button class="btn-danger" onclick="sessionsComponent.terminateSession('${session.id}')">
                                <i class="fas fa-times"></i> Terminate
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getTypeIcon(type) {
        const icons = {
            desktop: 'desktop',
            application: 'window-maximize',
            terminal: 'terminal'
        };
        return icons[type] || 'desktop';
    }

    getAppIcon(app) {
        const icons = {
            'Terminal': 'terminal',
            'File Manager': 'folder',
            'Text Editor': 'edit',
            'IDE': 'code',
            'Docker': 'cube',
            'Database Client': 'database',
            'Browser': 'globe',
            'Calculator': 'calculator',
            'SSH Terminal': 'terminal'
        };
        return icons[app] || 'window-maximize';
    }

    formatDuration(startTime) {
        const duration = Date.now() - startTime.getTime();
        const hours = Math.floor(duration / (1000 * 60 * 60));
        const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
        return `${hours}h ${minutes}m`;
    }

    formatLastActivity(lastActivity) {
        const diff = Date.now() - lastActivity.getTime();
        const minutes = Math.floor(diff / (1000 * 60));
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        return `${hours}h ago`;
    }

    getActiveSessionsCount() {
        return this.sessions.filter(s => s.status === 'active').length;
    }

    getIdleSessionsCount() {
        return this.sessions.filter(s => s.status === 'idle').length;
    }

    getUniqueUsersCount() {
        return new Set(this.sessions.map(s => s.user)).size;
    }

    getTotalResourceUsage() {
        const totalCpu = this.sessions.reduce((sum, s) => sum + s.cpuUsage, 0);
        return Math.round(totalCpu / this.sessions.length);
    }

    selectSession(sessionId) {
        this.selectedSession = sessionId;
        document.getElementById('session-details').innerHTML = this.renderSessionDetails();
        
        // Update selected row styling
        document.querySelectorAll('.table-row').forEach(row => row.classList.remove('selected'));
        event.target.closest('.table-row').classList.add('selected');
    }

    filterByStatus(status) {
        this.activeFilter = status;
        document.getElementById('sessions-table-body').innerHTML = this.renderSessionsTable();
    }

    searchSessions(query) {
        const filteredSessions = this.sessions.filter(session => 
            session.user.toLowerCase().includes(query.toLowerCase()) ||
            session.nodeName.toLowerCase().includes(query.toLowerCase()) ||
            session.type.toLowerCase().includes(query.toLowerCase())
        );
        
        document.getElementById('sessions-table-body').innerHTML = 
            this.renderSessionsTableForSessions(filteredSessions);
    }

    renderSessionsTableForSessions(sessions) {
        return sessions.map(session => `
            <div class="table-row ${session.status}" onclick="sessionsComponent.selectSession('${session.id}')">
                <div class="col-user">
                    <div class="user-info">
                        <i class="fas fa-user-circle"></i>
                        <span class="user-name">${session.user}</span>
                    </div>
                </div>
                <div class="col-node">
                    <div class="node-info">
                        <span class="node-name">${session.nodeName}</span>
                        <span class="node-id">${session.nodeId}</span>
                    </div>
                </div>
                <div class="col-type">
                    <span class="session-type ${session.type}">
                        <i class="fas fa-${this.getTypeIcon(session.type)}"></i>
                        ${session.type}
                    </span>
                </div>
                <div class="col-status">
                    <span class="status-indicator ${session.status}"></span>
                    ${session.status}
                </div>
                <div class="col-duration">
                    <span class="duration">${this.formatDuration(session.startTime)}</span>
                    <span class="last-activity">Last: ${this.formatLastActivity(session.lastActivity)}</span>
                </div>
                <div class="col-resources">
                    <div class="resource-mini">
                        <div class="resource-item">
                            <label>CPU</label>
                            <span>${session.cpuUsage}%</span>
                        </div>
                        <div class="resource-item">
                            <label>RAM</label>
                            <span>${session.memoryUsage}MB</span>
                        </div>
                        <div class="resource-item">
                            <label>Net</label>
                            <span>${session.bandwidth}MB/s</span>
                        </div>
                    </div>
                </div>
                <div class="col-actions">
                    <button class="btn-sm" onclick="sessionsComponent.connectToSession('${session.id}')" title="Connect">
                        <i class="fas fa-desktop"></i>
                    </button>
                    <button class="btn-sm" onclick="sessionsComponent.terminateSession('${session.id}')" title="Terminate">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    createSession() {
        console.log('Creating new session...');
        // Implementation for creating a new session
    }

    refreshSessions() {
        console.log('Refreshing sessions...');
        this.updateSessionMetrics();
        document.getElementById('sessions-table-body').innerHTML = this.renderSessionsTable();
    }

    connectToSession(sessionId) {
        console.log(`Connecting to session: ${sessionId}`);
        // Implementation for connecting to session
    }

    shareSession(sessionId) {
        console.log(`Sharing session: ${sessionId}`);
        // Implementation for sharing session
    }

    takeScreenshot(sessionId) {
        console.log(`Taking screenshot of session: ${sessionId}`);
        // Implementation for taking screenshot
    }

    suspendSession(sessionId) {
        if (confirm(`Are you sure you want to suspend session ${sessionId}?`)) {
            console.log(`Suspending session: ${sessionId}`);
            // Implementation for suspending session
        }
    }

    terminateSession(sessionId) {
        if (confirm(`Are you sure you want to terminate session ${sessionId}?`)) {
            console.log(`Terminating session: ${sessionId}`);
            // Implementation for terminating session
        }
    }

    updateSessionMetrics() {
        this.sessions.forEach(session => {
            if (session.status === 'active') {
                session.cpuUsage = Math.floor(Math.random() * 50) + 10;
                session.memoryUsage = Math.floor(Math.random() * 1000) + 200;
                session.bandwidth = (Math.random() * 5).toFixed(1);
                session.lastActivity = new Date();
            }
        });
    }

    startMonitoring() {
        setInterval(() => {
            this.updateSessionMetrics();
            if (document.getElementById('sessions-table-body')) {
                document.getElementById('sessions-table-body').innerHTML = this.renderSessionsTable();
            }
            if (this.selectedSession && document.getElementById('session-details')) {
                document.getElementById('session-details').innerHTML = this.renderSessionDetails();
            }
        }, 30000);
    }
}

// Global instance
window.sessionsComponent = new SessionsComponent();
