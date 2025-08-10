// Security Management Component
class SecurityComponent {
    constructor() {
        this.securityData = {
            overview: {
                threatLevel: 'low',
                activeScans: 2,
                blockedAttempts: 147,
                firewall: 'active',
                encryption: 'enabled',
                lastScan: new Date(Date.now() - 30 * 60 * 1000),
                vulnerabilities: { critical: 0, high: 2, medium: 5, low: 12 }
            },
            firewall: {
                status: 'active',
                rules: [
                    { id: 1, action: 'allow', protocol: 'tcp', port: '22', source: '192.168.1.0/24', description: 'SSH Access' },
                    { id: 2, action: 'allow', protocol: 'tcp', port: '80,443', source: 'any', description: 'Web Traffic' },
                    { id: 3, action: 'deny', protocol: 'tcp', port: '23', source: 'any', description: 'Block Telnet' },
                    { id: 4, action: 'allow', protocol: 'udp', port: '53', source: 'any', description: 'DNS Queries' }
                ],
                blockedConnections: [
                    { ip: '203.0.113.45', attempts: 25, lastAttempt: new Date(), reason: 'Brute force SSH' },
                    { ip: '198.51.100.89', attempts: 12, lastAttempt: new Date(Date.now() - 10 * 60 * 1000), reason: 'Port scanning' }
                ]
            },
            certificates: [
                { name: 'omega-control.local', type: 'SSL/TLS', expiry: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000), status: 'valid' },
                { name: 'api.omega.local', type: 'SSL/TLS', expiry: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000), status: 'expiring' },
                { name: 'root-ca', type: 'Root CA', expiry: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000), status: 'valid' }
            ],
            auditLogs: [
                { time: new Date(), user: 'admin', action: 'Login successful', source: '192.168.1.50', level: 'info' },
                { time: new Date(Date.now() - 5 * 60 * 1000), user: 'unknown', action: 'Failed login attempt', source: '203.0.113.45', level: 'warning' },
                { time: new Date(Date.now() - 10 * 60 * 1000), user: 'system', action: 'Firewall rule added', source: 'localhost', level: 'info' },
                { time: new Date(Date.now() - 15 * 60 * 1000), user: 'admin', action: 'Configuration changed', source: '192.168.1.50', level: 'warning' }
            ]
        };
        
        this.activeTab = 'overview';
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="security-container">
                <div class="security-header">
                    <h2><i class="fas fa-shield-alt"></i> Security Center</h2>
                    <div class="security-controls">
                        <button class="btn-primary" onclick="securityComponent.runFullScan()">
                            <i class="fas fa-search"></i> Full Scan
                        </button>
                        <button class="btn-secondary" onclick="securityComponent.updateSecurityRules()">
                            <i class="fas fa-sync-alt"></i> Update Rules
                        </button>
                    </div>
                </div>
                
                <div class="security-tabs">
                    <button class="tab-btn ${this.activeTab === 'overview' ? 'active' : ''}" onclick="securityComponent.switchTab('overview')">
                        <i class="fas fa-shield-alt"></i> Overview
                    </button>
                    <button class="tab-btn ${this.activeTab === 'firewall' ? 'active' : ''}" onclick="securityComponent.switchTab('firewall')">
                        <i class="fas fa-fire"></i> Firewall
                    </button>
                    <button class="tab-btn ${this.activeTab === 'certificates' ? 'active' : ''}" onclick="securityComponent.switchTab('certificates')">
                        <i class="fas fa-certificate"></i> Certificates
                    </button>
                    <button class="tab-btn ${this.activeTab === 'audit' ? 'active' : ''}" onclick="securityComponent.switchTab('audit')">
                        <i class="fas fa-clipboard-list"></i> Audit Logs
                    </button>
                </div>
                
                <div class="security-content" id="security-content">
                    ${this.renderTabContent()}
                </div>
            </div>
        `;
    }

    renderTabContent() {
        switch (this.activeTab) {
            case 'overview': return this.renderOverview();
            case 'firewall': return this.renderFirewall();
            case 'certificates': return this.renderCertificates();
            case 'audit': return this.renderAuditLogs();
            default: return this.renderOverview();
        }
    }

    renderOverview() {
        const overview = this.securityData.overview;
        
        return `
            <div class="security-overview">
                <div class="threat-level-card">
                    <div class="threat-indicator ${overview.threatLevel}">
                        <i class="fas fa-shield-alt"></i>
                        <div class="threat-info">
                            <h3>Threat Level: ${overview.threatLevel.toUpperCase()}</h3>
                            <p>System security status is currently ${overview.threatLevel}</p>
                        </div>
                    </div>
                </div>
                
                <div class="security-stats">
                    <div class="stat-card">
                        <i class="fas fa-search"></i>
                        <div class="stat-info">
                            <span class="stat-value">${overview.activeScans}</span>
                            <span class="stat-label">Active Scans</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-ban"></i>
                        <div class="stat-info">
                            <span class="stat-value">${overview.blockedAttempts}</span>
                            <span class="stat-label">Blocked Attempts</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-fire"></i>
                        <div class="stat-info">
                            <span class="stat-value status ${overview.firewall}">${overview.firewall}</span>
                            <span class="stat-label">Firewall</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-lock"></i>
                        <div class="stat-info">
                            <span class="stat-value status ${overview.encryption}">${overview.encryption}</span>
                            <span class="stat-label">Encryption</span>
                        </div>
                    </div>
                </div>
                
                <div class="vulnerabilities-summary">
                    <h3>Vulnerability Summary</h3>
                    <div class="vuln-grid">
                        <div class="vuln-item critical">
                            <span class="vuln-count">${overview.vulnerabilities.critical}</span>
                            <span class="vuln-label">Critical</span>
                        </div>
                        <div class="vuln-item high">
                            <span class="vuln-count">${overview.vulnerabilities.high}</span>
                            <span class="vuln-label">High</span>
                        </div>
                        <div class="vuln-item medium">
                            <span class="vuln-count">${overview.vulnerabilities.medium}</span>
                            <span class="vuln-label">Medium</span>
                        </div>
                        <div class="vuln-item low">
                            <span class="vuln-count">${overview.vulnerabilities.low}</span>
                            <span class="vuln-label">Low</span>
                        </div>
                    </div>
                </div>
                
                <div class="recent-events">
                    <h3>Recent Security Events</h3>
                    <div class="events-list">
                        ${this.securityData.auditLogs.slice(0, 5).map(log => `
                            <div class="event-item ${log.level}">
                                <i class="fas fa-${this.getLogIcon(log.level)}"></i>
                                <div class="event-content">
                                    <span class="event-action">${log.action}</span>
                                    <span class="event-details">User: ${log.user} | Source: ${log.source}</span>
                                    <span class="event-time">${this.formatTime(log.time)}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderFirewall() {
        const firewall = this.securityData.firewall;
        
        return `
            <div class="firewall-section">
                <div class="firewall-status">
                    <h3>Firewall Status: <span class="status ${firewall.status}">${firewall.status.toUpperCase()}</span></h3>
                    <div class="firewall-controls">
                        <button class="btn-primary" onclick="securityComponent.addFirewallRule()">
                            <i class="fas fa-plus"></i> Add Rule
                        </button>
                        <button class="btn-secondary" onclick="securityComponent.reloadFirewall()">
                            <i class="fas fa-sync-alt"></i> Reload
                        </button>
                    </div>
                </div>
                
                <div class="firewall-rules">
                    <h4>Firewall Rules</h4>
                    <div class="rules-table">
                        <div class="table-header">
                            <div class="col-action">Action</div>
                            <div class="col-protocol">Protocol</div>
                            <div class="col-port">Port</div>
                            <div class="col-source">Source</div>
                            <div class="col-description">Description</div>
                            <div class="col-actions">Actions</div>
                        </div>
                        <div class="table-body">
                            ${firewall.rules.map(rule => `
                                <div class="table-row">
                                    <div class="col-action">
                                        <span class="action ${rule.action}">${rule.action}</span>
                                    </div>
                                    <div class="col-protocol">${rule.protocol.toUpperCase()}</div>
                                    <div class="col-port">${rule.port}</div>
                                    <div class="col-source">${rule.source}</div>
                                    <div class="col-description">${rule.description}</div>
                                    <div class="col-actions">
                                        <button class="btn-sm" onclick="securityComponent.editRule(${rule.id})" title="Edit">
                                            <i class="fas fa-edit"></i>
                                        </button>
                                        <button class="btn-sm btn-danger" onclick="securityComponent.deleteRule(${rule.id})" title="Delete">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                
                <div class="blocked-connections">
                    <h4>Recently Blocked Connections</h4>
                    <div class="blocked-list">
                        ${firewall.blockedConnections.map(conn => `
                            <div class="blocked-item">
                                <div class="blocked-ip">${conn.ip}</div>
                                <div class="blocked-details">
                                    <span>Attempts: ${conn.attempts}</span>
                                    <span>Reason: ${conn.reason}</span>
                                    <span>Last: ${this.formatTime(conn.lastAttempt)}</span>
                                </div>
                                <button class="btn-sm" onclick="securityComponent.unblockIP('${conn.ip}')" title="Unblock">
                                    <i class="fas fa-unlock"></i>
                                </button>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderCertificates() {
        return `
            <div class="certificates-section">
                <div class="certificates-toolbar">
                    <button class="btn-primary" onclick="securityComponent.generateCertificate()">
                        <i class="fas fa-plus"></i> Generate Certificate
                    </button>
                    <button class="btn-secondary" onclick="securityComponent.importCertificate()">
                        <i class="fas fa-upload"></i> Import
                    </button>
                </div>
                
                <div class="certificates-table">
                    <div class="table-header">
                        <div class="col-name">Certificate Name</div>
                        <div class="col-type">Type</div>
                        <div class="col-expiry">Expiry Date</div>
                        <div class="col-status">Status</div>
                        <div class="col-actions">Actions</div>
                    </div>
                    <div class="table-body">
                        ${this.securityData.certificates.map((cert, index) => `
                            <div class="table-row">
                                <div class="col-name">
                                    <i class="fas fa-certificate"></i>
                                    ${cert.name}
                                </div>
                                <div class="col-type">${cert.type}</div>
                                <div class="col-expiry">
                                    ${cert.expiry.toLocaleDateString()}
                                    <span class="days-left">(${this.getDaysUntilExpiry(cert.expiry)} days)</span>
                                </div>
                                <div class="col-status">
                                    <span class="cert-status ${cert.status}">${cert.status}</span>
                                </div>
                                <div class="col-actions">
                                    <button class="btn-sm" onclick="securityComponent.viewCertificate(${index})" title="View">
                                        <i class="fas fa-eye"></i>
                                    </button>
                                    <button class="btn-sm" onclick="securityComponent.renewCertificate(${index})" title="Renew">
                                        <i class="fas fa-sync-alt"></i>
                                    </button>
                                    <button class="btn-sm" onclick="securityComponent.exportCertificate(${index})" title="Export">
                                        <i class="fas fa-download"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderAuditLogs() {
        return `
            <div class="audit-section">
                <div class="audit-toolbar">
                    <div class="filter-controls">
                        <select onchange="securityComponent.filterLogs(this.value)">
                            <option value="all">All Levels</option>
                            <option value="critical">Critical</option>
                            <option value="warning">Warning</option>
                            <option value="info">Info</option>
                        </select>
                        <input type="date" onchange="securityComponent.filterByDate(this.value)">
                    </div>
                    <div class="export-controls">
                        <button class="btn-secondary" onclick="securityComponent.exportAuditLogs()">
                            <i class="fas fa-download"></i> Export Logs
                        </button>
                    </div>
                </div>
                
                <div class="audit-logs">
                    <div class="logs-table">
                        <div class="table-header">
                            <div class="col-time">Timestamp</div>
                            <div class="col-user">User</div>
                            <div class="col-action">Action</div>
                            <div class="col-source">Source</div>
                            <div class="col-level">Level</div>
                        </div>
                        <div class="table-body">
                            ${this.securityData.auditLogs.map(log => `
                                <div class="table-row ${log.level}">
                                    <div class="col-time">${this.formatTime(log.time)}</div>
                                    <div class="col-user">${log.user}</div>
                                    <div class="col-action">${log.action}</div>
                                    <div class="col-source">${log.source}</div>
                                    <div class="col-level">
                                        <span class="log-level ${log.level}">
                                            <i class="fas fa-${this.getLogIcon(log.level)}"></i>
                                            ${log.level.toUpperCase()}
                                        </span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getLogIcon(level) {
        const icons = {
            critical: 'times-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[level] || 'info-circle';
    }

    formatTime(date) {
        return date.toLocaleString();
    }

    getDaysUntilExpiry(expiryDate) {
        const now = new Date();
        const diffTime = expiryDate - now;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        return diffDays;
    }

    switchTab(tab) {
        this.activeTab = tab;
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.tab-btn').classList.add('active');
        document.getElementById('security-content').innerHTML = this.renderTabContent();
    }

    // Action handlers
    runFullScan() { console.log('Running full security scan...'); }
    updateSecurityRules() { console.log('Updating security rules...'); }
    addFirewallRule() { console.log('Adding firewall rule...'); }
    reloadFirewall() { console.log('Reloading firewall...'); }
    editRule(id) { console.log(`Editing rule: ${id}`); }
    deleteRule(id) { console.log(`Deleting rule: ${id}`); }
    unblockIP(ip) { console.log(`Unblocking IP: ${ip}`); }
    generateCertificate() { console.log('Generating certificate...'); }
    importCertificate() { console.log('Importing certificate...'); }
    viewCertificate(index) { console.log(`Viewing certificate: ${index}`); }
    renewCertificate(index) { console.log(`Renewing certificate: ${index}`); }
    exportCertificate(index) { console.log(`Exporting certificate: ${index}`); }
    filterLogs(level) { console.log(`Filtering logs by level: ${level}`); }
    filterByDate(date) { console.log(`Filtering logs by date: ${date}`); }
    exportAuditLogs() { console.log('Exporting audit logs...'); }

    startMonitoring() {
        setInterval(() => {
            // Update security metrics
            this.securityData.overview.blockedAttempts += Math.floor(Math.random() * 3);
            
            // Add new audit log occasionally
            if (Math.random() < 0.1) {
                this.securityData.auditLogs.unshift({
                    time: new Date(),
                    user: 'system',
                    action: 'Security scan completed',
                    source: 'localhost',
                    level: 'info'
                });
                
                if (this.securityData.auditLogs.length > 20) {
                    this.securityData.auditLogs.pop();
                }
            }
            
            if (document.getElementById('security-content')) {
                document.getElementById('security-content').innerHTML = this.renderTabContent();
            }
        }, 30000);
    }
}

window.securityComponent = new SecurityComponent();
