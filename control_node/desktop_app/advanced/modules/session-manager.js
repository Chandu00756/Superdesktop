/**
 * Omega SuperDesktop v2.0 - Session Manager Module
 * Extracted from omega-control-center.html - Handles session creation, management, and monitoring
 */

class SessionManager extends EventTarget {
    constructor() {
        super();
        this.sessions = new Map();
        this.selectedSession = null;
        this.sessionTemplates = new Map();
        this.performanceMetrics = new Map();
        this.initializeSampleSessions();
        this.initializeSessionTemplates();
    }

    initialize() {
        console.log('💻 Initializing Session Manager...');
        this.setupEventListeners();
        this.startPerformanceMonitoring();
        this.loadSessionsFromBackend();
        console.log('✅ Session Manager initialized');
        this.dispatchEvent(new CustomEvent('sessionManagerInitialized'));
    }

    initializeSampleSessions() {
        const sampleSessions = [
            {
                id: 'vm-ubuntu-01',
                name: 'Ubuntu 22.04 Desktop',
                type: 'vm',
                category: 'development',
                status: 'running',
                node: 'Node-01',
                resources: { cpu: 45, memory: 3200, network: 15 },
                os: 'Ubuntu 22.04 LTS',
                uptime: '2h 35m',
                created: new Date(Date.now() - 2.5 * 60 * 60 * 1000),
                owner: 'current_user'
            },
            {
                id: 'vm-windows-gaming',
                name: 'Windows 11 Gaming',
                type: 'vm',
                category: 'gaming',
                status: 'running',
                node: 'Node-02',
                resources: { cpu: 78, memory: 12100, gpu: 89, network: 45 },
                os: 'Windows 11 Pro',
                uptime: '4h 12m',
                created: new Date(Date.now() - 4.2 * 60 * 60 * 1000),
                owner: 'current_user'
            },
            {
                id: 'container-pytorch',
                name: 'PyTorch ML Training',
                type: 'container',
                category: 'ai',
                status: 'running',
                node: 'Node-03',
                resources: { cpu: 95, memory: 28000, gpu: 98, network: 25 },
                uptime: '2h 15m',
                created: new Date(Date.now() - 2.25 * 60 * 60 * 1000),
                owner: 'current_user',
                image: 'pytorch/pytorch:latest'
            },
            {
                id: 'vm-kali-security',
                name: 'Kali Linux Security',
                type: 'vm',
                category: 'security',
                status: 'paused',
                node: 'Node-01',
                resources: { cpu: 0, memory: 4096, network: 0 },
                os: 'Kali Linux 2023.4',
                uptime: '0m',
                created: new Date(Date.now() - 1 * 60 * 60 * 1000),
                owner: 'current_user'
            }
        ];

        sampleSessions.forEach(session => {
            this.sessions.set(session.id, session);
            this.performanceMetrics.set(session.id, {
                history: [],
                lastUpdate: new Date()
            });
        });
    }

    initializeSessionTemplates() {
        const templates = [
            {
                id: 'ubuntu-dev',
                name: 'Ubuntu Development',
                type: 'vm',
                category: 'development',
                os: 'Ubuntu 22.04 LTS',
                defaultResources: { cpu: 4, memory: 8192, storage: 50 },
                preinstalledSoftware: ['git', 'nodejs', 'python3', 'docker', 'vscode']
            },
            {
                id: 'windows-gaming',
                name: 'Windows Gaming',
                type: 'vm',
                category: 'gaming',
                os: 'Windows 11 Pro',
                defaultResources: { cpu: 8, memory: 16384, storage: 100, gpu: true },
                preinstalledSoftware: ['steam', 'discord', 'obs-studio']
            },
            {
                id: 'pytorch-ai',
                name: 'PyTorch ML Environment',
                type: 'container',
                category: 'ai',
                image: 'pytorch/pytorch:latest',
                defaultResources: { cpu: 8, memory: 32768, gpu: true },
                preinstalledSoftware: ['jupyter', 'numpy', 'pandas', 'opencv']
            },
            {
                id: 'kali-security',
                name: 'Kali Security Testing',
                type: 'vm',
                category: 'security',
                os: 'Kali Linux 2023.4',
                defaultResources: { cpu: 4, memory: 8192, storage: 40 },
                preinstalledSoftware: ['nmap', 'metasploit', 'wireshark', 'burpsuite']
            }
        ];

        templates.forEach(template => {
            this.sessionTemplates.set(template.id, template);
        });
    }

    async loadSessionsFromBackend() {
        try {
            if (window.superDesktopDataManager) {
                const sessionsData = await window.superDesktopDataManager.loadData('sessions');
                if (sessionsData && sessionsData.sessions) {
                    // Merge backend sessions with local sessions
                    sessionsData.sessions.forEach(session => {
                        this.sessions.set(session.id, session);
                    });
                    this.updateSessionDisplay();
                }
            }
        } catch (error) {
            console.warn('Failed to load sessions from backend:', error);
        }
    }

    setupEventListeners() {
        // Listen for session action buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-session-action]')) {
                const action = e.target.dataset.sessionAction;
                const sessionId = e.target.dataset.sessionId || this.selectedSession?.id;
                
                if (sessionId) {
                    this.handleSessionAction(action, sessionId);
                }
            }
        });

        // Listen for session selection
        document.addEventListener('click', (e) => {
            if (e.target.closest('.session-item')) {
                const sessionItem = e.target.closest('.session-item');
                const sessionId = sessionItem.dataset.sessionId;
                if (sessionId) {
                    this.selectSession(sessionId);
                }
            }
        });
    }

    handleSessionAction(action, sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        switch (action) {
            case 'start':
                this.startSession(sessionId);
                break;
            case 'pause':
                this.pauseSession(sessionId);
                break;
            case 'stop':
                this.stopSession(sessionId);
                break;
            case 'restart':
                this.restartSession(sessionId);
                break;
            case 'connect':
                this.connectToSession(sessionId);
                break;
            case 'delete':
                this.deleteSession(sessionId);
                break;
            case 'clone':
                this.cloneSession(sessionId);
                break;
            case 'export':
                this.exportSession(sessionId);
                break;
        }
    }

    async createSession(options) {
        const sessionId = `session-${Date.now()}`;
        const session = {
            id: sessionId,
            name: options.name || `New Session ${sessionId.slice(-6)}`,
            type: options.type || 'vm',
            category: options.category || 'general',
            status: 'creating',
            node: options.node || 'auto',
            resources: options.resources || { cpu: 0, memory: 0, network: 0 },
            os: options.os || 'Ubuntu 22.04 LTS',
            uptime: '0m',
            created: new Date(),
            owner: 'current_user',
            template: options.template
        };

        this.sessions.set(sessionId, session);
        this.performanceMetrics.set(sessionId, {
            history: [],
            lastUpdate: new Date()
        });

        // Simulate session creation
        setTimeout(() => {
            session.status = 'running';
            this.updateSessionDisplay();
            this.notifySessionChange('created', session);
        }, 2000);

        this.updateSessionDisplay();
        this.dispatchEvent(new CustomEvent('sessionCreated', { detail: session }));

        return sessionId;
    }

    createFromTemplate(templateId, customOptions = {}) {
        const template = this.sessionTemplates.get(templateId);
        if (!template) {
            throw new Error(`Template ${templateId} not found`);
        }

        const options = {
            ...template,
            ...customOptions,
            template: templateId,
            name: customOptions.name || template.name,
            resources: { ...template.defaultResources, ...customOptions.resources }
        };

        return this.createSession(options);
    }

    getAllSessions() {
        return Array.from(this.sessions.values());
    }

    getSessionsByStatus(status) {
        return this.getAllSessions().filter(session => session.status === status);
    }

    getSessionsByCategory(category) {
        return this.getAllSessions().filter(session => session.category === category);
    }

    getSessionsByNode(nodeId) {
        return this.getAllSessions().filter(session => session.node === nodeId);
    }

    getSession(id) {
        return this.sessions.get(id);
    }

    selectSession(id) {
        const session = this.sessions.get(id);
        if (!session) return;

        this.selectedSession = session;
        this.updateSessionUI();
        this.dispatchEvent(new CustomEvent('sessionSelected', { detail: session }));
    }

    async startSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        session.status = 'starting';
        this.updateSessionDisplay();

        try {
            // Call backend to start session
            if (window.superDesktopDataManager) {
                await window.superDesktopDataManager.makeRequest(`/api/sessions/${sessionId}/start`, {
                    method: 'POST'
                });
            }

            // Simulate startup time
            setTimeout(() => {
                session.status = 'running';
                session.uptime = '0m';
                this.updateSessionDisplay();
                this.notifySessionChange('started', session);
            }, 3000);

        } catch (error) {
            session.status = 'error';
            this.updateSessionDisplay();
            this.notifySessionChange('error', session, error.message);
        }
    }

    async pauseSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        session.status = 'pausing';
        this.updateSessionDisplay();

        try {
            if (window.superDesktopDataManager) {
                await window.superDesktopDataManager.makeRequest(`/api/sessions/${sessionId}/pause`, {
                    method: 'POST'
                });
            }

            setTimeout(() => {
                session.status = 'paused';
                this.updateSessionDisplay();
                this.notifySessionChange('paused', session);
            }, 1000);

        } catch (error) {
            session.status = 'error';
            this.updateSessionDisplay();
            this.notifySessionChange('error', session, error.message);
        }
    }

    async stopSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        session.status = 'stopping';
        this.updateSessionDisplay();

        try {
            if (window.superDesktopDataManager) {
                await window.superDesktopDataManager.makeRequest(`/api/sessions/${sessionId}/stop`, {
                    method: 'POST'
                });
            }

            setTimeout(() => {
                session.status = 'stopped';
                session.resources = { cpu: 0, memory: 0, network: 0 };
                this.updateSessionDisplay();
                this.notifySessionChange('stopped', session);
            }, 2000);

        } catch (error) {
            session.status = 'error';
            this.updateSessionDisplay();
            this.notifySessionChange('error', session, error.message);
        }
    }

    async restartSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        await this.stopSession(sessionId);
        setTimeout(() => {
            this.startSession(sessionId);
        }, 3000);
    }

    connectToSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session || session.status !== 'running') return;

        this.dispatchEvent(new CustomEvent('sessionConnectionRequested', {
            detail: { session, connectionType: 'vnc' }
        }));

        this.notifySessionChange('connected', session);
    }

    async deleteSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        if (!confirm(`Are you sure you want to delete session "${session.name}"?`)) {
            return;
        }

        try {
            if (window.superDesktopDataManager) {
                await window.superDesktopDataManager.makeRequest(`/api/sessions/${sessionId}`, {
                    method: 'DELETE'
                });
            }

            this.sessions.delete(sessionId);
            this.performanceMetrics.delete(sessionId);
            
            if (this.selectedSession?.id === sessionId) {
                this.selectedSession = null;
            }

            this.updateSessionDisplay();
            this.notifySessionChange('deleted', session);

        } catch (error) {
            this.notifySessionChange('error', session, 'Failed to delete session');
        }
    }

    cloneSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        const cloneOptions = {
            name: `${session.name} (Clone)`,
            type: session.type,
            category: session.category,
            os: session.os,
            resources: { ...session.resources }
        };

        this.createSession(cloneOptions);
        this.notifySessionChange('cloned', session);
    }

    exportSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        const exportData = {
            session: { ...session },
            metrics: this.performanceMetrics.get(sessionId),
            exported: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `session-${session.name.replace(/\s+/g, '-')}-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.notifySessionChange('exported', session);
    }

    pauseAllSessions() {
        const runningSessions = this.getSessionsByStatus('running');
        runningSessions.forEach(session => {
            this.pauseSession(session.id);
        });
    }

    resumeAllSessions() {
        const pausedSessions = this.getSessionsByStatus('paused');
        pausedSessions.forEach(session => {
            this.startSession(session.id);
        });
    }

    terminateAllSessions() {
        const activeSessions = this.getAllSessions().filter(s => 
            ['running', 'paused', 'starting'].includes(s.status)
        );
        activeSessions.forEach(session => {
            this.stopSession(session.id);
        });
    }

    updateSessionUI() {
        if (!this.selectedSession) return;

        // Update session details in UI
        this.updateElementText('selectedSessionTitle', this.selectedSession.name);
        this.updateElementText('selectedSessionInfo', 
            `${this.selectedSession.type} • ${this.selectedSession.node} • Running for ${this.selectedSession.uptime}`);

        // Update session information panel
        this.updateElementText('sessionTypeInfo', this.selectedSession.type);
        this.updateElementText('sessionOSInfo', this.selectedSession.os || 'N/A');
        this.updateElementText('sessionUptimeInfo', this.selectedSession.uptime);
        this.updateElementText('sessionNodeInfo', this.selectedSession.node);
        this.updateElementText('sessionStatusInfo', this.selectedSession.status);

        // Update resource usage
        const resources = this.selectedSession.resources;
        this.updateElementText('sessionCPUUsage', `${resources.cpu || 0}%`);
        this.updateElementText('sessionMemoryUsage', `${resources.memory || 0} MB`);
        this.updateElementText('sessionNetworkUsage', `${resources.network || 0} MB/s`);

        // Update session action buttons
        this.updateSessionActionButtons();
    }

    updateSessionActionButtons() {
        if (!this.selectedSession) return;

        const status = this.selectedSession.status;
        const actions = {
            start: ['stopped', 'paused', 'error'],
            pause: ['running'],
            stop: ['running', 'paused'],
            restart: ['running', 'stopped', 'error'],
            connect: ['running']
        };

        Object.keys(actions).forEach(action => {
            const button = document.querySelector(`[data-session-action="${action}"]`);
            if (button) {
                button.disabled = !actions[action].includes(status);
            }
        });
    }

    updateSessionDisplay() {
        // Update session list in UI
        const sessionsList = document.getElementById('sessionsList');
        if (sessionsList) {
            this.renderSessionsList(sessionsList);
        }

        // Update session count
        const sessionCount = this.getAllSessions().length;
        const runningCount = this.getSessionsByStatus('running').length;
        
        this.updateElementText('totalSessions', sessionCount);
        this.updateElementText('runningSessions', runningCount);
        this.updateElementText('activeSessions', runningCount);
    }

    renderSessionsList(container) {
        const sessions = this.getAllSessions();
        
        container.innerHTML = sessions.map(session => `
            <div class="session-item ${session.status} ${this.selectedSession?.id === session.id ? 'selected' : ''}" 
                 data-session-id="${session.id}">
                <div class="session-header">
                    <div class="session-name">${session.name}</div>
                    <div class="session-status ${session.status}">${session.status}</div>
                </div>
                <div class="session-details">
                    <span class="session-type">${session.type}</span>
                    <span class="session-node">${session.node}</span>
                    <span class="session-uptime">${session.uptime}</span>
                </div>
                <div class="session-resources">
                    <div class="resource-bar">
                        <label>CPU: ${session.resources.cpu || 0}%</label>
                        <div class="bar"><div class="fill" style="width: ${session.resources.cpu || 0}%"></div></div>
                    </div>
                    <div class="resource-bar">
                        <label>Memory: ${session.resources.memory || 0} MB</label>
                        <div class="bar"><div class="fill" style="width: ${Math.min(100, (session.resources.memory || 0) / 100)}%"></div></div>
                    </div>
                </div>
                <div class="session-actions">
                    <button data-session-action="connect" data-session-id="${session.id}" 
                            ${session.status !== 'running' ? 'disabled' : ''}>Connect</button>
                    <button data-session-action="pause" data-session-id="${session.id}"
                            ${session.status !== 'running' ? 'disabled' : ''}>Pause</button>
                    <button data-session-action="stop" data-session-id="${session.id}"
                            ${!['running', 'paused'].includes(session.status) ? 'disabled' : ''}>Stop</button>
                </div>
            </div>
        `).join('');
    }

    startPerformanceMonitoring() {
        setInterval(() => {
            this.updateSessionMetrics();
        }, 5000);
    }

    updateSessionMetrics() {
        this.sessions.forEach((session, sessionId) => {
            if (session.status === 'running') {
                // Simulate performance metrics updates
                const metrics = this.performanceMetrics.get(sessionId);
                if (metrics) {
                    const newMetric = {
                        timestamp: new Date(),
                        cpu: session.resources.cpu + (Math.random() - 0.5) * 10,
                        memory: session.resources.memory + (Math.random() - 0.5) * 100,
                        network: session.resources.network + (Math.random() - 0.5) * 5
                    };

                    metrics.history.push(newMetric);
                    if (metrics.history.length > 100) {
                        metrics.history.shift();
                    }

                    // Update current values
                    session.resources.cpu = Math.max(0, Math.min(100, newMetric.cpu));
                    session.resources.memory = Math.max(0, newMetric.memory);
                    session.resources.network = Math.max(0, newMetric.network);

                    // Update uptime
                    const uptimeMs = Date.now() - session.created.getTime();
                    session.uptime = this.formatUptime(uptimeMs);
                }
            }
        });

        this.updateSessionDisplay();
        if (this.selectedSession) {
            this.updateSessionUI();
        }
    }

    formatUptime(ms) {
        const hours = Math.floor(ms / (1000 * 60 * 60));
        const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }

    notifySessionChange(action, session, message = null) {
        const notification = {
            title: `Session ${action}`,
            message: message || `Session "${session.name}" has been ${action}`,
            type: action === 'error' ? 'error' : 'info'
        };

        if (window.menuBarManager) {
            window.menuBarManager.showNotification(
                notification.title,
                notification.message,
                notification.type
            );
        }
    }

    getSessionTemplates() {
        return Array.from(this.sessionTemplates.values());
    }

    getPerformanceMetrics(sessionId) {
        return this.performanceMetrics.get(sessionId);
    }

    updateElementText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // Public API
    getSessionStats() {
        const sessions = this.getAllSessions();
        return {
            total: sessions.length,
            running: sessions.filter(s => s.status === 'running').length,
            paused: sessions.filter(s => s.status === 'paused').length,
            stopped: sessions.filter(s => s.status === 'stopped').length,
            error: sessions.filter(s => s.status === 'error').length
        };
    }

    dispose() {
        console.log('🧹 Session Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.SessionManager = SessionManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SessionManager;
}
