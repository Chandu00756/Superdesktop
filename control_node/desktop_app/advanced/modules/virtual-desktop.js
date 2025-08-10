// Virtual Desktop Manager - Extracted from omega-control-center.html
class VirtualDesktopManager extends EventTarget {
    constructor() {
        super();
        this.sessions = new Map();
        this.activeSessionId = null;
        this.sessionTemplates = new Map();
        this.isWebRTCEnabled = false;
        this.socketConnection = null;
        this.init();
    }

    init() {
        console.log('🖥️ Virtual Desktop Manager initializing...');
        this.loadSessionTemplates();
        this.setupWebRTCConnection();
        this.initializeEventListeners();
        this.createDefaultSession();
        this.startSessionMonitoring();
        console.log('🖥️ Virtual Desktop Manager ready');
    }

    loadSessionTemplates() {
        const templates = [
            {
                id: 'ubuntu-desktop',
                name: 'Ubuntu Desktop',
                type: 'vm',
                image: 'ubuntu:22.04-desktop',
                specs: { cpu: 2, memory: '4GB', storage: '20GB' },
                features: ['webrtc', 'gpu-acceleration', 'clipboard-sync'],
                category: 'linux'
            },
            {
                id: 'windows-10',
                name: 'Windows 10',
                type: 'vm',
                image: 'windows:10-pro',
                specs: { cpu: 4, memory: '8GB', storage: '40GB' },
                features: ['rdp', 'enhanced-session', 'usb-redirection'],
                category: 'windows'
            },
            {
                id: 'macos-monterey',
                name: 'macOS Monterey',
                type: 'vm',
                image: 'macos:monterey',
                specs: { cpu: 4, memory: '8GB', storage: '60GB' },
                features: ['vnc', 'screen-sharing', 'audio-sync'],
                category: 'macos'
            },
            {
                id: 'kali-security',
                name: 'Kali Linux Security',
                type: 'container',
                image: 'kali:latest',
                specs: { cpu: 2, memory: '4GB', storage: '10GB' },
                features: ['security-tools', 'network-analysis', 'penetration-testing'],
                category: 'security'
            },
            {
                id: 'dev-environment',
                name: 'Development Environment',
                type: 'container',
                image: 'omega-dev:latest',
                specs: { cpu: 4, memory: '6GB', storage: '15GB' },
                features: ['vscode', 'docker', 'node', 'python', 'git'],
                category: 'development'
            }
        ];

        templates.forEach(template => {
            this.sessionTemplates.set(template.id, template);
        });
    }

    setupWebRTCConnection() {
        if (typeof window !== 'undefined' && window.RTCPeerConnection) {
            this.isWebRTCEnabled = true;
            console.log('✅ WebRTC support detected');
        } else {
            console.log('❌ WebRTC not supported, falling back to VNC/RDP');
        }
    }

    initializeEventListeners() {
        // Listen for backend events
        if (typeof io !== 'undefined') {
            this.socketConnection = io('ws://localhost:8443/desktop');
            
            this.socketConnection.on('session-created', (data) => {
                this.handleSessionCreated(data);
            });

            this.socketConnection.on('session-destroyed', (data) => {
                this.handleSessionDestroyed(data);
            });

            this.socketConnection.on('session-error', (data) => {
                this.handleSessionError(data);
            });
        }
    }

    createDefaultSession() {
        const defaultSession = {
            id: 'local-desktop',
            name: 'Local Desktop',
            type: 'local',
            status: 'connected',
            protocol: 'native',
            created: new Date(),
            lastAccessed: new Date(),
            applications: [],
            performance: {
                cpu: 0,
                memory: 0,
                network: 0
            },
            features: ['native-performance', 'full-access', 'hardware-acceleration']
        };

        this.sessions.set(defaultSession.id, defaultSession);
        this.activeSessionId = defaultSession.id;
        
        this.dispatchEvent(new CustomEvent('sessionCreated', {
            detail: { session: defaultSession }
        }));
    }

    async createSession(config) {
        try {
            // Generate cryptographically secure session ID
            const randomValues = new Uint32Array(2);
            crypto.getRandomValues(randomValues);
            const sessionId = `session_${Date.now()}_${randomValues[0].toString(36)}${randomValues[1].toString(36)}`;
            
            // Get template if specified
            const template = config.templateId ? this.sessionTemplates.get(config.templateId) : null;
            
            // Create session object
            const session = {
                id: sessionId,
                name: config.name || (template ? template.name : 'New Session'),
                type: config.type || (template ? template.type : 'vm'),
                status: 'creating',
                protocol: this.determineProtocol(config),
                template: template,
                specs: config.specs || (template ? template.specs : { cpu: 2, memory: '2GB', storage: '10GB' }),
                created: new Date(),
                lastAccessed: new Date(),
                applications: [],
                performance: {
                    cpu: 0,
                    memory: 0,
                    network: 0
                },
                features: config.features || (template ? template.features : []),
                config: config
            };

            // Add to local sessions
            this.sessions.set(sessionId, session);

            // Dispatch creation event
            this.dispatchEvent(new CustomEvent('sessionCreated', {
                detail: { session }
            }));

            // Start creation process
            await this.startSessionCreation(session);

            return session;
        } catch (error) {
            console.error('Failed to create session:', error);
            this.dispatchEvent(new CustomEvent('sessionError', {
                detail: { error: error.message }
            }));
            throw error;
        }
    }

    determineProtocol(config) {
        if (config.protocol) return config.protocol;
        if (this.isWebRTCEnabled && config.type === 'vm') return 'webrtc';
        if (config.type === 'container') return 'vnc';
        return 'rdp';
    }

    async startSessionCreation(session) {
        try {
            // Update status
            session.status = 'provisioning';
            this.updateSession(session.id, { status: 'provisioning' });

            // Simulate provisioning process
            await this.simulateProvisioning(session);

            // Connect to session
            await this.connectToSession(session);

            // Update status
            session.status = 'connected';
            this.updateSession(session.id, { status: 'connected', lastAccessed: new Date() });

            this.dispatchEvent(new CustomEvent('sessionConnected', {
                detail: { session }
            }));

        } catch (error) {
            session.status = 'error';
            session.error = error.message;
            this.updateSession(session.id, { status: 'error', error: error.message });
            throw error;
        }
    }

    async simulateProvisioning(session) {
        // Simulate different provisioning times based on type
        const provisioningTime = session.type === 'container' ? 2000 : 5000;
        
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // 95% success rate
                if (Math.random() < 0.95) {
                    resolve();
                } else {
                    reject(new Error('Provisioning failed'));
                }
            }, provisioningTime);
        });
    }

    async connectToSession(session) {
        // Establish connection based on protocol
        switch (session.protocol) {
            case 'webrtc':
                await this.establishWebRTCConnection(session);
                break;
            case 'vnc':
                await this.establishVNCConnection(session);
                break;
            case 'rdp':
                await this.establishRDPConnection(session);
                break;
            case 'native':
                // Already connected
                break;
            default:
                throw new Error(`Unknown protocol: ${session.protocol}`);
        }
    }

    async establishWebRTCConnection(session) {
        // WebRTC connection logic
        console.log(`Establishing WebRTC connection for session ${session.id}`);
        
        if (!this.isWebRTCEnabled) {
            throw new Error('WebRTC not supported');
        }

        // Simulate WebRTC handshake
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        session.connectionDetails = {
            protocol: 'webrtc',
            quality: 'high',
            latency: 'low',
            bandwidth: 'unlimited'
        };
    }

    async establishVNCConnection(session) {
        console.log(`Establishing VNC connection for session ${session.id}`);
        
        // Simulate VNC connection
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        session.connectionDetails = {
            protocol: 'vnc',
            port: 5900 + Math.floor(Math.random() * 100),
            quality: 'medium',
            compression: true
        };
    }

    async establishRDPConnection(session) {
        console.log(`Establishing RDP connection for session ${session.id}`);
        
        // Simulate RDP connection
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        session.connectionDetails = {
            protocol: 'rdp',
            port: 3389,
            quality: 'high',
            remoteApp: true
        };
    }

    switchToSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        if (session.status !== 'connected') {
            throw new Error(`Session ${sessionId} is not connected`);
        }

        this.activeSessionId = sessionId;
        session.lastAccessed = new Date();

        this.dispatchEvent(new CustomEvent('sessionSwitched', {
            detail: { sessionId, session }
        }));
    }

    async closeSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        try {
            // Update status
            session.status = 'terminating';
            this.updateSession(sessionId, { status: 'terminating' });

            // Close connection
            await this.closeConnection(session);

            // Clean up resources
            await this.cleanupSession(session);

            // Remove from sessions
            this.sessions.delete(sessionId);

            // Switch to another session if this was active
            if (this.activeSessionId === sessionId) {
                const remainingSessions = Array.from(this.sessions.values());
                this.activeSessionId = remainingSessions.length > 0 ? remainingSessions[0].id : null;
            }

            this.dispatchEvent(new CustomEvent('sessionClosed', {
                detail: { sessionId, session }
            }));

        } catch (error) {
            console.error('Error closing session:', error);
            this.dispatchEvent(new CustomEvent('sessionError', {
                detail: { sessionId, error: error.message }
            }));
        }
    }

    async closeConnection(session) {
        console.log(`Closing connection for session ${session.id}`);
        // Simulate connection cleanup
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    async cleanupSession(session) {
        console.log(`Cleaning up resources for session ${session.id}`);
        // Simulate resource cleanup
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    updateSession(sessionId, updates) {
        const session = this.sessions.get(sessionId);
        if (session) {
            Object.assign(session, updates);
            this.dispatchEvent(new CustomEvent('sessionUpdated', {
                detail: { sessionId, session, updates }
            }));
        }
    }

    launchApplication(sessionId, appConfig) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        const app = {
            id: `app_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: appConfig.name,
            type: appConfig.type,
            executable: appConfig.executable,
            arguments: appConfig.arguments || [],
            window: {
                title: appConfig.name,
                width: appConfig.width || 800,
                height: appConfig.height || 600,
                x: appConfig.x || 100,
                y: appConfig.y || 100
            },
            launched: new Date(),
            pid: Math.floor(Math.random() * 10000) + 1000,
            status: 'running'
        };

        session.applications.push(app);

        this.dispatchEvent(new CustomEvent('applicationLaunched', {
            detail: { sessionId, app, session }
        }));

        return app;
    }

    closeApplication(sessionId, appId) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        const appIndex = session.applications.findIndex(app => app.id === appId);
        if (appIndex !== -1) {
            const app = session.applications.splice(appIndex, 1)[0];
            app.status = 'terminated';
            app.terminated = new Date();

            this.dispatchEvent(new CustomEvent('applicationClosed', {
                detail: { sessionId, app, session }
            }));
        }
    }

    startSessionMonitoring() {
        setInterval(() => {
            this.updateSessionMetrics();
        }, 2000);
    }

    updateSessionMetrics() {
        this.sessions.forEach(session => {
            if (session.status === 'connected') {
                // Simulate performance metrics
                session.performance = {
                    cpu: Math.floor(Math.random() * 80) + 10,
                    memory: Math.floor(Math.random() * 70) + 20,
                    network: Math.floor(Math.random() * 1000) + 100,
                    applications: session.applications.length
                };

                this.dispatchEvent(new CustomEvent('sessionMetricsUpdated', {
                    detail: { sessionId: session.id, metrics: session.performance }
                }));
            }
        });
    }

    // Event handlers for backend events
    handleSessionCreated(data) {
        console.log('Backend session created:', data);
    }

    handleSessionDestroyed(data) {
        console.log('Backend session destroyed:', data);
        if (this.sessions.has(data.sessionId)) {
            this.closeSession(data.sessionId);
        }
    }

    handleSessionError(data) {
        console.error('Backend session error:', data);
        this.dispatchEvent(new CustomEvent('sessionError', {
            detail: data
        }));
    }

    // Getters
    getActiveSessions() {
        return Array.from(this.sessions.values()).filter(session => 
            session.status === 'connected' || session.status === 'creating'
        );
    }

    getActiveSession() {
        return this.activeSessionId ? this.sessions.get(this.activeSessionId) : null;
    }

    getAllSessions() {
        return Array.from(this.sessions.values());
    }

    getSessionTemplates() {
        return Array.from(this.sessionTemplates.values());
    }

    getSessionById(sessionId) {
        return this.sessions.get(sessionId);
    }

    // Utility methods
    getSessionsByType(type) {
        return Array.from(this.sessions.values()).filter(session => session.type === type);
    }

    getSessionsByStatus(status) {
        return Array.from(this.sessions.values()).filter(session => session.status === status);
    }

    getSessionCount() {
        return this.sessions.size;
    }

    getActiveSessionCount() {
        return this.getActiveSessions().length;
    }
}

// Export for use
window.VirtualDesktopManager = VirtualDesktopManager;