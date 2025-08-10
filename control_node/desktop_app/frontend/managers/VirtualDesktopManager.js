/**
 * VirtualDesktopManager - Manages virtual desktop sessions
 */
class VirtualDesktopManager {
    constructor() {
        this.sessions = new Map();
        this.activeSessionId = null;
        this.sessionCounter = 0;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSessions();
    }

    /**
     * Create a new session
     */
    async createSession(config = {}) {
        const sessionId = `session-${++this.sessionCounter}-${Date.now()}`;
        
        const session = {
            id: sessionId,
            name: config.name || `Session ${this.sessionCounter}`,
            type: config.type || 'local',
            status: 'creating',
            created: new Date(),
            lastActive: new Date(),
            config: {
                resolution: config.resolution || '1920x1080',
                codec: config.codec || 'h264',
                bitrate: config.bitrate || 5000,
                fps: config.fps || 60,
                monitors: config.monitors || 1,
                ...config
            },
            metrics: {
                cpu: 0,
                memory: 0,
                network: 0,
                latency: 0,
                fps: 0,
                bitrate: 0
            },
            desktop: {
                canvas: null,
                stream: null,
                connection: null
            }
        };

        this.sessions.set(sessionId, session);

        try {
            // Initialize session based on type
            switch (session.type) {
                case 'remote':
                    await this.initializeRemoteSession(session);
                    break;
                case 'rdp':
                    await this.initializeRDPSession(session);
                    break;
                case 'vnc':
                    await this.initializeVNCSession(session);
                    break;
                default:
                    await this.initializeLocalSession(session);
            }

            session.status = 'connected';
            
            if (window.EventBus) {
                window.EventBus.emit('sessionCreated', session);
            }

            if (window.AppState) {
                window.AppState.dispatch({
                    type: 'SESSION_CREATED',
                    payload: { sessionId, session }
                });
            }

            return session;
        } catch (error) {
            session.status = 'error';
            session.error = error.message;
            
            if (window.EventBus) {
                window.EventBus.emit('sessionError', { sessionId, error: error.message });
            }
            
            throw error;
        }
    }

    /**
     * Get active sessions
     */
    getActiveSessions() {
        return Array.from(this.sessions.values()).filter(s => 
            s.status === 'connected' || s.status === 'paused'
        );
    }

    /**
     * Get session by ID
     */
    getSession(sessionId) {
        return this.sessions.get(sessionId);
    }

    /**
     * Switch to session
     */
    switchToSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        this.activeSessionId = sessionId;
        session.lastActive = new Date();

        if (window.EventBus) {
            window.EventBus.emit('sessionActivated', session);
        }

        if (window.AppState) {
            window.AppState.dispatch({
                type: 'SESSION_ACTIVATED',
                payload: { sessionId }
            });
        }

        return session;
    }

    /**
     * Pause session
     */
    async pauseSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) throw new Error(`Session ${sessionId} not found`);

        session.status = 'pausing';
        
        try {
            if (session.desktop.connection) {
                await session.desktop.connection.pause();
            }
            
            session.status = 'paused';
            
            if (window.EventBus) {
                window.EventBus.emit('sessionPaused', session);
            }

            return session;
        } catch (error) {
            session.status = 'error';
            throw error;
        }
    }

    /**
     * Resume session
     */
    async resumeSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) throw new Error(`Session ${sessionId} not found`);

        session.status = 'resuming';
        
        try {
            if (session.desktop.connection) {
                await session.desktop.connection.resume();
            }
            
            session.status = 'connected';
            session.lastActive = new Date();
            
            if (window.EventBus) {
                window.EventBus.emit('sessionResumed', session);
            }

            return session;
        } catch (error) {
            session.status = 'error';
            throw error;
        }
    }

    /**
     * Create session snapshot
     */
    async createSnapshot(sessionId, name) {
        const session = this.sessions.get(sessionId);
        if (!session) throw new Error(`Session ${sessionId} not found`);

        const snapshot = {
            id: `snapshot-${Date.now()}`,
            name: name || `Snapshot ${new Date().toLocaleString()}`,
            sessionId,
            created: new Date(),
            size: 0 // Will be set by backend
        };

        try {
            // Call backend to create snapshot
            const response = await fetch(`/api/sessions/${sessionId}/snapshots`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: snapshot.name })
            });

            if (!response.ok) {
                throw new Error('Failed to create snapshot');
            }

            const result = await response.json();
            snapshot.id = result.snapshotId;
            snapshot.size = result.size;

            if (!session.snapshots) session.snapshots = [];
            session.snapshots.push(snapshot);

            if (window.EventBus) {
                window.EventBus.emit('snapshotCreated', { session, snapshot });
            }

            return snapshot;
        } catch (error) {
            if (window.EventBus) {
                window.EventBus.emit('snapshotError', { sessionId, error: error.message });
            }
            throw error;
        }
    }

    /**
     * Terminate session
     */
    async terminateSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) throw new Error(`Session ${sessionId} not found`);

        session.status = 'terminating';

        try {
            // Clean up desktop connection
            if (session.desktop.connection) {
                await session.desktop.connection.disconnect();
                session.desktop.connection = null;
            }

            // Clean up canvas and stream
            if (session.desktop.canvas) {
                session.desktop.canvas.destroy();
                session.desktop.canvas = null;
            }

            if (session.desktop.stream) {
                session.desktop.stream.getTracks().forEach(track => track.stop());
                session.desktop.stream = null;
            }

            // Call backend to terminate
            await fetch(`/api/sessions/${sessionId}/terminate`, {
                method: 'POST'
            });

            this.sessions.delete(sessionId);

            if (this.activeSessionId === sessionId) {
                this.activeSessionId = null;
            }

            if (window.EventBus) {
                window.EventBus.emit('sessionTerminated', { sessionId });
            }

            if (window.AppState) {
                window.AppState.dispatch({
                    type: 'SESSION_TERMINATED',
                    payload: { sessionId }
                });
            }

        } catch (error) {
            session.status = 'error';
            throw error;
        }
    }

    /**
     * Update session metrics
     */
    updateMetrics(sessionId, metrics) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        session.metrics = { ...session.metrics, ...metrics };
        session.lastActive = new Date();

        if (window.EventBus) {
            window.EventBus.emit('sessionMetricsUpdated', { sessionId, metrics: session.metrics });
        }

        if (window.AppState) {
            window.AppState.updatePath(`metrics.sessions.${sessionId}`, session.metrics);
        }
    }

    // Private methods
    async initializeLocalSession(session) {
        // For local sessions, create a simple desktop interface
        session.config.endpoint = 'local';
        session.desktop.connection = new LocalDesktopConnection(session);
        await session.desktop.connection.connect();
    }

    async initializeRemoteSession(session) {
        // Initialize WebRTC connection for remote session
        const connection = new WebRTCDesktopConnection(session);
        session.desktop.connection = connection;
        
        await connection.connect();
    }

    async initializeRDPSession(session) {
        // Initialize RDP connection
        const connection = new RDPDesktopConnection(session);
        session.desktop.connection = connection;
        
        await connection.connect();
    }

    async initializeVNCSession(session) {
        // Initialize VNC connection
        const connection = new VNCDesktopConnection(session);
        session.desktop.connection = connection;
        
        await connection.connect();
    }

    bindEvents() {
        // Listen for window unload to clean up sessions
        window.addEventListener('beforeunload', () => {
            this.sessions.forEach(session => {
                if (session.desktop.connection) {
                    session.desktop.connection.disconnect();
                }
            });
        });

        // Start metrics collection
        this.startMetricsCollection();
    }

    startMetricsCollection() {
        setInterval(() => {
            this.sessions.forEach(session => {
                if (session.status === 'connected' && session.desktop.connection) {
                    const metrics = session.desktop.connection.getMetrics();
                    if (metrics) {
                        this.updateMetrics(session.id, metrics);
                    }
                }
            });
        }, 2000);
    }

    loadSessions() {
        // Load persisted sessions from localStorage if any
        try {
            const saved = localStorage.getItem('omega-sessions');
            if (saved) {
                const sessions = JSON.parse(saved);
                // Only restore session metadata, not active connections
                sessions.forEach(sessionData => {
                    if (sessionData.status === 'connected') {
                        sessionData.status = 'disconnected';
                    }
                    this.sessions.set(sessionData.id, sessionData);
                });
            }
        } catch (error) {
            console.warn('Failed to load sessions:', error);
        }
    }

    saveSessions() {
        try {
            const sessionsData = Array.from(this.sessions.values()).map(session => {
                // Don't persist connection objects or streams
                const { desktop, ...sessionData } = session;
                return {
                    ...sessionData,
                    desktop: {
                        canvas: null,
                        stream: null,
                        connection: null
                    }
                };
            });
            
            localStorage.setItem('omega-sessions', JSON.stringify(sessionsData));
        } catch (error) {
            console.warn('Failed to save sessions:', error);
        }
    }
}

/**
 * Base Desktop Connection class
 */
class DesktopConnection {
    constructor(session) {
        this.session = session;
        this.status = 'disconnected';
        this.metrics = {
            latency: 0,
            fps: 0,
            bitrate: 0,
            packetLoss: 0
        };
    }

    async connect() {
        throw new Error('connect() must be implemented by subclass');
    }

    async disconnect() {
        this.status = 'disconnected';
    }

    async pause() {
        this.status = 'paused';
    }

    async resume() {
        this.status = 'connected';
    }

    getMetrics() {
        return this.metrics;
    }

    updateQuality(settings) {
        // Override in subclasses
    }

    sendInput(inputData) {
        // Override in subclasses
    }
}

/**
 * Local Desktop Connection (for demo/development)
 */
class LocalDesktopConnection extends DesktopConnection {
    async connect() {
        this.status = 'connecting';
        
        // Simulate connection delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        this.status = 'connected';
        this.startMetricsSimulation();
    }

    startMetricsSimulation() {
        setInterval(() => {
            this.metrics = {
                latency: Math.random() * 50 + 10, // 10-60ms
                fps: 58 + Math.random() * 4, // 58-62 fps
                bitrate: 4800 + Math.random() * 400, // 4.8-5.2 Mbps
                packetLoss: Math.random() * 0.1 // 0-0.1%
            };
        }, 1000);
    }
}

// Global instance
window.VirtualDesktopManager = new VirtualDesktopManager();

export default VirtualDesktopManager;
