/**
 * WebRTCClient - Real-time desktop streaming via WebRTC
 */
class WebRTCClient {
    constructor(config = {}) {
        this.config = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
                ...config.iceServers || []
            ],
            signalingUrl: config.signalingUrl || '/api/signaling',
            token: config.token || null,
            ...config
        };

        this.peerConnection = null;
        this.dataChannel = null;
        this.localStream = null;
        this.remoteStream = null;
        this.socket = null;
        this.sessionId = null;
        this.status = 'disconnected';
        
        this.stats = {
            rtt: 0,
            bitrate: 0,
            packetLoss: 0,
            framesPerSecond: 0,
            timestamp: 0
        };

        this.callbacks = {
            onConnected: null,
            onDisconnected: null,
            onStream: null,
            onStats: null,
            onError: null,
            onDataChannel: null
        };

        this.init();
    }

    init() {
        this.setupPeerConnection();
        this.startStatsCollection();
    }

    /**
     * Connect to remote desktop session
     */
    async connect(sessionConfig) {
        try {
            this.sessionId = sessionConfig.sessionId;
            this.status = 'connecting';

            // Connect to signaling server
            await this.connectSignaling();

            // Create offer and start connection
            await this.createOffer();

            if (this.callbacks.onConnected) {
                this.callbacks.onConnected();
            }

            if (window.EventBus) {
                window.EventBus.emit('webrtcConnected', { sessionId: this.sessionId });
            }

        } catch (error) {
            this.status = 'error';
            this.handleError(error);
            throw error;
        }
    }

    /**
     * Disconnect from session
     */
    async disconnect() {
        this.status = 'disconnecting';

        try {
            // Close data channel
            if (this.dataChannel) {
                this.dataChannel.close();
                this.dataChannel = null;
            }

            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }

            // Close signaling connection
            if (this.socket) {
                this.socket.close();
                this.socket = null;
            }

            // Stop local stream
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => track.stop());
                this.localStream = null;
            }

            this.status = 'disconnected';

            if (this.callbacks.onDisconnected) {
                this.callbacks.onDisconnected();
            }

            if (window.EventBus) {
                window.EventBus.emit('webrtcDisconnected', { sessionId: this.sessionId });
            }

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Send input events (mouse, keyboard)
     */
    sendInput(inputData) {
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            try {
                this.dataChannel.send(JSON.stringify({
                    type: 'input',
                    data: inputData,
                    timestamp: Date.now()
                }));
            } catch (error) {
                console.error('Failed to send input:', error);
            }
        }
    }

    /**
     * Send clipboard data
     */
    sendClipboard(clipboardData) {
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            try {
                this.dataChannel.send(JSON.stringify({
                    type: 'clipboard',
                    data: clipboardData,
                    timestamp: Date.now()
                }));
            } catch (error) {
                console.error('Failed to send clipboard:', error);
            }
        }
    }

    /**
     * Update quality settings
     */
    async updateQuality(settings) {
        if (!this.peerConnection) return;

        try {
            const senders = this.peerConnection.getSenders();
            
            for (const sender of senders) {
                if (sender.track && sender.track.kind === 'video') {
                    const params = sender.getParameters();
                    
                    if (params.encodings && params.encodings.length > 0) {
                        const encoding = params.encodings[0];
                        
                        if (settings.maxBitrate) {
                            encoding.maxBitrate = settings.maxBitrate * 1000; // Convert to bps
                        }
                        
                        if (settings.maxFramerate) {
                            encoding.maxFramerate = settings.maxFramerate;
                        }
                        
                        await sender.setParameters(params);
                    }
                }
            }

            if (window.EventBus) {
                window.EventBus.emit('webrtcQualityUpdated', { sessionId: this.sessionId, settings });
            }

        } catch (error) {
            console.error('Failed to update quality:', error);
        }
    }

    /**
     * Get connection statistics
     */
    getStats() {
        return { ...this.stats };
    }

    /**
     * Set event callbacks
     */
    on(event, callback) {
        if (this.callbacks.hasOwnProperty(`on${event.charAt(0).toUpperCase() + event.slice(1)}`)) {
            this.callbacks[`on${event.charAt(0).toUpperCase() + event.slice(1)}`] = callback;
        }
    }

    // Private methods
    setupPeerConnection() {
        this.peerConnection = new RTCPeerConnection({
            iceServers: this.config.iceServers
        });

        // Handle remote stream
        this.peerConnection.ontrack = (event) => {
            this.remoteStream = event.streams[0];
            if (this.callbacks.onStream) {
                this.callbacks.onStream(this.remoteStream);
            }
        };

        // Handle ICE candidates
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.socket) {
                this.socket.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate,
                    sessionId: this.sessionId
                }));
            }
        };

        // Handle connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            const state = this.peerConnection.connectionState;
            
            if (state === 'connected') {
                this.status = 'connected';
            } else if (state === 'disconnected' || state === 'failed') {
                this.status = 'disconnected';
                this.handleError(new Error(`Connection ${state}`));
            }

            if (window.EventBus) {
                window.EventBus.emit('webrtcStateChange', { 
                    sessionId: this.sessionId, 
                    state,
                    status: this.status 
                });
            }
        };

        // Create data channel for input and clipboard
        this.dataChannel = this.peerConnection.createDataChannel('omega-control', {
            ordered: true
        });

        this.dataChannel.onopen = () => {
            if (this.callbacks.onDataChannel) {
                this.callbacks.onDataChannel();
            }
        };

        this.dataChannel.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleDataChannelMessage(message);
            } catch (error) {
                console.error('Failed to parse data channel message:', error);
            }
        };
    }

    async connectSignaling() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}${this.config.signalingUrl}`;
            
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                // Join session
                this.socket.send(JSON.stringify({
                    type: 'join-session',
                    sessionId: this.sessionId,
                    token: this.config.token
                }));
                resolve();
            };

            this.socket.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    await this.handleSignalingMessage(message);
                } catch (error) {
                    console.error('Signaling message error:', error);
                }
            };

            this.socket.onerror = (error) => {
                reject(new Error('Signaling connection failed'));
            };

            this.socket.onclose = () => {
                if (this.status === 'connected') {
                    this.handleError(new Error('Signaling connection lost'));
                }
            };
        });
    }

    async createOffer() {
        try {
            const offer = await this.peerConnection.createOffer({
                offerToReceiveVideo: true,
                offerToReceiveAudio: true
            });

            await this.peerConnection.setLocalDescription(offer);

            this.socket.send(JSON.stringify({
                type: 'offer',
                offer: offer,
                sessionId: this.sessionId
            }));
        } catch (error) {
            throw new Error(`Failed to create offer: ${error.message}`);
        }
    }

    async handleSignalingMessage(message) {
        switch (message.type) {
            case 'answer':
                await this.peerConnection.setRemoteDescription(new RTCSessionDescription(message.answer));
                break;

            case 'ice-candidate':
                await this.peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                break;

            case 'error':
                this.handleError(new Error(message.error));
                break;

            case 'session-ready':
                // Session is ready to receive connections
                break;

            default:
                console.warn('Unknown signaling message type:', message.type);
        }
    }

    handleDataChannelMessage(message) {
        switch (message.type) {
            case 'clipboard':
                if (window.EventBus) {
                    window.EventBus.emit('clipboardReceived', message.data);
                }
                break;

            case 'file-transfer':
                if (window.EventBus) {
                    window.EventBus.emit('fileTransferReceived', message.data);
                }
                break;

            case 'system-info':
                if (window.EventBus) {
                    window.EventBus.emit('remoteSystemInfo', message.data);
                }
                break;

            default:
                console.warn('Unknown data channel message type:', message.type);
        }
    }

    startStatsCollection() {
        setInterval(async () => {
            if (this.peerConnection && this.status === 'connected') {
                try {
                    const stats = await this.peerConnection.getStats();
                    this.processStats(stats);
                } catch (error) {
                    console.error('Failed to get stats:', error);
                }
            }
        }, 1000);
    }

    processStats(stats) {
        let inboundRtp = null;
        let remoteInboundRtp = null;
        let candidatePair = null;

        stats.forEach((report) => {
            if (report.type === 'inbound-rtp' && report.kind === 'video') {
                inboundRtp = report;
            } else if (report.type === 'remote-inbound-rtp' && report.kind === 'video') {
                remoteInboundRtp = report;
            } else if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                candidatePair = report;
            }
        });

        // Calculate metrics
        const currentTime = Date.now();
        const timeDelta = currentTime - this.stats.timestamp;

        if (inboundRtp && timeDelta > 0) {
            // Calculate bitrate
            const bytesDelta = inboundRtp.bytesReceived - (this.stats.lastBytesReceived || 0);
            this.stats.bitrate = Math.round((bytesDelta * 8) / (timeDelta / 1000) / 1000); // kbps
            this.stats.lastBytesReceived = inboundRtp.bytesReceived;

            // Calculate frame rate
            const framesDelta = inboundRtp.framesDecoded - (this.stats.lastFramesDecoded || 0);
            this.stats.framesPerSecond = Math.round(framesDelta / (timeDelta / 1000));
            this.stats.lastFramesDecoded = inboundRtp.framesDecoded;

            // Packet loss
            if (inboundRtp.packetsLost !== undefined) {
                this.stats.packetLoss = inboundRtp.packetsLost;
            }
        }

        // RTT from candidate pair
        if (candidatePair && candidatePair.currentRoundTripTime !== undefined) {
            this.stats.rtt = Math.round(candidatePair.currentRoundTripTime * 1000); // ms
        }

        this.stats.timestamp = currentTime;

        // Notify callbacks
        if (this.callbacks.onStats) {
            this.callbacks.onStats(this.stats);
        }

        if (window.EventBus) {
            window.EventBus.emit('webrtcStats', {
                sessionId: this.sessionId,
                stats: this.stats
            });
        }
    }

    handleError(error) {
        console.error('WebRTC Error:', error);
        
        if (this.callbacks.onError) {
            this.callbacks.onError(error);
        }

        if (window.EventBus) {
            window.EventBus.emit('webrtcError', {
                sessionId: this.sessionId,
                error: error.message
            });
        }
    }
}

/**
 * WebRTC Desktop Connection implementation
 */
class WebRTCDesktopConnection extends DesktopConnection {
    constructor(session) {
        super(session);
        this.client = null;
    }

    async connect() {
        this.status = 'connecting';

        try {
            // Get connection config from backend
            const response = await fetch('/api/sessions/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: 'webrtc',
                    config: this.session.config
                })
            });

            if (!response.ok) {
                throw new Error('Failed to create session');
            }

            const config = await response.json();

            // Initialize WebRTC client
            this.client = new WebRTCClient({
                signalingUrl: config.signalingUrl,
                iceServers: config.iceServers,
                token: config.token
            });

            // Set up event handlers
            this.client.on('connected', () => {
                this.status = 'connected';
            });

            this.client.on('disconnected', () => {
                this.status = 'disconnected';
            });

            this.client.on('stream', (stream) => {
                this.session.desktop.stream = stream;
                if (window.EventBus) {
                    window.EventBus.emit('desktopStreamReceived', {
                        sessionId: this.session.id,
                        stream
                    });
                }
            });

            this.client.on('stats', (stats) => {
                this.metrics = {
                    latency: stats.rtt,
                    fps: stats.framesPerSecond,
                    bitrate: stats.bitrate,
                    packetLoss: stats.packetLoss
                };
            });

            this.client.on('error', (error) => {
                this.status = 'error';
                if (window.EventBus) {
                    window.EventBus.emit('sessionError', {
                        sessionId: this.session.id,
                        error: error.message
                    });
                }
            });

            // Connect to session
            await this.client.connect({
                sessionId: config.sessionId
            });

        } catch (error) {
            this.status = 'error';
            throw error;
        }
    }

    async disconnect() {
        if (this.client) {
            await this.client.disconnect();
            this.client = null;
        }
        await super.disconnect();
    }

    async pause() {
        // For WebRTC, pause means stopping the stream temporarily
        if (this.client && this.session.desktop.stream) {
            this.session.desktop.stream.getTracks().forEach(track => {
                track.enabled = false;
            });
        }
        await super.pause();
    }

    async resume() {
        // Resume the stream
        if (this.client && this.session.desktop.stream) {
            this.session.desktop.stream.getTracks().forEach(track => {
                track.enabled = true;
            });
        }
        await super.resume();
    }

    updateQuality(settings) {
        if (this.client) {
            this.client.updateQuality(settings);
        }
    }

    sendInput(inputData) {
        if (this.client) {
            this.client.sendInput(inputData);
        }
    }

    sendClipboard(data) {
        if (this.client) {
            this.client.sendClipboard(data);
        }
    }

    getMetrics() {
        return this.client ? this.client.getStats() : this.metrics;
    }
}

// Export classes
window.WebRTCClient = WebRTCClient;
window.WebRTCDesktopConnection = WebRTCDesktopConnection;

export { WebRTCClient, WebRTCDesktopConnection };
