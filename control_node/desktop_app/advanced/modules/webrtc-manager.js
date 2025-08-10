/**
 * Omega SuperDesktop v2.0 - WebRTC Session Manager Module
 * Extracted from omega-control-center.html - Handles WebRTC connections for remote sessions
 */

class WebRTCSessionManager extends EventTarget {
    constructor() {
        super();
        this.peerConnections = new Map();
        this.dataChannels = new Map();
        this.videoStreams = new Map();
        this.configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'turn:superdesktop-turn.local:3478', username: 'omega', credential: 'cluster' }
            ]
        };
        this.baseURL = window.location.origin;
    }

    initialize() {
        console.log('🌐 Initializing WebRTC Session Manager...');
        this.setupEventListeners();
        console.log('✅ WebRTC Session Manager initialized');
        this.dispatchEvent(new CustomEvent('webrtcManagerInitialized'));
    }

    setupEventListeners() {
        window.connectWebRTC = (sessionId) => {
            this.connectWebRTC(sessionId);
        };

        window.disconnectWebRTC = (sessionId) => {
            this.disconnectSession(sessionId);
        };

        window.toggleClipboard = (sessionId) => {
            this.toggleClipboard(sessionId);
        };

        window.toggleFileSharing = (sessionId) => {
            this.toggleFileSharing(sessionId);
        };

        window.takeScreenshot = (sessionId) => {
            this.takeScreenshot(sessionId);
        };

        window.closeSessionWindow = (sessionId) => {
            this.closeSessionWindow(sessionId);
        };
    }

    async connectWebRTC(sessionId) {
        try {
            const peerConnection = new RTCPeerConnection(this.configuration);
            this.peerConnections.set(sessionId, peerConnection);

            // Set up data channel for control commands
            const dataChannel = peerConnection.createDataChannel('control', {
                ordered: true
            });
            this.dataChannels.set(sessionId, dataChannel);

            dataChannel.onopen = () => {
                console.log(`Data channel opened for session ${sessionId}`);
                this.sendControlCommand(sessionId, { type: 'init', timestamp: Date.now() });
            };

            dataChannel.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleControlMessage(sessionId, message);
            };

            // Handle incoming video stream
            peerConnection.ontrack = (event) => {
                const [stream] = event.streams;
                this.videoStreams.set(sessionId, stream);
                this.displayVideoStream(sessionId, stream);
            };

            // Create offer and establish connection
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            // Simulate WebRTC connection (replace with actual backend call)
            const mockAnswer = {
                type: 'answer',
                sdp: 'v=0\r\no=- 123456789 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
            };

            setTimeout(async () => {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(mockAnswer));
                this.showSessionWindow(sessionId);
                this.showNotification(`WebRTC connection established for ${sessionId}`, 'success');
            }, 1000);

        } catch (error) {
            console.error('WebRTC connection failed:', error);
            this.showNotification(`Failed to connect to ${sessionId}`, 'error');
        }
    }

    displayVideoStream(sessionId, stream) {
        const canvas = document.querySelector(`#preview-${sessionId} .session-thumbnail`);
        if (canvas) {
            const video = document.createElement('video');
            video.srcObject = stream;
            video.autoplay = true;
            video.muted = true;
            
            video.onloadedmetadata = () => {
                const ctx = canvas.getContext('2d');
                const drawFrame = () => {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    requestAnimationFrame(drawFrame);
                };
                drawFrame();
            };
        }
    }

    sendControlCommand(sessionId, command) {
        const dataChannel = this.dataChannels.get(sessionId);
        if (dataChannel && dataChannel.readyState === 'open') {
            dataChannel.send(JSON.stringify(command));
        }
    }

    handleControlMessage(sessionId, message) {
        switch (message.type) {
            case 'metrics':
                this.updateSessionMetricsUI(sessionId, message.data);
                break;
            case 'screenshot':
                this.updateSessionThumbnail(sessionId, message.data);
                break;
            case 'status':
                this.updateSessionStatus(sessionId, message.data);
                break;
        }
    }

    showSessionWindow(sessionId) {
        const sessionWindow = document.createElement('div');
        sessionWindow.className = 'session-fullscreen-window';
        sessionWindow.id = `window-${sessionId}`;
        sessionWindow.innerHTML = `
            <div class="session-window-header">
                <div class="session-window-title">
                    <h3>Session: ${sessionId}</h3>
                    <div class="session-window-controls">
                        <button type="button" onclick="window.toggleClipboard('${sessionId}')">
                            <i class="fas fa-clipboard"></i>
                        </button>
                        <button type="button" onclick="window.toggleFileSharing('${sessionId}')">
                            <i class="fas fa-folder"></i>
                        </button>
                        <button type="button" onclick="window.takeScreenshot('${sessionId}')">
                            <i class="fas fa-camera"></i>
                        </button>
                        <button type="button" onclick="window.closeSessionWindow('${sessionId}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            </div>
            <div class="session-window-content">
                <video id="video-${sessionId}" autoplay controls></video>
            </div>
        `;
        
        document.body.appendChild(sessionWindow);
        
        const video = document.getElementById(`video-${sessionId}`);
        const stream = this.videoStreams.get(sessionId);
        if (video && stream) {
            video.srcObject = stream;
        }
    }

    closeSessionWindow(sessionId) {
        const window = document.getElementById(`window-${sessionId}`);
        if (window) {
            window.remove();
        }
    }

    toggleClipboard(sessionId) {
        this.showNotification(`Clipboard sharing toggled for ${sessionId}`, 'info');
    }

    toggleFileSharing(sessionId) {
        this.showNotification(`File sharing toggled for ${sessionId}`, 'info');
    }

    takeScreenshot(sessionId) {
        this.showNotification(`Screenshot taken for ${sessionId}`, 'success');
    }

    disconnectSession(sessionId) {
        const peerConnection = this.peerConnections.get(sessionId);
        if (peerConnection) {
            peerConnection.close();
            this.peerConnections.delete(sessionId);
        }
        
        const dataChannel = this.dataChannels.get(sessionId);
        if (dataChannel) {
            dataChannel.close();
            this.dataChannels.delete(sessionId);
        }
        
        this.videoStreams.delete(sessionId);
        this.closeSessionWindow(sessionId);
        
        this.showNotification(`Disconnected from ${sessionId}`, 'info');
    }

    showNotification(message, type) {
        if (window.menuBarManager) {
            window.menuBarManager.showNotification('WebRTC', message, type);
        } else {
            console.log(`WebRTC: ${message}`);
        }
    }

    dispose() {
        this.peerConnections.forEach((connection, sessionId) => {
            this.disconnectSession(sessionId);
        });
        console.log('🧹 WebRTC Session Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.WebRTCSessionManager = WebRTCSessionManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebRTCSessionManager;
}
