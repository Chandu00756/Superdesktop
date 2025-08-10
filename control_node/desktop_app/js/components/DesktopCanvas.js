/**
 * DesktopCanvas - Desktop rendering and interaction component
 */
class DesktopCanvas {
    constructor(container, session) {
        this.container = container;
        this.session = session;
        this.canvas = null;
        this.ctx = null;
        this.video = null;
        this.stream = null;
        
        this.state = {
            isFullscreen: false,
            pointerLocked: false,
            scale: 1,
            offsetX: 0,
            offsetY: 0,
            fitMode: 'contain' // contain, cover, actual
        };

        this.input = {
            keys: new Set(),
            mouseButtons: new Set(),
            lastMousePos: { x: 0, y: 0 }
        };

        this.hud = null;
        this.controls = null;
        
        this.init();
    }

    init() {
        this.createCanvas();
        this.createVideo();
        this.createHUD();
        this.createControls();
        this.bindEvents();
    }

    /**
     * Set video stream for the desktop
     */
    setStream(stream) {
        this.stream = stream;
        this.video.srcObject = stream;
        this.video.play();
        
        this.video.onloadedmetadata = () => {
            this.updateCanvasSize();
            this.drawFrame();
        };
    }

    /**
     * Update canvas to match video dimensions
     */
    updateCanvasSize() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;

        const containerRect = this.container.getBoundingClientRect();
        const videoAspect = this.video.videoWidth / this.video.videoHeight;
        const containerAspect = containerRect.width / containerRect.height;

        let canvasWidth, canvasHeight;

        switch (this.state.fitMode) {
            case 'cover':
                if (videoAspect > containerAspect) {
                    canvasHeight = containerRect.height;
                    canvasWidth = canvasHeight * videoAspect;
                } else {
                    canvasWidth = containerRect.width;
                    canvasHeight = canvasWidth / videoAspect;
                }
                break;
                
            case 'actual':
                canvasWidth = this.video.videoWidth;
                canvasHeight = this.video.videoHeight;
                break;
                
            default: // contain
                if (videoAspect > containerAspect) {
                    canvasWidth = containerRect.width;
                    canvasHeight = canvasWidth / videoAspect;
                } else {
                    canvasHeight = containerRect.height;
                    canvasWidth = canvasHeight * videoAspect;
                }
        }

        this.canvas.width = canvasWidth;
        this.canvas.height = canvasHeight;
        
        // Center the canvas
        this.state.offsetX = (containerRect.width - canvasWidth) / 2;
        this.state.offsetY = (containerRect.height - canvasHeight) / 2;
        
        this.canvas.style.left = `${this.state.offsetX}px`;
        this.canvas.style.top = `${this.state.offsetY}px`;
        
        this.state.scale = canvasWidth / this.video.videoWidth;
    }

    /**
     * Draw current video frame to canvas
     */
    drawFrame() {
        if (!this.ctx || !this.video || this.video.readyState < 2) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw video frame
        this.ctx.drawImage(
            this.video,
            0, 0, this.video.videoWidth, this.video.videoHeight,
            0, 0, this.canvas.width, this.canvas.height
        );

        // Draw overlays (cursor, selection, etc.)
        this.drawOverlays();

        // Continue animation loop
        if (this.stream && this.stream.active) {
            requestAnimationFrame(() => this.drawFrame());
        }
    }

    /**
     * Draw overlays on the canvas
     */
    drawOverlays() {
        // Draw remote cursor if available
        if (this.session.remoteCursor) {
            this.drawRemoteCursor(this.session.remoteCursor);
        }

        // Draw selection if active
        if (this.session.selection) {
            this.drawSelection(this.session.selection);
        }
    }

    /**
     * Toggle fullscreen mode
     */
    async toggleFullscreen() {
        try {
            if (!this.state.isFullscreen) {
                await this.container.requestFullscreen();
                this.state.isFullscreen = true;
            } else {
                await document.exitFullscreen();
                this.state.isFullscreen = false;
            }
            
            this.updateCanvasSize();
            this.updateHUD();
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    }

    /**
     * Toggle pointer lock
     */
    async togglePointerLock() {
        try {
            if (!this.state.pointerLocked) {
                await this.canvas.requestPointerLock();
                this.state.pointerLocked = true;
            } else {
                document.exitPointerLock();
                this.state.pointerLocked = false;
            }
            
            this.updateControls();
        } catch (error) {
            console.error('Pointer lock toggle failed:', error);
        }
    }

    /**
     * Set fit mode for the desktop
     */
    setFitMode(mode) {
        this.state.fitMode = mode;
        this.updateCanvasSize();
        this.updateControls();
    }

    /**
     * Get mouse position relative to remote desktop
     */
    getRemoteMousePosition(clientX, clientY) {
        const canvasRect = this.canvas.getBoundingClientRect();
        const relativeX = (clientX - canvasRect.left) / this.state.scale;
        const relativeY = (clientY - canvasRect.top) / this.state.scale;
        
        return {
            x: Math.round(relativeX),
            y: Math.round(relativeY)
        };
    }

    /**
     * Send input to remote desktop
     */
    sendInput(inputData) {
        if (this.session.desktop.connection) {
            this.session.desktop.connection.sendInput(inputData);
        }
    }

    /**
     * Destroy the canvas and clean up
     */
    destroy() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        if (this.state.isFullscreen) {
            document.exitFullscreen().catch(() => {});
        }

        if (this.state.pointerLocked) {
            document.exitPointerLock();
        }

        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }

        if (this.video && this.video.parentNode) {
            this.video.parentNode.removeChild(this.video);
        }

        if (this.hud && this.hud.destroy) {
            this.hud.destroy();
        }

        if (this.controls && this.controls.destroy) {
            this.controls.destroy();
        }
    }

    // Private methods
    createCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'desktop-canvas';
        this.canvas.style.position = 'absolute';
        this.canvas.style.cursor = 'none';
        this.canvas.style.backgroundColor = '#000';
        
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
    }

    createVideo() {
        this.video = document.createElement('video');
        this.video.style.display = 'none';
        this.video.autoplay = true;
        this.video.muted = true;
        this.video.playsInline = true;
        
        this.container.appendChild(this.video);
    }

    createHUD() {
        this.hud = new DesktopHUD(this.container, this);
    }

    createControls() {
        this.controls = new DesktopControls(this.container, this);
    }

    bindEvents() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Keyboard events
        this.canvas.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.canvas.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // Focus events
        this.canvas.addEventListener('focus', () => {
            this.canvas.style.outline = '2px solid var(--omega-cyan)';
        });
        
        this.canvas.addEventListener('blur', () => {
            this.canvas.style.outline = 'none';
            this.input.keys.clear();
            this.input.mouseButtons.clear();
        });

        // Fullscreen events
        document.addEventListener('fullscreenchange', () => {
            this.state.isFullscreen = !!document.fullscreenElement;
            this.updateCanvasSize();
            this.updateHUD();
        });

        // Pointer lock events
        document.addEventListener('pointerlockchange', () => {
            this.state.pointerLocked = document.pointerLockElement === this.canvas;
            this.updateControls();
        });

        // Resize events
        const resizeObserver = new ResizeObserver(() => {
            this.updateCanvasSize();
        });
        resizeObserver.observe(this.container);

        // Make canvas focusable
        this.canvas.tabIndex = 0;
    }

    handleMouseDown(e) {
        e.preventDefault();
        this.canvas.focus();
        
        this.input.mouseButtons.add(e.button);
        const pos = this.getRemoteMousePosition(e.clientX, e.clientY);
        
        this.sendInput({
            type: 'mousedown',
            button: e.button,
            x: pos.x,
            y: pos.y,
            timestamp: Date.now()
        });
    }

    handleMouseUp(e) {
        e.preventDefault();
        
        this.input.mouseButtons.delete(e.button);
        const pos = this.getRemoteMousePosition(e.clientX, e.clientY);
        
        this.sendInput({
            type: 'mouseup',
            button: e.button,
            x: pos.x,
            y: pos.y,
            timestamp: Date.now()
        });
    }

    handleMouseMove(e) {
        const pos = this.getRemoteMousePosition(e.clientX, e.clientY);
        
        // Calculate movement delta for pointer lock mode
        const deltaX = this.state.pointerLocked ? e.movementX / this.state.scale : 0;
        const deltaY = this.state.pointerLocked ? e.movementY / this.state.scale : 0;
        
        this.sendInput({
            type: 'mousemove',
            x: pos.x,
            y: pos.y,
            deltaX: deltaX,
            deltaY: deltaY,
            pointerLocked: this.state.pointerLocked,
            timestamp: Date.now()
        });

        this.input.lastMousePos = pos;
    }

    handleWheel(e) {
        e.preventDefault();
        
        const pos = this.getRemoteMousePosition(e.clientX, e.clientY);
        
        this.sendInput({
            type: 'wheel',
            x: pos.x,
            y: pos.y,
            deltaX: e.deltaX,
            deltaY: e.deltaY,
            deltaMode: e.deltaMode,
            timestamp: Date.now()
        });
    }

    handleKeyDown(e) {
        // Don't prevent certain browser shortcuts
        if (e.ctrlKey && ['r', 'f5', 'f12'].includes(e.key.toLowerCase())) {
            return;
        }

        e.preventDefault();
        
        if (!this.input.keys.has(e.code)) {
            this.input.keys.add(e.code);
            
            this.sendInput({
                type: 'keydown',
                code: e.code,
                key: e.key,
                altKey: e.altKey,
                ctrlKey: e.ctrlKey,
                shiftKey: e.shiftKey,
                metaKey: e.metaKey,
                timestamp: Date.now()
            });
        }
    }

    handleKeyUp(e) {
        e.preventDefault();
        
        this.input.keys.delete(e.code);
        
        this.sendInput({
            type: 'keyup',
            code: e.code,
            key: e.key,
            altKey: e.altKey,
            ctrlKey: e.ctrlKey,
            shiftKey: e.shiftKey,
            metaKey: e.metaKey,
            timestamp: Date.now()
        });
    }

    drawRemoteCursor(cursor) {
        if (!cursor.visible) return;

        const x = cursor.x * this.state.scale;
        const y = cursor.y * this.state.scale;

        this.ctx.save();
        
        // Draw cursor shadow
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x + 1, y + 1, 12, 20);
        
        // Draw cursor
        this.ctx.fillStyle = cursor.color || '#ffffff';
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x, y + 18);
        this.ctx.lineTo(x + 4, y + 14);
        this.ctx.lineTo(x + 8, y + 16);
        this.ctx.lineTo(x + 10, y + 12);
        this.ctx.closePath();
        this.ctx.fill();
        
        this.ctx.restore();
    }

    drawSelection(selection) {
        if (!selection.visible) return;

        const x1 = selection.x1 * this.state.scale;
        const y1 = selection.y1 * this.state.scale;
        const x2 = selection.x2 * this.state.scale;
        const y2 = selection.y2 * this.state.scale;

        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(0, 245, 255, 0.8)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        this.ctx.restore();
    }

    updateHUD() {
        if (this.hud) {
            this.hud.update({
                fullscreen: this.state.isFullscreen,
                pointerLocked: this.state.pointerLocked,
                fitMode: this.state.fitMode,
                scale: this.state.scale
            });
        }
    }

    updateControls() {
        if (this.controls) {
            this.controls.update({
                fullscreen: this.state.isFullscreen,
                pointerLocked: this.state.pointerLocked,
                fitMode: this.state.fitMode
            });
        }
    }
}

/**
 * DesktopHUD - Heads-up display for desktop session
 */
class DesktopHUD {
    constructor(container, canvas) {
        this.container = container;
        this.canvas = canvas;
        this.element = null;
        this.visible = true;
        this.autoHideTimeout = null;
        
        this.init();
    }

    init() {
        this.createElement();
        this.bindEvents();
        this.startAutoHide();
    }

    createElement() {
        this.element = document.createElement('div');
        this.element.className = 'desktop-hud';
        this.element.innerHTML = `
            <div class="hud-section hud-stats">
                <div class="stat-item">
                    <span class="stat-label">FPS:</span>
                    <span class="stat-value" id="hud-fps">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Latency:</span>
                    <span class="stat-value" id="hud-latency">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Bitrate:</span>
                    <span class="stat-value" id="hud-bitrate">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Loss:</span>
                    <span class="stat-value" id="hud-loss">--</span>
                </div>
            </div>
            
            <div class="hud-section hud-status">
                <div class="status-indicator" id="hud-connection-status">
                    <i class="fas fa-circle"></i>
                    <span id="hud-connection-text">Connected</span>
                </div>
                <div class="status-indicator" id="hud-pointer-status">
                    <i class="fas fa-mouse-pointer"></i>
                    <span id="hud-pointer-text">Free</span>
                </div>
            </div>
            
            <div class="hud-section hud-controls">
                <button class="hud-btn" id="hud-fullscreen" title="Toggle Fullscreen (F11)">
                    <i class="fas fa-expand"></i>
                </button>
                <button class="hud-btn" id="hud-pointer-lock" title="Toggle Pointer Lock">
                    <i class="fas fa-lock"></i>
                </button>
                <button class="hud-btn" id="hud-fit-mode" title="Change Fit Mode">
                    <i class="fas fa-arrows-alt"></i>
                </button>
                <button class="hud-btn" id="hud-settings" title="Desktop Settings">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
        `;

        this.container.appendChild(this.element);
    }

    update(state) {
        // Update connection status
        const connectionStatus = this.element.querySelector('#hud-connection-status');
        const connectionText = this.element.querySelector('#hud-connection-text');
        
        connectionStatus.className = `status-indicator status-${this.canvas.session.status}`;
        connectionText.textContent = this.canvas.session.status.charAt(0).toUpperCase() + this.canvas.session.status.slice(1);

        // Update pointer status
        const pointerStatus = this.element.querySelector('#hud-pointer-status');
        const pointerText = this.element.querySelector('#hud-pointer-text');
        
        if (state.pointerLocked) {
            pointerStatus.className = 'status-indicator status-locked';
            pointerText.textContent = 'Locked';
        } else {
            pointerStatus.className = 'status-indicator status-free';
            pointerText.textContent = 'Free';
        }

        // Update fullscreen button
        const fullscreenBtn = this.element.querySelector('#hud-fullscreen i');
        fullscreenBtn.className = state.fullscreen ? 'fas fa-compress' : 'fas fa-expand';

        // Update fit mode button
        const fitModeBtn = this.element.querySelector('#hud-fit-mode');
        fitModeBtn.title = `Fit Mode: ${state.fitMode}`;
    }

    updateStats(stats) {
        this.element.querySelector('#hud-fps').textContent = Math.round(stats.fps || 0);
        this.element.querySelector('#hud-latency').textContent = `${Math.round(stats.latency || 0)}ms`;
        this.element.querySelector('#hud-bitrate').textContent = `${(stats.bitrate || 0).toFixed(1)}Mbps`;
        this.element.querySelector('#hud-loss').textContent = `${(stats.packetLoss || 0).toFixed(1)}%`;
    }

    show() {
        this.visible = true;
        this.element.style.opacity = '1';
        this.startAutoHide();
    }

    hide() {
        this.visible = false;
        this.element.style.opacity = '0';
        this.clearAutoHide();
    }

    toggle() {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }

    destroy() {
        this.clearAutoHide();
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }

    // Private methods
    bindEvents() {
        // Control buttons
        this.element.querySelector('#hud-fullscreen').addEventListener('click', () => {
            this.canvas.toggleFullscreen();
        });

        this.element.querySelector('#hud-pointer-lock').addEventListener('click', () => {
            this.canvas.togglePointerLock();
        });

        this.element.querySelector('#hud-fit-mode').addEventListener('click', () => {
            this.cycleFitMode();
        });

        this.element.querySelector('#hud-settings').addEventListener('click', () => {
            this.openDesktopSettings();
        });

        // Show HUD on mouse movement
        this.container.addEventListener('mousemove', () => {
            if (!this.visible) {
                this.show();
            }
        });

        // Listen for metrics updates
        if (window.EventBus) {
            window.EventBus.on('webrtcStats', (data) => {
                if (data.sessionId === this.canvas.session.id) {
                    this.updateStats(data.stats);
                }
            });
        }
    }

    startAutoHide() {
        this.clearAutoHide();
        this.autoHideTimeout = setTimeout(() => {
            if (!this.element.matches(':hover')) {
                this.hide();
            }
        }, 3000);
    }

    clearAutoHide() {
        if (this.autoHideTimeout) {
            clearTimeout(this.autoHideTimeout);
            this.autoHideTimeout = null;
        }
    }

    cycleFitMode() {
        const modes = ['contain', 'cover', 'actual'];
        const currentIndex = modes.indexOf(this.canvas.state.fitMode);
        const nextMode = modes[(currentIndex + 1) % modes.length];
        
        this.canvas.setFitMode(nextMode);
    }

    openDesktopSettings() {
        if (window.EventBus) {
            window.EventBus.emit('openDesktopSettings', { sessionId: this.canvas.session.id });
        }
    }
}

/**
 * DesktopControls - Additional desktop control panel
 */
class DesktopControls {
    constructor(container, canvas) {
        this.container = container;
        this.canvas = canvas;
        this.element = null;
        this.visible = false;
        
        this.init();
    }

    init() {
        this.createElement();
        this.bindEvents();
    }

    createElement() {
        this.element = document.createElement('div');
        this.element.className = 'desktop-controls';
        this.element.innerHTML = `
            <div class="controls-header">
                <h4>Desktop Controls</h4>
                <button class="btn-close" id="controls-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="controls-content">
                <div class="control-section">
                    <h5>Quality</h5>
                    <div class="quality-controls">
                        <label>Bitrate (Mbps):</label>
                        <input type="range" id="bitrate-slider" min="1" max="50" value="5" step="0.5">
                        <span id="bitrate-value">5.0</span>
                        
                        <label>FPS:</label>
                        <input type="range" id="fps-slider" min="15" max="120" value="60" step="15">
                        <span id="fps-value">60</span>
                        
                        <label>Codec:</label>
                        <select id="codec-select">
                            <option value="h264">H.264</option>
                            <option value="vp8">VP8</option>
                            <option value="vp9">VP9</option>
                            <option value="av1">AV1</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-section">
                    <h5>Display</h5>
                    <div class="display-controls">
                        <label>Fit Mode:</label>
                        <select id="fit-mode-select">
                            <option value="contain">Contain</option>
                            <option value="cover">Cover</option>
                            <option value="actual">Actual Size</option>
                        </select>
                        
                        <label>Resolution:</label>
                        <select id="resolution-select">
                            <option value="1920x1080">1920x1080</option>
                            <option value="2560x1440">2560x1440</option>
                            <option value="3840x2160">3840x2160</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-section">
                    <h5>Input</h5>
                    <div class="input-controls">
                        <label>
                            <input type="checkbox" id="clipboard-sync">
                            Clipboard Sync
                        </label>
                        
                        <label>
                            <input type="checkbox" id="file-transfer">
                            File Transfer
                        </label>
                        
                        <label>
                            <input type="checkbox" id="audio-capture">
                            Audio Capture
                        </label>
                    </div>
                </div>
            </div>
        `;

        this.container.appendChild(this.element);
        this.hide();
    }

    show() {
        this.visible = true;
        this.element.style.display = 'block';
        setTimeout(() => {
            this.element.classList.add('visible');
        }, 10);
    }

    hide() {
        this.visible = false;
        this.element.classList.remove('visible');
        setTimeout(() => {
            this.element.style.display = 'none';
        }, 300);
    }

    toggle() {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }

    update(state) {
        // Update fit mode select
        this.element.querySelector('#fit-mode-select').value = state.fitMode;
    }

    destroy() {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }

    // Private methods
    bindEvents() {
        // Close button
        this.element.querySelector('#controls-close').addEventListener('click', () => {
            this.hide();
        });

        // Quality controls
        const bitrateSlider = this.element.querySelector('#bitrate-slider');
        const bitrateValue = this.element.querySelector('#bitrate-value');
        
        bitrateSlider.addEventListener('input', () => {
            bitrateValue.textContent = bitrateSlider.value;
            this.updateQuality();
        });

        const fpsSlider = this.element.querySelector('#fps-slider');
        const fpsValue = this.element.querySelector('#fps-value');
        
        fpsSlider.addEventListener('input', () => {
            fpsValue.textContent = fpsSlider.value;
            this.updateQuality();
        });

        this.element.querySelector('#codec-select').addEventListener('change', () => {
            this.updateQuality();
        });

        // Display controls
        this.element.querySelector('#fit-mode-select').addEventListener('change', (e) => {
            this.canvas.setFitMode(e.target.value);
        });

        this.element.querySelector('#resolution-select').addEventListener('change', (e) => {
            this.updateResolution(e.target.value);
        });

        // Input controls
        this.element.querySelector('#clipboard-sync').addEventListener('change', (e) => {
            this.updateInputSettings('clipboard', e.target.checked);
        });

        this.element.querySelector('#file-transfer').addEventListener('change', (e) => {
            this.updateInputSettings('fileTransfer', e.target.checked);
        });

        this.element.querySelector('#audio-capture').addEventListener('change', (e) => {
            this.updateInputSettings('audioCapture', e.target.checked);
        });
    }

    updateQuality() {
        const settings = {
            maxBitrate: parseFloat(this.element.querySelector('#bitrate-slider').value),
            maxFramerate: parseInt(this.element.querySelector('#fps-slider').value),
            codec: this.element.querySelector('#codec-select').value
        };

        if (this.canvas.session.desktop.connection) {
            this.canvas.session.desktop.connection.updateQuality(settings);
        }
    }

    updateResolution(resolution) {
        if (resolution === 'custom') {
            // Show custom resolution dialog
            const width = prompt('Enter width:', '1920');
            const height = prompt('Enter height:', '1080');
            
            if (width && height) {
                resolution = `${width}x${height}`;
            } else {
                return;
            }
        }

        // Send resolution change request
        if (window.EventBus) {
            window.EventBus.emit('resolutionChangeRequested', {
                sessionId: this.canvas.session.id,
                resolution
            });
        }
    }

    updateInputSettings(setting, enabled) {
        if (window.EventBus) {
            window.EventBus.emit('inputSettingsChanged', {
                sessionId: this.canvas.session.id,
                setting,
                enabled
            });
        }
    }
}

// Export classes
window.DesktopCanvas = DesktopCanvas;
window.DesktopHUD = DesktopHUD;
window.DesktopControls = DesktopControls;

export { DesktopCanvas, DesktopHUD, DesktopControls };
