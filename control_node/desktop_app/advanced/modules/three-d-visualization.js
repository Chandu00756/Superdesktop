/**
 * Omega SuperDesktop v2.0 - Three.js Visualization Manager Module
 * Extracted from omega-control-center.html - Advanced 3D data visualization
 */

class ThreeDVisualizationManager extends EventTarget {
    constructor() {
        super();
        this.scenes = new Map();
        this.renderers = new Map();
        this.cameras = new Map();
        this.animationFrames = new Map();
        this.loadedModels = new Map();
        this.isLoading = false;
    }

    initialize() {
        console.log('🎬 Initializing 3D Visualization Manager...');
        this.setupEventListeners();
        this.initializeVisualizationTab();
        console.log('✅ 3D Visualization Manager initialized');
        this.dispatchEvent(new CustomEvent('threeDManagerInitialized'));
    }

    setupEventListeners() {
        window.showVisualization = (type) => {
            this.showVisualization(type);
        };

        window.updateVisualization = () => {
            this.updateVisualization();
        };

        window.resetVisualization = () => {
            this.resetVisualization();
        };

        window.exportVisualization = (format) => {
            this.exportVisualization(format);
        };
    }

    initializeVisualizationTab() {
        const visualizationContent = document.getElementById('visualization-content');
        if (!visualizationContent) return;

        visualizationContent.innerHTML = `
            <div class="visualization-controls">
                <div class="viz-control-group">
                    <button type="button" onclick="window.showVisualization('network')" class="btn btn-primary">
                        <i class="fas fa-network-wired"></i> Network Topology
                    </button>
                    <button type="button" onclick="window.showVisualization('performance')" class="btn btn-primary">
                        <i class="fas fa-chart-line"></i> Performance 3D
                    </button>
                    <button type="button" onclick="window.showVisualization('data')" class="btn btn-primary">
                        <i class="fas fa-database"></i> Data Flow
                    </button>
                    <button type="button" onclick="window.showVisualization('cluster')" class="btn btn-primary">
                        <i class="fas fa-cubes"></i> Cluster View
                    </button>
                </div>
                <div class="viz-control-group">
                    <button type="button" onclick="window.updateVisualization()" class="btn btn-secondary">
                        <i class="fas fa-sync"></i> Update
                    </button>
                    <button type="button" onclick="window.resetVisualization()" class="btn btn-secondary">
                        <i class="fas fa-undo"></i> Reset
                    </button>
                    <div class="dropdown">
                        <button class="btn btn-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" onclick="window.exportVisualization('image')">PNG Image</a></li>
                            <li><a class="dropdown-item" onclick="window.exportVisualization('model')">3D Model</a></li>
                            <li><a class="dropdown-item" onclick="window.exportVisualization('video')">Video</a></li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="visualization-container">
                <canvas id="three-js-canvas" width="1200" height="800"></canvas>
            </div>
            
            <div class="visualization-info">
                <div class="info-panel">
                    <h5>Visualization Info</h5>
                    <div id="viz-stats">
                        <div>Objects: <span id="object-count">0</span></div>
                        <div>Triangles: <span id="triangle-count">0</span></div>
                        <div>FPS: <span id="fps-counter">60</span></div>
                    </div>
                </div>
            </div>
        `;
    }

    showVisualization(type) {
        this.clearCurrentVisualization();
        
        switch (type) {
            case 'network':
                this.createNetworkTopology();
                break;
            case 'performance':
                this.createPerformanceVisualization();
                break;
            case 'data':
                this.createDataFlowVisualization();
                break;
            case 'cluster':
                this.createClusterVisualization();
                break;
        }
    }

    createNetworkTopology() {
        const canvas = document.getElementById('three-js-canvas');
        if (!canvas) return;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(canvas.width, canvas.height);

        // Network nodes
        const nodeGeometry = new THREE.SphereGeometry(0.5, 32, 32);
        const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
        
        const nodes = [];
        for (let i = 0; i < 20; i++) {
            const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
            node.position.set(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            );
            scene.add(node);
            nodes.push(node);
        }

        // Connections
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x4444ff });
        nodes.forEach((node, i) => {
            if (i < nodes.length - 1) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    node.position,
                    nodes[i + 1].position
                ]);
                const line = new THREE.Line(geometry, lineMaterial);
                scene.add(line);
            }
        });

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        camera.position.z = 30;

        this.scenes.set('current', scene);
        this.renderers.set('current', renderer);
        this.cameras.set('current', camera);

        this.animate('current', nodes);
    }

    createPerformanceVisualization() {
        const canvas = document.getElementById('three-js-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2a2a3e);
        
        const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(canvas.width, canvas.height);

        // Performance bars
        const bars = [];
        for (let i = 0; i < 10; i++) {
            const height = Math.random() * 5 + 1;
            const geometry = new THREE.BoxGeometry(1, height, 1);
            const material = new THREE.MeshPhongMaterial({ 
                color: new THREE.Color().setHSL(height / 6, 0.8, 0.5)
            });
            const bar = new THREE.Mesh(geometry, material);
            bar.position.set(i * 2 - 9, height / 2, 0);
            scene.add(bar);
            bars.push(bar);
        }

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        camera.position.set(0, 5, 15);
        camera.lookAt(0, 0, 0);

        this.scenes.set('current', scene);
        this.renderers.set('current', renderer);
        this.cameras.set('current', camera);

        this.animatePerformance('current', bars);
    }

    createDataFlowVisualization() {
        const canvas = document.getElementById('three-js-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e3e);
        
        const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(canvas.width, canvas.height);

        // Data particles
        const particleGeometry = new THREE.BufferGeometry();
        const particleCount = 1000;
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i++) {
            positions[i] = (Math.random() - 0.5) * 50;
        }
        
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color: 0x88aaff,
            size: 0.1
        });
        
        const particles = new THREE.Points(particleGeometry, particleMaterial);
        scene.add(particles);

        camera.position.z = 30;

        this.scenes.set('current', scene);
        this.renderers.set('current', renderer);
        this.cameras.set('current', camera);

        this.animateParticles('current', particles);
    }

    createClusterVisualization() {
        const canvas = document.getElementById('three-js-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2e1a2e);
        
        const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(canvas.width, canvas.height);

        // Cluster nodes
        const clusterGroup = new THREE.Group();
        
        for (let i = 0; i < 8; i++) {
            const nodeGeometry = new THREE.BoxGeometry(2, 2, 2);
            const nodeMaterial = new THREE.MeshPhongMaterial({ 
                color: new THREE.Color().setHSL(i / 8, 0.7, 0.6)
            });
            const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
            
            const angle = (i / 8) * Math.PI * 2;
            node.position.set(
                Math.cos(angle) * 8,
                Math.sin(angle) * 2,
                Math.sin(angle) * 8
            );
            
            clusterGroup.add(node);
        }
        
        scene.add(clusterGroup);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        camera.position.set(0, 10, 20);
        camera.lookAt(0, 0, 0);

        this.scenes.set('current', scene);
        this.renderers.set('current', renderer);
        this.cameras.set('current', camera);

        this.animateCluster('current', clusterGroup);
    }

    animate(sceneKey, objects) {
        const scene = this.scenes.get(sceneKey);
        const renderer = this.renderers.get(sceneKey);
        const camera = this.cameras.get(sceneKey);
        
        if (!scene || !renderer || !camera) return;

        const animateFrame = () => {
            objects.forEach(obj => {
                obj.rotation.x += 0.01;
                obj.rotation.y += 0.01;
            });

            renderer.render(scene, camera);
            this.animationFrames.set(sceneKey, requestAnimationFrame(animateFrame));
        };
        
        animateFrame();
    }

    animatePerformance(sceneKey, bars) {
        const scene = this.scenes.get(sceneKey);
        const renderer = this.renderers.get(sceneKey);
        const camera = this.cameras.get(sceneKey);
        
        if (!scene || !renderer || !camera) return;

        const animateFrame = () => {
            bars.forEach((bar, i) => {
                const newHeight = Math.random() * 5 + 1;
                bar.scale.y = newHeight / bar.geometry.parameters.height;
                bar.material.color.setHSL(newHeight / 6, 0.8, 0.5);
            });

            renderer.render(scene, camera);
            this.animationFrames.set(sceneKey, requestAnimationFrame(animateFrame));
        };
        
        animateFrame();
    }

    animateParticles(sceneKey, particles) {
        const scene = this.scenes.get(sceneKey);
        const renderer = this.renderers.get(sceneKey);
        const camera = this.cameras.get(sceneKey);
        
        if (!scene || !renderer || !camera) return;

        const animateFrame = () => {
            particles.rotation.x += 0.005;
            particles.rotation.y += 0.01;

            renderer.render(scene, camera);
            this.animationFrames.set(sceneKey, requestAnimationFrame(animateFrame));
        };
        
        animateFrame();
    }

    animateCluster(sceneKey, clusterGroup) {
        const scene = this.scenes.get(sceneKey);
        const renderer = this.renderers.get(sceneKey);
        const camera = this.cameras.get(sceneKey);
        
        if (!scene || !renderer || !camera) return;

        const animateFrame = () => {
            clusterGroup.rotation.y += 0.01;
            clusterGroup.children.forEach(node => {
                node.rotation.x += 0.02;
                node.rotation.z += 0.01;
            });

            renderer.render(scene, camera);
            this.animationFrames.set(sceneKey, requestAnimationFrame(animateFrame));
        };
        
        animateFrame();
    }

    updateVisualization() {
        this.showNotification('Visualization updated', 'success');
    }

    resetVisualization() {
        this.clearCurrentVisualization();
        this.showNotification('Visualization reset', 'info');
    }

    exportVisualization(format) {
        const renderer = this.renderers.get('current');
        if (!renderer) return;

        switch (format) {
            case 'image':
                const dataURL = renderer.domElement.toDataURL('image/png');
                const link = document.createElement('a');
                link.download = 'visualization.png';
                link.href = dataURL;
                link.click();
                break;
            case 'model':
                this.showNotification('3D model export not implemented', 'warning');
                break;
            case 'video':
                this.showNotification('Video export not implemented', 'warning');
                break;
        }
    }

    clearCurrentVisualization() {
        const animationFrame = this.animationFrames.get('current');
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            this.animationFrames.delete('current');
        }

        this.scenes.delete('current');
        this.renderers.delete('current');
        this.cameras.delete('current');
    }

    showNotification(message, type) {
        if (window.menuBarManager) {
            window.menuBarManager.showNotification('3D Visualization', message, type);
        } else {
            console.log(`3D Viz: ${message}`);
        }
    }

    dispose() {
        this.animationFrames.forEach(frame => cancelAnimationFrame(frame));
        this.scenes.clear();
        this.renderers.clear();
        this.cameras.clear();
        this.animationFrames.clear();
        console.log('🧹 3D Visualization Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.ThreeDVisualizationManager = ThreeDVisualizationManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThreeDVisualizationManager;
}
