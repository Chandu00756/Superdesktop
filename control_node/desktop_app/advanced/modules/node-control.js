/**
 * Omega SuperDesktop v2.0 - Node Control Manager Module
 * Extracted from omega-control-center.html - Handles node management, control, and monitoring
 */

class NodeControlManager extends EventTarget {
    constructor() {
        super();
        this.nodeStates = new Map();
        this.nodeMetrics = new Map();
        this.activeConnections = new Map();
        this.refreshInterval = null;
        this.numaConfiguration = {
            memoryBinding: 'strict',
            cpuAffinity: 'numa-node',
            balancingPolicy: 'load-balanced'
        };
    }

    initialize() {
        console.log('🖥️ Initializing Node Control Manager...');
        this.initializeNodeStates();
        this.setupEventListeners();
        this.startNodeMonitoring();
        console.log('✅ Node Control Manager initialized');
        this.dispatchEvent(new CustomEvent('nodeControlManagerInitialized'));
    }

    initializeNodeStates() {
        // Initialize sample node states
        const sampleNodes = [
            {
                id: 'gpu-node-01',
                name: 'GPU Cluster Node 01',
                type: 'gpu',
                status: 'online',
                resources: ['gpu', 'memory', 'cpu'],
                specifications: {
                    gpus: 8,
                    gpuModel: 'H100 80GB',
                    cpu: '128 cores',
                    memory: '1TB',
                    storage: '10TB NVMe'
                },
                location: { rack: 'R1', position: 'U1-4' },
                lastSeen: new Date(),
                workloads: 3,
                utilization: { cpu: 65, memory: 72, gpu: 89 }
            },
            {
                id: 'cpu-node-02',
                name: 'CPU Compute Node 02',
                type: 'cpu',
                status: 'online',
                resources: ['cpu', 'memory'],
                specifications: {
                    cpu: '256 cores',
                    memory: '2TB',
                    storage: '5TB NVMe'
                },
                location: { rack: 'R2', position: 'U5-8' },
                lastSeen: new Date(),
                workloads: 12,
                utilization: { cpu: 45, memory: 58 }
            },
            {
                id: 'storage-node-03',
                name: 'Storage Node 03',
                type: 'storage',
                status: 'maintenance',
                resources: ['storage', 'memory'],
                specifications: {
                    storage: '100TB',
                    memory: '512GB',
                    network: '100GbE'
                },
                location: { rack: 'R3', position: 'U1-2' },
                lastSeen: new Date(Date.now() - 300000), // 5 minutes ago
                workloads: 0,
                utilization: { storage: 73, memory: 25 }
            },
            {
                id: 'edge-node-04',
                name: 'Edge Processing Node 04',
                type: 'edge',
                status: 'warning',
                resources: ['cpu', 'memory', 'gpu'],
                specifications: {
                    cpu: '32 cores',
                    memory: '128GB',
                    gpu: '4x RTX 4090',
                    network: '10GbE'
                },
                location: { rack: 'R1', position: 'U20-22' },
                lastSeen: new Date(),
                workloads: 2,
                utilization: { cpu: 85, memory: 90, gpu: 45 }
            }
        ];

        sampleNodes.forEach(node => {
            this.nodeStates.set(node.id, node);
            this.nodeMetrics.set(node.id, {
                history: this.generateMetricHistory(),
                lastUpdate: new Date()
            });
        });
    }

    generateMetricHistory() {
        const history = [];
        for (let i = 0; i < 60; i++) {
            history.push({
                timestamp: new Date(Date.now() - i * 60000), // 1 minute intervals
                cpu: Math.random() * 100,
                memory: Math.random() * 100,
                network: Math.random() * 100,
                temperature: 25 + Math.random() * 50
            });
        }
        return history.reverse();
    }

    setupEventListeners() {
        // Node control actions
        this.setupNodeActions();
        
        // Resource pool actions
        this.setupResourcePoolActions();
        
        // NUMA and topology actions
        this.setupNUMAActions();
        
        // Monitoring and diagnostics
        this.setupMonitoringActions();
    }

    setupNodeActions() {
        window.restartNode = (nodeId) => {
            this.restartNode(nodeId);
        };

        window.maintainNode = (nodeId) => {
            this.setNodeMaintenance(nodeId);
        };

        window.quarantineNode = (nodeId) => {
            this.quarantineNode(nodeId);
        };

        window.drainNode = (nodeId) => {
            this.drainNode(nodeId);
        };

        window.removeNode = (nodeId) => {
            this.removeNode(nodeId);
        };

        window.viewNodeLogs = (nodeId) => {
            this.openNodeLogs(nodeId);
        };

        window.viewNodeMetrics = (nodeId) => {
            this.openNodeMetrics(nodeId);
        };

        window.connectToNode = (nodeId) => {
            this.openNodeTerminal(nodeId);
        };
    }

    setupResourcePoolActions() {
        window.configureGPUSharing = (nodeId) => {
            this.openResourceConfiguration(nodeId, 'gpu');
        };

        window.configureCPUSharing = (nodeId) => {
            this.openResourceConfiguration(nodeId, 'cpu');
        };

        window.configureMemorySharing = (nodeId) => {
            this.openResourceConfiguration(nodeId, 'memory');
        };

        window.viewGPUJobs = (nodeId) => {
            this.openJobViewer(nodeId, 'gpu');
        };

        window.viewCPUJobs = (nodeId) => {
            this.openJobViewer(nodeId, 'cpu');
        };

        window.viewMemoryJobs = (nodeId) => {
            this.openJobViewer(nodeId, 'memory');
        };

        window.toggleGPUPool = (nodeId) => {
            this.toggleResourcePool(nodeId, 'gpu');
        };

        window.toggleCPUPool = (nodeId) => {
            this.toggleResourcePool(nodeId, 'cpu');
        };

        window.toggleMemoryPool = (nodeId) => {
            this.toggleResourcePool(nodeId, 'memory');
        };

        window.refreshNodeResources = (nodeId) => {
            this.refreshNodeResources(nodeId);
        };
    }

    setupNUMAActions() {
        window.refreshNUMAVisualization = () => {
            this.refreshNUMAVisualization();
        };

        window.exportNUMAData = () => {
            this.exportNUMAData();
        };

        window.configureNUMASettings = () => {
            this.openNUMASettings();
        };
    }

    setupMonitoringActions() {
        window.pauseJob = (jobId) => {
            this.pauseJob(jobId);
        };

        window.killJob = (jobId) => {
            this.killJob(jobId);
        };

        window.downloadLogs = (nodeId) => {
            this.downloadNodeLogs(nodeId);
        };

        window.clearLogs = (nodeId) => {
            this.clearNodeLogs(nodeId);
        };
    }

    async restartNode(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        this.showNodeAction(nodeId, 'Restarting', 'Node restart initiated...', 'warning');
        
        try {
            node.status = 'restarting';
            this.updateNodeDisplay();
            
            // Simulate restart process
            await this.simulateNodeOperation(nodeId, 'restart', 5000);
            
            node.status = 'online';
            node.lastSeen = new Date();
            this.updateNodeDisplay();
            
            this.showNodeAction(nodeId, 'Restart Complete', 'Node has been successfully restarted.', 'success');
            
            this.dispatchEvent(new CustomEvent('nodeRestarted', { detail: { nodeId, node } }));
            
        } catch (error) {
            node.status = 'error';
            this.updateNodeDisplay();
            this.showNodeAction(nodeId, 'Restart Failed', error.message, 'error');
        }
    }

    async setNodeMaintenance(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        node.status = 'maintenance';
        node.workloads = 0; // Drain workloads
        this.updateNodeDisplay();
        
        this.showNodeAction(nodeId, 'Maintenance Mode', 'Node entered maintenance mode.', 'info');
        this.dispatchEvent(new CustomEvent('nodeMaintenanceSet', { detail: { nodeId, node } }));
    }

    async quarantineNode(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        node.status = 'quarantined';
        node.workloads = 0;
        this.updateNodeDisplay();
        
        this.showNodeAction(nodeId, 'Quarantined', 'Node has been quarantined for security.', 'warning');
        this.dispatchEvent(new CustomEvent('nodeQuarantined', { detail: { nodeId, node } }));
    }

    async drainNode(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        this.showNodeAction(nodeId, 'Draining', 'Node is being drained of workloads.', 'info');
        
        // Simulate gradual workload draining
        const originalWorkloads = node.workloads;
        for (let i = originalWorkloads; i >= 0; i--) {
            node.workloads = i;
            this.updateNodeDisplay();
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        this.showNodeAction(nodeId, 'Drained', 'All workloads have been migrated.', 'success');
        this.dispatchEvent(new CustomEvent('nodeDrained', { detail: { nodeId, node } }));
    }

    async removeNode(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        if (!confirm(`Are you sure you want to remove node ${nodeId}? This action cannot be undone.`)) {
            return;
        }

        this.showNodeAction(nodeId, 'Removing', 'Node removal in progress...', 'error');
        
        try {
            // Simulate removal process
            await this.simulateNodeOperation(nodeId, 'remove', 3000);
            
            this.nodeStates.delete(nodeId);
            this.nodeMetrics.delete(nodeId);
            this.activeConnections.delete(nodeId);
            
            this.updateNodeDisplay();
            this.showNodeAction(nodeId, 'Removed', 'Node has been removed from cluster.', 'success');
            
            this.dispatchEvent(new CustomEvent('nodeRemoved', { detail: { nodeId } }));
            
        } catch (error) {
            this.showNodeAction(nodeId, 'Removal Failed', error.message, 'error');
        }
    }

    openNodeLogs(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        const logs = this.generateNodeLogs(nodeId);
        const modal = this.createModal(`Node Logs - ${nodeId}`, `
            <div class="logs-container">
                <div class="logs-header">
                    <h3>System Logs for ${nodeId}</h3>
                    <div class="logs-controls">
                        <button type="button" class="btn btn-secondary" onclick="window.downloadLogs('${nodeId}')">
                            <i class="fas fa-download"></i> Download
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="window.clearLogs('${nodeId}')">
                            <i class="fas fa-trash"></i> Clear
                        </button>
                        <button type="button" class="btn btn-primary" onclick="this.refreshLogs()">
                            <i class="fas fa-refresh"></i> Refresh
                        </button>
                    </div>
                </div>
                <div class="logs-content">
                    <pre class="log-output" id="logOutput-${nodeId}">${logs}</pre>
                </div>
            </div>
        `);

        // Auto-refresh logs
        const logOutput = modal.querySelector(`#logOutput-${nodeId}`);
        setInterval(() => {
            if (modal.parentNode) {
                logOutput.textContent = this.generateNodeLogs(nodeId);
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        }, 2000);
    }

    generateNodeLogs(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return '';

        const logs = [
            `[${new Date().toISOString()}] INFO: Node ${nodeId} status: ${node.status}`,
            `[${new Date().toISOString()}] INFO: Active workloads: ${node.workloads}`,
            `[${new Date().toISOString()}] INFO: CPU utilization: ${node.utilization?.cpu || 0}%`,
            `[${new Date().toISOString()}] INFO: Memory utilization: ${node.utilization?.memory || 0}%`
        ];

        if (node.type === 'gpu') {
            logs.push(`[${new Date().toISOString()}] INFO: GPU utilization: ${node.utilization?.gpu || 0}%`);
            logs.push(`[${new Date().toISOString()}] DEBUG: GPU temperature: ${Math.round(65 + Math.random() * 20)}°C`);
        }

        logs.push(`[${new Date().toISOString()}] DEBUG: Heartbeat sent to cluster controller`);
        logs.push(`[${new Date().toISOString()}] DEBUG: Resource allocation updated`);

        return logs.join('\n');
    }

    openNodeMetrics(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        const modal = this.createModal(`Node Metrics - ${nodeId}`, `
            <div class="metrics-dashboard">
                <h3>Performance Metrics for ${nodeId}</h3>
                <div class="metrics-overview">
                    <div class="metric-card">
                        <h4>Current Status</h4>
                        <div class="status-indicator status-${node.status}">${node.status}</div>
                        <p>Last seen: ${node.lastSeen.toLocaleString()}</p>
                    </div>
                    <div class="metric-card">
                        <h4>Workloads</h4>
                        <div class="metric-value">${node.workloads}</div>
                        <p>Active jobs</p>
                    </div>
                    <div class="metric-card">
                        <h4>Utilization</h4>
                        <div class="utilization-bars">
                            <div class="util-bar">
                                <label>CPU: ${node.utilization?.cpu || 0}%</label>
                                <div class="bar"><div class="fill" style="width: ${node.utilization?.cpu || 0}%"></div></div>
                            </div>
                            <div class="util-bar">
                                <label>Memory: ${node.utilization?.memory || 0}%</label>
                                <div class="bar"><div class="fill" style="width: ${node.utilization?.memory || 0}%"></div></div>
                            </div>
                            ${node.utilization?.gpu ? `
                                <div class="util-bar">
                                    <label>GPU: ${node.utilization.gpu}%</label>
                                    <div class="bar"><div class="fill" style="width: ${node.utilization.gpu}%"></div></div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                <div class="metrics-charts">
                    <div class="chart-container">
                        <h4>CPU Usage Over Time</h4>
                        <canvas id="nodeMetricsCPU-${nodeId}"></canvas>
                    </div>
                    <div class="chart-container">
                        <h4>Memory Usage Over Time</h4>
                        <canvas id="nodeMetricsMemory-${nodeId}"></canvas>
                    </div>
                    <div class="chart-container">
                        <h4>Network I/O</h4>
                        <canvas id="nodeMetricsNetwork-${nodeId}"></canvas>
                    </div>
                </div>
            </div>
        `);

        // Initialize charts if Chart.js is available
        setTimeout(() => {
            this.initializeNodeMetricsCharts(nodeId, modal);
        }, 100);
    }

    initializeNodeMetricsCharts(nodeId, modal) {
        if (typeof Chart === 'undefined') return;

        const metrics = this.nodeMetrics.get(nodeId);
        if (!metrics) return;

        ['CPU', 'Memory', 'Network'].forEach(type => {
            const canvas = modal.querySelector(`#nodeMetrics${type}-${nodeId}`);
            if (canvas) {
                new Chart(canvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: metrics.history.map((_, i) => `${60-i}m`),
                        datasets: [{
                            label: `${type} Usage`,
                            data: metrics.history.map(h => h[type.toLowerCase()] || 0),
                            borderColor: '#00ff7f',
                            backgroundColor: 'rgba(0, 255, 127, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { min: 0, max: 100 }
                        }
                    }
                });
            }
        });
    }

    openNodeTerminal(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        const modal = this.createModal(`Terminal - ${nodeId}`, `
            <div class="terminal-container">
                <div class="terminal-header">
                    <span class="terminal-title">root@${nodeId}:~#</span>
                    <div class="terminal-controls">
                        <button type="button" onclick="this.clearTerminal()">Clear</button>
                        <button type="button" onclick="this.closeTerminal()">Disconnect</button>
                    </div>
                </div>
                <div class="terminal-output" id="terminal-${nodeId}">
                    <div class="terminal-line">Welcome to ${nodeId}</div>
                    <div class="terminal-line">Node Type: ${node.type}</div>
                    <div class="terminal-line">Status: ${node.status}</div>
                    <div class="terminal-line">Type 'help' for available commands</div>
                    <div class="terminal-line"></div>
                </div>
                <div class="terminal-input-container">
                    <span class="prompt-text">root@${nodeId}:~# </span>
                    <input type="text" class="terminal-input" id="terminalInput-${nodeId}" 
                           placeholder="Enter command..." onkeydown="this.handleTerminalInput(event, '${nodeId}')">
                </div>
            </div>
        `);

        // Set up terminal input handling
        const input = modal.querySelector(`#terminalInput-${nodeId}`);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.handleTerminalCommand(nodeId, input.value, modal);
                input.value = '';
            }
        });

        // Focus input
        input.focus();
        
        // Track active connection
        this.activeConnections.set(nodeId, { modal, type: 'terminal', startTime: new Date() });
    }

    handleTerminalCommand(nodeId, command, modal) {
        const output = modal.querySelector(`#terminal-${nodeId}`);
        const commandLine = document.createElement('div');
        commandLine.className = 'terminal-line';
        commandLine.innerHTML = `<span class="prompt">root@${nodeId}:~#</span> ${command}`;
        output.appendChild(commandLine);

        // Process command
        const response = this.processTerminalCommand(nodeId, command);
        const responseLine = document.createElement('div');
        responseLine.className = 'terminal-line terminal-response';
        responseLine.textContent = response;
        output.appendChild(responseLine);

        // Scroll to bottom
        output.scrollTop = output.scrollHeight;
    }

    processTerminalCommand(nodeId, command) {
        const node = this.nodeStates.get(nodeId);
        
        switch (command.toLowerCase().trim()) {
            case 'help':
                return 'Available commands: status, top, free, df, nvidia-smi, uptime, whoami, clear, exit';
            case 'status':
                return `Node Status: ${node.status}\nWorkloads: ${node.workloads}\nLast seen: ${node.lastSeen.toLocaleString()}`;
            case 'top':
                return `CPU: ${node.utilization?.cpu || 0}%\nMemory: ${node.utilization?.memory || 0}%`;
            case 'free':
                return `Memory: ${node.specifications?.memory || 'Unknown'}\nUsage: ${node.utilization?.memory || 0}%`;
            case 'df':
                return `Storage: ${node.specifications?.storage || 'Unknown'}\nUsage: ${node.utilization?.storage || 0}%`;
            case 'nvidia-smi':
                if (node.type === 'gpu') {
                    return `GPU: ${node.specifications?.gpuModel || 'Unknown'}\nCount: ${node.specifications?.gpus || 0}\nUtilization: ${node.utilization?.gpu || 0}%`;
                }
                return 'No NVIDIA GPUs found';
            case 'uptime':
                const uptime = Math.floor((Date.now() - node.lastSeen.getTime()) / 60000);
                return `Uptime: ${Math.max(0, 1440 - uptime)} minutes`;
            case 'whoami':
                return 'root';
            case 'clear':
                return '[CLEAR]'; // Special command to clear terminal
            case 'exit':
                return 'Connection closed';
            default:
                return `Command not found: ${command}`;
        }
    }

    openResourceConfiguration(nodeId, resourceType) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        const modal = this.createModal(`${resourceType.toUpperCase()} Configuration - ${nodeId}`, `
            <div class="resource-config">
                <h3>${resourceType.toUpperCase()} Resource Pool Configuration</h3>
                <div class="config-form">
                    <div class="form-group">
                        <label>Sharing Policy:</label>
                        <select id="sharingPolicy-${nodeId}">
                            <option value="exclusive">Exclusive</option>
                            <option value="shared">Shared</option>
                            <option value="time-sliced">Time Sliced</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Job Assignment:</label>
                        <select id="jobAssignment-${nodeId}">
                            <option value="load-balanced">Load Balanced</option>
                            <option value="priority-queue">Priority Queue</option>
                            <option value="numa-aware">NUMA Aware</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Maximum Concurrent Jobs:</label>
                        <input type="number" id="maxJobs-${nodeId}" value="8" min="1" max="32">
                    </div>
                    <div class="form-group">
                        <label>Resource Overcommit:</label>
                        <input type="checkbox" id="overcommit-${nodeId}">
                        <span>Allow resource overcommitment</span>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn btn-primary" onclick="this.applyResourceConfig('${nodeId}', '${resourceType}')">
                            Apply Changes
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `);
    }

    openJobViewer(nodeId, resourceType) {
        const node = this.nodeStates.get(nodeId);
        if (!node) return;

        const jobs = this.generateActiveJobs(nodeId, resourceType);
        const modal = this.createModal(`Active ${resourceType.toUpperCase()} Jobs - ${nodeId}`, `
            <div class="jobs-viewer">
                <h3>Active ${resourceType.toUpperCase()} Jobs on ${nodeId}</h3>
                <div class="jobs-controls">
                    <button type="button" class="btn btn-secondary" onclick="this.refreshJobs()">
                        <i class="fas fa-refresh"></i> Refresh
                    </button>
                    <button type="button" class="btn btn-primary" onclick="this.exportJobs()">
                        <i class="fas fa-download"></i> Export
                    </button>
                </div>
                <div class="jobs-list">
                    ${jobs.map(job => `
                        <div class="job-item" data-job-id="${job.id}">
                            <div class="job-header">
                                <h4>${job.name}</h4>
                                <div class="job-status status-${job.status}">${job.status}</div>
                            </div>
                            <div class="job-details">
                                <p>${job.description}</p>
                                <div class="job-stats">
                                    <span>CPU: ${job.resources.cpu}%</span>
                                    <span>Memory: ${job.resources.memory}</span>
                                    <span>Runtime: ${job.runtime}</span>
                                    ${job.resources.gpu ? `<span>GPU: ${job.resources.gpu}%</span>` : ''}
                                </div>
                            </div>
                            <div class="job-actions">
                                <button type="button" class="btn btn-warning" onclick="window.pauseJob('${job.id}')">
                                    <i class="fas fa-pause"></i> Pause
                                </button>
                                <button type="button" class="btn btn-danger" onclick="window.killJob('${job.id}')">
                                    <i class="fas fa-stop"></i> Kill
                                </button>
                                <button type="button" class="btn btn-info" onclick="this.viewJobDetails('${job.id}')">
                                    <i class="fas fa-info"></i> Details
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `);
    }

    generateActiveJobs(nodeId, resourceType) {
        const jobTypes = {
            gpu: ['ml-training', 'rendering', 'simulation'],
            cpu: ['data-processing', 'compilation', 'analysis'],
            memory: ['in-memory-db', 'cache-warming', 'buffer-operations']
        };

        const jobs = [];
        const typeJobs = jobTypes[resourceType] || ['generic-job'];
        
        for (let i = 0; i < Math.random() * 5 + 1; i++) {
            const type = typeJobs[Math.floor(Math.random() * typeJobs.length)];
            jobs.push({
                id: `${type}-${Date.now()}-${i}`,
                name: `${type.replace('-', ' ').toUpperCase()} ${i + 1}`,
                description: `${resourceType.toUpperCase()} intensive ${type} workload`,
                status: ['running', 'queued', 'paused'][Math.floor(Math.random() * 3)],
                resources: {
                    cpu: Math.floor(Math.random() * 100),
                    memory: `${Math.floor(Math.random() * 32)}GB`,
                    gpu: resourceType === 'gpu' ? Math.floor(Math.random() * 100) : null
                },
                runtime: `${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m`
            });
        }
        
        return jobs;
    }

    pauseJob(jobId) {
        this.showNotification('Job Paused', `Job ${jobId} has been paused`, 'info');
    }

    killJob(jobId) {
        if (confirm(`Are you sure you want to kill job ${jobId}?`)) {
            this.showNotification('Job Terminated', `Job ${jobId} has been terminated`, 'warning');
        }
    }

    downloadNodeLogs(nodeId) {
        const logs = this.generateNodeLogs(nodeId);
        const blob = new Blob([logs], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${nodeId}-logs-${Date.now()}.log`;
        a.click();
        URL.revokeObjectURL(url);
    }

    clearNodeLogs(nodeId) {
        this.showNotification('Logs Cleared', `Logs for ${nodeId} have been cleared`, 'info');
    }

    toggleResourcePool(nodeId, resourceType) {
        this.showNodeAction(nodeId, 'Resource Pool', `${resourceType.toUpperCase()} pool status toggled.`, 'info');
    }

    refreshNodeResources(nodeId) {
        const node = this.nodeStates.get(nodeId);
        if (node) {
            // Simulate resource refresh
            node.utilization = {
                cpu: Math.random() * 100,
                memory: Math.random() * 100,
                gpu: node.type === 'gpu' ? Math.random() * 100 : undefined
            };
            this.updateNodeDisplay();
        }
        this.showNodeAction(nodeId, 'Resources Updated', 'Resource information refreshed.', 'info');
    }

    refreshNUMAVisualization() {
        // Update NUMA node states with random data
        const numaNodes = document.querySelectorAll('.numa-node');
        numaNodes.forEach(node => {
            const classes = ['active', 'busy', 'overloaded', 'idle'];
            const randomClass = classes[Math.floor(Math.random() * classes.length)];
            node.className = `numa-node ${randomClass}`;
        });
        
        this.showNotification('NUMA Updated', 'NUMA topology visualization refreshed', 'info');
    }

    exportNUMAData() {
        const numaData = {
            timestamp: new Date().toISOString(),
            nodes: Array.from(this.nodeStates.values()),
            topology: 'NUMA-aware',
            configuration: this.numaConfiguration
        };
        
        const blob = new Blob([JSON.stringify(numaData, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `numa-topology-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification('NUMA Exported', 'NUMA topology data exported successfully', 'success');
    }

    openNUMASettings() {
        const modal = this.createModal('NUMA Configuration', `
            <div class="numa-settings">
                <h3>NUMA & Rack Configuration</h3>
                <div class="settings-form">
                    <div class="form-group">
                        <label>Memory Binding Policy:</label>
                        <select id="memoryBinding">
                            <option value="strict" ${this.numaConfiguration.memoryBinding === 'strict' ? 'selected' : ''}>Strict</option>
                            <option value="preferred" ${this.numaConfiguration.memoryBinding === 'preferred' ? 'selected' : ''}>Preferred</option>
                            <option value="interleaved" ${this.numaConfiguration.memoryBinding === 'interleaved' ? 'selected' : ''}>Interleaved</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>CPU Affinity:</label>
                        <select id="cpuAffinity">
                            <option value="numa-node" ${this.numaConfiguration.cpuAffinity === 'numa-node' ? 'selected' : ''}>NUMA Node</option>
                            <option value="socket" ${this.numaConfiguration.cpuAffinity === 'socket' ? 'selected' : ''}>Socket</option>
                            <option value="core" ${this.numaConfiguration.cpuAffinity === 'core' ? 'selected' : ''}>Core</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Load Balancing:</label>
                        <select id="balancingPolicy">
                            <option value="load-balanced" ${this.numaConfiguration.balancingPolicy === 'load-balanced' ? 'selected' : ''}>Load Balanced</option>
                            <option value="round-robin" ${this.numaConfiguration.balancingPolicy === 'round-robin' ? 'selected' : ''}>Round Robin</option>
                            <option value="least-loaded" ${this.numaConfiguration.balancingPolicy === 'least-loaded' ? 'selected' : ''}>Least Loaded</option>
                        </select>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn btn-primary" onclick="this.applyNUMAConfiguration()">
                            Apply Configuration
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="this.resetNUMADefaults()">
                            Reset to Defaults
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `);
    }

    startNodeMonitoring() {
        this.refreshInterval = setInterval(() => {
            this.updateNodeMetrics();
            this.updateNodeDisplay();
        }, 5000);
    }

    updateNodeMetrics() {
        this.nodeStates.forEach((node, nodeId) => {
            if (node.status === 'online') {
                // Update utilization with some variation
                node.utilization.cpu = Math.max(0, Math.min(100, 
                    node.utilization.cpu + (Math.random() - 0.5) * 10));
                node.utilization.memory = Math.max(0, Math.min(100, 
                    node.utilization.memory + (Math.random() - 0.5) * 5));
                    
                if (node.utilization.gpu !== undefined) {
                    node.utilization.gpu = Math.max(0, Math.min(100, 
                        node.utilization.gpu + (Math.random() - 0.5) * 15));
                }

                // Update metrics history
                const metrics = this.nodeMetrics.get(nodeId);
                if (metrics) {
                    metrics.history.shift();
                    metrics.history.push({
                        timestamp: new Date(),
                        cpu: node.utilization.cpu,
                        memory: node.utilization.memory,
                        network: Math.random() * 100,
                        temperature: 25 + Math.random() * 50
                    });
                    metrics.lastUpdate = new Date();
                }
            }
        });
    }

    updateNodeDisplay() {
        // Update node list display
        const nodesList = document.getElementById('nodesList');
        if (nodesList) {
            this.renderNodesList(nodesList);
        }

        // Update node count
        this.updateElementText('totalNodes', this.nodeStates.size);
        this.updateElementText('onlineNodes', 
            Array.from(this.nodeStates.values()).filter(n => n.status === 'online').length);
    }

    renderNodesList(container) {
        const nodes = Array.from(this.nodeStates.values());
        
        container.innerHTML = nodes.map(node => `
            <div class="node-item ${node.status}" data-node-id="${node.id}">
                <div class="node-header">
                    <div class="node-name">${node.name}</div>
                    <div class="node-status status-${node.status}">${node.status}</div>
                </div>
                <div class="node-details">
                    <span class="node-type">${node.type}</span>
                    <span class="node-location">${node.location.rack}-${node.location.position}</span>
                    <span class="node-workloads">${node.workloads} jobs</span>
                </div>
                <div class="node-utilization">
                    <div class="util-bar">
                        <label>CPU: ${Math.round(node.utilization?.cpu || 0)}%</label>
                        <div class="bar"><div class="fill" style="width: ${node.utilization?.cpu || 0}%"></div></div>
                    </div>
                    <div class="util-bar">
                        <label>Memory: ${Math.round(node.utilization?.memory || 0)}%</label>
                        <div class="bar"><div class="fill" style="width: ${node.utilization?.memory || 0}%"></div></div>
                    </div>
                </div>
                <div class="node-actions">
                    <button onclick="window.viewNodeMetrics('${node.id}')" class="btn btn-info">Metrics</button>
                    <button onclick="window.connectToNode('${node.id}')" class="btn btn-primary">Connect</button>
                    <button onclick="window.restartNode('${node.id}')" class="btn btn-warning">Restart</button>
                </div>
            </div>
        `).join('');
    }

    // Utility methods
    async simulateNodeOperation(nodeId, operation, duration) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (Math.random() > 0.1) { // 90% success rate
                    resolve();
                } else {
                    reject(new Error(`Failed to ${operation} node ${nodeId}`));
                }
            }, duration);
        });
    }

    showNodeAction(nodeId, title, message, type) {
        if (window.menuBarManager) {
            window.menuBarManager.showNotification(title, `${nodeId}: ${message}`, type);
        } else {
            console.log(`${title}: ${nodeId}: ${message}`);
        }
    }

    showNotification(title, message, type) {
        if (window.menuBarManager) {
            window.menuBarManager.showNotification(title, message, type);
        } else {
            console.log(`${title}: ${message}`);
        }
    }

    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button type="button" class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        return modal;
    }

    updateElementText(id, text) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = text;
        }
    }

    // Public API
    getNodeStates() {
        return new Map(this.nodeStates);
    }

    getNodeMetrics(nodeId) {
        return this.nodeMetrics.get(nodeId);
    }

    getAllNodeMetrics() {
        return new Map(this.nodeMetrics);
    }

    getActiveConnections() {
        return new Map(this.activeConnections);
    }

    dispose() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.activeConnections.clear();
        console.log('🧹 Node Control Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.NodeControlManager = NodeControlManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeControlManager;
}
