// Nodes Management Component
class NodesComponent {
    constructor() {
        this.nodes = [
            {
                id: 'node-01',
                name: 'Control Node 01',
                type: 'control',
                status: 'healthy',
                cpu: 45,
                memory: 62,
                load: 1.2,
                uptime: '5d 12h',
                sessions: 3,
                ip: '192.168.1.10'
            },
            {
                id: 'node-02',
                name: 'Compute Node 02',
                type: 'compute',
                status: 'healthy',
                cpu: 78,
                memory: 83,
                load: 2.1,
                uptime: '3d 8h',
                sessions: 5,
                ip: '192.168.1.11'
            },
            {
                id: 'node-03',
                name: 'Storage Node 03',
                type: 'storage',
                status: 'warning',
                cpu: 23,
                memory: 91,
                load: 0.8,
                uptime: '7d 2h',
                sessions: 1,
                ip: '192.168.1.12'
            },
            {
                id: 'node-04',
                name: 'Edge Node 04',
                type: 'edge',
                status: 'healthy',
                cpu: 34,
                memory: 56,
                load: 1.5,
                uptime: '2d 15h',
                sessions: 2,
                ip: '192.168.1.13'
            }
        ];
        
        this.selectedNode = null;
        this.init();
    }

    init() {
        this.startMonitoring();
    }

    render() {
        return `
            <div class="nodes-container">
                <div class="nodes-header">
                    <h2><i class="fas fa-server"></i> Node Management</h2>
                    <div class="nodes-controls">
                        <button class="btn-primary" onclick="nodesComponent.addNode()">
                            <i class="fas fa-plus"></i> Add Node
                        </button>
                        <button class="btn-secondary" onclick="nodesComponent.refreshNodes()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                
                <div class="nodes-grid">
                    <div class="nodes-list">
                        <div class="nodes-toolbar">
                            <div class="filter-controls">
                                <label>Filter by type:</label>
                                <select onchange="nodesComponent.filterByType(this.value)">
                                    <option value="all">All Types</option>
                                    <option value="control">Control</option>
                                    <option value="compute">Compute</option>
                                    <option value="storage">Storage</option>
                                    <option value="edge">Edge</option>
                                </select>
                            </div>
                            <div class="search-controls">
                                <input type="text" placeholder="Search nodes..." onkeyup="nodesComponent.searchNodes(this.value)">
                            </div>
                        </div>
                        
                        <div class="nodes-table">
                            <div class="table-header">
                                <div class="col-name">Node</div>
                                <div class="col-type">Type</div>
                                <div class="col-status">Status</div>
                                <div class="col-resources">Resources</div>
                                <div class="col-sessions">Sessions</div>
                                <div class="col-actions">Actions</div>
                            </div>
                            <div class="table-body" id="nodes-table-body">
                                ${this.renderNodesTable()}
                            </div>
                        </div>
                    </div>
                    
                    <div class="node-details" id="node-details">
                        ${this.renderNodeDetails()}
                    </div>
                </div>
            </div>
        `;
    }

    renderNodesTable() {
        return this.nodes.map(node => `
            <div class="table-row ${node.status}" onclick="nodesComponent.selectNode('${node.id}')">
                <div class="col-name">
                    <div class="node-info">
                        <i class="fas fa-${this.getNodeIcon(node.type)}"></i>
                        <div>
                            <div class="node-name">${node.name}</div>
                            <div class="node-ip">${node.ip}</div>
                        </div>
                    </div>
                </div>
                <div class="col-type">
                    <span class="node-type ${node.type}">${node.type}</span>
                </div>
                <div class="col-status">
                    <span class="status-indicator ${node.status}"></span>
                    ${node.status}
                </div>
                <div class="col-resources">
                    <div class="resource-bars">
                        <div class="resource-bar">
                            <label>CPU</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${node.cpu}%"></div>
                            </div>
                            <span>${node.cpu}%</span>
                        </div>
                        <div class="resource-bar">
                            <label>RAM</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${node.memory}%"></div>
                            </div>
                            <span>${node.memory}%</span>
                        </div>
                    </div>
                </div>
                <div class="col-sessions">
                    <span class="session-count">${node.sessions}</span>
                </div>
                <div class="col-actions">
                    <button class="btn-sm" onclick="nodesComponent.manageNode('${node.id}')" title="Manage Node">
                        <i class="fas fa-cog"></i>
                    </button>
                    <button class="btn-sm" onclick="nodesComponent.restartNode('${node.id}')" title="Restart Node">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    renderNodeDetails() {
        if (!this.selectedNode) {
            return `
                <div class="no-selection">
                    <i class="fas fa-server"></i>
                    <h3>Select a Node</h3>
                    <p>Choose a node from the list to view detailed information and management options.</p>
                </div>
            `;
        }

        const node = this.nodes.find(n => n.id === this.selectedNode);
        if (!node) return '<div class="no-selection">Node not found</div>';

        return `
            <div class="node-details-content">
                <div class="details-header">
                    <h3><i class="fas fa-${this.getNodeIcon(node.type)}"></i> ${node.name}</h3>
                    <span class="status-badge ${node.status}">${node.status}</span>
                </div>
                
                <div class="details-sections">
                    <div class="details-section">
                        <h4>System Information</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <label>Node ID</label>
                                <span>${node.id}</span>
                            </div>
                            <div class="info-item">
                                <label>Type</label>
                                <span class="node-type ${node.type}">${node.type}</span>
                            </div>
                            <div class="info-item">
                                <label>IP Address</label>
                                <span>${node.ip}</span>
                            </div>
                            <div class="info-item">
                                <label>Uptime</label>
                                <span>${node.uptime}</span>
                            </div>
                            <div class="info-item">
                                <label>Load Average</label>
                                <span>${node.load}</span>
                            </div>
                            <div class="info-item">
                                <label>Active Sessions</label>
                                <span>${node.sessions}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-section">
                        <h4>Resource Usage</h4>
                        <div class="resource-details">
                            <div class="resource-item">
                                <div class="resource-header">
                                    <label>CPU Usage</label>
                                    <span>${node.cpu}%</span>
                                </div>
                                <div class="progress-bar large">
                                    <div class="progress-fill" style="width: ${node.cpu}%"></div>
                                </div>
                            </div>
                            <div class="resource-item">
                                <div class="resource-header">
                                    <label>Memory Usage</label>
                                    <span>${node.memory}%</span>
                                </div>
                                <div class="progress-bar large">
                                    <div class="progress-fill" style="width: ${node.memory}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-section">
                        <h4>Management Actions</h4>
                        <div class="action-buttons">
                            <button class="btn-primary" onclick="nodesComponent.openTerminal('${node.id}')">
                                <i class="fas fa-terminal"></i> Open Terminal
                            </button>
                            <button class="btn-secondary" onclick="nodesComponent.viewLogs('${node.id}')">
                                <i class="fas fa-file-alt"></i> View Logs
                            </button>
                            <button class="btn-secondary" onclick="nodesComponent.configureNode('${node.id}')">
                                <i class="fas fa-cog"></i> Configure
                            </button>
                            <button class="btn-warning" onclick="nodesComponent.restartNode('${node.id}')">
                                <i class="fas fa-redo"></i> Restart
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getNodeIcon(type) {
        const icons = {
            control: 'laptop-code',
            compute: 'microchip',
            storage: 'hdd',
            edge: 'wifi'
        };
        return icons[type] || 'server';
    }

    selectNode(nodeId) {
        this.selectedNode = nodeId;
        document.getElementById('node-details').innerHTML = this.renderNodeDetails();
        
        // Update selected row styling
        document.querySelectorAll('.table-row').forEach(row => row.classList.remove('selected'));
        event.target.closest('.table-row').classList.add('selected');
    }

    filterByType(type) {
        const filteredNodes = type === 'all' ? this.nodes : this.nodes.filter(node => node.type === type);
        document.getElementById('nodes-table-body').innerHTML = this.renderNodesTableForNodes(filteredNodes);
    }

    searchNodes(query) {
        const filteredNodes = this.nodes.filter(node => 
            node.name.toLowerCase().includes(query.toLowerCase()) ||
            node.ip.includes(query) ||
            node.type.toLowerCase().includes(query.toLowerCase())
        );
        document.getElementById('nodes-table-body').innerHTML = this.renderNodesTableForNodes(filteredNodes);
    }

    renderNodesTableForNodes(nodes) {
        return nodes.map(node => `
            <div class="table-row ${node.status}" onclick="nodesComponent.selectNode('${node.id}')">
                <div class="col-name">
                    <div class="node-info">
                        <i class="fas fa-${this.getNodeIcon(node.type)}"></i>
                        <div>
                            <div class="node-name">${node.name}</div>
                            <div class="node-ip">${node.ip}</div>
                        </div>
                    </div>
                </div>
                <div class="col-type">
                    <span class="node-type ${node.type}">${node.type}</span>
                </div>
                <div class="col-status">
                    <span class="status-indicator ${node.status}"></span>
                    ${node.status}
                </div>
                <div class="col-resources">
                    <div class="resource-bars">
                        <div class="resource-bar">
                            <label>CPU</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${node.cpu}%"></div>
                            </div>
                            <span>${node.cpu}%</span>
                        </div>
                        <div class="resource-bar">
                            <label>RAM</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${node.memory}%"></div>
                            </div>
                            <span>${node.memory}%</span>
                        </div>
                    </div>
                </div>
                <div class="col-sessions">
                    <span class="session-count">${node.sessions}</span>
                </div>
                <div class="col-actions">
                    <button class="btn-sm" onclick="nodesComponent.manageNode('${node.id}')" title="Manage Node">
                        <i class="fas fa-cog"></i>
                    </button>
                    <button class="btn-sm" onclick="nodesComponent.restartNode('${node.id}')" title="Restart Node">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    addNode() {
        console.log('Adding new node...');
        // Implementation for adding a new node
    }

    refreshNodes() {
        console.log('Refreshing nodes...');
        // Simulate data refresh
        this.updateNodeMetrics();
        document.getElementById('nodes-table-body').innerHTML = this.renderNodesTable();
    }

    manageNode(nodeId) {
        console.log(`Managing node: ${nodeId}`);
        // Implementation for node management
    }

    restartNode(nodeId) {
        if (confirm(`Are you sure you want to restart ${nodeId}?`)) {
            console.log(`Restarting node: ${nodeId}`);
            // Implementation for node restart
        }
    }

    openTerminal(nodeId) {
        console.log(`Opening terminal for node: ${nodeId}`);
        // Implementation for opening terminal
    }

    viewLogs(nodeId) {
        console.log(`Viewing logs for node: ${nodeId}`);
        // Implementation for viewing logs
    }

    configureNode(nodeId) {
        console.log(`Configuring node: ${nodeId}`);
        // Implementation for node configuration
    }

    updateNodeMetrics() {
        this.nodes.forEach(node => {
            node.cpu = Math.floor(Math.random() * 40) + 20;
            node.memory = Math.floor(Math.random() * 50) + 30;
        });
    }

    startMonitoring() {
        setInterval(() => {
            this.updateNodeMetrics();
            if (document.getElementById('nodes-table-body')) {
                document.getElementById('nodes-table-body').innerHTML = this.renderNodesTable();
            }
            if (this.selectedNode && document.getElementById('node-details')) {
                document.getElementById('node-details').innerHTML = this.renderNodeDetails();
            }
        }, 30000);
    }
}

// Global instance
window.nodesComponent = new NodesComponent();
