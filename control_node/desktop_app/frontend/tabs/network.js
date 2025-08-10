export function renderNetwork(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.network) {
    console.log('[Network] State structure missing:', { state, data: state?.data, network: state?.data?.network });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  // Advanced network monitoring interface
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- Network Status Header -->
      <div class="network-status-header">
        <div class="network-overview">
          <div class="network-metric">
            <i class="fas fa-ethernet"></i>
            <span>Interfaces: <strong id="interface-count">0</strong></span>
          </div>
          <div class="network-metric">
            <i class="fas fa-exchange-alt"></i>
            <span>Throughput: <strong id="total-throughput">0 MB/s</strong></span>
          </div>
          <div class="network-metric">
            <i class="fas fa-clock"></i>
            <span>Latency: <strong id="avg-latency">0ms</strong></span>
          </div>
          <div class="network-metric">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Errors: <strong id="error-count">0</strong></span>
          </div>
        </div>
        <div class="network-actions">
          <button onclick="refreshNetworkData()" class="network-btn">
            <i class="fas fa-sync"></i>
            Refresh
          </button>
          <button onclick="runNetworkDiagnostics()" class="network-btn">
            <i class="fas fa-stethoscope"></i>
            Diagnostics
          </button>
          <button onclick="exportNetworkConfig()" class="network-btn">
            <i class="fas fa-download"></i>
            Export
          </button>
        </div>
      </div>

      <!-- Network Sub-tabs -->
      <div class="network-subtabs">
        <button class="network-subtab active" onclick="switchNetworkTab('topology')">
          <i class="fas fa-sitemap"></i>
          <span>Topology</span>
        </button>
        <button class="network-subtab" onclick="switchNetworkTab('interfaces')">
          <i class="fas fa-ethernet"></i>
          <span>Interfaces</span>
        </button>
        <button class="network-subtab" onclick="switchNetworkTab('traffic')">
          <i class="fas fa-chart-line"></i>
          <span>Traffic</span>
        </button>
        <button class="network-subtab" onclick="switchNetworkTab('routing')">
          <i class="fas fa-route"></i>
          <span>Routing</span>
        </button>
        <button class="network-subtab" onclick="switchNetworkTab('security')">
          <i class="fas fa-shield-alt"></i>
          <span>Security</span>
        </button>
        <button class="network-subtab" onclick="switchNetworkTab('monitoring')">
          <i class="fas fa-radar"></i>
          <span>Monitoring</span>
        </button>
      </div>

      <!-- Network Content Panels -->
      <div class="network-content">
        <!-- Topology Panel -->
        <div id="network-topology" class="network-panel active">
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="topology-canvas-container">
              <div class="topology-controls">
                <div class="topology-tools">
                  <button onclick="zoomIn()" class="topo-btn"><i class="fas fa-search-plus"></i></button>
                  <button onclick="zoomOut()" class="topo-btn"><i class="fas fa-search-minus"></i></button>
                  <button onclick="resetView()" class="topo-btn"><i class="fas fa-home"></i></button>
                  <button onclick="autoLayout()" class="topo-btn"><i class="fas fa-project-diagram"></i></button>
                </div>
                <div class="topology-view-options">
                  <select id="topology-view">
                    <option value="physical">Physical View</option>
                    <option value="logical">Logical View</option>
                    <option value="cluster">Cluster View</option>
                  </select>
                </div>
              </div>
              <div id="topology-canvas" class="topology-canvas">
                <!-- Interactive network topology will be rendered here -->
              </div>
            </div>
            <div class="topology-sidebar">
              <div class="topology-info-panel">
                <h5>Node Information</h5>
                <div id="node-details" class="node-details">
                  <div class="no-selection">Select a node to view details</div>
                </div>
              </div>
              <div class="topology-legend">
                <h5>Legend</h5>
                <div class="legend-items">
                  <div class="legend-item">
                    <div class="legend-color" style="background: var(--omega-cyan);"></div>
                    <span>Control Node</span>
                  </div>
                  <div class="legend-item">
                    <div class="legend-color" style="background: var(--omega-green);"></div>
                    <span>Compute Node</span>
                  </div>
                  <div class="legend-item">
                    <div class="legend-color" style="background: var(--omega-blue);"></div>
                    <span>Storage Node</span>
                  </div>
                  <div class="legend-item">
                    <div class="legend-color" style="background: var(--omega-yellow);"></div>
                    <span>Edge Node</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Interfaces Panel -->
        <div id="network-interfaces" class="network-panel">
          <div class="interfaces-header">
            <div class="interfaces-filter">
              <input type="text" id="interface-search" placeholder="Search interfaces..." class="interface-search">
              <select id="interface-filter" class="interface-filter-select">
                <option value="all">All Interfaces</option>
                <option value="active">Active Only</option>
                <option value="ethernet">Ethernet</option>
                <option value="wireless">Wireless</option>
              </select>
            </div>
            <div class="interfaces-actions">
              <button onclick="addInterface()" class="interface-btn primary">
                <i class="fas fa-plus"></i>
                Add Interface
              </button>
            </div>
          </div>
          <div class="interfaces-table-container">
            <table class="interfaces-table">
              <thead>
                <tr>
                  <th>Interface</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Speed</th>
                  <th>IP Address</th>
                  <th>RX/TX</th>
                  <th>Errors</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="interfaces-table-body">
                <!-- Populated by renderInterfacesTable() -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Traffic Panel -->
        <div id="network-traffic" class="network-panel">
          <div class="traffic-controls">
            <div class="traffic-filters">
              <select id="traffic-timeframe">
                <option value="realtime">Real-time</option>
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
              </select>
              <select id="traffic-granularity">
                <option value="second">Per Second</option>
                <option value="minute">Per Minute</option>
                <option value="hour">Per Hour</option>
              </select>
            </div>
            <div class="traffic-actions">
              <button onclick="startTrafficCapture()" class="traffic-btn primary">
                <i class="fas fa-play"></i>
                Start Capture
              </button>
              <button onclick="stopTrafficCapture()" class="traffic-btn">
                <i class="fas fa-stop"></i>
                Stop
              </button>
              <button onclick="exportTrafficData()" class="traffic-btn">
                <i class="fas fa-download"></i>
                Export
              </button>
            </div>
          </div>
          <div style="display: grid; grid-template-rows: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="traffic-charts">
              <div class="traffic-chart-container">
                <h5>Bandwidth Utilization</h5>
                <canvas id="bandwidth-chart" style="width: 100%; height: 150px;"></canvas>
              </div>
            </div>
            <div class="traffic-analysis">
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div class="traffic-stats">
                  <h5>Traffic Statistics</h5>
                  <div id="traffic-stats-content">
                    <!-- Populated by renderTrafficStats() -->
                  </div>
                </div>
                <div class="protocol-breakdown">
                  <h5>Protocol Breakdown</h5>
                  <canvas id="protocol-chart" style="width: 100%; height: 150px;"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Routing Panel -->
        <div id="network-routing" class="network-panel">
          <div class="routing-header">
            <div class="routing-tabs">
              <button class="routing-tab active" onclick="switchRoutingView('table')">Routing Table</button>
              <button class="routing-tab" onclick="switchRoutingView('static')">Static Routes</button>
              <button class="routing-tab" onclick="switchRoutingView('dynamic')">Dynamic Routes</button>
            </div>
            <div class="routing-actions">
              <button onclick="addStaticRoute()" class="routing-btn primary">
                <i class="fas fa-plus"></i>
                Add Route
              </button>
              <button onclick="refreshRoutes()" class="routing-btn">
                <i class="fas fa-sync"></i>
                Refresh
              </button>
            </div>
          </div>
          <div id="routing-content" class="routing-content">
            <!-- Populated by renderRoutingTable() -->
          </div>
        </div>

        <!-- Security Panel -->
        <div id="network-security" class="network-panel">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="security-rules">
              <div class="security-header">
                <h5>Firewall Rules</h5>
                <button onclick="addFirewallRule()" class="security-btn primary">
                  <i class="fas fa-plus"></i>
                  Add Rule
                </button>
              </div>
              <div class="firewall-rules-list" id="firewall-rules">
                <!-- Populated by renderFirewallRules() -->
              </div>
            </div>
            <div class="security-monitoring">
              <div class="security-header">
                <h5>Security Events</h5>
                <button onclick="clearSecurityEvents()" class="security-btn">
                  <i class="fas fa-trash"></i>
                  Clear
                </button>
              </div>
              <div class="security-events-list" id="security-events">
                <!-- Populated by renderSecurityEvents() -->
              </div>
            </div>
          </div>
        </div>

        <!-- Monitoring Panel -->
        <div id="network-monitoring" class="network-panel">
          <div class="monitoring-dashboard">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px;">
              <div class="monitoring-metric">
                <div class="metric-icon"><i class="fas fa-tachometer-alt"></i></div>
                <div class="metric-info">
                  <div class="metric-label">Average Latency</div>
                  <div class="metric-value" id="avg-latency-metric">--</div>
                </div>
              </div>
              <div class="monitoring-metric">
                <div class="metric-icon"><i class="fas fa-exchange-alt"></i></div>
                <div class="metric-info">
                  <div class="metric-label">Packet Loss</div>
                  <div class="metric-value" id="packet-loss-metric">--</div>
                </div>
              </div>
              <div class="monitoring-metric">
                <div class="metric-icon"><i class="fas fa-wifi"></i></div>
                <div class="metric-info">
                  <div class="metric-label">Signal Quality</div>
                  <div class="metric-value" id="signal-quality-metric">--</div>
                </div>
              </div>
            </div>
            <div style="display: grid; grid-template-rows: 1fr 1fr; gap: 16px; height: 300px;">
              <div class="monitoring-chart">
                <h5>Network Health Over Time</h5>
                <canvas id="network-health-chart" style="width: 100%; height: 120px;"></canvas>
              </div>
              <div class="monitoring-alerts">
                <h5>Active Network Alerts</h5>
                <div id="network-alerts-list" class="alerts-container">
                  <!-- Populated by renderNetworkAlerts() -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize network data
  updateNetworkStatus(state);
  renderTopologyCanvas(state);
  renderInterfacesTable(state);
  renderTrafficStats(state);
  renderRoutingTable(state);
  renderFirewallRules(state);
  renderSecurityEvents(state);
  renderNetworkAlerts(state);
  initializeNetworkCharts(state);
}

function updateNetworkStatus(state) {
  const networkData = state.data.network || {};
  const interfaces = networkData.statistics?.interfaces || [];
  
  // Update header metrics
  document.getElementById('interface-count').textContent = interfaces.length;
  
  const totalRx = interfaces.reduce((sum, iface) => sum + (iface.bytes_recv || 0), 0);
  const totalTx = interfaces.reduce((sum, iface) => sum + (iface.bytes_sent || 0), 0);
  const throughput = ((totalRx + totalTx) / (1024 * 1024)).toFixed(2);
  document.getElementById('total-throughput').textContent = `${throughput} MB/s`;
  
  const errors = interfaces.reduce((sum, iface) => sum + (iface.errin || 0) + (iface.errout || 0), 0);
  document.getElementById('error-count').textContent = errors;
  
  // Simulate latency for now
  document.getElementById('avg-latency').textContent = '2.3ms';
}

function renderTopologyCanvas(state) {
  const canvas = document.getElementById('topology-canvas');
  const networkData = state.data.network || {};
  const topology = networkData.topology || { nodes: [], connections: [] };
  
  // Clear existing content
  canvas.innerHTML = '';
  
  // Create SVG for connections
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  svg.style.position = 'absolute';
  svg.style.top = '0';
  svg.style.left = '0';
  svg.style.pointerEvents = 'none';
  canvas.appendChild(svg);
  
  // Add sample nodes if none exist
  const nodes = topology.nodes.length > 0 ? topology.nodes : [
    { id: 'control-01', type: 'control', x: 200, y: 100, status: 'active' },
    { id: 'compute-01', type: 'compute', x: 100, y: 200, status: 'active' },
    { id: 'compute-02', type: 'compute', x: 300, y: 200, status: 'active' },
    { id: 'storage-01', type: 'storage', x: 200, y: 300, status: 'active' },
    { id: 'edge-01', type: 'edge', x: 50, y: 50, status: 'standby' }
  ];
  
  // Add sample connections
  const connections = topology.connections.length > 0 ? topology.connections : [
    { source: 'control-01', target: 'compute-01' },
    { source: 'control-01', target: 'compute-02' },
    { source: 'control-01', target: 'storage-01' },
    { source: 'compute-01', target: 'storage-01' },
    { source: 'compute-02', target: 'storage-01' },
    { source: 'control-01', target: 'edge-01' }
  ];
  
  // Draw connections
  connections.forEach(conn => {
    const sourceNode = nodes.find(n => n.id === conn.source);
    const targetNode = nodes.find(n => n.id === conn.target);
    
    if (sourceNode && targetNode) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sourceNode.x + 30);
      line.setAttribute('y1', sourceNode.y + 20);
      line.setAttribute('x2', targetNode.x + 30);
      line.setAttribute('y2', targetNode.y + 20);
      line.setAttribute('stroke', 'var(--omega-gray-1)');
      line.setAttribute('stroke-width', '2');
      svg.appendChild(line);
    }
  });
  
  // Draw nodes
  nodes.forEach(node => {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'topology-node';
    nodeEl.style.left = node.x + 'px';
    nodeEl.style.top = node.y + 'px';
    nodeEl.onclick = () => selectTopologyNode(node);
    
    const typeColors = {
      control: 'var(--omega-cyan)',
      compute: 'var(--omega-green)',
      storage: 'var(--omega-blue)',
      edge: 'var(--omega-yellow)'
    };
    
    nodeEl.innerHTML = `
      <div class="node-visual" style="border-color: ${typeColors[node.type] || 'var(--omega-gray-1)'};">
        <i class="fas fa-${getNodeIcon(node.type)}"></i>
      </div>
      <div class="node-label">${node.id}</div>
      <div class="node-status ${node.status || 'unknown'}"></div>
    `;
    
    canvas.appendChild(nodeEl);
  });
}

function getNodeIcon(type) {
  const icons = {
    control: 'crown',
    compute: 'microchip',
    storage: 'database',
    edge: 'satellite'
  };
  return icons[type] || 'server';
}

function selectTopologyNode(node) {
  const detailsEl = document.getElementById('node-details');
  detailsEl.innerHTML = `
    <div class="node-detail-header">
      <h6>${node.id}</h6>
      <span class="node-type-badge ${node.type}">${node.type.toUpperCase()}</span>
    </div>
    <div class="node-detail-content">
      <div class="detail-item">
        <span class="detail-label">Status:</span>
        <span class="detail-value ${node.status}">${(node.status || 'unknown').toUpperCase()}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">IP Address:</span>
        <span class="detail-value">192.168.1.${Math.floor(Math.random() * 254) + 1}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">Uptime:</span>
        <span class="detail-value">${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">Load:</span>
        <span class="detail-value">${(Math.random() * 100).toFixed(1)}%</span>
      </div>
    </div>
    <div class="node-actions">
      <button onclick="pingNode('${node.id}')" class="node-action-btn">
        <i class="fas fa-satellite-dish"></i>
        Ping
      </button>
      <button onclick="restartNode('${node.id}')" class="node-action-btn">
        <i class="fas fa-redo"></i>
        Restart
      </button>
    </div>
  `;
}

function renderInterfacesTable(state) {
  const tbody = document.getElementById('interfaces-table-body');
  const interfaces = state.data.network?.statistics?.interfaces || [];
  
  // Add sample data if none exists
  const sampleInterfaces = interfaces.length > 0 ? interfaces : [
    { name: 'eth0', type: 'ethernet', isup: true, speed: '1000', addrs: [{ address: '192.168.1.100' }], bytes_recv: 1024000, bytes_sent: 512000, errin: 0, errout: 0 },
    { name: 'eth1', type: 'ethernet', isup: true, speed: '1000', addrs: [{ address: '10.0.0.5' }], bytes_recv: 2048000, bytes_sent: 1024000, errin: 2, errout: 1 },
    { name: 'lo', type: 'loopback', isup: true, speed: 'N/A', addrs: [{ address: '127.0.0.1' }], bytes_recv: 1000, bytes_sent: 1000, errin: 0, errout: 0 }
  ];
  
  tbody.innerHTML = sampleInterfaces.map(iface => `
    <tr class="interface-row">
      <td>
        <div class="interface-name">
          <i class="fas fa-${iface.type === 'ethernet' ? 'ethernet' : 'circle'}"></i>
          <span>${iface.name}</span>
        </div>
      </td>
      <td><span class="interface-type">${iface.type}</span></td>
      <td>
        <span class="interface-status ${iface.isup ? 'up' : 'down'}">
          ${iface.isup ? 'UP' : 'DOWN'}
        </span>
      </td>
      <td>${iface.speed} Mbps</td>
      <td>${iface.addrs?.[0]?.address || 'N/A'}</td>
      <td>
        <div class="traffic-stats">
          <div>RX: ${formatBytes(iface.bytes_recv || 0)}</div>
          <div>TX: ${formatBytes(iface.bytes_sent || 0)}</div>
        </div>
      </td>
      <td>
        <span class="error-count ${(iface.errin + iface.errout) > 0 ? 'has-errors' : ''}">
          ${(iface.errin || 0) + (iface.errout || 0)}
        </span>
      </td>
      <td>
        <div class="interface-actions">
          <button onclick="configureInterface('${iface.name}')" class="action-btn">
            <i class="fas fa-cog"></i>
          </button>
          <button onclick="toggleInterface('${iface.name}')" class="action-btn">
            <i class="fas fa-power-off"></i>
          </button>
        </div>
      </td>
    </tr>
  `).join('');
}

function renderTrafficStats(state) {
  const statsEl = document.getElementById('traffic-stats-content');
  if (!statsEl) return;
  
  const interfaces = state.data.network?.statistics?.interfaces || [];
  const totalRx = interfaces.reduce((sum, iface) => sum + (iface.bytes_recv || 0), 0);
  const totalTx = interfaces.reduce((sum, iface) => sum + (iface.bytes_sent || 0), 0);
  
  statsEl.innerHTML = `
    <div class="traffic-stat">
      <div class="stat-label">Total Received</div>
      <div class="stat-value">${formatBytes(totalRx)}</div>
    </div>
    <div class="traffic-stat">
      <div class="stat-label">Total Transmitted</div>
      <div class="stat-value">${formatBytes(totalTx)}</div>
    </div>
    <div class="traffic-stat">
      <div class="stat-label">Peak Bandwidth</div>
      <div class="stat-value">125.3 MB/s</div>
    </div>
    <div class="traffic-stat">
      <div class="stat-label">Average Utilization</div>
      <div class="stat-value">23.7%</div>
    </div>
  `;
}

function renderRoutingTable(state) {
  const contentEl = document.getElementById('routing-content');
  if (!contentEl) return;
  
  // Sample routing table
  const routes = [
    { destination: '0.0.0.0/0', gateway: '192.168.1.1', interface: 'eth0', metric: 100, flags: 'UG' },
    { destination: '192.168.1.0/24', gateway: '0.0.0.0', interface: 'eth0', metric: 0, flags: 'U' },
    { destination: '10.0.0.0/8', gateway: '10.0.0.1', interface: 'eth1', metric: 200, flags: 'UG' },
    { destination: '127.0.0.0/8', gateway: '0.0.0.0', interface: 'lo', metric: 0, flags: 'U' }
  ];
  
  contentEl.innerHTML = `
    <table class="routing-table">
      <thead>
        <tr>
          <th>Destination</th>
          <th>Gateway</th>
          <th>Interface</th>
          <th>Metric</th>
          <th>Flags</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${routes.map(route => `
          <tr>
            <td><code>${route.destination}</code></td>
            <td><code>${route.gateway}</code></td>
            <td><span class="interface-badge">${route.interface}</span></td>
            <td>${route.metric}</td>
            <td><code>${route.flags}</code></td>
            <td>
              <button onclick="editRoute('${route.destination}')" class="route-btn">
                <i class="fas fa-edit"></i>
              </button>
              <button onclick="deleteRoute('${route.destination}')" class="route-btn danger">
                <i class="fas fa-trash"></i>
              </button>
            </td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

function renderFirewallRules(state) {
  const rulesEl = document.getElementById('firewall-rules');
  if (!rulesEl) return;
  
  const sampleRules = [
    { id: 1, action: 'ALLOW', protocol: 'TCP', source: 'any', destination: '22', description: 'SSH Access' },
    { id: 2, action: 'ALLOW', protocol: 'TCP', source: 'any', destination: '80', description: 'HTTP Traffic' },
    { id: 3, action: 'ALLOW', protocol: 'TCP', source: 'any', destination: '443', description: 'HTTPS Traffic' },
    { id: 4, action: 'DENY', protocol: 'TCP', source: 'any', destination: '23', description: 'Block Telnet' }
  ];
  
  rulesEl.innerHTML = sampleRules.map(rule => `
    <div class="firewall-rule">
      <div class="rule-header">
        <span class="rule-action ${rule.action.toLowerCase()}">${rule.action}</span>
        <span class="rule-protocol">${rule.protocol}</span>
        <button onclick="deleteFirewallRule(${rule.id})" class="rule-delete">
          <i class="fas fa-times"></i>
        </button>
      </div>
      <div class="rule-details">
        <div class="rule-detail">
          <span class="detail-label">Source:</span>
          <code>${rule.source}</code>
        </div>
        <div class="rule-detail">
          <span class="detail-label">Destination:</span>
          <code>${rule.destination}</code>
        </div>
      </div>
      <div class="rule-description">${rule.description}</div>
    </div>
  `).join('');
}

function renderSecurityEvents(state) {
  const eventsEl = document.getElementById('security-events');
  if (!eventsEl) return;
  
  const sampleEvents = [
    { time: '14:32:15', type: 'warning', source: '192.168.1.50', event: 'Multiple failed login attempts' },
    { time: '14:28:42', type: 'info', source: '10.0.0.100', event: 'Port scan detected' },
    { time: '14:25:33', type: 'error', source: '192.168.1.200', event: 'Suspicious traffic pattern' },
    { time: '14:20:18', type: 'info', source: 'firewall', event: 'Rule updated successfully' }
  ];
  
  eventsEl.innerHTML = sampleEvents.map(event => `
    <div class="security-event ${event.type}">
      <div class="event-time">${event.time}</div>
      <div class="event-source">${event.source}</div>
      <div class="event-description">${event.event}</div>
    </div>
  `).join('');
}

function renderNetworkAlerts(state) {
  const alertsEl = document.getElementById('network-alerts-list');
  if (!alertsEl) return;
  
  const sampleAlerts = [
    { type: 'warning', message: 'High latency detected on eth1', time: '2 min ago' },
    { type: 'info', message: 'Interface eth0 recovered', time: '5 min ago' }
  ];
  
  if (sampleAlerts.length === 0) {
    alertsEl.innerHTML = '<div class="no-alerts">No active network alerts</div>';
    return;
  }
  
  alertsEl.innerHTML = sampleAlerts.map(alert => `
    <div class="network-alert ${alert.type}">
      <div class="alert-icon">
        <i class="fas fa-${alert.type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
      </div>
      <div class="alert-content">
        <div class="alert-message">${alert.message}</div>
        <div class="alert-time">${alert.time}</div>
      </div>
    </div>
  `).join('');
}

function initializeNetworkCharts(state) {
  // Initialize bandwidth chart
  const bandwidthCanvas = document.getElementById('bandwidth-chart');
  if (bandwidthCanvas) {
    drawBandwidthChart(bandwidthCanvas);
  }
  
  // Initialize protocol chart
  const protocolCanvas = document.getElementById('protocol-chart');
  if (protocolCanvas) {
    drawProtocolChart(protocolCanvas);
  }
  
  // Initialize network health chart
  const healthCanvas = document.getElementById('network-health-chart');
  if (healthCanvas) {
    drawNetworkHealthChart(healthCanvas);
  }
}

function drawBandwidthChart(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  
  // Clear canvas
  ctx.fillStyle = 'var(--omega-dark-3)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw sample bandwidth data
  const data = [20, 25, 30, 45, 40, 35, 50, 60, 55, 45];
  ctx.strokeStyle = 'var(--omega-cyan)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  data.forEach((value, index) => {
    const x = (index / (data.length - 1)) * canvas.width;
    const y = canvas.height - (value / 100) * canvas.height;
    
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  
  ctx.stroke();
}

function drawProtocolChart(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = Math.min(centerX, centerY) - 10;
  
  const protocols = [
    { name: 'TCP', value: 60, color: 'var(--omega-cyan)' },
    { name: 'UDP', value: 25, color: 'var(--omega-green)' },
    { name: 'ICMP', value: 10, color: 'var(--omega-yellow)' },
    { name: 'Other', value: 5, color: 'var(--omega-red)' }
  ];
  
  let currentAngle = -Math.PI / 2;
  
  protocols.forEach(protocol => {
    const sliceAngle = (protocol.value / 100) * 2 * Math.PI;
    
    ctx.fillStyle = protocol.color;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
    ctx.fill();
    
    currentAngle += sliceAngle;
  });
}

function drawNetworkHealthChart(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  
  // Clear canvas
  ctx.fillStyle = 'var(--omega-dark-3)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw sample health data
  const healthData = [95, 97, 94, 96, 98, 92, 95, 97, 96, 94];
  ctx.strokeStyle = 'var(--omega-green)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  healthData.forEach((value, index) => {
    const x = (index / (healthData.length - 1)) * canvas.width;
    const y = canvas.height - ((value - 90) / 10) * canvas.height;
    
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  
  ctx.stroke();
}

// Helper functions
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Global action functions
window.switchNetworkTab = (tabName) => {
  // Remove active class from all tabs and panels
  document.querySelectorAll('.network-subtab').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.network-panel').forEach(panel => panel.classList.remove('active'));
  
  // Add active class to selected tab and panel
  document.querySelector(`[onclick="switchNetworkTab('${tabName}')"]`).classList.add('active');
  document.getElementById(`network-${tabName}`).classList.add('active');
};

window.refreshNetworkData = async () => {
  try {
    window.notify('info', 'Network', 'Refreshing network data...');
    // Refresh implementation would go here
  } catch (e) {
    window.notify('error', 'Network', e.message);
  }
};

window.runNetworkDiagnostics = async () => {
  try {
    window.notify('info', 'Network', 'Running diagnostics...');
    // Diagnostics implementation would go here
  } catch (e) {
    window.notify('error', 'Network', e.message);
  }
};

// Add network-specific CSS
if (!document.getElementById('network-styles')) {
  const style = document.createElement('style');
  style.id = 'network-styles';
  style.textContent = `
    .network-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .network-overview {
      display: flex;
      gap: 24px;
    }
    
    .network-metric {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .network-metric i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .network-actions {
      display: flex;
      gap: 8px;
    }
    
    .network-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
    }
    
    .network-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .network-subtabs {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .network-subtab {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-bottom: none;
      color: var(--omega-light-1);
      padding: 8px 16px;
      cursor: pointer;
      transition: all 0.15s ease;
      font: 400 11px var(--font-mono);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .network-subtab:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .network-subtab.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .network-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .network-panel {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: var(--omega-dark-2);
      padding: 16px;
      opacity: 0;
      transform: translateX(20px);
      transition: all 0.2s ease;
      pointer-events: none;
      overflow: auto;
    }
    
    .network-panel.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .topology-canvas-container {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      position: relative;
      overflow: hidden;
    }
    
    .topology-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: var(--omega-dark-4);
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .topology-tools {
      display: flex;
      gap: 4px;
    }
    
    .topo-btn {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 6px;
      border-radius: 2px;
      cursor: pointer;
      font-size: 10px;
      transition: all 0.15s ease;
    }
    
    .topo-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .topology-canvas {
      position: relative;
      height: 400px;
      background: var(--omega-dark-2);
      overflow: hidden;
    }
    
    .topology-node {
      position: absolute;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }
    
    .topology-node:hover {
      transform: scale(1.05);
    }
    
    .node-visual {
      width: 60px;
      height: 40px;
      background: var(--omega-dark-4);
      border: 2px solid var(--omega-gray-1);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--omega-white);
      font-size: 16px;
      margin-bottom: 4px;
    }
    
    .node-label {
      font: 600 9px var(--font-mono);
      color: var(--omega-white);
      text-align: center;
      margin-bottom: 2px;
    }
    
    .node-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin: 0 auto;
    }
    
    .node-status.active {
      background: var(--omega-green);
    }
    
    .node-status.standby {
      background: var(--omega-yellow);
    }
    
    .node-status.unknown {
      background: var(--omega-red);
    }
    
    .topology-sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .topology-info-panel {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px;
      flex: 1;
    }
    
    .topology-info-panel h5 {
      margin: 0 0 8px 0;
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .node-details .no-selection {
      text-align: center;
      color: var(--omega-light-1);
      font: 400 10px var(--font-mono);
      padding: 20px;
    }
    
    .node-detail-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    
    .node-detail-header h6 {
      margin: 0;
      font: 600 12px var(--font-mono);
      color: var(--omega-white);
    }
    
    .node-type-badge {
      padding: 2px 6px;
      border-radius: 2px;
      font: 600 8px var(--font-mono);
      text-transform: uppercase;
    }
    
    .node-type-badge.control {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .node-type-badge.compute {
      background: var(--omega-green);
      color: var(--omega-black);
    }
    
    .node-type-badge.storage {
      background: var(--omega-blue);
      color: var(--omega-white);
    }
    
    .node-type-badge.edge {
      background: var(--omega-yellow);
      color: var(--omega-black);
    }
    
    .detail-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 6px;
    }
    
    .detail-label {
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .detail-value {
      font: 600 9px var(--font-mono);
      color: var(--omega-white);
    }
    
    .detail-value.active {
      color: var(--omega-green);
    }
    
    .detail-value.standby {
      color: var(--omega-yellow);
    }
    
    .node-actions {
      display: flex;
      gap: 4px;
      margin-top: 12px;
    }
    
    .node-action-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 8px;
      border-radius: 2px;
      cursor: pointer;
      font: 400 8px var(--font-mono);
      transition: all 0.15s ease;
      flex: 1;
    }
    
    .node-action-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .topology-legend {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px;
    }
    
    .topology-legend h5 {
      margin: 0 0 8px 0;
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .legend-items {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .legend-color {
      width: 12px;
      height: 12px;
      border-radius: 2px;
    }
    
    .legend-item span {
      font: 400 9px var(--font-mono);
      color: var(--omega-white);
    }
    
    .interfaces-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    
    .interfaces-filter {
      display: flex;
      gap: 8px;
    }
    
    .interface-search {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 8px;
      border-radius: 3px;
      font: 400 10px var(--font-mono);
      width: 200px;
    }
    
    .interface-search::placeholder {
      color: var(--omega-light-1);
    }
    
    .interface-filter-select {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 8px;
      border-radius: 3px;
      font: 400 10px var(--font-mono);
    }
    
    .interface-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
    }
    
    .interface-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .interface-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .interfaces-table-container {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      overflow: auto;
      max-height: 400px;
    }
    
    .interfaces-table {
      width: 100%;
      border-collapse: collapse;
      font: 400 10px var(--font-mono);
    }
    
    .interfaces-table th {
      background: var(--omega-dark-4);
      color: var(--omega-cyan);
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--omega-gray-1);
      font-weight: 600;
      position: sticky;
      top: 0;
    }
    
    .interfaces-table td {
      padding: 8px 12px;
      border-bottom: 1px solid var(--omega-gray-2);
      color: var(--omega-white);
    }
    
    .interface-row:hover {
      background: var(--omega-dark-4);
    }
    
    .interface-name {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .interface-name i {
      color: var(--omega-cyan);
    }
    
    .interface-type {
      padding: 2px 6px;
      background: var(--omega-dark-4);
      border-radius: 2px;
      font-size: 8px;
      text-transform: uppercase;
    }
    
    .interface-status {
      padding: 2px 6px;
      border-radius: 2px;
      font-weight: 600;
      font-size: 8px;
    }
    
    .interface-status.up {
      background: var(--omega-green);
      color: var(--omega-black);
    }
    
    .interface-status.down {
      background: var(--omega-red);
      color: var(--omega-white);
    }
    
    .traffic-stats {
      font-size: 9px;
      line-height: 1.2;
    }
    
    .error-count {
      font-weight: 600;
    }
    
    .error-count.has-errors {
      color: var(--omega-red);
    }
    
    .interface-actions {
      display: flex;
      gap: 4px;
    }
    
    .action-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 3px 6px;
      border-radius: 2px;
      cursor: pointer;
      font-size: 8px;
      transition: all 0.15s ease;
    }
    
    .action-btn:hover {
      border-color: var(--omega-cyan);
    }
  `;
  document.head.appendChild(style);
}
