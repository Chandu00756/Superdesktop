export function renderDashboard(root, state) {
  try {
    console.log('[Dashboard] Rendering dashboard with state:', state?.data?.dashboard);
    
    // Ensure state structure exists
    if (!state || !state.data || !state.data.dashboard) {
      console.log('[Dashboard] State structure missing:', { state, data: state?.data, dashboard: state?.data?.dashboard });
      root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
      return;
    }
    
    // Advanced dashboard with full design specification compliance
    root.innerHTML = `
      <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 16px; height: 100%; padding: 8px;">
        <!-- Top Status Bar -->
        <div class="dashboard-status-bar">
          <div class="status-group">
            <div class="status-item">
              <i class="fas fa-server"></i>
              <span>Cluster Status</span>
              <span id="cluster-status" class="status-value">OPERATIONAL</span>
            </div>
            <div class="status-item">
              <i class="fas fa-users"></i>
              <span>Active Users</span>
              <span id="active-users" class="status-value">12</span>
          </div>
          <div class="status-item">
            <i class="fas fa-desktop"></i>
            <span>Sessions</span>
            <span id="total-sessions" class="status-value">8</span>
          </div>
          <div class="status-item">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Alerts</span>
            <span id="alert-count" class="status-value warning">3</span>
          </div>
        </div>
        <div class="status-actions">
          <button onclick="refreshDashboard()" class="status-btn">
            <i class="fas fa-sync"></i>
          </button>
          <button onclick="openClusterSettings()" class="status-btn">
            <i class="fas fa-cog"></i>
          </button>
        </div>
      </div>

      <!-- Overview Cards Grid -->
      <div class="overview-cards-grid">
        <!-- Cluster Overview -->
        <div class="overview-card primary">
          <div class="card-header">
            <div class="card-title">
              <i class="fas fa-cube"></i>
              <span>Cluster Overview</span>
            </div>
            <div class="card-actions">
              <button onclick="viewTopology()" class="card-btn">
                <i class="fas fa-sitemap"></i>
              </button>
            </div>
          </div>
          <div class="card-content" id="cluster-overview">
            <!-- Populated by renderClusterOverview() -->
          </div>
        </div>

        <!-- Performance Summary -->
        <div class="overview-card">
          <div class="card-header">
            <div class="card-title">
              <i class="fas fa-tachometer-alt"></i>
              <span>Performance</span>
            </div>
            <div class="card-actions">
              <button onclick="openPerformanceTab()" class="card-btn">
                <i class="fas fa-external-link-alt"></i>
              </button>
            </div>
          </div>
          <div class="card-content" id="performance-summary">
            <!-- Populated by renderPerformanceSummary() -->
          </div>
        </div>

        <!-- Resource Utilization -->
        <div class="overview-card">
          <div class="card-header">
            <div class="card-title">
              <i class="fas fa-chart-pie"></i>
              <span>Resources</span>
            </div>
            <div class="card-actions">
              <button onclick="openResourcesTab()" class="card-btn">
                <i class="fas fa-external-link-alt"></i>
              </button>
            </div>
          </div>
          <div class="card-content">
            <canvas id="resource-donut-chart" style="width: 100%; height: 150px;"></canvas>
          </div>
        </div>

        <!-- Network Status -->
        <div class="overview-card">
          <div class="card-header">
            <div class="card-title">
              <i class="fas fa-network-wired"></i>
              <span>Network</span>
            </div>
            <div class="card-actions">
              <button onclick="openNetworkTab()" class="card-btn">
                <i class="fas fa-external-link-alt"></i>
              </button>
            </div>
          </div>
          <div class="card-content" id="network-summary">
            <!-- Populated by renderNetworkSummary() -->
          </div>
        </div>
      </div>

      <!-- Main Dashboard Content -->
      <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
        <!-- Left Panel: Charts and Analytics -->
        <div style="display: grid; grid-template-rows: 1fr 1fr; gap: 16px;">
          <!-- Performance Trends Chart -->
          <div class="dashboard-panel">
            <div class="panel-header">
              <div class="panel-title">
                <i class="fas fa-chart-line"></i>
                <span>Performance Trends</span>
              </div>
              <div class="panel-controls">
                <select id="chart-timeframe" onchange="updateChart()">
                  <option value="1h">1 Hour</option>
                  <option value="6h">6 Hours</option>
                  <option value="24h" selected>24 Hours</option>
                  <option value="7d">7 Days</option>
                </select>
                <button onclick="exportChart()" class="panel-btn">
                  <i class="fas fa-download"></i>
                </button>
              </div>
            </div>
            <div class="panel-content">
              <canvas id="performance-trends-chart" style="width: 100%; height: 200px;"></canvas>
            </div>
          </div>

          <!-- System Health Matrix -->
          <div class="dashboard-panel">
            <div class="panel-header">
              <div class="panel-title">
                <i class="fas fa-heartbeat"></i>
                <span>System Health Matrix</span>
              </div>
              <div class="panel-controls">
                <button onclick="runHealthCheck()" class="panel-btn primary">
                  <i class="fas fa-stethoscope"></i>
                  Run Check
                </button>
              </div>
            </div>
            <div class="panel-content">
              <div id="health-matrix" class="health-matrix">
                <!-- Populated by renderHealthMatrix() -->
              </div>
            </div>
          </div>
        </div>

        <!-- Right Panel: Activity and Alerts -->
        <div style="display: grid; grid-template-rows: 1fr 1fr; gap: 16px;">
          <!-- Activity Feed -->
          <div class="dashboard-panel">
            <div class="panel-header">
              <div class="panel-title">
                <i class="fas fa-stream"></i>
                <span>Live Activity</span>
              </div>
              <div class="panel-controls">
                <div class="activity-filter">
                  <button class="filter-btn active" data-filter="all">All</button>
                  <button class="filter-btn" data-filter="errors">Errors</button>
                  <button class="filter-btn" data-filter="warnings">Warnings</button>
                </div>
              </div>
            </div>
            <div class="panel-content">
              <div id="activity-feed" class="activity-feed">
                <!-- Populated by renderActivityFeed() -->
              </div>
            </div>
          </div>

          <!-- Quick Actions & Alerts -->
          <div class="dashboard-panel">
            <div class="panel-header">
              <div class="panel-title">
                <i class="fas fa-bolt"></i>
                <span>Quick Actions</span>
              </div>
            </div>
            <div class="panel-content">
              <div class="quick-actions-grid">
                <button onclick="discoverNodes()" class="quick-action-btn">
                  <i class="fas fa-search"></i>
                  <span>Discover Nodes</span>
                </button>
                <button onclick="runBenchmark()" class="quick-action-btn">
                  <i class="fas fa-tachometer-alt"></i>
                  <span>Benchmark</span>
                </button>
                <button onclick="createSession()" class="quick-action-btn">
                  <i class="fas fa-plus"></i>
                  <span>New Session</span>
                </button>
                <button onclick="generateReport()" class="quick-action-btn">
                  <i class="fas fa-file-alt"></i>
                  <span>Report</span>
                </button>
              </div>
              
              <div class="alerts-section" style="margin-top: 16px;">
                <h5 style="color: var(--omega-cyan); font-size: 11px; margin-bottom: 8px;">
                  <i class="fas fa-exclamation-triangle"></i> ACTIVE ALERTS
                </h5>
                <div id="alerts-list" class="alerts-list">
                  <!-- Populated by renderAlerts() -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize dashboard components
  renderClusterOverview(state);
  renderPerformanceSummary(state);
  renderNetworkSummary(state);
  renderHealthMatrix(state);
  renderActivityFeed(state);
  renderAlerts(state);
  initializeCharts(state);
  setupEventHandlers();
  
  console.log('[Dashboard] Dashboard rendered successfully');
  } catch (error) {
    console.error('[Dashboard] Error rendering dashboard:', error);
    root.innerHTML = `<div style="padding: 20px; color: #ff4444;">Error loading dashboard: ${error.message}</div>`;
  }
}

function renderClusterOverview(state) {
  const container = document.getElementById('cluster-overview');
  const cluster = state.data.dashboard?.cluster || {};
  
  container.innerHTML = `
    <div class="cluster-stats">
      <div class="cluster-stat">
        <div class="stat-label">Cluster Name</div>
        <div class="stat-value">${cluster.name || 'Local-Cluster'}</div>
      </div>
      <div class="cluster-stat">
        <div class="stat-label">Status</div>
        <div class="stat-value ${(cluster.status || 'DEGRADED').toLowerCase()}">${cluster.status || 'DEGRADED'}</div>
      </div>
      <div class="cluster-stat">
        <div class="stat-label">Uptime</div>
        <div class="stat-value">${cluster.uptime || '0h 0m'}</div>
      </div>
      <div class="cluster-stat">
        <div class="stat-label">Nodes</div>
        <div class="stat-value">${cluster.active_nodes || 0} / ${(cluster.active_nodes || 0) + (cluster.standby_nodes || 0)}</div>
      </div>
      <div class="cluster-stat">
        <div class="stat-label">Sessions</div>
        <div class="stat-value">${cluster.total_sessions || 0}</div>
      </div>
      <div class="cluster-stat">
        <div class="stat-label">Load</div>
        <div class="stat-value">${(cluster.cpu_usage || 0).toFixed(1)}%</div>
      </div>
    </div>
  `;
}

function renderPerformanceSummary(state) {
  const container = document.getElementById('performance-summary');
  const perf = state.data.dashboard?.performance || {};
  
  container.innerHTML = `
    <div class="perf-metrics">
      <div class="perf-metric">
        <div class="metric-header">
          <span class="metric-label">CPU</span>
          <span class="metric-value">${(perf.cpu_utilization || 0).toFixed(1)}%</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill" style="width: ${perf.cpu_utilization || 0}%; background: var(--omega-cyan);"></div>
        </div>
      </div>
      <div class="perf-metric">
        <div class="metric-header">
          <span class="metric-label">Memory</span>
          <span class="metric-value">${(perf.memory_utilization || 0).toFixed(1)}%</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill" style="width: ${perf.memory_utilization || 0}%; background: var(--omega-green);"></div>
        </div>
      </div>
      <div class="perf-metric">
        <div class="metric-header">
          <span class="metric-label">GPU</span>
          <span class="metric-value">${(perf.gpu_utilization || 0).toFixed(1)}%</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill" style="width: ${perf.gpu_utilization || 0}%; background: var(--omega-yellow);"></div>
        </div>
      </div>
      <div class="perf-metric">
        <div class="metric-header">
          <span class="metric-label">Network</span>
          <span class="metric-value">${formatBytes((perf.network_rx || 0) + (perf.network_tx || 0))}/s</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill" style="width: 45%; background: var(--omega-blue);"></div>
        </div>
      </div>
    </div>
  `;
}

function renderNetworkSummary(state) {
  const container = document.getElementById('network-summary');
  const network = state.data.network || {};
  const interfaces = network.statistics?.interfaces || [];
  
  const totalRx = interfaces.reduce((sum, iface) => sum + (iface.bytes_recv || 0), 0);
  const totalTx = interfaces.reduce((sum, iface) => sum + (iface.bytes_sent || 0), 0);
  
  container.innerHTML = `
    <div class="network-stats">
      <div class="network-stat">
        <div class="stat-icon"><i class="fas fa-download"></i></div>
        <div class="stat-info">
          <div class="stat-label">Download</div>
          <div class="stat-value">${formatBytes(totalRx)}</div>
        </div>
      </div>
      <div class="network-stat">
        <div class="stat-icon"><i class="fas fa-upload"></i></div>
        <div class="stat-info">
          <div class="stat-label">Upload</div>
          <div class="stat-value">${formatBytes(totalTx)}</div>
        </div>
      </div>
      <div class="network-stat">
        <div class="stat-icon"><i class="fas fa-ethernet"></i></div>
        <div class="stat-info">
          <div class="stat-label">Interfaces</div>
          <div class="stat-value">${interfaces.length}</div>
        </div>
      </div>
      <div class="network-stat">
        <div class="stat-icon"><i class="fas fa-clock"></i></div>
        <div class="stat-info">
          <div class="stat-label">Latency</div>
          <div class="stat-value">2.1ms</div>
        </div>
      </div>
    </div>
  `;
}

function renderHealthMatrix(state) {
  const container = document.getElementById('health-matrix');
  const nodes = state.data.nodes?.nodes || [];
  
  container.innerHTML = `
    <div class="health-grid">
      ${nodes.slice(0, 12).map(node => `
        <div class="health-node" onclick="selectNode('${node.node_id}')">
          <div class="node-header">
            <span class="node-name">${node.node_id}</span>
            <div class="node-status ${node.status || 'unknown'}"></div>
          </div>
          <div class="node-metrics">
            <div class="node-metric">
              <span>CPU</span>
              <span>${(node.metrics?.cpu_usage || 0).toFixed(1)}%</span>
            </div>
            <div class="node-metric">
              <span>MEM</span>
              <span>${(node.metrics?.memory_usage || 0).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderActivityFeed(state) {
  const container = document.getElementById('activity-feed');
  const alerts = state.data.dashboard?.alerts || [];
  
  // Generate recent activity entries
  const activities = [
    { time: '14:32', type: 'info', message: 'Session omega-session-001 started successfully', icon: 'fas fa-play' },
    { time: '14:28', type: 'warning', message: 'High CPU usage detected on node-03', icon: 'fas fa-exclamation-triangle' },
    { time: '14:25', type: 'success', message: 'Node node-08 joined cluster', icon: 'fas fa-server' },
    { time: '14:20', type: 'info', message: 'Health check completed - All systems nominal', icon: 'fas fa-check-circle' },
    { time: '14:15', type: 'error', message: 'Connection timeout to node-05', icon: 'fas fa-times-circle' }
  ];
  
  container.innerHTML = `
    <div class="activity-entries">
      ${activities.map(activity => `
        <div class="activity-entry ${activity.type}">
          <div class="activity-time">${activity.time}</div>
          <div class="activity-icon">
            <i class="${activity.icon}"></i>
          </div>
          <div class="activity-message">${activity.message}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderAlerts(state) {
  const container = document.getElementById('alerts-list');
  const alerts = state.data.dashboard?.alerts || [];
  
  if (alerts.length === 0) {
    container.innerHTML = `
      <div class="no-alerts">
        <i class="fas fa-shield-alt"></i>
        <span>No active alerts</span>
      </div>
    `;
    return;
  }
  
  container.innerHTML = alerts.slice(0, 3).map(alert => `
    <div class="alert-item ${alert.type || 'info'}">
      <div class="alert-header">
        <span class="alert-type">${(alert.type || 'info').toUpperCase()}</span>
        <button onclick="dismissAlert('${alert.id}')" class="alert-dismiss">
          <i class="fas fa-times"></i>
        </button>
      </div>
      <div class="alert-message">${alert.message}</div>
      <div class="alert-actions">
        <button onclick="viewAlert('${alert.id}')" class="alert-btn">View</button>
        <button onclick="resolveAlert('${alert.id}')" class="alert-btn primary">Resolve</button>
      </div>
    </div>
  `).join('');
}

function initializeCharts(state) {
  // Initialize performance trends chart
  const perfChart = document.getElementById('performance-trends-chart');
  if (perfChart) {
    drawPerformanceTrends(perfChart, state);
  }
  
  // Initialize resource donut chart
  const resourceChart = document.getElementById('resource-donut-chart');
  if (resourceChart) {
    drawResourceDonut(resourceChart, state);
  }
}

function drawPerformanceTrends(canvas, state) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  
  // Clear canvas
  ctx.fillStyle = 'var(--omega-dark-3)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.lineWidth = 1;
  
  for (let i = 0; i <= 5; i++) {
    const y = (canvas.height / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  
  // Draw performance lines (simplified)
  const perf = state.data.dashboard?.performance || {};
  drawLine(ctx, canvas, [20, 25, 30, 45, 40, 35, 50], '#00f5ff', 'CPU');
  drawLine(ctx, canvas, [60, 65, 70, 67, 72, 68, 75], '#00ff7f', 'Memory');
  drawLine(ctx, canvas, [10, 15, 12, 20, 18, 25, 22], '#ffaa00', 'GPU');
}

function drawLine(ctx, canvas, data, color, label) {
  ctx.strokeStyle = color;
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

function drawResourceDonut(canvas, state) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = Math.min(centerX, centerY) - 10;
  const innerRadius = radius * 0.6;
  
  const data = [
    { label: 'CPU', value: 45, color: '#00f5ff' },
    { label: 'Memory', value: 67, color: '#00ff7f' },
    { label: 'GPU', value: 23, color: '#ffaa00' },
    { label: 'Storage', value: 78, color: '#ff6b6b' }
  ];
  
  let currentAngle = -Math.PI / 2;
  
  data.forEach(segment => {
    const sliceAngle = (segment.value / 100) * 2 * Math.PI * 0.25; // Quarter for each metric
    
    ctx.fillStyle = segment.color;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
    ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
    ctx.fill();
    
    currentAngle += sliceAngle;
  });
}

function setupEventHandlers() {
  // Activity filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterActivity(btn.dataset.filter);
    };
  });
}

function filterActivity(filter) {
  const entries = document.querySelectorAll('.activity-entry');
  entries.forEach(entry => {
    if (filter === 'all' || entry.classList.contains(filter)) {
      entry.style.display = 'flex';
    } else {
      entry.style.display = 'none';
    }
  });
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
window.refreshDashboard = async () => {
  try {
    window.notify('info', 'Dashboard', 'Refreshing dashboard data...');
    const dashboardData = await window.api.getDashboard();
    window.state.setState('data.dashboard', dashboardData);
    
    // Re-render dashboard components with new data
    renderClusterOverview(window.state);
    renderPerformanceMetrics(window.state);
    renderSystemAlerts(window.state);
    updateNetworkTopology(window.state);
    
    window.notify('success', 'Dashboard', 'Data refreshed successfully');
  } catch (e) {
    window.notify('error', 'Dashboard', e.message);
  }
};

window.discoverNodes = async () => {
  try {
    await window.omegaAPI.secureAction('discover_nodes', {});
    window.notify('success', 'Node Discovery', 'Node discovery completed');
  } catch (e) {
    window.notify('error', 'Node Discovery', e.message);
  }
};

window.runBenchmark = async () => {
  try {
    await window.omegaAPI.secureAction('run_benchmark', {});
    window.notify('success', 'Benchmark', 'Benchmark completed');
  } catch (e) {
    window.notify('error', 'Benchmark', e.message);
  }
};

export function applyDashboardDelta(state, delta) {
  const cluster = state.data.dashboard.cluster || (state.data.dashboard.cluster = {});
  if (delta.system_stats) {
    if (delta.system_stats.cpu_usage !== undefined) cluster.cpu_usage = delta.system_stats.cpu_usage;
    if (delta.system_stats.memory_usage !== undefined) cluster.memory_usage = delta.system_stats.memory_usage;
  }
  
  // Re-render affected components
  if (document.getElementById('cluster-overview')) {
    renderClusterOverview(state);
    renderPerformanceSummary(state);
  }
}

// Add dashboard-specific CSS
if (!document.getElementById('dashboard-styles')) {
  const style = document.createElement('style');
  style.id = 'dashboard-styles';
  style.textContent = `
    .dashboard-status-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .status-group {
      display: flex;
      gap: 24px;
    }
    
    .status-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .status-item i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .status-value {
      font-weight: 600;
      color: var(--omega-white);
    }
    
    .status-value.warning {
      color: var(--omega-yellow);
    }
    
    .status-value.error {
      color: var(--omega-red);
    }
    
    .status-actions {
      display: flex;
      gap: 8px;
    }
    
    .status-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 8px;
      border-radius: 3px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    
    .status-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .overview-cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 16px;
    }
    
    .overview-card {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      overflow: hidden;
      transition: all 0.15s ease;
    }
    
    .overview-card:hover {
      border-color: var(--omega-cyan);
      transform: translateY(-2px);
    }
    
    .overview-card.primary {
      border-color: var(--omega-cyan);
    }
    
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: var(--omega-dark-4);
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 600 12px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .card-actions {
      display: flex;
      gap: 4px;
    }
    
    .card-btn {
      background: transparent;
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 6px;
      border-radius: 2px;
      cursor: pointer;
      font-size: 10px;
      transition: all 0.15s ease;
    }
    
    .card-btn:hover {
      border-color: var(--omega-cyan);
      background: var(--omega-dark-3);
    }
    
    .card-content {
      padding: 16px;
    }
    
    .cluster-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    
    .cluster-stat {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    
    .stat-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .stat-value {
      font: 600 14px var(--font-mono);
      color: var(--omega-white);
    }
    
    .stat-value.operational {
      color: var(--omega-green);
    }
    
    .stat-value.degraded {
      color: var(--omega-yellow);
    }
    
    .stat-value.offline {
      color: var(--omega-red);
    }
    
    .perf-metrics {
      display: grid;
      gap: 12px;
    }
    
    .perf-metric {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .metric-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
    }
    
    .metric-value {
      font: 600 12px var(--font-mono);
      color: var(--omega-white);
    }
    
    .metric-bar {
      height: 4px;
      background: var(--omega-dark-2);
      border-radius: 2px;
      overflow: hidden;
    }
    
    .metric-fill {
      height: 100%;
      transition: width 0.3s ease;
    }
    
    .network-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    
    .network-stat {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .stat-icon {
      width: 24px;
      height: 24px;
      background: var(--omega-dark-2);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--omega-cyan);
      font-size: 10px;
    }
    
    .stat-info {
      flex: 1;
    }
    
    .dashboard-panel {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: var(--omega-dark-4);
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .panel-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 600 12px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .panel-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .panel-btn {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 8px;
      border-radius: 2px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
    }
    
    .panel-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .panel-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .panel-content {
      flex: 1;
      padding: 16px;
      overflow: auto;
    }
    
    .health-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 8px;
    }
    
    .health-node {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-2);
      border-radius: 3px;
      padding: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    
    .health-node:hover {
      border-color: var(--omega-cyan);
      transform: scale(1.02);
    }
    
    .node-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    
    .node-name {
      font: 600 10px var(--font-mono);
      color: var(--omega-white);
    }
    
    .node-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
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
    
    .node-metrics {
      display: grid;
      gap: 2px;
    }
    
    .node-metric {
      display: flex;
      justify-content: space-between;
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .activity-feed {
      max-height: 200px;
      overflow-y: auto;
    }
    
    .activity-entries {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    .activity-entry {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 8px;
      background: var(--omega-dark-4);
      border-radius: 3px;
      border-left: 3px solid var(--omega-gray-1);
    }
    
    .activity-entry.info {
      border-left-color: var(--omega-cyan);
    }
    
    .activity-entry.warning {
      border-left-color: var(--omega-yellow);
    }
    
    .activity-entry.error {
      border-left-color: var(--omega-red);
    }
    
    .activity-entry.success {
      border-left-color: var(--omega-green);
    }
    
    .activity-time {
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
      width: 40px;
    }
    
    .activity-icon {
      width: 16px;
      color: var(--omega-cyan);
      font-size: 10px;
    }
    
    .activity-message {
      flex: 1;
      font: 400 10px var(--font-mono);
      color: var(--omega-white);
    }
    
    .quick-actions-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 16px;
    }
    
    .quick-action-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 12px;
      border-radius: 3px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }
    
    .quick-action-btn:hover {
      border-color: var(--omega-cyan);
      background: var(--omega-dark-3);
      transform: translateY(-1px);
    }
    
    .quick-action-btn i {
      font-size: 16px;
      color: var(--omega-cyan);
    }
    
    .quick-action-btn span {
      font: 400 9px var(--font-mono);
    }
    
    .alerts-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    .alert-item {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      border-radius: 3px;
      padding: 8px;
    }
    
    .alert-item.warning {
      border-left: 3px solid var(--omega-yellow);
    }
    
    .alert-item.error {
      border-left: 3px solid var(--omega-red);
    }
    
    .alert-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }
    
    .alert-type {
      font: 600 8px var(--font-mono);
      color: var(--omega-yellow);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .alert-dismiss {
      background: transparent;
      border: none;
      color: var(--omega-light-1);
      cursor: pointer;
      font-size: 8px;
      padding: 2px;
    }
    
    .alert-message {
      font: 400 10px var(--font-mono);
      color: var(--omega-white);
      margin-bottom: 6px;
      line-height: 1.3;
    }
    
    .alert-actions {
      display: flex;
      gap: 4px;
    }
    
    .alert-btn {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 3px 6px;
      border-radius: 2px;
      cursor: pointer;
      font: 400 8px var(--font-mono);
      transition: all 0.15s ease;
    }
    
    .alert-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .alert-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .no-alerts {
      text-align: center;
      color: var(--omega-light-1);
      font: 400 10px var(--font-mono);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 16px;
    }
    
    .no-alerts i {
      font-size: 24px;
      opacity: 0.3;
    }
    
    .activity-filter {
      display: flex;
      gap: 4px;
    }
    
    .filter-btn {
      background: transparent;
      border: 1px solid var(--omega-gray-2);
      color: var(--omega-light-1);
      padding: 3px 6px;
      border-radius: 2px;
      cursor: pointer;
      font: 400 8px var(--font-mono);
      transition: all 0.15s ease;
    }
    
    .filter-btn.active,
    .filter-btn:hover {
      border-color: var(--omega-cyan);
      color: var(--omega-cyan);
    }
  `;
  document.head.appendChild(style);
}
