// Omega Super Desktop Console - Advanced Renderer
// Initial prototype frontend with real-time monitoring and AI optimization

const { ipcRenderer } = require('electron');

// Application State
let currentUser = null;
let authToken = null;
let wsConnection = null;
let charts = {};
let updateIntervals = {};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

async function initializeApp() {
  try {
    // Show loading screen
    showLoadingScreen();
    
    // Check authentication
    const savedToken = localStorage.getItem('omega_auth_token');
    if (savedToken) {
      authToken = savedToken;
      await loadApplication();
    } else {
      showLoginModal();
    }
  } catch (error) {
    console.error('App initialization error:', error);
    showError('Failed to initialize application');
  }
}

function showLoadingScreen() {
  const loadingScreen = document.getElementById('loading-screen');
  const app = document.getElementById('app');
  
  loadingScreen.classList.remove('hidden');
  app.classList.add('hidden');
}

function hideLoadingScreen() {
  const loadingScreen = document.getElementById('loading-screen');
  const app = document.getElementById('app');
  
  loadingScreen.classList.add('hidden');
  app.classList.remove('hidden');
}

function showLoginModal() {
  hideLoadingScreen();
  const modal = document.getElementById('login-modal');
  modal.classList.remove('hidden');
  
  // Setup login form
  const form = document.getElementById('login-form');
  form.addEventListener('submit', handleLogin);
}

function hideLoginModal() {
  const modal = document.getElementById('login-modal');
  modal.classList.add('hidden');
}

async function handleLogin(event) {
  event.preventDefault();
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  
  try {
    const result = await ipcRenderer.invoke('auth-login', { username, password });
    
    if (result.success) {
      authToken = result.token;
      localStorage.setItem('omega_auth_token', authToken);
      hideLoginModal();
      await loadApplication();
    } else {
      showError(result.error || 'Login failed');
    }
  } catch (error) {
    console.error('Login error:', error);
    showError('Login failed');
  }
}

async function loadApplication() {
  try {
    showLoadingScreen();
    
    // Initialize UI components
    initializeNavigation();
    initializeCharts();
    initializeModals();
    initializeEventListeners();
    
    // Load initial data
    await loadInitialData();
    
    // Start real-time updates
    startRealtimeUpdates();
    
    // Show main app
    hideLoadingScreen();
    
    // Show dashboard by default
    showTab('dashboard');
    
  } catch (error) {
    console.error('App loading error:', error);
    showError('Failed to load application');
  }
}

function initializeNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab');
      showTab(tabId);
      
      // Update active nav item
      navItems.forEach(nav => nav.classList.remove('active'));
      item.classList.add('active');
    });
  });
}

function showTab(tabId) {
  // Hide all tabs
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => tab.classList.remove('active'));
  
  // Show selected tab
  const selectedTab = document.getElementById(tabId);
  if (selectedTab) {
    selectedTab.classList.add('active');
    
    // Load tab-specific data
    loadTabData(tabId);
  }
}

async function loadTabData(tabId) {
  switch (tabId) {
    case 'dashboard':
      await updateDashboard();
      break;
    case 'nodes':
      await updateNodes();
      break;
    case 'sessions':
      await updateSessions();
      break;
    case 'tasks':
      await updateTasks();
      break;
    case 'monitoring':
      await updateMonitoring();
      break;
    case 'storage':
      await updateStorage();
      break;
    case 'ai':
      await updateAI();
      break;
  }
}

function initializeCharts() {
  // Performance chart
  const performanceCtx = document.getElementById('performance-chart');
  if (performanceCtx) {
    charts.performance = new Chart(performanceCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'CPU Usage %',
            data: [],
            borderColor: '#0066ff',
            backgroundColor: 'rgba(0, 102, 255, 0.1)',
            tension: 0.4
          },
          {
            label: 'Memory Usage %',
            data: [],
            borderColor: '#28a745',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            tension: 0.4
          },
          {
            label: 'Latency (ms)',
            data: [],
            borderColor: '#ffc107',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            tension: 0.4,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#cbd5e0' }
          }
        },
        scales: {
          x: {
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            max: 100,
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#718096' }
          }
        }
      }
    });
  }
  
  // Resource utilization chart
  const resourceCtx = document.getElementById('resource-chart');
  if (resourceCtx) {
    charts.resource = new Chart(resourceCtx, {
      type: 'doughnut',
      data: {
        labels: ['CPU', 'Memory', 'GPU', 'Storage', 'Network'],
        datasets: [{
          data: [0, 0, 0, 0, 0],
          backgroundColor: [
            '#0066ff',
            '#28a745',
            '#dc3545',
            '#ffc107',
            '#17a2b8'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#cbd5e0' }
          }
        }
      }
    });
  }
  
  // Storage tier chart
  const storageCtx = document.getElementById('storage-tier-chart');
  if (storageCtx) {
    charts.storage = new Chart(storageCtx, {
      type: 'bar',
      data: {
        labels: ['Hot', 'Warm', 'Cold'],
        datasets: [{
          label: 'Storage Usage (GB)',
          data: [0, 0, 0],
          backgroundColor: ['#dc3545', '#ffc107', '#17a2b8']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#cbd5e0' }
          }
        },
        scales: {
          x: {
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          },
          y: {
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          }
        }
      }
    });
  }
  
  // Prediction accuracy chart
  const predictionCtx = document.getElementById('prediction-chart');
  if (predictionCtx) {
    charts.prediction = new Chart(predictionCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Prediction Accuracy %',
          data: [],
          borderColor: '#0066ff',
          backgroundColor: 'rgba(0, 102, 255, 0.1)',
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          },
          x: {
            ticks: { color: '#718096' },
            grid: { color: '#2d3748' }
          }
        },
        plugins: {
          legend: {
            labels: { color: '#cbd5e0' }
          }
        }
      }
    });
  }
}

function initializeModals() {
  // Session creation modal
  const sessionModal = document.getElementById('session-modal');
  const createSessionBtn = document.getElementById('create-session-btn');
  const newSessionBtn = document.getElementById('new-session-btn');
  const cancelSessionBtn = document.getElementById('cancel-session');
  const sessionForm = document.getElementById('session-form');
  
  [createSessionBtn, newSessionBtn].forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        sessionModal.classList.remove('hidden');
      });
    }
  });
  
  if (cancelSessionBtn) {
    cancelSessionBtn.addEventListener('click', () => {
      sessionModal.classList.add('hidden');
    });
  }
  
  if (sessionForm) {
    sessionForm.addEventListener('submit', handleSessionCreation);
  }
  
  // Close modals on background click
  document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  });
  
  // Close buttons
  document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const modal = e.target.closest('.modal');
      if (modal) {
        modal.classList.add('hidden');
      }
    });
  });
}

function initializeEventListeners() {
  // Refresh buttons
  document.getElementById('refresh-dashboard')?.addEventListener('click', updateDashboard);
  document.getElementById('refresh-nodes')?.addEventListener('click', updateNodes);
  document.getElementById('refresh-sessions')?.addEventListener('click', updateSessions);
  document.getElementById('refresh-tasks')?.addEventListener('click', updateTasks);
  
  // Menu event listeners
  ipcRenderer.on('menu-new-session', () => {
    document.getElementById('session-modal').classList.remove('hidden');
  });
  
  ipcRenderer.on('menu-add-node', () => {
    showTab('nodes');
    // Could open an add node dialog here
  });
  
  ipcRenderer.on('menu-diagnostics', () => {
    showTab('monitoring');
  });
  
  ipcRenderer.on('menu-performance', () => {
    showTab('ai');
  });
}

async function loadInitialData() {
  try {
    await Promise.all([
      updateDashboard(),
      updateNodes(),
      updateSessions()
    ]);
  } catch (error) {
    console.error('Error loading initial data:', error);
  }
}

function startRealtimeUpdates() {
  // Update dashboard every 2 seconds
  updateIntervals.dashboard = setInterval(updateDashboard, 2000);
  
  // Update nodes every 5 seconds
  updateIntervals.nodes = setInterval(updateNodes, 5000);
  
  // Update sessions every 3 seconds
  updateIntervals.sessions = setInterval(updateSessions, 3000);
  
  // Try to establish WebSocket connection for real-time updates
  connectWebSocket();
}

function connectWebSocket() {
  try {
    wsConnection = new WebSocket('ws://localhost:8443/ws/realtime');
    
    wsConnection.onopen = () => {
      console.log('WebSocket connected');
      updateConnectionStatus(true);
    };
    
    wsConnection.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealtimeUpdate(data);
    };
    
    wsConnection.onclose = () => {
      console.log('WebSocket disconnected');
      updateConnectionStatus(false);
      
      // Reconnect after 5 seconds
      setTimeout(connectWebSocket, 5000);
    };
    
    wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
      updateConnectionStatus(false);
    };
  } catch (error) {
    console.error('WebSocket connection error:', error);
    updateConnectionStatus(false);
  }
}

function updateConnectionStatus(connected) {
  const statusIndicator = document.getElementById('connection-status');
  const statusDot = statusIndicator.querySelector('.status-dot');
  const statusText = statusIndicator.querySelector('.status-text');
  
  if (connected) {
    statusDot.style.background = '#28a745';
    statusText.textContent = 'Connected';
  } else {
    statusDot.style.background = '#dc3545';
    statusText.textContent = 'Disconnected';
  }
}

function handleRealtimeUpdate(data) {
  // Update quick stats
  if (data.nodes !== undefined) {
    document.getElementById('node-count').textContent = data.nodes;
  }
  if (data.sessions !== undefined) {
    document.getElementById('session-count').textContent = data.sessions;
  }
  
  // Update system metrics
  updateSystemMetrics({
    cpu_percent: data.cpu_percent,
    memory_percent: data.memory_percent
  });
  
  // Update performance chart
  updatePerformanceChart(data);
}

async function updateDashboard() {
  try {
    const response = await fetch('/api/dashboard/metrics');
    const metrics = await response.json();
    
    if (!metrics || metrics.error) {
      console.error('Error fetching metrics:', metrics?.error);
      return;
    }
    
    // Update system overview with real Mac data
    updateSystemMetrics(metrics);
    
    // Update performance chart with real data
    updatePerformanceChart(metrics);
    
    // Update real-time system information
    updateSystemInformation(metrics);
    
    // Update activity feed with real process data
    updateActivityFeed(metrics);
    
  } catch (error) {
    console.error('Error updating dashboard:', error);
  }
}

function updateSystemInformation(data) {
  // Update system details with real Mac information
  if (data.system) {
    const systemInfo = document.querySelector('.system-info');
    if (systemInfo) {
      systemInfo.innerHTML = `
        <h3><i class="fas fa-desktop"></i> System Information</h3>
        <div class="info-row">
          <span>Hostname:</span>
          <span>${data.system.hostname || 'Unknown'}</span>
        </div>
        <div class="info-row">
          <span>Platform:</span>
          <span>${data.system.platform?.system || 'Unknown'} ${data.system.platform?.release || ''}</span>
        </div>
        <div class="info-row">
          <span>Architecture:</span>
          <span>${data.system.platform?.machine || 'Unknown'}</span>
        </div>
        <div class="info-row">
          <span>Processor:</span>
          <span>${data.system.platform?.processor || 'Unknown'}</span>
        </div>
        <div class="info-row">
          <span>Uptime:</span>
          <span>${data.system.uptime_human || '0:00:00'}</span>
        </div>
        <div class="info-row">
          <span>Load Average:</span>
          <span>1m: ${data.system.load_avg?.['1min']?.toFixed(2) || '0.00'}, 
                5m: ${data.system.load_avg?.['5min']?.toFixed(2) || '0.00'}, 
                15m: ${data.system.load_avg?.['15min']?.toFixed(2) || '0.00'}</span>
        </div>
        <div class="info-row">
          <span>Processes:</span>
          <span>${data.system.process_count || 0}</span>
        </div>
      `;
    }
  }
  
  // Update CPU details
  if (data.cpu) {
    const cpuInfo = document.querySelector('.cpu-details');
    if (cpuInfo) {
      cpuInfo.innerHTML = `
        <h4>CPU Details</h4>
        <div class="info-row">
          <span>Usage:</span>
          <span>${data.cpu.usage_percent?.toFixed(1) || 0}%</span>
        </div>
        <div class="info-row">
          <span>Logical Cores:</span>
          <span>${data.cpu.logical_cores || 0}</span>
        </div>
        <div class="info-row">
          <span>Physical Cores:</span>
          <span>${data.cpu.physical_cores || 0}</span>
        </div>
        <div class="info-row">
          <span>Frequency:</span>
          <span>${data.cpu.frequency_mhz ? (data.cpu.frequency_mhz / 1000).toFixed(2) + ' GHz' : 'Unknown'}</span>
        </div>
      `;
    }
  }
  
  // Update Memory details
  if (data.memory) {
    const memInfo = document.querySelector('.memory-details');
    if (memInfo) {
      memInfo.innerHTML = `
        <h4>Memory Details</h4>
        <div class="info-row">
          <span>Usage:</span>
          <span>${data.memory.usage_percent?.toFixed(1) || 0}%</span>
        </div>
        <div class="info-row">
          <span>Total:</span>
          <span>${data.memory.total_gb || 0} GB</span>
        </div>
        <div class="info-row">
          <span>Available:</span>
          <span>${data.memory.available_gb || 0} GB</span>
        </div>
        <div class="info-row">
          <span>Used:</span>
          <span>${data.memory.used_gb || 0} GB</span>
        </div>
        ${data.memory.wired ? `<div class="info-row"><span>Wired:</span><span>${(data.memory.wired / (1024**3)).toFixed(2)} GB</span></div>` : ''}
        ${data.memory.active ? `<div class="info-row"><span>Active:</span><span>${(data.memory.active / (1024**3)).toFixed(2)} GB</span></div>` : ''}
        ${data.memory.inactive ? `<div class="info-row"><span>Inactive:</span><span>${(data.memory.inactive / (1024**3)).toFixed(2)} GB</span></div>` : ''}
      `;
    }
  }
}

function updateActivityFeed(data) {
  const activityFeed = document.querySelector('.activity-list');
  if (!activityFeed || !data.processes) return;
  
  // Clear existing activity
  activityFeed.innerHTML = '';
  
  // Add real process information
  if (data.processes.top_cpu && data.processes.top_cpu.length > 0) {
    const cpuHeader = document.createElement('div');
    cpuHeader.className = 'activity-header';
    cpuHeader.innerHTML = '<h4>Top CPU Processes</h4>';
    activityFeed.appendChild(cpuHeader);
    
    data.processes.top_cpu.forEach(process => {
      const activityItem = document.createElement('div');
      activityItem.className = 'activity-item';
      activityItem.innerHTML = `
        <div class="activity-icon cpu-activity"></div>
        <div class="activity-details">
          <div class="activity-title">${process.name} (PID: ${process.pid})</div>
          <div class="activity-time">CPU: ${process.cpu_percent}%</div>
        </div>
      `;
      activityFeed.appendChild(activityItem);
    });
  }
  
  if (data.processes.top_memory && data.processes.top_memory.length > 0) {
    const memHeader = document.createElement('div');
    memHeader.className = 'activity-header';
    memHeader.innerHTML = '<h4>Top Memory Processes</h4>';
    activityFeed.appendChild(memHeader);
    
    data.processes.top_memory.forEach(process => {
      const activityItem = document.createElement('div');
      activityItem.className = 'activity-item';
      activityItem.innerHTML = `
        <div class="activity-icon memory-activity"></div>
        <div class="activity-details">
          <div class="activity-title">${process.name} (PID: ${process.pid})</div>
          <div class="activity-time">Memory: ${process.memory_percent}%</div>
        </div>
      `;
      activityFeed.appendChild(activityItem);
    });
  }
}

function updateSystemMetrics(systemData) {
  if (!systemData) return;
  
  // Update main dashboard metrics with real data
  document.getElementById('cpu-usage').textContent = `${Math.round(systemData.cpu?.usage_percent || 0)}%`;
  document.getElementById('memory-usage').textContent = `${Math.round(systemData.memory?.usage_percent || 0)}%`;
  document.getElementById('node-count').textContent = systemData.cluster?.active_nodes || 0;
  document.getElementById('session-count').textContent = systemData.cluster?.total_sessions || 0;
  
  // Update cluster nodes display with real data
  if (systemData.cluster) {
    const clusterNodes = document.getElementById('cluster-nodes');
    if (clusterNodes) {
      const activeNodes = systemData.cluster.active_nodes || 1;
      const standbyNodes = systemData.cluster.standby_nodes || 0;
      clusterNodes.textContent = `${activeNodes} Active, ${standbyNodes} Standby`;
    }
  }
  
  // Update uptime with real Mac system uptime
  if (systemData.system?.uptime_human) {
    const uptimeElement = document.getElementById('uptime');
    if (uptimeElement) uptimeElement.textContent = systemData.system.uptime_human;
  }
  
  // Update performance bars with real Mac data
  if (systemData.cpu) {
    const cpuPercent = Math.round(systemData.cpu.usage_percent);
    const cpuProgressFill = document.getElementById('cpu-progress-fill');
    const cpuPerfValue = document.getElementById('cpu-perf-value');
    if (cpuProgressFill) cpuProgressFill.style.width = `${cpuPercent}%`;
    if (cpuPerfValue) cpuPerfValue.textContent = `${systemData.cpu.logical_cores} logical cores (${cpuPercent}%)`;
  }
  
  if (systemData.memory) {
    const memPercent = Math.round(systemData.memory.usage_percent);
    const ramProgressFill = document.getElementById('ram-progress-fill');
    const ramPerfValue = document.getElementById('ram-perf-value');
    if (ramProgressFill) ramProgressFill.style.width = `${memPercent}%`;
    if (ramPerfValue) ramPerfValue.textContent = `${systemData.memory.used_gb}GB/${systemData.memory.total_gb}GB (${memPercent}%)`;
  }
  
  if (systemData.disk) {
    const diskPercent = Math.round(systemData.disk.usage_percent);
    const storageProgressFill = document.getElementById('storage-progress-fill');
    const storagePerfValue = document.getElementById('storage-perf-value');
    if (storageProgressFill) storageProgressFill.style.width = `${diskPercent}%`;
    if (storagePerfValue) storagePerfValue.textContent = `${systemData.disk.used_gb}GB/${systemData.disk.total_gb}GB (${diskPercent}%)`;
  }
  
  // Update GPU info (placeholder since we don't have real GPU data)
  const gpuProgressFill = document.getElementById('gpu-progress-fill');
  const gpuPerfValue = document.getElementById('gpu-perf-value');
  if (gpuProgressFill) gpuProgressFill.style.width = '0%';
  if (gpuPerfValue) gpuPerfValue.textContent = 'N/A (No GPU detected)';
  
  // Update uptime
  if (systemData.system?.uptime_human) {
    const uptimeElement = document.getElementById('uptime');
    if (uptimeElement) uptimeElement.textContent = systemData.system.uptime_human;
  }
  
  // Update resource chart with real Mac system data
  if (charts.resource) {
    charts.resource.data.datasets[0].data = [
      systemData.cpu?.usage_percent || 0,
      systemData.memory?.usage_percent || 0,
      systemData.disk?.usage_percent || 0,  // Real disk usage
      (systemData.network?.bytes_sent + systemData.network?.bytes_recv) / 1000000 || 0,  // Network MB/s
      systemData.system?.load_avg?.['1min'] * 10 || 0  // Load average as %
    ];
    charts.resource.update('none');
  }
}

function updatePerformanceChart(data) {
  if (!charts.performance) return;
  
  const now = new Date().toLocaleTimeString();
  const maxPoints = 50;
  
  // Add new data point with real metrics
  charts.performance.data.labels.push(now);
  charts.performance.data.datasets[0].data.push(data.cpu?.usage_percent || 0);
  charts.performance.data.datasets[1].data.push(data.memory?.usage_percent || 0);
  charts.performance.data.datasets[2].data.push(data.system?.load_avg?.['1min'] || 0); // Real load average
  
  // Remove old data points
  if (charts.performance.data.labels.length > maxPoints) {
    charts.performance.data.labels.shift();
    charts.performance.data.datasets.forEach(dataset => dataset.data.shift());
  }
  
  charts.performance.update('none');
}

async function updateNodes() {
  try {
    const nodes = await ipcRenderer.invoke('get-nodes');
    renderNodes(nodes);
  } catch (error) {
    console.error('Error updating nodes:', error);
  }
}

function renderNodes(nodes) {
  const container = document.getElementById('nodes-grid');
  if (!container) return;
  
  container.innerHTML = nodes.map(node => `
    <div class="node-card">
      <div class="node-header">
        <div class="node-title">${node.node_id}</div>
        <div class="node-status ${node.status}">${node.status}</div>
      </div>
      <div class="node-type">${node.node_type}</div>
      <div class="node-metrics">
        <div class="node-metric">
          <span class="metric-label">CPU Cores</span>
          <span class="metric-value">${node.resources?.cpu_cores || 'N/A'}</span>
        </div>
        <div class="node-metric">
          <span class="metric-label">Memory</span>
          <span class="metric-value">${formatBytes(node.resources?.memory_total || 0)}</span>
        </div>
        <div class="node-metric">
          <span class="metric-label">GPU Count</span>
          <span class="metric-value">${node.resources?.gpu_count || 0}</span>
        </div>
        <div class="node-metric">
          <span class="metric-label">Network</span>
          <span class="metric-value">${node.resources?.network || 'N/A'}</span>
        </div>
      </div>
    </div>
  `).join('');
}

async function updateSessions() {
  try {
    const sessions = await ipcRenderer.invoke('get-sessions');
    renderSessions(sessions);
  } catch (error) {
    console.error('Error updating sessions:', error);
  }
}

function renderSessions(sessions) {
  const container = document.getElementById('sessions-list');
  const dashboardContainer = document.getElementById('dashboard-sessions');
  
  const sessionHTML = sessions.map(session => `
    <div class="session-item">
      <div class="item-header">
        <div class="item-title">${session.session_id}</div>
        <div class="item-actions">
          <button class="btn btn-secondary btn-sm" onclick="viewSessionDetails('${session.session_id}')">Details</button>
          <button class="btn btn-danger btn-sm" onclick="terminateSession('${session.session_id}')">Terminate</button>
        </div>
      </div>
      <div class="item-details">
        <div class="detail-item">
          <div class="detail-label">User</div>
          <div class="detail-value">${session.user_id}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Application</div>
          <div class="detail-value">${session.app_uri}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Node</div>
          <div class="detail-value">${session.node_id}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Status</div>
          <div class="detail-value">${session.status}</div>
        </div>
      </div>
    </div>
  `).join('');
  
  if (container) container.innerHTML = sessionHTML;
  if (dashboardContainer) {
    dashboardContainer.innerHTML = sessions.slice(0, 5).map(session => `
      <div class="session-summary">
        <strong>${session.session_id}</strong> - ${session.status}
      </div>
    `).join('');
  }
}

async function updateTasks() {
  // Implementation for task updates
  // This would fetch and display running tasks
}

async function updateMonitoring() {
  // Implementation for monitoring updates
  // This would update latency heatmaps and health matrices
}

async function updateStorage() {
  // Implementation for storage updates
  // This would update storage tier charts and statistics
}

async function updateAI() {
  // Implementation for AI optimization updates
  // This would update prediction accuracy and model performance
}

function updateActivityFeed() {
  const container = document.getElementById('activity-feed');
  if (!container) return;
  
  // Simulate activity feed (in initial prototype, this would come from backend)
  const activities = [
    { type: 'success', icon: '[COMPLETE]', text: 'Session session_123 started successfully', time: '2 minutes ago' },
    { type: 'warning', icon: '[WARNING]', text: 'High CPU usage detected on node_cpu_1', time: '5 minutes ago' },
    { type: 'success', icon: '[LAUNCH]', text: 'GPU task completed on node_gpu_2', time: '8 minutes ago' },
    { type: 'error', icon: '[ERROR]', text: 'Network timeout on node_storage_3', time: '12 minutes ago' }
  ];
  
  container.innerHTML = activities.map(activity => `
    <div class="activity-item">
      <div class="activity-icon ${activity.type}">${activity.icon}</div>
      <div class="activity-content">
        <div class="activity-text">${activity.text}</div>
        <div class="activity-time">${activity.time}</div>
      </div>
    </div>
  `).join('');
}

async function handleSessionCreation(event) {
  event.preventDefault();
  
  const formData = new FormData(event.target);
  const sessionRequest = {
    user_id: currentUser || 'admin',
    app_uri: formData.get('app') || document.getElementById('session-app').value,
    cpu_cores: parseInt(document.getElementById('session-cpu').value),
    gpu_units: parseInt(document.getElementById('session-gpu').value),
    ram_bytes: parseInt(document.getElementById('session-ram').value) * 1024 * 1024 * 1024,
    low_latency: document.getElementById('session-low-latency').checked
  };
  
  try {
    const result = await ipcRenderer.invoke('create-session', sessionRequest);
    if (result) {
      showSuccess('Session created successfully');
      document.getElementById('session-modal').classList.add('hidden');
      event.target.reset();
      await updateSessions();
    } else {
      showError('Failed to create session');
    }
  } catch (error) {
    console.error('Session creation error:', error);
    showError('Failed to create session');
  }
}

async function terminateSession(sessionId) {
  if (!confirm('Are you sure you want to terminate this session?')) return;
  
  try {
    const success = await ipcRenderer.invoke('terminate-session', sessionId);
    if (success) {
      showSuccess('Session terminated successfully');
      await updateSessions();
    } else {
      showError('Failed to terminate session');
    }
  } catch (error) {
    console.error('Session termination error:', error);
    showError('Failed to terminate session');
  }
}

function viewSessionDetails(sessionId) {
  // Implementation for viewing session details
  showInfo(`Session details for ${sessionId} would be shown here`);
}

// Utility functions
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
  // Simple error notification (could be replaced with a toast library)
  alert(`Error: ${message}`);
}

function showSuccess(message) {
  // Simple success notification (could be replaced with a toast library)
  console.log(`Success: ${message}`);
}

function showInfo(message) {
  // Simple info notification (could be replaced with a toast library)
  alert(`Info: ${message}`);
}

// Make functions globally available
window.terminateSession = terminateSession;
window.viewSessionDetails = viewSessionDetails;
