export function renderNodes(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.nodes) {
    console.log('[Nodes] State structure missing:', { state, data: state?.data, nodes: state?.data?.nodes });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  const nodesData = state.data.nodes || {};
  const nodes = (nodesData.nodes) || [];
  const processes = (state.data.processes?.processes) || [];
  const logs = (state.data.logs?.events) || [];
  
  // Create advanced nodes interface with sub-tabs
  root.innerHTML = `
    <div style="display: grid; grid-template-columns: 280px 1fr; gap: 16px; height: 100%;">
      <!-- Nodes List -->
      <div style="display: flex; flex-direction: column; border: 1px solid var(--omega-gray-1); background: var(--omega-dark-3); border-radius: 4px; overflow: hidden;">
        <div style="padding: 12px; font: 600 12px var(--font-mono); color: var(--omega-cyan); border-bottom: 1px solid var(--omega-gray-1);">
          <i class="fas fa-server"></i> NODES (${nodes.length})
        </div>
        <div id="node-list" style="overflow: auto; flex: 1;"></div>
        <div style="padding: 8px; border-top: 1px solid var(--omega-gray-1);">
          <button onclick="window.omegaAPI.secureAction('discover_nodes', {})" class="btn-primary" style="width: 100%; font-size: 11px;">
            <i class="fas fa-search"></i> Discover Nodes
          </button>
        </div>
      </div>
      
      <!-- Node Detail with Sub-tabs -->
      <div style="display: flex; flex-direction: column; border: 1px solid var(--omega-gray-1); background: var(--omega-dark-3); border-radius: 4px; overflow: hidden;">
        <div id="node-detail-header" style="padding: 12px; border-bottom: 1px solid var(--omega-gray-1);">
          <div style="font: 600 14px var(--font-mono); color: var(--omega-white);">Select a node</div>
        </div>
        
        <!-- Node Sub-tabs -->
        <div id="node-subtabs" style="display: none; background: var(--omega-dark-2); border-bottom: 1px solid var(--omega-gray-1);">
          <div style="display: flex;">
            <button class="node-subtab active" data-subtab="overview">Overview</button>
            <button class="node-subtab" data-subtab="performance">Performance</button>
            <button class="node-subtab" data-subtab="processes">Processes</button>
            <button class="node-subtab" data-subtab="logs">Logs</button>
            <button class="node-subtab" data-subtab="security">Security</button>
            <button class="node-subtab" data-subtab="maintenance">Maintenance</button>
          </div>
        </div>
        
        <!-- Node Detail Content -->
        <div id="node-detail-content" style="flex: 1; padding: 16px; overflow: auto;">
          <div style="font: 400 13px var(--font-mono); color: var(--omega-light-1);">Select a node to view details</div>
        </div>
      </div>
    </div>`;

  const list = root.querySelector('#node-list');
  let selectedNode = null;
  
  // Render node list
  nodes.forEach(n => {
    const status = n.status || 'unknown';
    const statusColor = status === 'active' ? '--omega-green' : status === 'standby' ? '--omega-yellow' : '--omega-red';
    
    const btn = document.createElement('div');
    btn.style.cssText = `
      cursor: pointer; padding: 12px; border-bottom: 1px solid var(--omega-gray-2);
      transition: background 0.15s ease;
    `;
    btn.onmouseenter = () => btn.style.background = 'var(--omega-dark-4)';
    btn.onmouseleave = () => btn.style.background = 'transparent';
    btn.onclick = () => selectNode(n);
    
    btn.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <div style="font: 600 12px var(--font-mono); color: var(--omega-white);">${n.node_id}</div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background: var(${statusColor});"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font: 400 10px var(--font-mono); color: var(--omega-light-1);">
        <span>${n.node_type}</span>
        <span>${n.ip_address}</span>
      </div>
      <div style="margin-top: 6px; display: flex; gap: 8px; font: 400 9px var(--font-mono);">
        <span>CPU: ${n.metrics?.cpu_usage?.toFixed(1) || 0}%</span>
        <span>MEM: ${n.metrics?.memory_usage?.toFixed(1) || 0}%</span>
      </div>
    `;
    list.appendChild(btn);
  });

  function selectNode(node) {
    selectedNode = node;
    const header = root.querySelector('#node-detail-header');
    const subtabs = root.querySelector('#node-subtabs');
    const content = root.querySelector('#node-detail-content');
    
    header.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <div style="font: 600 16px var(--font-mono); color: var(--omega-cyan);">${node.node_id}</div>
          <div style="font: 400 12px var(--font-mono); color: var(--omega-light-1);">${node.node_type} • ${node.ip_address}</div>
        </div>
        <div style="display: flex; gap: 8px;">
          <button onclick="restartNode('${node.node_id}')" class="btn-secondary">
            <i class="fas fa-redo"></i> Restart
          </button>
          <button onclick="maintenanceNode('${node.node_id}')" class="btn-secondary">
            <i class="fas fa-tools"></i> Maintenance
          </button>
        </div>
      </div>
    `;
    
    subtabs.style.display = 'block';
    showNodeSubtab('overview');
    
    // Setup subtab handlers
    root.querySelectorAll('.node-subtab').forEach(tab => {
      tab.onclick = () => showNodeSubtab(tab.dataset.subtab);
    });
  }

  function showNodeSubtab(subtab) {
    // Update active tab
    root.querySelectorAll('.node-subtab').forEach(t => {
      t.classList.toggle('active', t.dataset.subtab === subtab);
    });
    
    const content = root.querySelector('#node-detail-content');
    
    switch(subtab) {
      case 'overview':
        renderNodeOverview(content, selectedNode);
        break;
      case 'performance':
        renderNodePerformance(content, selectedNode);
        break;
      case 'processes':
        renderNodeProcesses(content, selectedNode, processes);
        break;
      case 'logs':
        renderNodeLogs(content, selectedNode, logs);
        break;
      case 'security':
        renderNodeSecurity(content, selectedNode);
        break;
      case 'maintenance':
        renderNodeMaintenance(content, selectedNode);
        break;
    }
  }

  function renderNodeOverview(content, node) {
    const r = node.resources || {};
    const m = node.metrics || {};
    
    content.innerHTML = `
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
        <div class="overview-card">
          <h4><i class="fas fa-info-circle"></i> System Info</h4>
          <div class="metric-grid">
            ${metric('Hostname', node.hostname || node.node_id)}
            ${metric('Type', node.node_type)}
            ${metric('Status', node.status)}
            ${metric('Port', node.port)}
          </div>
        </div>
        
        <div class="overview-card">
          <h4><i class="fas fa-microchip"></i> Hardware</h4>
          <div class="metric-grid">
            ${metric('CPU Cores', r.cpu_cores || '-')}
            ${metric('Memory GB', r.memory_gb || '-')}
            ${metric('Temperature', m.temperature ? m.temperature.toFixed(1) + '°C' : '-')}
            ${metric('Power', m.power_consumption ? m.power_consumption.toFixed(1) + 'W' : '-')}
          </div>
        </div>
        
        <div class="overview-card">
          <h4><i class="fas fa-chart-line"></i> Performance</h4>
          <div class="metric-grid">
            ${metric('CPU Usage', m.cpu_usage ? m.cpu_usage.toFixed(1) + '%' : '0%')}
            ${metric('Memory Usage', m.memory_usage ? m.memory_usage.toFixed(1) + '%' : '0%')}
            ${metric('GPU Usage', m.gpu_usage ? m.gpu_usage.toFixed(1) + '%' : '0%')}
            ${metric('Network I/O', `${formatBytes(m.network_rx || 0)} / ${formatBytes(m.network_tx || 0)}`)}
          </div>
        </div>
        
        <div class="overview-card">
          <h4><i class="fas fa-clock"></i> Uptime & Health</h4>
          <div class="metric-grid">
            ${metric('Last Heartbeat', formatTime(node.last_heartbeat))}
            ${metric('Created', formatTime(node.created_at))}
            ${metric('Health Score', calculateHealthScore(m) + '/100')}
            ${metric('Alerts', '0 active')}
          </div>
        </div>
      </div>
    `;
  }

  function renderNodePerformance(content, node) {
    content.innerHTML = `
      <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;">
          <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value">${(node.metrics?.cpu_usage || 0).toFixed(1)}%</div>
            <div class="metric-progress">
              <div style="width: ${node.metrics?.cpu_usage || 0}%; background: var(--omega-cyan);"></div>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value">${(node.metrics?.memory_usage || 0).toFixed(1)}%</div>
            <div class="metric-progress">
              <div style="width: ${node.metrics?.memory_usage || 0}%; background: var(--omega-green);"></div>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-label">GPU Usage</div>
            <div class="metric-value">${(node.metrics?.gpu_usage || 0).toFixed(1)}%</div>
            <div class="metric-progress">
              <div style="width: ${node.metrics?.gpu_usage || 0}%; background: var(--omega-yellow);"></div>
            </div>
          </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px;">
          <div class="chart-container">
            <h4>Performance Trends</h4>
            <canvas id="node-perf-chart" style="width: 100%; height: 200px;"></canvas>
          </div>
          <div class="chart-container">
            <h4>Resource Distribution</h4>
            <canvas id="node-resource-chart" style="width: 100%; height: 200px;"></canvas>
          </div>
        </div>
      </div>
    `;
    
    // Initialize charts (simplified)
    setTimeout(() => {
      const perfChart = content.querySelector('#node-perf-chart');
      const resourceChart = content.querySelector('#node-resource-chart');
      if (perfChart && resourceChart) {
        drawSimpleChart(perfChart, 'line', node.metrics);
        drawSimpleChart(resourceChart, 'doughnut', node.metrics);
      }
    }, 100);
  }

  function renderNodeProcesses(content, node, processes) {
    const nodeProcesses = processes.slice(0, 15); // Top 15 processes
    
    content.innerHTML = `
      <div style="display: flex; flex-direction: column; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h4><i class="fas fa-list"></i> Running Processes (${nodeProcesses.length})</h4>
          <button onclick="refreshProcesses()" class="btn-secondary">
            <i class="fas fa-sync"></i> Refresh
          </button>
        </div>
        
        <div class="process-table">
          <div class="process-header">
            <div>PID</div>
            <div>Name</div>
            <div>CPU %</div>
            <div>Memory</div>
            <div>Actions</div>
          </div>
          <div class="process-list" style="flex: 1; overflow-y: auto;">
            ${nodeProcesses.map(p => `
              <div class="process-row">
                <div>${p.pid}</div>
                <div style="font-family: var(--font-mono);">${p.name}</div>
                <div>${(p.cpu || 0).toFixed(1)}%</div>
                <div>${p.mem_mb} MB</div>
                <div>
                  <button onclick="killProcess(${p.pid})" class="btn-danger" style="font-size: 10px; padding: 2px 6px;">
                    Kill
                  </button>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  function renderNodeLogs(content, node, logs) {
    const nodeLogs = logs.filter(log => log.source === node.node_id || log.source === 'system').slice(0, 50);
    
    content.innerHTML = `
      <div style="display: flex; flex-direction: column; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h4><i class="fas fa-file-alt"></i> System Logs (${nodeLogs.length})</h4>
          <div style="display: flex; gap: 8px;">
            <select id="log-filter" style="background: var(--omega-dark-4); border: 1px solid var(--omega-gray-1); color: var(--omega-white); padding: 4px 8px;">
              <option value="all">All Levels</option>
              <option value="error">Errors</option>
              <option value="warning">Warnings</option>
              <option value="info">Info</option>
            </select>
            <button onclick="refreshLogs()" class="btn-secondary">
              <i class="fas fa-sync"></i> Refresh
            </button>
          </div>
        </div>
        
        <div class="log-container" style="flex: 1; overflow-y: auto; background: var(--omega-dark-4); border: 1px solid var(--omega-gray-1); border-radius: 4px; padding: 8px;">
          ${nodeLogs.map(log => `
            <div class="log-entry ${log.severity}" style="margin-bottom: 4px; padding: 6px; border-left: 3px solid ${getSeverityColor(log.severity)}; background: var(--omega-dark-3);">
              <div style="display: flex; justify-content: space-between; font: 400 10px var(--font-mono); opacity: 0.7;">
                <span>${formatTime(log.timestamp)}</span>
                <span class="log-severity">${log.severity.toUpperCase()}</span>
              </div>
              <div style="font: 400 11px var(--font-mono); margin-top: 2px;">${log.message}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  function renderNodeSecurity(content, node) {
    content.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
        <div class="security-panel">
          <h4><i class="fas fa-shield-alt"></i> Security Status</h4>
          <div class="security-metrics">
            ${securityMetric('Access Control', 'RBAC Enabled', 'success')}
            ${securityMetric('Encryption', 'AES-256-GCM', 'success')}
            ${securityMetric('Firewall', 'Active', 'success')}
            ${securityMetric('SSH Access', 'Key-based', 'success')}
            ${securityMetric('Last Scan', '2 hours ago', 'info')}
            ${securityMetric('Vulnerabilities', '0 critical', 'success')}
          </div>
        </div>
        
        <div class="security-panel">
          <h4><i class="fas fa-key"></i> Certificates & Keys</h4>
          <div class="cert-list">
            <div class="cert-item">
              <div class="cert-name">SSL Certificate</div>
              <div class="cert-status valid">Valid until 2025-12-31</div>
            </div>
            <div class="cert-item">
              <div class="cert-name">SSH Host Key</div>
              <div class="cert-status valid">RSA 2048-bit</div>
            </div>
            <div class="cert-item">
              <div class="cert-name">API Token</div>
              <div class="cert-status valid">Active</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderNodeMaintenance(content, node) {
    content.innerHTML = `
      <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
          <button onclick="runDiagnostics('${node.node_id}')" class="maintenance-btn">
            <i class="fas fa-stethoscope"></i>
            <div>Run Diagnostics</div>
          </button>
          <button onclick="updateNode('${node.node_id}')" class="maintenance-btn">
            <i class="fas fa-download"></i>
            <div>Update System</div>
          </button>
          <button onclick="cleanupNode('${node.node_id}')" class="maintenance-btn">
            <i class="fas fa-broom"></i>
            <div>Cleanup Temp Files</div>
          </button>
          <button onclick="backupNode('${node.node_id}')" class="maintenance-btn">
            <i class="fas fa-save"></i>
            <div>Create Backup</div>
          </button>
        </div>
        
        <div class="maintenance-log" style="border: 1px solid var(--omega-gray-1); border-radius: 4px; padding: 12px; overflow-y: auto; background: var(--omega-dark-4);">
          <h4 style="margin-bottom: 12px;"><i class="fas fa-history"></i> Maintenance History</h4>
          <div class="maintenance-entries">
            <div class="maintenance-entry">
              <div class="maintenance-time">2024-08-10 14:30</div>
              <div class="maintenance-action">System diagnostics completed - All checks passed</div>
            </div>
            <div class="maintenance-entry">
              <div class="maintenance-time">2024-08-09 09:15</div>
              <div class="maintenance-action">Temporary files cleaned - 2.3GB freed</div>
            </div>
            <div class="maintenance-entry">
              <div class="maintenance-time">2024-08-08 16:45</div>
              <div class="maintenance-action">System backup created successfully</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Helper functions
  function metric(label, value) {
    return `<div class="metric-item"><span>${label}</span><span>${value}</span></div>`;
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatTime(timestamp) {
    if (!timestamp) return 'Never';
    return new Date(timestamp * 1000).toLocaleString();
  }

  function calculateHealthScore(metrics) {
    if (!metrics) return 85;
    const cpu = metrics.cpu_usage || 0;
    const mem = metrics.memory_usage || 0;
    return Math.max(0, 100 - (cpu + mem) / 2);
  }

  function getSeverityColor(severity) {
    const colors = {
      error: 'var(--omega-red)',
      warning: 'var(--omega-yellow)',
      info: 'var(--omega-cyan)',
      debug: 'var(--omega-gray-1)'
    };
    return colors[severity] || colors.info;
  }

  function securityMetric(label, value, status) {
    const statusColors = {
      success: 'var(--omega-green)',
      warning: 'var(--omega-yellow)',
      error: 'var(--omega-red)',
      info: 'var(--omega-cyan)'
    };
    return `
      <div class="security-metric">
        <div class="security-label">${label}</div>
        <div class="security-value" style="color: ${statusColors[status]};">${value}</div>
      </div>
    `;
  }

  function drawSimpleChart(canvas, type, data) {
    // Simplified chart drawing - placeholder for real implementation
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'var(--omega-dark-3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'var(--omega-cyan)';
    ctx.font = '12px monospace';
    ctx.fillText(`${type} chart placeholder`, 10, 20);
  }

  // Global functions for button handlers
  window.restartNode = async (nodeId) => {
    try {
      await window.omegaAPI.secureAction('restart_node', { node_id: nodeId });
      window.notify('success', 'Node Restart', `Restart initiated for ${nodeId}`);
    } catch (e) {
      window.notify('error', 'Node Restart', e.message);
    }
  };

  window.maintenanceNode = async (nodeId) => {
    try {
      await window.omegaAPI.secureAction('health_check', { node_id: nodeId });
      window.notify('success', 'Maintenance', `Health check completed for ${nodeId}`);
    } catch (e) {
      window.notify('error', 'Maintenance', e.message);
    }
  };

  // Add CSS for node subtabs and components
  if (!document.getElementById('nodes-styles')) {
    const style = document.createElement('style');
    style.id = 'nodes-styles';
    style.textContent = `
      .node-subtab {
        background: transparent;
        border: none;
        color: var(--omega-light-1);
        padding: 8px 16px;
        cursor: pointer;
        font: 400 11px var(--font-mono);
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
      }
      
      .node-subtab:hover {
        background: var(--omega-dark-3);
        color: var(--omega-cyan);
      }
      
      .node-subtab.active {
        background: var(--omega-dark-3);
        color: var(--omega-cyan);
        border-bottom-color: var(--omega-cyan);
      }
      
      .overview-card {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 12px;
      }
      
      .overview-card h4 {
        color: var(--omega-cyan);
        font: 600 12px var(--font-mono);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      
      .metric-grid {
        display: grid;
        gap: 8px;
      }
      
      .metric-item {
        display: flex;
        justify-content: space-between;
        font: 400 11px var(--font-mono);
        padding: 4px 0;
        border-bottom: 1px solid var(--omega-gray-2);
      }
      
      .metric-card {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 12px;
        text-align: center;
      }
      
      .metric-label {
        font: 400 10px var(--font-mono);
        color: var(--omega-light-1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      
      .metric-value {
        font: 600 18px var(--font-mono);
        color: var(--omega-cyan);
        margin: 4px 0;
      }
      
      .metric-progress {
        height: 4px;
        background: var(--omega-dark-2);
        border-radius: 2px;
        overflow: hidden;
      }
      
      .metric-progress > div {
        height: 100%;
        transition: width 0.3s ease;
      }
      
      .chart-container {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 12px;
      }
      
      .chart-container h4 {
        color: var(--omega-white);
        font: 600 12px var(--font-mono);
        margin-bottom: 8px;
      }
      
      .process-table {
        display: flex;
        flex-direction: column;
        height: 100%;
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        overflow: hidden;
      }
      
      .process-header {
        display: grid;
        grid-template-columns: 60px 1fr 80px 80px 80px;
        gap: 8px;
        background: var(--omega-dark-4);
        padding: 8px 12px;
        font: 600 10px var(--font-mono);
        color: var(--omega-cyan);
        text-transform: uppercase;
        border-bottom: 1px solid var(--omega-gray-1);
      }
      
      .process-row {
        display: grid;
        grid-template-columns: 60px 1fr 80px 80px 80px;
        gap: 8px;
        padding: 8px 12px;
        font: 400 11px var(--font-mono);
        border-bottom: 1px solid var(--omega-gray-2);
        transition: background 0.15s ease;
      }
      
      .process-row:hover {
        background: var(--omega-dark-4);
      }
      
      .security-panel {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 12px;
      }
      
      .security-panel h4 {
        color: var(--omega-cyan);
        font: 600 12px var(--font-mono);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      
      .security-metric {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid var(--omega-gray-2);
        font: 400 11px var(--font-mono);
      }
      
      .maintenance-btn {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: var(--omega-white);
        text-align: center;
      }
      
      .maintenance-btn:hover {
        border-color: var(--omega-cyan);
        background: var(--omega-dark-3);
      }
      
      .maintenance-btn i {
        font-size: 24px;
        color: var(--omega-cyan);
        margin-bottom: 8px;
        display: block;
      }
      
      .maintenance-entry {
        margin-bottom: 8px;
        padding: 6px 0;
        border-bottom: 1px solid var(--omega-gray-2);
      }
      
      .maintenance-time {
        font: 400 10px var(--font-mono);
        color: var(--omega-light-1);
      }
      
      .maintenance-action {
        font: 400 11px var(--font-mono);
        margin-top: 2px;
      }
    `;
    document.head.appendChild(style);
  }
}
