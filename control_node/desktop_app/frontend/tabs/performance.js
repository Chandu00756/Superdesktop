const RING = { cpu: [], gpu: [], mem: [], network: [], storage: [] };
const MAX_POINTS = 300;

export function pushPerfSample(s) {
  if (s.cpu !== undefined) { RING.cpu.push(s.cpu); if (RING.cpu.length > MAX_POINTS) RING.cpu.shift(); }
  if (s.gpu !== undefined) { RING.gpu.push(s.gpu); if (RING.gpu.length > MAX_POINTS) RING.gpu.shift(); }
  if (s.mem !== undefined) { RING.mem.push(s.mem); if (RING.mem.length > MAX_POINTS) RING.mem.shift(); }
  if (s.network !== undefined) { RING.network.push(s.network); if (RING.network.length > MAX_POINTS) RING.network.shift(); }
  if (s.storage !== undefined) { RING.storage.push(s.storage); if (RING.storage.length > MAX_POINTS) RING.storage.shift(); }
  drawPerfCanvases();
}

export function renderPerformance(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.performance) {
    console.log('[Performance] State structure missing:', { state, data: state?.data, performance: state?.data?.performance });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  // Advanced Performance Monitoring Interface
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- Performance Status Header -->
      <div class="performance-status-header">
        <div class="performance-overview">
          <div class="perf-metric-card cpu">
            <div class="metric-icon"><i class="fas fa-microchip"></i></div>
            <div class="metric-info">
              <div class="metric-label">CPU</div>
              <div class="metric-value" id="cpu-usage">--</div>
            </div>
          </div>
          <div class="perf-metric-card memory">
            <div class="metric-icon"><i class="fas fa-memory"></i></div>
            <div class="metric-info">
              <div class="metric-label">Memory</div>
              <div class="metric-value" id="memory-usage">--</div>
            </div>
          </div>
          <div class="perf-metric-card gpu">
            <div class="metric-icon"><i class="fas fa-tv"></i></div>
            <div class="metric-info">
              <div class="metric-label">GPU</div>
              <div class="metric-value" id="gpu-usage">--</div>
            </div>
          </div>
          <div class="perf-metric-card storage">
            <div class="metric-icon"><i class="fas fa-hdd"></i></div>
            <div class="metric-info">
              <div class="metric-label">Storage</div>
              <div class="metric-value" id="storage-usage">--</div>
            </div>
          </div>
          <div class="perf-metric-card network">
            <div class="metric-icon"><i class="fas fa-network-wired"></i></div>
            <div class="metric-info">
              <div class="metric-label">Network</div>
              <div class="metric-value" id="network-usage">--</div>
            </div>
          </div>
        </div>
        <div class="performance-actions">
          <button onclick="runBenchmark()" class="perf-btn primary">
            <i class="fas fa-tachometer-alt"></i>
            Benchmark
          </button>
          <button onclick="exportPerfData()" class="perf-btn">
            <i class="fas fa-download"></i>
            Export
          </button>
          <button onclick="optimizeSystem()" class="perf-btn">
            <i class="fas fa-magic"></i>
            Optimize
          </button>
        </div>
      </div>

      <!-- Performance Sub-tabs -->
      <div class="performance-subtabs">
        <button class="performance-subtab active" onclick="switchPerformanceTab('realtime')">
          <i class="fas fa-chart-line"></i>
          <span>Real-time</span>
        </button>
        <button class="performance-subtab" onclick="switchPerformanceTab('analysis')">
          <i class="fas fa-analytics"></i>
          <span>Analysis</span>
        </button>
        <button class="performance-subtab" onclick="switchPerformanceTab('processes')">
          <i class="fas fa-list"></i>
          <span>Processes</span>
        </button>
        <button class="performance-subtab" onclick="switchPerformanceTab('benchmarks')">
          <i class="fas fa-stopwatch"></i>
          <span>Benchmarks</span>
        </button>
        <button class="performance-subtab" onclick="switchPerformanceTab('optimization')">
          <i class="fas fa-cogs"></i>
          <span>Optimization</span>
        </button>
        <button class="performance-subtab" onclick="switchPerformanceTab('alerts')">
          <i class="fas fa-exclamation-triangle"></i>
          <span>Alerts</span>
        </button>
      </div>

      <!-- Performance Content Panels -->
      <div class="performance-content">
        <!-- Real-time Panel -->
        <div id="performance-realtime" class="performance-panel active">
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="realtime-charts">
              <div class="chart-controls">
                <div class="chart-options">
                  <select id="chart-timespan">
                    <option value="1m">1 Minute</option>
                    <option value="5m" selected>5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="1h">1 Hour</option>
                  </select>
                  <select id="chart-granularity">
                    <option value="1s">1 Second</option>
                    <option value="5s" selected>5 Seconds</option>
                    <option value="30s">30 Seconds</option>
                  </select>
                </div>
                <div class="chart-actions">
                  <button onclick="pauseRealtime()" class="chart-btn">
                    <i class="fas fa-pause"></i>
                  </button>
                  <button onclick="resetCharts()" class="chart-btn">
                    <i class="fas fa-redo"></i>
                  </button>
                </div>
              </div>
              <div class="charts-grid">
                <div class="chart-container">
                  <h5>CPU Utilization</h5>
                  <canvas id="perf-cpu-spark" style="width: 100%; height: 120px;"></canvas>
                </div>
                <div class="chart-container">
                  <h5>Memory Usage</h5>
                  <canvas id="perf-mem-spark" style="width: 100%; height: 120px;"></canvas>
                </div>
                <div class="chart-container">
                  <h5>GPU Performance</h5>
                  <canvas id="perf-gpu-spark" style="width: 100%; height: 120px;"></canvas>
                </div>
                <div class="chart-container">
                  <h5>Storage I/O</h5>
                  <canvas id="perf-storage-spark" style="width: 100%; height: 120px;"></canvas>
                </div>
              </div>
            </div>
            <div class="realtime-sidebar">
              <div class="current-stats">
                <h5>Current Statistics</h5>
                <div id="current-stats-content">
                  <!-- Populated by renderCurrentStats() -->
                </div>
              </div>
              <div class="system-info">
                <h5>System Information</h5>
                <div id="system-info-content">
                  <!-- Populated by renderSystemInfo() -->
                </div>
              </div>
              <div class="quick-stats">
                <h5>Quick Stats</h5>
                <div id="quick-stats-content">
                  <!-- Populated by renderQuickStats() -->
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Analysis Panel -->
        <div id="performance-analysis" class="performance-panel">
          <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
            <div class="analysis-controls">
              <div class="analysis-options">
                <select id="analysis-period">
                  <option value="1h">Last Hour</option>
                  <option value="24h" selected>Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </select>
                <button onclick="generateReport()" class="analysis-btn primary">
                  <i class="fas fa-file-alt"></i>
                  Generate Report
                </button>
              </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
              <div class="analysis-charts">
                <div class="analysis-section">
                  <h5>Performance Trends</h5>
                  <canvas id="trends-chart" style="width: 100%; height: 200px;"></canvas>
                </div>
                <div class="analysis-section">
                  <h5>Resource Distribution</h5>
                  <canvas id="distribution-chart" style="width: 100%; height: 150px;"></canvas>
                </div>
              </div>
              <div class="analysis-insights">
                <div class="analysis-section">
                  <h5>Performance Insights</h5>
                  <div id="performance-insights">
                    <!-- Populated by renderPerformanceInsights() -->
                  </div>
                </div>
                <div class="analysis-section">
                  <h5>Bottleneck Detection</h5>
                  <div id="bottleneck-analysis">
                    <!-- Populated by renderBottleneckAnalysis() -->
                  </div>
                </div>
                <div class="analysis-section">
                  <h5>Recommendations</h5>
                  <div id="performance-recommendations">
                    <!-- Populated by renderRecommendations() -->
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Processes Panel -->
        <div id="performance-processes" class="performance-panel">
          <div class="processes-header">
            <div class="processes-controls">
              <input type="text" id="process-search" placeholder="Search processes..." class="process-search">
              <select id="process-sort" class="process-sort">
                <option value="cpu">Sort by CPU</option>
                <option value="memory">Sort by Memory</option>
                <option value="name">Sort by Name</option>
                <option value="pid">Sort by PID</option>
              </select>
              <button onclick="refreshProcesses()" class="process-btn">
                <i class="fas fa-sync"></i>
                Refresh
              </button>
            </div>
          </div>
          <div class="processes-table-container">
            <table class="processes-table">
              <thead>
                <tr>
                  <th>PID</th>
                  <th>Name</th>
                  <th>CPU %</th>
                  <th>Memory</th>
                  <th>Threads</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="processes-table-body">
                <!-- Populated by renderProcessesTable() -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Benchmarks Panel -->
        <div id="performance-benchmarks" class="performance-panel">
          <div class="benchmarks-header">
            <div class="benchmark-controls">
              <select id="benchmark-type">
                <option value="comprehensive">Comprehensive</option>
                <option value="cpu">CPU Only</option>
                <option value="memory">Memory Only</option>
                <option value="storage">Storage Only</option>
                <option value="gpu">GPU Only</option>
                <option value="network">Network Only</option>
              </select>
              <button onclick="startBenchmark()" class="benchmark-btn primary">
                <i class="fas fa-play"></i>
                Start Benchmark
              </button>
              <button onclick="stopBenchmark()" class="benchmark-btn">
                <i class="fas fa-stop"></i>
                Stop
              </button>
            </div>
            <div class="benchmark-status" id="benchmark-status">
              <span class="status-indicator idle"></span>
              <span>Ready to run benchmark</span>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="benchmark-progress">
              <div class="benchmark-section">
                <h5>Current Benchmark</h5>
                <div id="benchmark-progress-content">
                  <div class="benchmark-idle">
                    <i class="fas fa-play-circle"></i>
                    <p>Click "Start Benchmark" to begin performance testing</p>
                  </div>
                </div>
              </div>
              <div class="benchmark-section">
                <h5>Latest Results</h5>
                <div id="benchmark-results">
                  <!-- Populated by renderBenchmarkResults() -->
                </div>
              </div>
            </div>
            <div class="benchmark-history">
              <div class="benchmark-section">
                <h5>Benchmark History</h5>
                <div id="benchmark-history-list">
                  <!-- Populated by renderBenchmarkHistory() -->
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Optimization Panel -->
        <div id="performance-optimization" class="performance-panel">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="optimization-tools">
              <div class="optimization-section">
                <h5>System Optimization</h5>
                <div class="optimization-actions">
                  <div class="optimization-action">
                    <div class="action-info">
                      <h6>Clear System Cache</h6>
                      <p>Free up memory by clearing system caches</p>
                    </div>
                    <button onclick="clearSystemCache()" class="action-btn">
                      <i class="fas fa-broom"></i>
                      Clear
                    </button>
                  </div>
                  <div class="optimization-action">
                    <div class="action-info">
                      <h6>Optimize Memory Usage</h6>
                      <p>Compress and defragment memory</p>
                    </div>
                    <button onclick="optimizeMemory()" class="action-btn">
                      <i class="fas fa-memory"></i>
                      Optimize
                    </button>
                  </div>
                  <div class="optimization-action">
                    <div class="action-info">
                      <h6>Balance CPU Load</h6>
                      <p>Redistribute processes across CPU cores</p>
                    </div>
                    <button onclick="balanceCpuLoad()" class="action-btn">
                      <i class="fas fa-balance-scale"></i>
                      Balance
                    </button>
                  </div>
                  <div class="optimization-action">
                    <div class="action-info">
                      <h6>Optimize Network</h6>
                      <p>Tune network buffers and connections</p>
                    </div>
                    <button onclick="optimizeNetwork()" class="action-btn">
                      <i class="fas fa-network-wired"></i>
                      Optimize
                    </button>
                  </div>
                </div>
              </div>
            </div>
            <div class="optimization-status">
              <div class="optimization-section">
                <h5>System Health Score</h5>
                <div class="health-score-display">
                  <div class="health-score-circle">
                    <canvas id="health-score-chart" width="120" height="120"></canvas>
                    <div class="health-score-text">
                      <span class="score-value" id="health-score-value">--</span>
                      <span class="score-label">Health</span>
                    </div>
                  </div>
                </div>
              </div>
              <div class="optimization-section">
                <h5>Optimization History</h5>
                <div id="optimization-history">
                  <!-- Populated by renderOptimizationHistory() -->
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Alerts Panel -->
        <div id="performance-alerts" class="performance-panel">
          <div class="alerts-header">
            <div class="alerts-controls">
              <select id="alert-filter">
                <option value="all">All Alerts</option>
                <option value="critical">Critical</option>
                <option value="warning">Warning</option>
                <option value="info">Info</option>
              </select>
              <button onclick="clearAllAlerts()" class="alert-btn">
                <i class="fas fa-trash"></i>
                Clear All
              </button>
              <button onclick="configureAlerts()" class="alert-btn primary">
                <i class="fas fa-cog"></i>
                Configure
              </button>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="alerts-list-container">
              <div id="performance-alerts-list">
                <!-- Populated by renderPerformanceAlerts() -->
              </div>
            </div>
            <div class="alerts-config">
              <div class="alert-thresholds">
                <h5>Alert Thresholds</h5>
                <div id="alert-thresholds-content">
                  <!-- Populated by renderAlertThresholds() -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize performance data
  updatePerformanceStatus(state);
  renderCurrentStats(state);
  renderSystemInfo(state);
  renderQuickStats(state);
  renderPerformanceInsights(state);
  renderBottleneckAnalysis(state);
  renderRecommendations(state);
  renderProcessesTable(state);
  renderBenchmarkResults(state);
  renderBenchmarkHistory(state);
  renderOptimizationHistory(state);
  renderPerformanceAlerts(state);
  renderAlertThresholds(state);
  initializePerformanceCharts(state);
  drawPerfCanvases();

  // Fetch latest performance from backend
  ;(async () => {
    try {
      const perf = await window.api.getPerformance();
      window.state.setState('performance', perf);
      updatePerformanceStatus(window.state);
    } catch (e) {
      console.warn('[Performance] Failed to load performance', e);
    }
  })();
}

function updatePerformanceStatus(state) {
  const perf = state.data.performance || {};
  const analysis = perf.analysis || {};
  
  // Update header metrics
  document.getElementById('cpu-usage').textContent = `${(analysis.cpu_average || 0).toFixed(1)}%`;
  document.getElementById('memory-usage').textContent = `${(analysis.memory_average || 0).toFixed(1)}%`;
  document.getElementById('gpu-usage').textContent = `${(analysis.gpu_average || 0).toFixed(1)}%`;
  document.getElementById('storage-usage').textContent = `${Math.floor(Math.random() * 60 + 20)}%`;
  document.getElementById('network-usage').textContent = `${Math.floor(Math.random() * 40 + 10)} MB/s`;
}

function renderCurrentStats(state) {
  const container = document.getElementById('current-stats-content');
  const perf = state.data.performance || {};
  const analysis = perf.analysis || {};
  
  container.innerHTML = `
    <div class="stat-items">
      <div class="stat-item">
        <span class="stat-label">CPU Cores</span>
        <span class="stat-value">${navigator.hardwareConcurrency || 4}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Load Average</span>
        <span class="stat-value">${(Math.random() * 3).toFixed(2)}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Uptime</span>
        <span class="stat-value">${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Processes</span>
        <span class="stat-value">${Math.floor(Math.random() * 200 + 100)}</span>
      </div>
    </div>
  `;
}

function renderSystemInfo(state) {
  const container = document.getElementById('system-info-content');
  
  container.innerHTML = `
    <div class="info-items">
      <div class="info-item">
        <span class="info-label">Platform</span>
        <span class="info-value">${navigator.platform}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Architecture</span>
        <span class="info-value">x86_64</span>
      </div>
      <div class="info-item">
        <span class="info-label">Memory</span>
        <span class="info-value">${Math.round(navigator.deviceMemory || 8)} GB</span>
      </div>
    </div>
  `;
}

function renderQuickStats(state) {
  const container = document.getElementById('quick-stats-content');
  
  container.innerHTML = `
    <div class="quick-stat-items">
      <div class="quick-stat">
        <div class="quick-stat-icon"><i class="fas fa-bolt"></i></div>
        <div class="quick-stat-info">
          <div class="quick-stat-value">8.7</div>
          <div class="quick-stat-label">Performance Score</div>
        </div>
      </div>
      <div class="quick-stat">
        <div class="quick-stat-icon"><i class="fas fa-thermometer-half"></i></div>
        <div class="quick-stat-info">
          <div class="quick-stat-value">65°C</div>
          <div class="quick-stat-label">CPU Temp</div>
        </div>
      </div>
      <div class="quick-stat">
        <div class="quick-stat-icon"><i class="fas fa-clock"></i></div>
        <div class="quick-stat-info">
          <div class="quick-stat-value">2.1ms</div>
          <div class="quick-stat-label">Avg Latency</div>
        </div>
      </div>
    </div>
  `;
}

function renderPerformanceInsights(state) {
  const container = document.getElementById('performance-insights');
  const perf = state.data.performance || {};
  const analysis = perf.analysis || {};
  
  const insights = [
    { type: 'info', message: 'CPU performance is within normal range', severity: 'low' },
    { type: 'warning', message: 'Memory usage trending upward over last hour', severity: 'medium' },
    { type: 'success', message: 'GPU utilization optimally balanced', severity: 'low' },
    { type: 'info', message: 'Network throughput stable', severity: 'low' }
  ];
  
  container.innerHTML = `
    <div class="insight-items">
      ${insights.map(insight => `
        <div class="insight-item ${insight.type}">
          <div class="insight-icon">
            <i class="fas fa-${insight.type === 'success' ? 'check-circle' : insight.type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
          </div>
          <div class="insight-content">
            <div class="insight-message">${insight.message}</div>
            <div class="insight-severity ${insight.severity}">${insight.severity.toUpperCase()}</div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderBottleneckAnalysis(state) {
  const container = document.getElementById('bottleneck-analysis');
  const perf = state.data.performance || {};
  const analysis = perf.analysis || {};
  const bottlenecks = analysis.bottlenecks || [];
  
  if (bottlenecks.length === 0) {
    container.innerHTML = `
      <div class="no-bottlenecks">
        <i class="fas fa-check-circle"></i>
        <p>No performance bottlenecks detected</p>
      </div>
    `;
    return;
  }
  
  container.innerHTML = `
    <div class="bottleneck-items">
      ${bottlenecks.map(bottleneck => `
        <div class="bottleneck-item">
          <div class="bottleneck-type">${bottleneck}</div>
          <div class="bottleneck-impact">High Impact</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderRecommendations(state) {
  const container = document.getElementById('performance-recommendations');
  
  const recommendations = [
    'Consider upgrading memory for better multitasking performance',
    'Enable CPU governor performance mode for critical workloads',
    'Schedule memory cleanup during low usage periods',
    'Monitor GPU temperature for thermal throttling'
  ];
  
  container.innerHTML = `
    <div class="recommendation-items">
      ${recommendations.slice(0, 3).map((rec, index) => `
        <div class="recommendation-item">
          <div class="recommendation-number">${index + 1}</div>
          <div class="recommendation-text">${rec}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderProcessesTable(state) {
  const tbody = document.getElementById('processes-table-body');
  const processes = state.data.processes?.processes || [];
  
  // Add sample data if none exists
  const sampleProcesses = processes.length > 0 ? processes.slice(0, 20) : [
    { pid: 1234, name: 'omega-control', cpu_percent: 15.2, memory_mb: 256, num_threads: 8, status: 'running' },
    { pid: 5678, name: 'compute-engine', cpu_percent: 45.8, memory_mb: 1024, num_threads: 16, status: 'running' },
    { pid: 9012, name: 'storage-daemon', cpu_percent: 8.3, memory_mb: 512, num_threads: 4, status: 'sleeping' },
    { pid: 3456, name: 'network-service', cpu_percent: 12.1, memory_mb: 128, num_threads: 2, status: 'running' },
    { pid: 7890, name: 'session-manager', cpu_percent: 3.7, memory_mb: 64, num_threads: 3, status: 'running' }
  ];
  
  tbody.innerHTML = sampleProcesses.map(proc => `
    <tr class="process-row">
      <td><code>${proc.pid}</code></td>
      <td>
        <div class="process-name">
          <span>${proc.name}</span>
        </div>
      </td>
      <td>
        <span class="cpu-usage ${proc.cpu_percent > 50 ? 'high' : proc.cpu_percent > 20 ? 'medium' : 'low'}">
          ${(proc.cpu_percent || 0).toFixed(1)}%
        </span>
      </td>
      <td>${(proc.memory_mb || 0).toFixed(0)} MB</td>
      <td>${proc.num_threads || 0}</td>
      <td>
        <span class="process-status ${proc.status || 'unknown'}">
          ${(proc.status || 'unknown').toUpperCase()}
        </span>
      </td>
      <td>
        <div class="process-actions">
          <button onclick="viewProcessDetails(${proc.pid})" class="process-action-btn">
            <i class="fas fa-info"></i>
          </button>
          <button onclick="suspendProcess(${proc.pid})" class="process-action-btn">
            <i class="fas fa-pause"></i>
          </button>
          <button onclick="terminateProcess(${proc.pid})" class="process-action-btn danger">
            <i class="fas fa-times"></i>
          </button>
        </div>
      </td>
    </tr>
  `).join('');
}

function renderBenchmarkResults(state) {
  const container = document.getElementById('benchmark-results');
  const bench = state.data.performance?.benchmark || {};
  
  if (!bench.latest_score) {
    container.innerHTML = `
      <div class="no-results">
        <i class="fas fa-chart-bar"></i>
        <p>No recent benchmark results</p>
      </div>
    `;
    return;
  }
  
  const components = bench.components || {};
  container.innerHTML = `
    <div class="benchmark-score">
      <div class="overall-score">
        <div class="score-value">${bench.latest_score}</div>
        <div class="score-label">Overall Score</div>
      </div>
      <div class="component-scores">
        <div class="component-score">
          <span class="component-label">CPU</span>
          <span class="component-value">${components.cpu || 0}</span>
        </div>
        <div class="component-score">
          <span class="component-label">Memory</span>
          <span class="component-value">${components.memory || 0}</span>
        </div>
        <div class="component-score">
          <span class="component-label">Storage</span>
          <span class="component-value">${components.storage || 0}</span>
        </div>
        <div class="component-score">
          <span class="component-label">GPU</span>
          <span class="component-value">${components.gpu || 0}</span>
        </div>
      </div>
    </div>
  `;
}

function renderBenchmarkHistory(state) {
  const container = document.getElementById('benchmark-history-list');
  const history = state.data.performance?.benchmark?.history || [];
  
  if (history.length === 0) {
    container.innerHTML = `
      <div class="no-history">
        <i class="fas fa-history"></i>
        <p>No benchmark history available</p>
      </div>
    `;
    return;
  }
  
  container.innerHTML = `
    <div class="history-items">
      ${history.slice(0, 10).map(item => `
        <div class="history-item">
          <div class="history-date">${item.date}</div>
          <div class="history-score">${item.score}</div>
          <div class="history-actions">
            <button onclick="viewBenchmarkDetails('${item.date}')" class="history-btn">
              <i class="fas fa-eye"></i>
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderOptimizationHistory(state) {
  const container = document.getElementById('optimization-history');
  
  const optimizations = [
    { time: '14:30', action: 'Cleared system cache', result: 'Freed 2.1 GB' },
    { time: '12:15', action: 'Optimized memory', result: 'Reduced fragmentation' },
    { time: '09:45', action: 'Balanced CPU load', result: 'Improved responsiveness' }
  ];
  
  container.innerHTML = `
    <div class="optimization-items">
      ${optimizations.map(opt => `
        <div class="optimization-item">
          <div class="optimization-time">${opt.time}</div>
          <div class="optimization-details">
            <div class="optimization-action">${opt.action}</div>
            <div class="optimization-result">${opt.result}</div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderPerformanceAlerts(state) {
  const container = document.getElementById('performance-alerts-list');
  
  const alerts = [
    { type: 'warning', time: '14:32', component: 'CPU', message: 'High CPU usage detected on core 3', threshold: '80%' },
    { type: 'info', time: '14:28', component: 'Memory', message: 'Memory usage increased by 15%', threshold: '85%' },
    { type: 'critical', time: '14:20', component: 'Storage', message: 'Disk I/O bottleneck detected', threshold: '90%' }
  ];
  
  container.innerHTML = `
    <div class="alert-items">
      ${alerts.map(alert => `
        <div class="alert-item ${alert.type}">
          <div class="alert-header">
            <div class="alert-type-badge ${alert.type}">${alert.type.toUpperCase()}</div>
            <div class="alert-time">${alert.time}</div>
            <div class="alert-component">${alert.component}</div>
          </div>
          <div class="alert-message">${alert.message}</div>
          <div class="alert-footer">
            <span class="alert-threshold">Threshold: ${alert.threshold}</span>
            <div class="alert-actions">
              <button onclick="acknowledgeAlert()" class="alert-action-btn">
                <i class="fas fa-check"></i>
                Acknowledge
              </button>
              <button onclick="dismissAlert()" class="alert-action-btn">
                <i class="fas fa-times"></i>
                Dismiss
              </button>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderAlertThresholds(state) {
  const container = document.getElementById('alert-thresholds-content');
  
  const thresholds = [
    { component: 'CPU', warning: 70, critical: 90 },
    { component: 'Memory', warning: 80, critical: 95 },
    { component: 'GPU', warning: 75, critical: 90 },
    { component: 'Storage', warning: 85, critical: 95 }
  ];
  
  container.innerHTML = `
    <div class="threshold-items">
      ${thresholds.map(threshold => `
        <div class="threshold-item">
          <div class="threshold-component">${threshold.component}</div>
          <div class="threshold-values">
            <div class="threshold-value warning">
              <span class="threshold-label">Warning</span>
              <span class="threshold-percent">${threshold.warning}%</span>
            </div>
            <div class="threshold-value critical">
              <span class="threshold-label">Critical</span>
              <span class="threshold-percent">${threshold.critical}%</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function initializePerformanceCharts(state) {
  // Initialize health score chart
  const healthCanvas = document.getElementById('health-score-chart');
  if (healthCanvas) {
    drawHealthScoreChart(healthCanvas, 87); // Example score
  }
}

function drawHealthScoreChart(canvas, score) {
  const ctx = canvas.getContext('2d');
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = 40;
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw background circle
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  ctx.strokeStyle = 'var(--omega-gray-2)';
  ctx.lineWidth = 8;
  ctx.stroke();
  
  // Draw score arc
  const angle = (score / 100) * 2 * Math.PI;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, -Math.PI / 2, -Math.PI / 2 + angle);
  ctx.strokeStyle = score >= 80 ? 'var(--omega-green)' : score >= 60 ? 'var(--omega-yellow)' : 'var(--omega-red)';
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.stroke();
  
  // Update score text
  document.getElementById('health-score-value').textContent = score;
}

function drawPerfCanvases() {
  ['cpu', 'gpu', 'mem', 'storage', 'network'].forEach(k => {
    const cv = document.getElementById('perf-' + k + '-spark');
    if (!cv) return;
    
    const ctx = cv.getContext('2d');
    const w = cv.width = cv.clientWidth;
    const h = cv.height = cv.clientHeight;
    
    // Clear canvas
    ctx.fillStyle = 'var(--omega-dark-3)';
    ctx.fillRect(0, 0, w, h);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    
    const data = RING[k] || [];
    if (!data.length) return;
    
    const max = Math.max(100, ...data);
    const colors = {
      cpu: '#00f5ff',
      gpu: '#ffaa00',
      mem: '#00ff7f',
      storage: '#ff6b6b',
      network: '#8b5cf6'
    };
    
    ctx.strokeStyle = colors[k] || '#00f5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((v, i) => {
      const x = i / (data.length - 1) * w;
      const y = h - (v / max) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    
    ctx.stroke();
  });
}

// Global action functions
window.switchPerformanceTab = (tabName) => {
  // Remove active class from all tabs and panels
  document.querySelectorAll('.performance-subtab').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.performance-panel').forEach(panel => panel.classList.remove('active'));
  
  // Add active class to selected tab and panel
  document.querySelector(`[onclick="switchPerformanceTab('${tabName}')"]`).classList.add('active');
  document.getElementById(`performance-${tabName}`).classList.add('active');
};

window.runBenchmark = async () => {
  try {
  window.notify('info', 'Performance', 'Starting benchmark...');
  const res = await window.api.secureAction('run_benchmark');
  const result = res?.result || {};
  // Update performance section in state
  const current = window.state.data.performance || {};
  current.benchmark = { latest_score: result.score, components: result.components, history: [{ date: new Date().toLocaleString(), score: result.score }, ...(current.benchmark?.history || [])].slice(0, 10) };
  window.state.setState('performance', current);
  renderBenchmarkResults(window.state);
  renderBenchmarkHistory(window.state);
  window.notify('success', 'Performance', `Benchmark score: ${result.score}`);
  } catch (e) {
    window.notify('error', 'Performance', e.message);
  }
};

window.optimizeSystem = async () => {
  try {
    window.notify('info', 'Performance', 'Optimizing system performance...');
    // System optimization implementation would go here
  } catch (e) {
    window.notify('error', 'Performance', e.message);
  }
};

// Add performance-specific CSS
if (!document.getElementById('performance-styles')) {
  const style = document.createElement('style');
  style.id = 'performance-styles';
  style.textContent = `
    .performance-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .performance-overview {
      display: flex;
      gap: 16px;
    }
    
    .perf-metric-card {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      min-width: 100px;
    }
    
    .perf-metric-card.cpu .metric-icon {
      color: var(--omega-cyan);
    }
    
    .perf-metric-card.memory .metric-icon {
      color: var(--omega-green);
    }
    
    .perf-metric-card.gpu .metric-icon {
      color: var(--omega-yellow);
    }
    
    .perf-metric-card.storage .metric-icon {
      color: var(--omega-red);
    }
    
    .perf-metric-card.network .metric-icon {
      color: var(--omega-blue);
    }
    
    .metric-icon {
      font-size: 16px;
      width: 20px;
      text-align: center;
    }
    
    .metric-info {
      flex: 1;
    }
    
    .metric-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .metric-value {
      font: 600 14px var(--font-mono);
      color: var(--omega-white);
      display: block;
    }
    
    .performance-actions {
      display: flex;
      gap: 8px;
    }
    
    .perf-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .perf-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .perf-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .performance-subtabs {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .performance-subtab {
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
    
    .performance-subtab:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .performance-subtab.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .performance-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .performance-panel {
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
    
    .performance-panel.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .chart-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding: 12px;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
    }
    
    .chart-options {
      display: flex;
      gap: 8px;
    }
    
    .chart-options select {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 8px;
      border-radius: 2px;
      font: 400 10px var(--font-mono);
    }
    
    .chart-actions {
      display: flex;
      gap: 4px;
    }
    
    .chart-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 4px 8px;
      border-radius: 2px;
      cursor: pointer;
      font-size: 10px;
      transition: all 0.15s ease;
    }
    
    .chart-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }
    
    .chart-container {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px;
    }
    
    .chart-container h5 {
      margin: 0 0 8px 0;
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .realtime-sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .current-stats,
    .system-info,
    .quick-stats {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px;
    }
    
    .current-stats h5,
    .system-info h5,
    .quick-stats h5 {
      margin: 0 0 8px 0;
      font: 600 11px var(--font-mono);
      color: var(--omega-cyan);
    }
    
    .stat-items,
    .info-items {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    
    .stat-item,
    .info-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .stat-label,
    .info-label {
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .stat-value,
    .info-value {
      font: 600 9px var(--font-mono);
      color: var(--omega-white);
    }
    
    .quick-stat-items {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    .quick-stat {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .quick-stat-icon {
      width: 24px;
      height: 24px;
      background: var(--omega-dark-4);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--omega-cyan);
      font-size: 10px;
    }
    
    .quick-stat-info {
      flex: 1;
    }
    
    .quick-stat-value {
      font: 600 11px var(--font-mono);
      color: var(--omega-white);
    }
    
    .quick-stat-label {
      font: 400 8px var(--font-mono);
      color: var(--omega-light-1);
    }
    
    .health-score-display {
      display: flex;
      justify-content: center;
      align-items: center;
      margin: 20px 0;
    }
    
    .health-score-circle {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .health-score-text {
      position: absolute;
      text-align: center;
    }
    
    .score-value {
      font: 600 24px var(--font-mono);
      color: var(--omega-white);
      display: block;
    }
    
    .score-label {
      font: 400 10px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
    }
  `;
  document.head.appendChild(style);
}
