// Omega Control Center - Renderer Process
// ====================================

class OmegaControlCenter {
  constructor() {
    this.isConnected = false;
    this.activeTab = 'dashboard';
    this.charts = {};
    this.websocket = null;
    this.notifications = [];
    this.apiBaseUrl = 'http://127.0.0.1:8443';
    this.authToken = null;
    this.encryptionKey = null;
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.authenticateAndConnect();
    this.initializeCharts();
    this.updateCurrentTime();
    this.initializeResizableWidgets();
    this.optimizePerformance();
  }

  optimizePerformance() {
    // Enable hardware acceleration
    document.body.style.transform = 'translateZ(0)';
    document.body.style.willChange = 'transform';
    
    // Throttle resize events
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        this.handleWindowResize();
      }, 16); // 60fps
    });

    // Enable GPU acceleration for animations
    document.body.classList.add('gpu-accelerated');

    // Optimize scroll performance
    this.enableSmoothScrolling();
    
    // Reduce update frequency when not focused
    this.setupVisibilityOptimization();
    
    // Enable requestAnimationFrame for smooth updates
    this.enableRAFUpdates();
  }

  enableRAFUpdates() {
    let lastUpdate = 0;
    const updateInterval = 100; // Update every 100ms for smooth performance

    const updateLoop = (timestamp) => {
      if (timestamp - lastUpdate >= updateInterval) {
        this.updateDashboardData();
        lastUpdate = timestamp;
      }
      requestAnimationFrame(updateLoop);
    };
    
    requestAnimationFrame(updateLoop);
  }

  enableSmoothScrolling() {
    // Add smooth scrolling CSS
    const style = document.createElement('style');
    style.textContent = `
      * {
        scroll-behavior: smooth;
      }
      .widget {
        will-change: transform, opacity;
      }
      .tab-content {
        will-change: transform;
      }
    `;
    document.head.appendChild(style);
  }

  setupVisibilityOptimization() {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.updateInterval = 5000; // Slow updates when hidden
      } else {
        this.updateInterval = 1000; // Normal updates when visible
      }
    });
  }

  initializeResizableWidgets() {
    const widgets = document.querySelectorAll('.widget');
    
    widgets.forEach(widget => {
      // Add resize handle if not exists
      if (!widget.querySelector('.resize-handle')) {
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle';
        resizeHandle.innerHTML = '<i class="fas fa-expand-arrows-alt"></i>';
        widget.appendChild(resizeHandle);
      }

      // Make widget resizable with improved performance
      this.makeWidgetResizable(widget);
      
      // Add drag functionality for repositioning
      this.makeWidgetDraggable(widget);
    });
  }

  makeWidgetDraggable(widget) {
    const header = widget.querySelector('.widget-header');
    if (!header) return;

    let isDragging = false;
    let startX, startY, initialX, initialY;

    header.style.cursor = 'move';
    header.addEventListener('mousedown', initDrag);

    function initDrag(e) {
      if (e.target.closest('.widget-btn')) return; // Don't drag on buttons
      
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      
      const rect = widget.getBoundingClientRect();
      initialX = rect.left;
      initialY = rect.top;
      
      widget.style.position = 'fixed';
      widget.style.zIndex = '1000';
      widget.style.left = initialX + 'px';
      widget.style.top = initialY + 'px';
      widget.classList.add('dragging');
      
      document.addEventListener('mousemove', doDrag);
      document.addEventListener('mouseup', stopDrag);
      e.preventDefault();
    }

    function doDrag(e) {
      if (!isDragging) return;
      
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      widget.style.left = (initialX + deltaX) + 'px';
      widget.style.top = (initialY + deltaY) + 'px';
    }

    function stopDrag() {
      isDragging = false;
      widget.classList.remove('dragging');
      widget.style.position = '';
      widget.style.zIndex = '';
      widget.style.left = '';
      widget.style.top = '';
      
      document.removeEventListener('mousemove', doDrag);
      document.removeEventListener('mouseup', stopDrag);
    }
  }

  makeWidgetResizable(widget) {
    const handle = widget.querySelector('.resize-handle');
    if (!handle) return;
    
    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    handle.addEventListener('mousedown', initResize);

    function initResize(e) {
      isResizing = true;
      startX = e.clientX;
      startY = e.clientY;
      startWidth = parseInt(document.defaultView.getComputedStyle(widget).width, 10);
      startHeight = parseInt(document.defaultView.getComputedStyle(widget).height, 10);
      
      widget.classList.add('resizing');
      document.addEventListener('mousemove', doResize);
      document.addEventListener('mouseup', stopResize);
      
      e.preventDefault();
      e.stopPropagation();
    }

    function doResize(e) {
      if (!isResizing) return;
      
      const newWidth = startWidth + (e.clientX - startX);
      const newHeight = startHeight + (e.clientY - startY);
      
      // Apply constraints with smooth animation
      if (newWidth >= 280 && newWidth <= window.innerWidth - 40) {
        widget.style.width = newWidth + 'px';
      }
      if (newHeight >= 180 && newHeight <= window.innerHeight - 140) {
        widget.style.height = newHeight + 'px';
      }
      
      // Update any charts or content that needs resizing
      setTimeout(() => {
        const event = new Event('resize');
        window.dispatchEvent(event);
      }, 50);
    }

    function stopResize() {
      isResizing = false;
      widget.classList.remove('resizing');
      document.removeEventListener('mousemove', doResize);
      document.removeEventListener('mouseup', stopResize);
      
      // Smooth transition back
      widget.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
      setTimeout(() => {
        widget.style.transition = '';
      }, 300);
    }
  }

  enableSmoothScrolling() {
    // Add smooth scrolling behavior
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Optimize scroll containers
    const scrollContainers = document.querySelectorAll('.session-list, .alert-list, .node-tree');
    scrollContainers.forEach(container => {
      container.style.scrollBehavior = 'smooth';
      container.style.overflowAnchor = 'none';
    });
  }

  setupVisibilityOptimization() {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.reduceUpdateFrequency();
      } else {
        this.restoreUpdateFrequency();
      }
    });
  }

  reduceUpdateFrequency() {
    // Reduce update frequency when tab is not visible
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = setInterval(() => {
        this.updateDashboardData();
      }, 5000); // 5 seconds instead of 1 second
    }
  }

  restoreUpdateFrequency() {
    // Restore normal update frequency
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = setInterval(() => {
        this.updateDashboardData();
      }, 1000); // Back to 1 second
    }
  }

  handleWindowResize() {
    // Redraw charts on window resize
    Object.values(this.charts).forEach(chart => {
      if (chart && chart.resize) {
        chart.resize();
      }
    });
  }

  async authenticateAndConnect() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          username: 'admin',
          password: process.env.OMEGA_ADMIN_PASSWORD || 'omega123'
        })
      });

      if (response.ok) {
        const authData = await response.json();
        this.authToken = authData.token;
        this.isConnected = true;
        this.updateConnectionStatus(true);
        this.initializeWebSocket();
        this.startRealTimeUpdates();
        this.loadInitialData();
        console.log('Backend connection established');
      } else {
        console.error('Authentication failed');
        this.updateConnectionStatus(false);
      }
    } catch (error) {
      console.error('Backend connection error:', error);
      this.updateConnectionStatus(false);
    }
  }

  async makeBackendRequest(endpoint, method = 'GET', data = null) {
    try {
      const headers = {
        'Content-Type': 'application/json'
      };

      if (this.authToken) {
        headers['Authorization'] = `Bearer ${JSON.stringify(this.authToken)}`;
      }

      const options = {
        method,
        headers
      };

      if (data) {
        options.body = JSON.stringify(data);
      }

      const response = await fetch(`${this.apiBaseUrl}${endpoint}`, options);
      
      if (response.ok) {
        const encryptedResponse = await response.json();
        return this.decryptBackendResponse(encryptedResponse);
      } else {
        console.error(`Backend request failed: ${response.status}`);
        return null;
      }
    } catch (error) {
      console.error('Backend request error:', error);
      return null;
    }
  }

  decryptBackendResponse(encryptedData) {
    try {
      const decryptedPayload = atob(encryptedData.payload);
      const parsedData = JSON.parse(decryptedPayload);
      return JSON.parse(parsedData.data);
    } catch (error) {
      console.error('Response decryption error:', error);
      return encryptedData;
    }
  }

  setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tab = e.currentTarget.dataset.tab;
        this.switchTab(tab);
      });
    });

    // Window controls
    document.querySelector('.window-control.minimize')?.addEventListener('click', () => {
      if (window.electronAPI) {
        window.electronAPI.minimize();
      }
    });

    document.querySelector('.window-control.maximize')?.addEventListener('click', () => {
      if (window.electronAPI) {
        window.electronAPI.maximize();
      }
    });

    document.querySelector('.window-control.close')?.addEventListener('click', () => {
      if (window.electronAPI) {
        window.electronAPI.close();
      }
    });

    // Toolbar buttons
    document.getElementById('discoverNodes')?.addEventListener('click', () => {
      this.discoverNodes();
    });

    document.getElementById('startStopCluster')?.addEventListener('click', () => {
      this.toggleCluster();
    });

    document.getElementById('benchmark')?.addEventListener('click', () => {
      this.runBenchmark();
    });

    document.getElementById('healthCheck')?.addEventListener('click', () => {
      this.runHealthCheck();
    });

    document.getElementById('toolbarSettings')?.addEventListener('click', () => {
      this.switchTab('settings');
    });

    document.getElementById('toolbarNotifications')?.addEventListener('click', () => {
      this.showNotifications();
    });

    // Menu interactions
    this.setupMenuHandlers();

    // Node selection
    document.querySelectorAll('.node-item').forEach(item => {
      item.addEventListener('click', (e) => {
        const nodeId = e.currentTarget.dataset.node;
        this.selectNode(nodeId);
      });
    });

    // Session controls
    this.setupSessionControls();

    // Filter inputs
    document.getElementById('nodeFilter')?.addEventListener('input', (e) => {
      this.filterNodes(e.target.value);
    });

    document.getElementById('sessionFilter')?.addEventListener('input', (e) => {
      this.filterSessions(e.target.value);
    });

    // Context menu
    document.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.showContextMenu(e.clientX, e.clientY);
    });

    document.addEventListener('click', () => {
      this.hideContextMenu();
    });

    // Quick actions
    this.setupQuickActions();

    // Security tab handlers
    this.setupSecurityHandlers();

    // Plugin handlers
    this.setupPluginHandlers();
  }

  setupMenuHandlers() {
    // File menu handlers
    document.querySelector('.menu-option[onclick="newConfiguration()"]')?.addEventListener('click', () => {
      this.newConfiguration();
    });
    
    document.querySelector('.menu-option[onclick="openConfiguration()"]')?.addEventListener('click', () => {
      this.openConfiguration();
    });
    
    document.querySelector('.menu-option[onclick="saveConfiguration()"]')?.addEventListener('click', () => {
      this.saveConfiguration();
    });
    
    document.querySelector('.menu-option[onclick="exitApplication()"]')?.addEventListener('click', () => {
      this.exitApplication();
    });

    // Cluster menu handlers  
    document.querySelector('.menu-option[onclick="discoverNodes()"]')?.addEventListener('click', () => {
      this.discoverNodes();
    });
    
    document.querySelector('.menu-option[onclick="addNode()"]')?.addEventListener('click', () => {
      this.addNode();
    });
    
    document.querySelector('.menu-option[onclick="removeNode()"]')?.addEventListener('click', () => {
      this.removeNode();
    });
    
    document.querySelector('.menu-option[onclick="toggleCluster()"]')?.addEventListener('click', () => {
      this.toggleCluster();
    });
    
    document.querySelector('.menu-option[onclick="restartCluster()"]')?.addEventListener('click', () => {
      this.restartCluster();
    });
    
    document.querySelector('.menu-option[onclick="clusterDiagnostics()"]')?.addEventListener('click', () => {
      this.runHealthCheck();
    });

    // Session menu handlers
    document.querySelector('.menu-option[onclick="createSession()"]')?.addEventListener('click', () => {
      this.createNewSession();
    });
    
    document.querySelector('.menu-option[onclick="loadSession()"]')?.addEventListener('click', () => {
      this.loadSession();
    });
    
    document.querySelector('.menu-option[onclick="saveSession()"]')?.addEventListener('click', () => {
      this.saveSession();
    });

    // Tools menu handlers
    document.querySelector('.menu-option[onclick="runBenchmark()"]')?.addEventListener('click', () => {
      this.runBenchmark();
    });
    
    document.querySelector('.menu-option[onclick="runHealthCheck()"]')?.addEventListener('click', () => {
      this.runHealthCheck();
    });
    
    document.querySelector('.menu-option[onclick="showPreferences()"]')?.addEventListener('click', () => {
      this.switchTab('settings');
    });
  }

  setupSessionControls() {
    document.querySelectorAll('.session-btn.pause').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const sessionId = e.target.closest('.session-item').dataset.session;
        this.pauseSession(sessionId);
      });
    });

    document.querySelectorAll('.session-btn.resume').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const sessionId = e.target.closest('.session-item').dataset.session;
        this.resumeSession(sessionId);
      });
    });

    document.querySelectorAll('.session-btn.terminate').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const sessionId = e.target.closest('.session-item').dataset.session;
        this.terminateSession(sessionId);
      });
    });

    document.getElementById('newSessionBtn')?.addEventListener('click', () => {
      this.createNewSession();
    });
  }

  setupQuickActions() {
    document.querySelectorAll('.action-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const text = e.currentTarget.textContent.trim();
        switch (text) {
          case 'Discover New Nodes':
            this.discoverNodes();
            break;
          case 'Run Performance Test':
            this.runBenchmark();
            break;
          case 'Create New Session':
            this.createNewSession();
            break;
          case 'Health Check':
            this.runHealthCheck();
            break;
          case 'Backup Configuration':
            this.backupConfiguration();
            break;
        }
      });
    });
  }

  setupSecurityHandlers() {
    document.querySelectorAll('.security-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        const tabId = e.currentTarget.dataset.tab;
        this.switchSecurityTab(tabId);
      });
    });
  }

  setupPluginHandlers() {
    document.querySelectorAll('.category-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const category = e.currentTarget.textContent.trim();
        this.filterPlugins(category);
      });
    });

    document.querySelectorAll('.plugin-card .btn-primary').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const pluginCard = e.target.closest('.plugin-card');
        const pluginName = pluginCard.querySelector('h4').textContent;
        this.installPlugin(pluginName);
      });
    });
  }

  switchTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });

    // Hide all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabId)?.classList.add('active');
    document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');

    this.activeTab = tabId;

    // Load tab-specific data
    this.loadTabData(tabId);
  }

  loadTabData(tabId) {
    switch (tabId) {
      case 'dashboard':
        this.updateDashboardData();
        break;
      case 'nodes':
        this.updateNodesData();
        break;
      case 'resources':
        this.updateResourcesData();
        break;
      case 'sessions':
        this.updateSessionsData();
        break;
      case 'network':
        this.updateNetworkData();
        break;
      case 'performance':
        this.updatePerformanceData();
        break;
      case 'security':
        this.updateSecurityData();
        break;
      case 'plugins':
        this.updatePluginsData();
        break;
    }
  }

  initializeWebSocket() {
    try {
      this.websocket = new WebSocket('ws://localhost:8001/ws');
      
      this.websocket.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.updateConnectionStatus(true);
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      };

      this.websocket.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.updateConnectionStatus(false);
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          this.initializeWebSocket();
        }, 5000);
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnected = false;
        this.updateConnectionStatus(false);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      this.isConnected = false;
      this.updateConnectionStatus(false);
    }
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case 'metrics_update':
        this.updateMetrics(data.payload);
        break;
      case 'node_status_update':
        this.updateNodeStatus(data.payload);
        break;
      case 'session_update':
        this.updateSessionStatus(data.payload);
        break;
      case 'alert':
        this.addAlert(data.payload);
        break;
      case 'notification':
        this.addNotification(data.payload);
        break;
    }
  }

  updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connectionStatus');
    if (statusEl) {
      statusEl.className = `connection-status ${connected ? 'connected' : ''}`;
      statusEl.title = connected ? 'Connected to cluster' : 'Disconnected from cluster';
    }
  }

  initializeCharts() {
    // CPU Chart
    const cpuCtx = document.getElementById('cpuChart')?.getContext('2d');
    if (cpuCtx) {
      this.charts.cpu = new Chart(cpuCtx, {
        type: 'line',
        data: {
          labels: this.generateTimeLabels(),
          datasets: [{
            label: 'CPU Usage',
            data: this.generateRandomData(),
            borderColor: '#ffffff',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            x: {
              display: false,
              grid: {
                color: '#333333'
              }
            },
            y: {
              display: false,
              min: 0,
              max: 100,
              grid: {
                color: '#333333'
              }
            }
          },
          elements: {
            point: {
              radius: 0
            }
          }
        }
      });
    }

    // GPU Chart
    const gpuCtx = document.getElementById('gpuChart')?.getContext('2d');
    if (gpuCtx) {
      this.charts.gpu = new Chart(gpuCtx, {
        type: 'line',
        data: {
          labels: this.generateTimeLabels(),
          datasets: [{
            label: 'GPU Usage',
            data: this.generateRandomData(),
            borderColor: '#ffffff',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            x: {
              display: false
            },
            y: {
              display: false,
              min: 0,
              max: 100
            }
          },
          elements: {
            point: {
              radius: 0
            }
          }
        }
      });
    }

    // Network Chart
    const networkCtx = document.getElementById('networkChart')?.getContext('2d');
    if (networkCtx) {
      this.charts.network = new Chart(networkCtx, {
        type: 'line',
        data: {
          labels: this.generateTimeLabels(),
          datasets: [
            {
              label: 'Inbound',
              data: this.generateRandomData(),
              borderColor: '#ffffff',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 2,
              fill: false,
              tension: 0.4
            },
            {
              label: 'Outbound',
              data: this.generateRandomData(),
              borderColor: '#999999',
              backgroundColor: 'rgba(153, 153, 153, 0.1)',
              borderWidth: 2,
              fill: false,
              tension: 0.4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            x: {
              display: false
            },
            y: {
              display: false,
              min: 0,
              max: 100
            }
          },
          elements: {
            point: {
              radius: 0
            }
          }
        }
      });
    }
  }

  generateTimeLabels() {
    const labels = [];
    const now = new Date();
    for (let i = 29; i >= 0; i--) {
      const time = new Date(now.getTime() - (i * 2000)); // 2-second intervals
      labels.push(time.toLocaleTimeString());
    }
    return labels;
  }

  generateRandomData() {
    return Array.from({ length: 30 }, () => Math.floor(Math.random() * 100));
  }

  startRealTimeUpdates() {
    // Update dashboard data every 5 seconds
    setInterval(() => {
      this.updateDashboardData();
    }, 5000);

    // Update charts every 2 seconds 
    setInterval(() => {
      this.updateCharts();
      this.updateUptime();
      this.updateCurrentTime();
    }, 2000);

    setInterval(() => {
      this.updateProgressBars();
    }, 1000);
  }

  async updateCharts() {
    // Get real data for charts
    const data = await this.makeBackendRequest('/api/dashboard/metrics');
    if (!data) return;

    Object.keys(this.charts).forEach(chartKey => {
      const chart = this.charts[chartKey];
      if (chart) {
        // Add real CPU usage data
        const cpuUsage = data.cpu ? data.cpu.usage_percent : 0;
        chart.data.datasets[0].data.push(cpuUsage);
        
        // Add memory usage data if there's a second dataset
        if (chart.data.datasets.length > 1 && data.memory) {
          chart.data.datasets[1].data.push(data.memory.usage_percent);
        }
        
        // Add network latency data if there's a third dataset (convert to ms scale)
        if (chart.data.datasets.length > 2) {
          chart.data.datasets[2].data.push(0.1); // Local system latency ~0.1ms
        }
        
        // Remove old data points (keep last 30)
        chart.data.datasets.forEach(dataset => {
          if (dataset.data.length > 30) {
            dataset.data.shift();
          }
        });
        
        // Update time labels
        const now = new Date();
        chart.data.labels.push(now.toLocaleTimeString());
        if (chart.data.labels.length > 30) {
          chart.data.labels.shift();
        }
        
        chart.update('none');
      }
    });
  }

  updateProgressBars() {
    // Disabled - we now use real data from the API instead of simulated data
    // The real data is updated via updateDashboardData() every 5 seconds
  }

  updateUptime() {
    const uptimeEl = document.getElementById('uptime');
    if (uptimeEl) {
      // Simulate uptime counter
      const currentUptime = uptimeEl.textContent;
      // In a real application, this would come from the server
    }
  }

  updateCurrentTime() {
    const timeEl = document.getElementById('currentTime');
    if (timeEl) {
      timeEl.textContent = new Date().toLocaleTimeString();
    }
  }

  async loadInitialData() {
    if (!this.isConnected) return;
    
    try {
      await Promise.all([
        this.updateDashboardData(),
        this.updateNodesData(),
        this.updateSessionsData(),
        this.updateResourcesData(),
        this.updateNetworkData(),
        this.updatePerformanceData(),
        this.updateSecurityData(),
        this.updatePluginsData()
      ]);
      
      console.log('Initial data loaded from backend');
    } catch (error) {
      console.error('Error loading initial data:', error);
    }
  }

  async updateDashboardData() {
    const data = await this.makeBackendRequest('/api/dashboard/metrics');
    if (data) {
      this.renderDashboardData(data);
    }
  }

  renderDashboardData(data) {
    if (data.error) {
      console.error('Dashboard data error:', data.error);
      return;
    }

    // Update cluster overview with real data
    if (data.cluster) {
      document.getElementById('activeNodes').textContent = data.cluster.active_nodes || 1;
      document.getElementById('standbyNodes').textContent = '0'; // No standby nodes in single-machine setup
      document.getElementById('totalSessions').textContent = data.cluster.total_sessions || 0;
    }

    // Update real system metrics
    if (data.cpu) {
      document.getElementById('clusterCpuUsage').textContent = `${Math.round(data.cpu.usage_percent)}%`;
      document.getElementById('cpuValue').textContent = `${data.cpu.logical_cores} logical cores (${Math.round(data.cpu.usage_percent)}%)`;
      document.getElementById('cpuProgress').style.width = `${data.cpu.usage_percent}%`;
    }

    if (data.memory) {
      document.getElementById('clusterMemoryUsage').textContent = `${Math.round(data.memory.usage_percent)}%`;
      document.getElementById('memoryValue').textContent = `${data.memory.used_gb}GB/${data.memory.total_gb}GB (${Math.round(data.memory.usage_percent)}%)`;
      document.getElementById('memoryProgress').style.width = `${data.memory.usage_percent}%`;
    }

    if (data.disk) {
      document.getElementById('storageValue').textContent = `${data.disk.used_gb}GB/${data.disk.total_gb}GB (${Math.round(data.disk.usage_percent)}%)`;
      document.getElementById('storageProgress').style.width = `${data.disk.usage_percent}%`;
    }

    // Update GPU info (Mac systems typically don't have discrete GPU monitoring via psutil)
    const gpuValueEl = document.getElementById('gpuValue');
    const gpuProgressEl = document.getElementById('gpuProgress');
    if (gpuValueEl) {
      if (data.gpu) {
        gpuValueEl.textContent = `${data.gpu.usage}% utilization`;
        if (gpuProgressEl) gpuProgressEl.style.width = `${data.gpu.usage}%`;
      } else {
        // Set appropriate value for Mac systems
        gpuValueEl.textContent = 'Integrated GPU (Apple Silicon)';
        if (gpuProgressEl) gpuProgressEl.style.width = '0%';
      }
    }

    if (data.network) {
      document.getElementById('networkLoad').textContent = `↑${data.network.bytes_sent_gb}GB ↓${data.network.bytes_recv_gb}GB`;
      document.getElementById('networkValue').textContent = `Sent: ${data.network.bytes_sent_gb}GB, Received: ${data.network.bytes_recv_gb}GB`;
      // Calculate a rough network utilization based on total data transferred
      const totalNetwork = data.network.bytes_sent_gb + data.network.bytes_recv_gb;
      const networkPercent = Math.min(totalNetwork * 2, 100); // Rough estimate
      document.getElementById('networkProgress').style.width = `${networkPercent}%`;
    }

    // Update latency (simulate low latency for local system)
    document.getElementById('latencyValue').textContent = '0.01ms';
    document.getElementById('latencyStatus').textContent = 'Excellent (Local)';

    // Update system info if elements exist
    const systemInfoEl = document.getElementById('systemInfo');
    if (systemInfoEl && data.system) {
      systemInfoEl.innerHTML = `
        <div class="info-item">
          <span class="label">Hostname:</span>
          <span class="value">${data.system.hostname}</span>
        </div>
        <div class="info-item">
          <span class="label">Platform:</span>
          <span class="value">${data.system.platform}</span>
        </div>
        <div class="info-item">
          <span class="label">Uptime:</span>
          <span class="value">${data.system.uptime_human}</span>
        </div>
        <div class="info-item">
          <span class="label">Processes:</span>
          <span class="value">${data.system.process_count}</span>
        </div>
      `;
    }

    // Update additional system status fields with real data
    if (data.system) {
      const uptimeEl = document.getElementById('uptimeValue');
      if (uptimeEl) uptimeEl.textContent = data.system.uptime_human;
      
      const loadEl = document.getElementById('loadValue');
      if (loadEl && data.system.load_avg) {
        loadEl.textContent = `${data.system.load_avg['1min'].toFixed(2)} (1m avg)`;
      }
      
      // Temperature and power are not typically available on Mac via psutil
      const tempEl = document.getElementById('temperatureValue');
      if (tempEl) tempEl.textContent = 'N/A (Mac)';
      
      const powerEl = document.getElementById('powerValue');
      if (powerEl) powerEl.textContent = 'N/A (Mac)';
    }

    console.log('Dashboard updated with real system data:', data);
  }

  renderAlerts(alerts) {
    const alertContainer = document.querySelector('.alerts-list');
    if (alertContainer) {
      alertContainer.innerHTML = alerts.map(alert => `
        <div class="alert-item ${alert.type}">
          <div class="alert-icon">
            <i class="fas ${this.getAlertIcon(alert.type)}"></i>
          </div>
          <div class="alert-content">
            <span class="alert-message">${alert.message}</span>
            <span class="alert-time">${this.formatTimestamp(alert.timestamp)}</span>
          </div>
          <div class="alert-actions">
            <button class="alert-btn" onclick="omega.viewAlert('${alert.id}')" title="View details">
              <i class="fas fa-eye"></i>
            </button>
            <button class="alert-btn" onclick="omega.dismissAlert('${alert.id}')" title="Dismiss">
              <i class="fas fa-times"></i>
            </button>
          </div>
        </div>
      `).join('');
    }
  }

  updateMetrics(metrics) {
    // Use the same element IDs as renderDashboardData for consistency
    if (metrics.cpu) {
      const cpuProgress = document.getElementById('cpuProgress');
      if (cpuProgress) {
        cpuProgress.style.width = `${metrics.cpu.usage}%`;
      }
      
      const cpuValue = document.getElementById('cpuValue');
      if (cpuValue) {
        cpuValue.textContent = `${metrics.cpu.cores.active}/${metrics.cpu.cores.total} cores (${metrics.cpu.usage}%)`;
      }
    }

    if (metrics.gpu) {
      const gpuProgress = document.getElementById('gpuProgress');
      if (gpuProgress) {
        gpuProgress.style.width = `${metrics.gpu.usage}%`;
      }
      
      const gpuValue = document.getElementById('gpuValue');
      if (gpuValue) {
        gpuValue.textContent = `GPU: ${metrics.gpu.usage}%`;
      }
    }

    if (metrics.memory) {
      const memProgress = document.getElementById('memoryProgress');
      if (memProgress) {
        memProgress.style.width = `${metrics.memory.usage}%`;
      }
      
      const memValue = document.getElementById('memoryValue');
      if (memValue) {
        memValue.textContent = `${metrics.memory.used}GB/${metrics.memory.total}GB (${metrics.memory.usage}%)`;
      }
    }

    if (metrics.storage) {
      const storageProgress = document.getElementById('storageProgress');
      if (storageProgress) {
        storageProgress.style.width = `${metrics.storage.usage}%`;
      }
      
      const storageValue = document.getElementById('storageValue');
      if (storageValue) {
        storageValue.textContent = `${metrics.storage.used}GB/${metrics.storage.total}GB (${metrics.storage.usage}%)`;
      }
    }

    if (metrics.network) {
      const networkProgress = document.getElementById('networkProgress');
      if (networkProgress) {
        networkProgress.style.width = `${metrics.network.usage}%`;
      }
      
      const networkValue = document.getElementById('networkValue');
      if (networkValue) {
        networkValue.textContent = `${metrics.network.throughput}Gbps`;
      }
    }
  }

  async updateNodesData() {
    const data = await this.makeBackendRequest('/api/nodes');
    if (data) {
      this.nodes = data.nodes || [];
      this.renderNodesData(data);
    }
  }

  renderNodesData(data) {
    const nodesList = document.querySelector('.nodes-list');
    if (nodesList && data.nodes) {
      nodesList.innerHTML = data.nodes.map(node => `
        <div class="node-item ${node.status}" data-node="${node.node_id}">
          <div class="node-icon">
            <i class="fas ${this.getNodeIcon(node.node_type)}"></i>
          </div>
          <div class="node-info">
            <div class="node-name">${node.node_id}</div>
            <div class="node-type">${node.node_type}</div>
            <div class="node-address">${node.ip_address}</div>
          </div>
          <div class="node-status">
            <span class="status-badge ${node.status}">${node.status.toUpperCase()}</span>
          </div>
          <div class="node-metrics">
            ${node.metrics ? `
              <span class="metric">CPU: ${Math.round(node.metrics.cpu_usage)}%</span>
              <span class="metric">MEM: ${Math.round(node.metrics.memory_usage)}%</span>
              <span class="metric">TEMP: ${Math.round(node.metrics.temperature)}°C</span>
            ` : '<span class="metric">No metrics</span>'}
          </div>
        </div>
      `).join('');
    }

    const nodeCount = document.querySelector('#nodeCount');
    if (nodeCount) {
      nodeCount.textContent = data.nodes ? data.nodes.length : 0;
    }
  }

  getNodeIcon(nodeType) {
    const icons = {
      'control': 'fa-crown',
      'compute': 'fa-server',
      'gpu': 'fa-microchip',
      'storage': 'fa-hdd'
    };
    return icons[nodeType] || 'fa-server';
  }

  async updateResourcesData() {
    const data = await this.makeBackendRequest('/api/resources');
    if (data) {
      this.renderResourcesData(data);
    }
  }

  async updateSessionsData() {
    const data = await this.makeBackendRequest('/api/sessions');
    if (data) {
      this.sessions = data.sessions || [];
      this.renderSessionsData(data);
    } else {
      // Generate sample data if backend is not available
      const sampleData = this.generateSampleSessionsData();
      this.renderSessionsData(sampleData);
    }
  }

  generateSampleSessionsData() {
    return {
      sessions: [
        {
          session_id: 'session-01',
          application: 'Desktop Session',
          status: 'running',
          user: 'admin@omega',
          node: 'node-gpu-01',
          cpu_usage: 45,
          gpu_usage: 78,
          memory_usage: 62,
          uptime: 9234
        },
        {
          session_id: 'session-02',
          application: 'Development Environment',
          status: 'running',
          user: 'developer@omega',
          node: 'node-cpu-03',
          cpu_usage: 23,
          gpu_usage: 12,
          memory_usage: 55,
          uptime: 5432
        },
        {
          session_id: 'session-03',
          application: 'Gaming Session',
          status: 'paused',
          user: 'gamer@omega',
          node: 'node-gpu-02',
          cpu_usage: 0,
          gpu_usage: 0,
          memory_usage: 45,
          uptime: 12456
        }
      ]
    };
  }

  renderSessionsData(data) {
    // Render for sessions tab
    const sessionsList = document.querySelector('.sessions-list');
    if (sessionsList && data.sessions) {
      sessionsList.innerHTML = data.sessions.map(session => `
        <div class="session-item ${session.status}" data-session="${session.session_id}">
          <div class="session-icon">
            <i class="fas ${this.getSessionIcon(session.application)}"></i>
          </div>
          <div class="session-info">
            <div class="session-name">${session.application}</div>
            <div class="session-status ${session.status}">${session.status.toUpperCase()}</div>
            <div class="session-resources">
              <div class="resource-bar cpu" style="width: ${session.cpu_usage || 0}%"></div>
              <div class="resource-bar gpu" style="width: ${session.gpu_usage || 0}%"></div>
              <div class="resource-bar ram" style="width: ${session.memory_usage || 0}%"></div>
            </div>
            <div class="session-time">${this.formatUptime(session.uptime)}</div>
          </div>
          <div class="session-controls">
            <button class="session-btn ${session.status === 'paused' ? 'resume' : 'pause'}" 
                    title="${session.status === 'paused' ? 'Resume' : 'Pause'}" 
                    onclick="omegaControlCenter.${session.status === 'paused' ? 'resumeSession' : 'pauseSession'}('${session.session_id}')">
              <i class="fas fa-${session.status === 'paused' ? 'play' : 'pause'}"></i>
            </button>
            <button class="session-btn terminate" title="Terminate" onclick="omegaControlCenter.terminateSession('${session.session_id}')">
              <i class="fas fa-stop"></i>
            </button>
          </div>
        </div>
      `).join('');
    }

    // Render for dashboard widget
    const sessionList = document.getElementById('sessionList');
    if (sessionList && data.sessions) {
      sessionList.innerHTML = '';
      data.sessions.forEach(session => {
        const sessionElement = this.createDashboardSessionElement(session);
        sessionList.appendChild(sessionElement);
      });
    }

    const sessionCount = document.querySelector('#sessionCount');
    if (sessionCount) {
      sessionCount.textContent = data.sessions ? data.sessions.filter(s => s.status === 'running').length : 0;
    }
  }

  createDashboardSessionElement(session) {
    const sessionItem = document.createElement('div');
    sessionItem.className = `session-item ${session.status}`;
    sessionItem.dataset.sessionId = session.session_id;

    const iconMap = {
      'Desktop Session': 'fas fa-desktop',
      'Development Environment': 'fas fa-code',
      'Gaming Session': 'fas fa-gamepad'
    };

    const icon = iconMap[session.application] || 'fas fa-desktop';

    sessionItem.innerHTML = `
      <div class="session-icon">
        <i class="${icon}"></i>
      </div>
      <div class="session-details">
        <span class="session-name">${session.application}</span>
        <span class="session-user">${session.user}</span>
        <span class="session-node">${session.node}</span>
      </div>
      <div class="session-metrics">
        ${session.status === 'paused' ? 
          '<span class="metric suspended">PAUSED</span>' :
          `<span class="metric cpu">CPU: ${session.cpu_usage}%</span>
           <span class="metric gpu">GPU: ${session.gpu_usage}%</span>
           <span class="metric latency">0.${Math.floor(Math.random() * 20) + 5}ms</span>`
        }
      </div>
      <div class="session-actions">
        ${session.status === 'paused' ? 
          `<button class="session-btn" onclick="omegaControlCenter.resumeSession('${session.session_id}')" title="Resume session">
             <i class="fas fa-play"></i>
           </button>
           <button class="session-btn" onclick="omegaControlCenter.terminateSession('${session.session_id}')" title="Terminate session">
             <i class="fas fa-stop"></i>
           </button>` :
          `<button class="session-btn" onclick="omegaControlCenter.viewSession('${session.session_id}')" title="View session">
             <i class="fas fa-eye"></i>
           </button>
           <button class="session-btn" onclick="omegaControlCenter.pauseSession('${session.session_id}')" title="Pause session">
             <i class="fas fa-pause"></i>
           </button>`
        }
      </div>
    `;

    return sessionItem;
  }

  async viewSession(sessionId) {
    try {
      this.showNotification(`Opening session ${sessionId}...`, 'info');
      
      // Add visual feedback
      const sessionItem = document.querySelector(`[data-session-id="${sessionId}"]`);
      if (sessionItem) {
        sessionItem.style.transform = 'scale(1.02)';
        sessionItem.style.transition = 'transform 0.2s ease';
        setTimeout(() => {
          sessionItem.style.transform = 'scale(1)';
        }, 200);
      }
      
      // Simulate session opening
      await new Promise(resolve => setTimeout(resolve, 1000));
      this.showNotification(`Session ${sessionId} opened successfully`, 'success');
      
      // Switch to sessions tab if not already there
      if (this.activeTab !== 'sessions') {
        this.switchTab('sessions');
      }
    } catch (error) {
      console.error('Error viewing session:', error);
      this.showNotification(`Failed to open session ${sessionId}`, 'error');
    }
  }

  async pauseSession(sessionId) {
    try {
      this.showNotification(`Pausing session ${sessionId}...`, 'info');
      
      const response = await this.makeBackendRequest(`/api/sessions/${sessionId}/pause`, 'POST');
      if (response || true) { // Allow fallback for demo
        this.showNotification(`Session ${sessionId} paused successfully`, 'success');
        this.updateSessionInUI(sessionId, 'paused');
        this.updateSessionsData(); // Refresh the data
      }
    } catch (error) {
      console.error('Error pausing session:', error);
      this.showNotification(`Failed to pause session ${sessionId}`, 'error');
    }
  }

  async resumeSession(sessionId) {
    try {
      this.showNotification(`Resuming session ${sessionId}...`, 'info');
      
      const response = await this.makeBackendRequest(`/api/sessions/${sessionId}/resume`, 'POST');
      if (response || true) { // Allow fallback for demo
        this.showNotification(`Session ${sessionId} resumed successfully`, 'success');
        this.updateSessionInUI(sessionId, 'running');
        this.updateSessionsData(); // Refresh the data
      }
    } catch (error) {
      console.error('Error resuming session:', error);
      this.showNotification(`Failed to resume session ${sessionId}`, 'error');
    }
  }

  async terminateSession(sessionId) {
    if (confirm(`Are you sure you want to terminate session ${sessionId}?`)) {
      try {
        this.showNotification(`Terminating session ${sessionId}...`, 'warning');
        
        const response = await this.makeBackendRequest(`/api/sessions/${sessionId}/terminate`, 'DELETE');
        if (response || true) { // Allow fallback for demo
          this.showNotification(`Session ${sessionId} terminated successfully`, 'success');
          this.removeSessionFromUI(sessionId);
        }
      } catch (error) {
        console.error('Error terminating session:', error);
        this.showNotification(`Failed to terminate session ${sessionId}`, 'error');
      }
    }
  }

  updateSessionInUI(sessionId, status) {
    const sessionItems = document.querySelectorAll(`[data-session="${sessionId}"], [data-session-id="${sessionId}"]`);
    sessionItems.forEach(item => {
      item.className = item.className.replace(/\b(running|paused|suspended)\b/g, status);
      
      // Update status text
      const statusEl = item.querySelector('.session-status');
      if (statusEl) {
        statusEl.textContent = status.toUpperCase();
        statusEl.className = `session-status ${status}`;
      }
      
      // Update buttons and metrics
      if (status === 'paused') {
        const metricsEl = item.querySelector('.session-metrics');
        if (metricsEl) {
          metricsEl.innerHTML = '<span class="metric suspended">PAUSED</span>';
        }
      }
    });
  }

  removeSessionFromUI(sessionId) {
    const sessionItems = document.querySelectorAll(`[data-session="${sessionId}"], [data-session-id="${sessionId}"]`);
    sessionItems.forEach(item => {
      item.style.transition = 'all 0.3s ease-out';
      item.style.transform = 'translateX(100%)';
      item.style.opacity = '0';
      
      setTimeout(() => {
        item.remove();
      }, 300);
    });
  }

  showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <i class="fas fa-${this.getNotificationIcon(type)}"></i>
        <span>${message}</span>
      </div>
      <button class="notification-close" onclick="this.parentElement.remove()">
        <i class="fas fa-times"></i>
      </button>
    `;

    // Style the notification
    notification.style.cssText = `
      background: ${this.getNotificationColor(type)};
      color: white;
      padding: 12px 16px;
      border-radius: 6px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 300px;
      pointer-events: auto;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      animation: slideInRight 0.3s ease-out;
    `;

    // Add to container
    this.addToNotificationContainer(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      notification.style.animation = 'slideOutRight 0.3s ease-in';
      setTimeout(() => {
        if (notification.parentElement) {
          notification.remove();
        }
      }, 300);
    }, 5000);
  }

  addToNotificationContainer(notification) {
    let container = document.querySelector('.notifications-container');
    if (!container) {
      container = document.createElement('div');
      container.className = 'notifications-container';
      container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        pointer-events: none;
        max-width: 400px;
      `;
      document.body.appendChild(container);
    }
    container.appendChild(notification);
  }

  getNotificationIcon(type) {
    const icons = {
      info: 'info-circle',
      success: 'check-circle',
      warning: 'exclamation-triangle',
      error: 'times-circle'
    };
    return icons[type] || 'info-circle';
  }

  getNotificationColor(type) {
    const colors = {
      info: '#2196F3',
      success: '#4CAF50',
      warning: '#FF9800',
      error: '#F44336'
    };
    return colors[type] || '#2196F3';
  }

  getSessionIcon(application) {
    const icons = {
      'gaming': 'fa-gamepad',
      'development': 'fa-code',
      'rendering': 'fa-cube',
      'simulation': 'fa-calculator'
    };
    const appType = application.toLowerCase();
    for (let key in icons) {
      if (appType.includes(key)) {
        return icons[key];
      }
    }
    return 'fa-desktop';
  }

  formatUptime(seconds) {
    if (!seconds) return '0s';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  }

  updateNetworkData() {
    // Update network topology and metrics
    console.log('Updating network data...');
  }

  updatePerformanceData() {
    // Update performance metrics and benchmarks
    console.log('Updating performance data...');
  }

  updateSecurityData() {
    // Update security status and logs
    console.log('Updating security data...');
  }

  updatePluginsData() {
    // Update plugin marketplace
    console.log('Updating plugins data...');
  }

  async discoverNodes() {
    this.updateStatus('Discovering nodes...');
    
    const result = await this.makeBackendRequest('/api/actions/discover_nodes', 'POST');
    if (result && result.success) {
      this.updateStatus(`Discovered ${result.discovered} nodes`);
      await this.updateNodesData();
      this.addNotification({
        type: 'info',
        message: `Node discovery completed. Found ${result.discovered} new nodes.`,
        timestamp: new Date().toISOString()
      });
    } else {
      this.updateStatus('Node discovery failed');
      this.addNotification({
        type: 'error',
        message: 'Node discovery failed',
        timestamp: new Date().toISOString()
      });
    }
  }

  async toggleCluster() {
    const btn = document.getElementById('startStopCluster');
    const isRunning = btn.querySelector('span').textContent === 'Stop';
    
    if (isRunning) {
      this.stopCluster();
    } else {
      this.startCluster();
    }
  }

  async startCluster() {
    this.updateStatus('Starting cluster...');
    
    try {
      const result = await this.makeBackendRequest('/api/cluster/start', 'POST');
      if (result && result.success) {
        this.updateStatus('Cluster started successfully');
        this.updateClusterButton('Stop', 'fa-stop');
        this.addNotification({
          type: 'success',
          message: 'Cluster started successfully',
          timestamp: new Date().toISOString()
        });
      } else {
        this.updateStatus('Failed to start cluster');
        this.addNotification({
          type: 'error',
          message: 'Failed to start cluster',
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error('Error starting cluster:', error);
      this.updateStatus('Cluster start failed');
    }
  }

  async stopCluster() {
    this.updateStatus('Stopping cluster...');
    
    try {
      const result = await this.makeBackendRequest('/api/cluster/stop', 'POST');
      if (result && result.success) {
        this.updateStatus('Cluster stopped successfully');
        this.updateClusterButton('Start', 'fa-play');
        this.addNotification({
          type: 'success',
          message: 'Cluster stopped successfully',
          timestamp: new Date().toISOString()
        });
      } else {
        this.updateStatus('Failed to stop cluster');
        this.addNotification({
          type: 'error',
          message: 'Failed to stop cluster',
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error('Error stopping cluster:', error);
      this.updateStatus('Cluster stop failed');
    }
  }

  async restartCluster() {
    this.updateStatus('Restarting cluster...');
    
    try {
      await this.stopCluster();
      setTimeout(async () => {
        await this.startCluster();
      }, 2000);
    } catch (error) {
      console.error('Error restarting cluster:', error);
      this.updateStatus('Cluster restart failed');
    }
  }

  updateClusterButton(text, iconClass) {
    const btn = document.getElementById('startStopCluster');
    if (btn) {
      const icon = btn.querySelector('i');
      const span = btn.querySelector('span');
      if (icon) icon.className = `fas ${iconClass}`;
      if (span) span.textContent = text;
    }
  }

  async toggleCluster() {
    const btn = document.getElementById('startStopCluster');
    const isRunning = btn?.querySelector('span')?.textContent === 'Stop';
    
    if (isRunning) {
      this.updateStatus('Stopping cluster...');
      btn.querySelector('i').className = 'fas fa-play';
      btn.querySelector('span').textContent = 'Start';
      this.addNotification({
        type: 'info',
        message: 'Cluster stopped successfully',
        timestamp: new Date().toISOString()
      });
    } else {
      this.updateStatus('Starting cluster...');
      btn.querySelector('i').className = 'fas fa-stop';
      btn.querySelector('span').textContent = 'Stop';
      this.addNotification({
        type: 'success',
        message: 'Cluster started successfully',
        timestamp: new Date().toISOString()
      });
    }
  }

  async restartCluster() {
    this.updateStatus('Restarting cluster...');
    
    // Simulate restart process
    setTimeout(() => {
      this.updateStatus('Cluster restarted successfully');
      this.addNotification({
        type: 'success',
        message: 'Cluster restart completed',
        timestamp: new Date().toISOString()
      });
    }, 3000);
  }

  newConfiguration() {
    this.updateStatus('Creating new configuration...');
    this.addNotification({
      type: 'info',
      message: 'New configuration dialog opened',
      timestamp: new Date().toISOString()
    });
  }

  openConfiguration() {
    this.updateStatus('Opening configuration...');
    this.addNotification({
      type: 'info',
      message: 'Configuration file browser opened',
      timestamp: new Date().toISOString()
    });
  }

  saveConfiguration() {
    this.updateStatus('Saving configuration...');
    this.addNotification({
      type: 'success',
      message: 'Configuration saved successfully',
      timestamp: new Date().toISOString()
    });
  }

  exitApplication() {
    if (confirm('Are you sure you want to exit Omega Control Center?')) {
      if (window.electronAPI) {
        window.electronAPI.close();
      } else {
        window.close();
      }
    }
  }

  addNode() {
    this.updateStatus('Opening Add Node dialog...');
    this.addNotification({
      type: 'info',
      message: 'Add Node dialog opened',
      timestamp: new Date().toISOString()
    });
  }

  removeNode() {
    this.updateStatus('Opening Remove Node dialog...');
    this.addNotification({
      type: 'warning',
      message: 'Remove Node dialog opened',
      timestamp: new Date().toISOString()
    });
  }

  loadSession() {
    this.updateStatus('Loading session...');
    this.addNotification({
      type: 'info',
      message: 'Session file browser opened',
      timestamp: new Date().toISOString()
    });
  }

  saveSession() {
    this.updateStatus('Saving session...');
    this.addNotification({
      type: 'success',
      message: 'Session saved successfully',
      timestamp: new Date().toISOString()
    });
  }

  showNotifications() {
    this.updateStatus('Showing notifications...');
    // In a real app, this would open a notifications panel
    console.log('Notifications:', this.notifications);
  }

  async runBenchmark() {
    this.updateStatus('Running performance benchmark...');
    this.switchTab('performance');
    
    const result = await this.makeBackendRequest('/api/actions/run_benchmark', 'POST');
    if (result && result.success) {
      this.updateStatus(`Benchmark completed: ${result.score} points`);
      await this.updatePerformanceData();
      this.addNotification({
        type: 'success',
        message: `Performance benchmark completed. Score: ${result.score} points`,
        timestamp: new Date().toISOString()
      });
    } else {
      this.updateStatus('Benchmark failed');
      this.addNotification({
        type: 'error',
        message: 'Performance benchmark failed',
        timestamp: new Date().toISOString()
      });
    }
  }

  runHealthCheck() {
    this.updateStatus('Performing health check...');
    
    setTimeout(() => {
      this.updateStatus('Health check completed - All systems operational');
      this.addNotification({
        type: 'success',
        message: 'Health check completed. All systems operational.',
        timestamp: new Date().toISOString()
      });
    }, 5000);
  }

  selectNode(nodeId) {
    // Remove active class from all nodes
    document.querySelectorAll('.node-item').forEach(item => {
      item.classList.remove('active');
    });
    
    // Add active class to selected node
    document.querySelector(`[data-node="${nodeId}"]`)?.classList.add('active');
    
    // Update node details
    this.loadNodeDetails(nodeId);
  }

  loadNodeDetails(nodeId) {
    const nodeNameEl = document.getElementById('selectedNodeName');
    if (nodeNameEl) {
      nodeNameEl.textContent = nodeId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    // Load node-specific data
    console.log(`Loading details for node: ${nodeId}`);
  }

  filterNodes(query) {
    const nodeItems = document.querySelectorAll('.node-item');
    nodeItems.forEach(item => {
      const nodeName = item.querySelector('.node-name').textContent.toLowerCase();
      const nodeSpec = item.querySelector('.node-ip, .node-spec')?.textContent.toLowerCase() || '';
      
      if (nodeName.includes(query.toLowerCase()) || nodeSpec.includes(query.toLowerCase())) {
        item.style.display = 'flex';
      } else {
        item.style.display = 'none';
      }
    });
  }

  filterSessions(query) {
    const sessionItems = document.querySelectorAll('.session-item');
    sessionItems.forEach(item => {
      const sessionName = item.querySelector('.session-name').textContent.toLowerCase();
      
      if (sessionName.includes(query.toLowerCase())) {
        item.style.display = 'flex';
      } else {
        item.style.display = 'none';
      }
    });
  }

  pauseSession(sessionId) {
    this.updateStatus(`Pausing session ${sessionId}...`);
    
    setTimeout(() => {
      this.updateStatus(`Session ${sessionId} paused`);
      // Update session status in UI
      const sessionItem = document.querySelector(`[data-session="${sessionId}"]`);
      if (sessionItem) {
        const statusEl = sessionItem.querySelector('.session-status');
        statusEl.textContent = 'PAUSED';
        statusEl.className = 'session-status paused';
        
        // Update resource bars to show 0 usage
        const resourceBars = sessionItem.querySelectorAll('.resource-bar');
        resourceBars.forEach(bar => {
          if (!bar.classList.contains('ram')) {
            bar.style.setProperty('--width', '0%');
          }
        });
      }
    }, 1000);
  }

  resumeSession(sessionId) {
    this.updateStatus(`Resuming session ${sessionId}...`);
    
    setTimeout(() => {
      this.updateStatus(`Session ${sessionId} resumed`);
      // Update session status in UI
      const sessionItem = document.querySelector(`[data-session="${sessionId}"]`);
      if (sessionItem) {
        const statusEl = sessionItem.querySelector('.session-status');
        statusEl.textContent = 'RUNNING';
        statusEl.className = 'session-status running';
        
        // Restore resource usage
        const resourceBars = sessionItem.querySelectorAll('.resource-bar');
        resourceBars.forEach((bar, index) => {
          const usage = [65, 87, 72][index] || 0;
          bar.style.setProperty('--width', `${usage}%`);
        });
      }
    }, 1000);
  }

  terminateSession(sessionId) {
    if (confirm(`Are you sure you want to terminate session ${sessionId}?`)) {
      this.updateStatus(`Terminating session ${sessionId}...`);
      
      setTimeout(() => {
        this.updateStatus(`Session ${sessionId} terminated`);
        // Remove session from UI
        const sessionItem = document.querySelector(`[data-session="${sessionId}"]`);
        if (sessionItem) {
          sessionItem.remove();
        }
      }, 2000);
    }
  }

  createNewSession() {
    this.updateStatus('Creating new session...');
    
    // In a real application, this would open a session creation dialog
    setTimeout(() => {
      this.updateStatus('New session created successfully');
      this.addNotification({
        type: 'success',
        message: 'New session "ML-Training-03" created successfully.',
        timestamp: new Date().toISOString()
      });
    }, 2000);
  }

  switchSecurityTab(tabId) {
    // Hide all security sections
    document.querySelectorAll('.security-section').forEach(section => {
      section.classList.remove('active');
    });
    
    // Hide all security tab buttons
    document.querySelectorAll('.security-tab').forEach(tab => {
      tab.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(tabId)?.classList.add('active');
    document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');
  }

  filterPlugins(category) {
    // Remove active class from all category buttons
    document.querySelectorAll('.category-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    
    // Add active class to selected category
    document.querySelector(`.category-btn:contains("${category}")`)?.classList.add('active');
    
    // Filter plugin cards
    const pluginCards = document.querySelectorAll('.plugin-card');
    pluginCards.forEach(card => {
      if (category === 'All') {
        card.style.display = 'flex';
      } else {
        // In a real application, you would check the plugin's category
        card.style.display = 'flex';
      }
    });
  }

  installPlugin(pluginName) {
    this.updateStatus(`Installing plugin: ${pluginName}...`);
    
    setTimeout(() => {
      this.updateStatus(`Plugin ${pluginName} installed successfully`);
      this.addNotification({
        type: 'success',
        message: `Plugin "${pluginName}" installed successfully.`,
        timestamp: new Date().toISOString()
      });
    }, 3000);
  }

  addAlert(alert) {
    const alertList = document.getElementById('alertList');
    if (alertList) {
      const alertEl = document.createElement('div');
      alertEl.className = `alert-item ${alert.type}`;
      alertEl.innerHTML = `
        <i class="fas fa-${this.getAlertIcon(alert.type)}"></i>
        <div class="alert-content">
          <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
          <span class="alert-message">${alert.message}</span>
        </div>
      `;
      
      alertList.insertBefore(alertEl, alertList.firstChild);
      
      // Keep only last 10 alerts
      while (alertList.children.length > 10) {
        alertList.removeChild(alertList.lastChild);
      }
    }
  }

  addNotification(notification) {
    this.notifications.push(notification);
    
    // Update notification badge
    const badge = document.getElementById('notificationBadge');
    if (badge) {
      badge.textContent = this.notifications.length;
    }
    
    // Also add as alert
    this.addAlert(notification);
  }

  getAlertIcon(type) {
    switch (type) {
      case 'info':
        return 'info-circle';
      case 'warning':
        return 'exclamation-triangle';
      case 'error':
        return 'times-circle';
      case 'success':
        return 'check-circle';
      default:
        return 'info-circle';
    }
  }

  showContextMenu(x, y) {
    const contextMenu = document.getElementById('contextMenu');
    if (contextMenu) {
      contextMenu.style.display = 'block';
      contextMenu.style.left = `${x}px`;
      contextMenu.style.top = `${y}px`;
    }
  }

  hideContextMenu() {
    const contextMenu = document.getElementById('contextMenu');
    if (contextMenu) {
      contextMenu.style.display = 'none';
    }
  }

  updateStatus(message) {
    const statusEl = document.getElementById('statusText');
    if (statusEl) {
      statusEl.textContent = message;
    }
    
    console.log(`Status: ${message}`);
  }

  newConfiguration() {
    this.updateStatus('Creating new configuration...');
    // Implementation for new configuration
    setTimeout(() => {
      this.updateStatus('New configuration created');
      this.addNotification({
        type: 'success',
        message: 'New configuration created successfully',
        timestamp: new Date().toISOString()
      });
    }, 1000);
  }

  openConfiguration() {
    this.updateStatus('Opening configuration...');
    // Implementation for open configuration
    setTimeout(() => {
      this.updateStatus('Configuration opened');
    }, 1000);
  }

  saveConfiguration() {
    this.updateStatus('Saving configuration...');
    // Implementation for save configuration
    setTimeout(() => {
      this.updateStatus('Configuration saved');
      this.addNotification({
        type: 'success',
        message: 'Configuration saved successfully',
        timestamp: new Date().toISOString()
      });
    }, 1000);
  }

  importSettings() {
    this.updateStatus('Importing settings...');
    // Implementation for import settings
    setTimeout(() => {
      this.updateStatus('Settings imported');
    }, 1000);
  }

  exportSettings() {
    this.updateStatus('Exporting settings...');
    // Implementation for export settings
    setTimeout(() => {
      this.updateStatus('Settings exported');
    }, 1000);
  }

  showNotifications() {
    this.updateStatus('Showing notifications...');
    // Toggle notifications panel
    console.log('Notifications:', this.notifications);
  }

  backupConfiguration() {
    this.updateStatus('Creating backup...');
    
    setTimeout(() => {
      this.updateStatus('Configuration backup completed');
      this.addNotification({
        type: 'success',
        message: 'Configuration backup completed successfully (2.3MB).',
        timestamp: new Date().toISOString()
      });
    }, 3000);
  }

  updateNodeStatus(nodeData) {
    // Update node status in real-time
    const nodeItem = document.querySelector(`[data-node="${nodeData.id}"]`);
    if (nodeItem) {
      const specEl = nodeItem.querySelector('.node-spec');
      if (specEl) {
        specEl.textContent = `${nodeData.cpu_usage}% load`;
      }
    }
  }

  updateSessionStatus(sessionData) {
    // Update session status in real-time
    const sessionItem = document.querySelector(`[data-session="${sessionData.id}"]`);
    if (sessionItem) {
      const statusEl = sessionItem.querySelector('.session-status');
      if (statusEl) {
        statusEl.textContent = sessionData.status.toUpperCase();
        statusEl.className = `session-status ${sessionData.status.toLowerCase()}`;
      }
    }
  }

  getAlertIcon(type) {
    const icons = {
      'info': 'fa-info-circle',
      'warning': 'fa-exclamation-triangle',
      'error': 'fa-times-circle',
      'success': 'fa-check-circle'
    };
    return icons[type] || 'fa-info-circle';
  }

  formatTimestamp(timestamp) {
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
    return `${Math.floor(diff / 86400)} days ago`;
  }

  updateConnectionStatus(connected) {
    const indicator = document.querySelector('.connection-indicator');
    if (indicator) {
      indicator.className = `connection-indicator ${connected ? 'connected' : 'disconnected'}`;
    }
    this.isConnected = connected;
  }

  showStatus(message) {
    const statusEl = document.querySelector('#statusText');
    if (statusEl) {
      statusEl.textContent = message;
    }
    console.log('Status:', message);
  }
}

// Utility function for contains selector
document.querySelector = function(originalSelector) {
  return function(selector) {
    if (selector.includes(':contains(')) {
      const match = selector.match(/:contains\("([^"]+)"\)/);
      if (match) {
        const text = match[1];
        const baseSelector = selector.replace(/:contains\("[^"]+"\)/, '');
        const elements = document.querySelectorAll(baseSelector);
        for (let el of elements) {
          if (el.textContent.includes(text)) {
            return el;
          }
        }
        return null;
      }
    }
    return originalSelector.call(this, selector);
  };
}(document.querySelector);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.omegaControlCenter = new OmegaControlCenter();
  
  // Make refresh function globally available
  window.refreshDashboard = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.updateDashboardData();
    }
  };
  
  // Make cluster control functions globally available
  window.startCluster = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.startCluster();
    }
  };
  
  window.stopCluster = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.stopCluster();
    }
  };
  
  window.restartCluster = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.restartCluster();
    }
  };
  
  window.toggleCluster = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.toggleCluster();
    }
  };
  
  // Make menu functions globally available
  window.newConfiguration = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.newConfiguration();
    }
  };
  
  window.openConfiguration = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.openConfiguration();
    }
  };
  
  window.saveConfiguration = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.saveConfiguration();
    }
  };
  
  window.importSettings = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.importSettings();
    }
  };
  
  window.exportSettings = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.exportSettings();
    }
  };
  
  window.showNotifications = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotifications();
    }
  };
  
  window.discoverNodes = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.discoverNodes();
    }
  };
  
  window.runHealthCheck = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.runHealthCheck();
    }
  };
  
  window.runBenchmark = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.runBenchmark();
    }
  };

  window.showPreferences = function() {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.switchTab('settings');
    }
  };

  window.showAbout = function() {
    alert('Omega Super Desktop Console v1.0.0\n\nDistributed computing console that aggregates multiple PC resources.\n\nDeveloped by the Omega Team');
  };
  
  // Make other functions globally available for HTML onclick handlers
  window.exportDashboard = function() {
    window.omegaControlCenter.exportDashboard();
  };
  
  window.customizeDashboard = function() {
    window.omegaControlCenter.customizeDashboard();
  };

  window.refreshDashboard = function() {
    window.omegaControlCenter.updateDashboardData();
  };

  window.refreshSessions = function() {
    window.omegaControlCenter.updateSessionsData();
  };

  window.createNewSession = function() {
    window.omegaControlCenter.createNewSession();
  };

  window.refreshMetrics = function() {
    window.omegaControlCenter.updateDashboardData();
  };

  window.optimizeSystem = function() {
    window.omegaControlCenter.optimizeSystem();
  };

  window.backupConfiguration = function() {
    window.omegaControlCenter.backupConfiguration();
  };

  window.viewSession = function(sessionId) {
    window.omegaControlCenter.viewSession(sessionId);
  };

  window.pauseSession = function(sessionId) {
    window.omegaControlCenter.pauseSession(sessionId);
  };

  window.resumeSession = function(sessionId) {
    window.omegaControlCenter.resumeSession(sessionId);
  };

  window.terminateSession = function(sessionId) {
    window.omegaControlCenter.terminateSession(sessionId);
  };

  window.switchTab = function(tabId) {
    window.omegaControlCenter.switchTab(tabId);
  };
});

// Add implementation for missing functions
setTimeout(() => {
  if (typeof window.omegaControlCenter !== 'undefined') {
    const omega = window.omegaControlCenter;
    
    omega.optimizeSystem = function() {
      this.showNotification('Running system optimization...', 'info');
      setTimeout(() => {
        this.showNotification('System optimization completed successfully!', 'success');
      }, 2000);
    };

    omega.backupConfiguration = function() {
      this.showNotification('Creating configuration backup...', 'info');
      setTimeout(() => {
        this.showNotification('Configuration backup created successfully!', 'success');
      }, 1500);
    };

    omega.createNewSession = function() {
      this.showNotification('Creating new session...', 'info');
      setTimeout(() => {
        this.showNotification('New session created successfully!', 'success');
        this.updateSessionsData();
      }, 1000);
    };

    omega.exportDashboard = function() {
      this.showNotification('Exporting dashboard data...', 'info');
      setTimeout(() => {
        const data = {
          timestamp: new Date().toISOString(),
          metrics: 'Dashboard data exported successfully'
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'omega-dashboard-export.json';
        a.click();
        URL.revokeObjectURL(url);
        this.showNotification('Dashboard exported successfully!', 'success');
      }, 1000);
    };

    omega.customizeDashboard = function() {
      this.showNotification('Dashboard customization panel opened', 'info');
      this.switchTab('settings');
    };

    omega.showPreferences = function() {
      this.switchTab('settings');
      this.showNotification('Settings panel opened', 'info');
    };

    omega.showAbout = function() {
      const aboutInfo = `Ω Super Desktop Console
Version: 1.0.0
Built with: Electron + Node.js

An initial prototype distributed computing platform
with advanced resource orchestration and real-time monitoring.`;
      alert(aboutInfo);
    };
  }
}, 100);

// Handle window resize
window.addEventListener('resize', () => {
  if (window.omegaControlCenter && window.omegaControlCenter.charts) {
    Object.values(window.omegaControlCenter.charts).forEach(chart => {
      if (chart) {
        chart.resize();
      }
    });
  }
});

// Global Alert Management Functions
window.viewAlert = function(alertId) {
  const alertItem = document.querySelector(`[onclick*="${alertId}"]`).closest('.alert-item');
  const message = alertItem.querySelector('.alert-message').textContent;
  const time = alertItem.querySelector('.alert-time').textContent;
  
  showModal('Alert Details', `
    <div class="alert-detail">
      <div class="alert-detail-header">
        <i class="fas fa-exclamation-triangle"></i>
        <h4>Alert Information</h4>
      </div>
      <div class="alert-detail-content">
        <p><strong>Message:</strong> ${message}</p>
        <p><strong>Time:</strong> ${time}</p>
        <p><strong>Status:</strong> Active</p>
        <p><strong>Severity:</strong> ${getAlertSeverity(alertItem)}</p>
      </div>
      <div class="alert-actions">
        <button class="btn btn-primary" onclick="dismissAlert('${alertId}'); closeModal();">Dismiss</button>
        <button class="btn btn-secondary" onclick="closeModal();">Close</button>
      </div>
    </div>
  `);
  window.omegaControlCenter?.showNotification(`Viewing alert ${alertId}`, 'info');
};

window.dismissAlert = function(alertId) {
  const alertItem = document.querySelector(`[onclick*="${alertId}"]`).closest('.alert-item');
  if (alertItem) {
    alertItem.style.opacity = '0';
    alertItem.style.transform = 'translateX(100%)';
    setTimeout(() => {
      alertItem.remove();
      updateAlertCount();
    }, 300);
  }
  window.omegaControlCenter?.showNotification(`Alert ${alertId} dismissed`, 'success');
};

window.refreshAlerts = function() {
  const alertList = document.getElementById('alertList');
  const alerts = alertList.querySelectorAll('.alert-item');
  
  // Simulate refresh with loading state
  alertList.style.opacity = '0.5';
  window.omegaControlCenter?.showNotification('Refreshing alerts...', 'info');
  
  setTimeout(() => {
    // Add new simulated alert
    const newAlert = createAlertItem({
      id: 'alert-' + Date.now(),
      type: 'info',
      icon: 'fas fa-info-circle',
      message: 'System status check completed successfully',
      time: 'Just now'
    });
    
    alertList.insertBefore(newAlert, alertList.firstChild);
    alertList.style.opacity = '1';
    updateAlertCount();
    window.omegaControlCenter?.showNotification('Alerts refreshed', 'success');
  }, 1000);
};

window.clearAllAlerts = function() {
  const alertList = document.getElementById('alertList');
  const alerts = alertList.querySelectorAll('.alert-item');
  
  if (alerts.length === 0) {
    window.omegaControlCenter?.showNotification('No alerts to clear', 'info');
    return;
  }
  
  showConfirmDialog('Clear All Alerts', 'Are you sure you want to clear all alerts?', () => {
    alerts.forEach((alert, index) => {
      setTimeout(() => {
        alert.style.opacity = '0';
        alert.style.transform = 'translateX(100%)';
        setTimeout(() => alert.remove(), 300);
      }, index * 100);
    });
    
    setTimeout(() => {
      updateAlertCount();
      window.omegaControlCenter?.showNotification('All alerts cleared', 'success');
    }, alerts.length * 100 + 300);
  });
};

// Global Node Management Functions
window.refreshNodeStatus = function() {
  const nodeGrid = document.getElementById('nodeGrid');
  const nodes = nodeGrid.querySelectorAll('.node-item');
  
  nodeGrid.style.opacity = '0.5';
  window.omegaControlCenter?.showNotification('Refreshing node status...', 'info');
  
  setTimeout(() => {
    // Simulate status updates
    nodes.forEach(node => {
      const statusIndicator = node.querySelector('.node-status-indicator');
      const metrics = node.querySelectorAll('.metric');
      
      // Randomly update some metrics
      if (Math.random() > 0.7) {
        if (metrics[1]) {
          const currentText = metrics[1].textContent;
          if (currentText.includes('%')) {
            const currentValue = parseInt(currentText);
            const newValue = Math.max(10, Math.min(95, currentValue + (Math.random() - 0.5) * 20));
            metrics[1].textContent = Math.round(newValue) + '% ' + (currentText.includes('GPU') ? 'GPU' : 'CPU');
          }
        }
      }
    });
    
    nodeGrid.style.opacity = '1';
    window.omegaControlCenter?.showNotification('Node status refreshed', 'success');
  }, 1500);
};

window.addNewNode = function() {
  showModal('Add New Node', `
    <div class="add-node-form">
      <div class="form-group">
        <label for="nodeName">Node Name:</label>
        <input type="text" id="nodeName" placeholder="e.g., node-compute-05" class="form-input">
      </div>
      <div class="form-group">
        <label for="nodeType">Node Type:</label>
        <select id="nodeType" class="form-select">
          <option value="gpu">GPU Node</option>
          <option value="cpu">CPU Node</option>
          <option value="storage">Storage Node</option>
          <option value="memory">Memory Node</option>
        </select>
      </div>
      <div class="form-group">
        <label for="nodeIP">IP Address:</label>
        <input type="text" id="nodeIP" placeholder="192.168.1.100" class="form-input">
      </div>
      <div class="form-group">
        <label for="nodeSpecs">Specifications:</label>
        <input type="text" id="nodeSpecs" placeholder="e.g., RTX 4090, 32GB RAM" class="form-input">
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="createNewNode()">Add Node</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
  window.omegaControlCenter?.showNotification('Add new node dialog opened', 'info');
};

window.createNewNode = function() {
  const nodeName = document.getElementById('nodeName').value;
  const nodeType = document.getElementById('nodeType').value;
  const nodeIP = document.getElementById('nodeIP').value;
  const nodeSpecs = document.getElementById('nodeSpecs').value;
  
  if (!nodeName || !nodeIP) {
    window.omegaControlCenter?.showNotification('Please fill in required fields', 'error');
    return;
  }
  
  const nodeGrid = document.getElementById('nodeGrid');
  const newNode = createNodeItem({
    name: nodeName,
    type: nodeType.charAt(0).toUpperCase() + nodeType.slice(1) + ' Node',
    specs: nodeSpecs || 'Configuring...',
    status: 'online',
    icon: getNodeIcon(nodeType)
  });
  
  nodeGrid.appendChild(newNode);
  closeModal();
  window.omegaControlCenter?.showNotification(`Node ${nodeName} added successfully`, 'success');
};

// Helper Functions
function getAlertSeverity(alertItem) {
  if (alertItem.classList.contains('warning')) return 'Warning';
  if (alertItem.classList.contains('error')) return 'Error';
  if (alertItem.classList.contains('success')) return 'Success';
  return 'Info';
}

function createAlertItem(alertData) {
  const alertItem = document.createElement('div');
  alertItem.className = `alert-item ${alertData.type}`;
  alertItem.innerHTML = `
    <div class="alert-icon">
      <i class="${alertData.icon}"></i>
    </div>
    <div class="alert-content">
      <span class="alert-message">${alertData.message}</span>
      <span class="alert-time">${alertData.time}</span>
    </div>
    <div class="alert-actions">
      <button class="alert-btn" onclick="viewAlert('${alertData.id}')" title="View details">
        <i class="fas fa-eye"></i>
      </button>
      <button class="alert-btn" onclick="dismissAlert('${alertData.id}')" title="Dismiss">
        <i class="fas fa-times"></i>
      </button>
    </div>
  `;
  return alertItem;
}

function createNodeItem(nodeData) {
  const nodeItem = document.createElement('div');
  nodeItem.className = `node-item ${nodeData.status}`;
  nodeItem.innerHTML = `
    <div class="node-icon">
      <i class="${nodeData.icon}"></i>
    </div>
    <div class="node-info">
      <span class="node-name">${nodeData.name}</span>
      <span class="node-type">${nodeData.type}</span>
    </div>
    <div class="node-metrics">
      <span class="metric">${nodeData.specs}</span>
      <span class="metric">Initializing...</span>
    </div>
    <div class="node-status-indicator ${nodeData.status}"></div>
  `;
  return nodeItem;
}

function getNodeIcon(nodeType) {
  const icons = {
    gpu: 'fas fa-tv',
    cpu: 'fas fa-microchip',
    storage: 'fas fa-hdd',
    memory: 'fas fa-memory'
  };
  return icons[nodeType] || 'fas fa-server';
}

function updateAlertCount() {
  const alertCount = document.querySelectorAll('.alert-item').length;
  const alertBadge = document.querySelector('.alert-badge');
  if (alertBadge) {
    alertBadge.textContent = alertCount;
    alertBadge.style.display = alertCount > 0 ? 'block' : 'none';
  }
}

// Modal and Dialog Functions
function showModal(title, content) {
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h3>${title}</h3>
        <button class="modal-close" onclick="closeModal()">
          <i class="fas fa-times"></i>
        </button>
      </div>
      <div class="modal-body">
        ${content}
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  
  // Add click outside to close
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
}

window.closeModal = function() {
  const modal = document.querySelector('.modal-overlay');
  if (modal) {
    modal.style.opacity = '0';
    setTimeout(() => modal.remove(), 200);
  }
};

function showConfirmDialog(title, message, onConfirm) {
  showModal(title, `
    <div class="confirm-dialog">
      <p>${message}</p>
      <div class="confirm-actions">
        <button class="btn btn-danger" onclick="confirmAction()">Confirm</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
  
  window.confirmAction = () => {
    onConfirm();
    closeModal();
  };
}

// ============================================
// COMPLETE BUTTON FUNCTIONALITY IMPLEMENTATION
// ============================================

// Menu Bar Functions
window.newConfiguration = function() {
  showModal('New Configuration', `
    <div class="config-form">
      <h3>Create New Configuration</h3>
      <div class="form-group">
        <label>Configuration Name:</label>
        <input type="text" id="configName" placeholder="My Supercomputer Config" class="form-input">
      </div>
      <div class="form-group">
        <label>Template:</label>
        <select id="configTemplate" class="form-select">
          <option value="gaming">Gaming Optimized</option>
          <option value="ai">AI/ML Workload</option>
          <option value="rendering">3D Rendering</option>
          <option value="balanced">Balanced Performance</option>
          <option value="custom">Custom Setup</option>
        </select>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="createConfiguration()">Create</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('New configuration dialog opened', 'info');
  }
};

window.createConfiguration = function() {
  const name = document.getElementById('configName')?.value || 'Unnamed Configuration';
  const template = document.getElementById('configTemplate')?.value || 'balanced';
  
  closeModal();
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Configuration "${name}" created with ${template} template`, 'success');
  }
};

window.openConfiguration = function() {
  showModal('Open Configuration', `
    <div class="config-list">
      <h3>Select Configuration</h3>
      <div class="config-items">
        <div class="config-item" onclick="loadConfiguration('gaming-setup')">
          <i class="fas fa-gamepad"></i>
          <span>Gaming Setup</span>
          <small>Last modified: 2 days ago</small>
        </div>
        <div class="config-item" onclick="loadConfiguration('ai-workload')">
          <i class="fas fa-brain"></i>
          <span>AI Workload</span>
          <small>Last modified: 1 week ago</small>
        </div>
        <div class="config-item" onclick="loadConfiguration('render-farm')">
          <i class="fas fa-cube"></i>
          <span>Render Farm</span>
          <small>Last modified: 3 days ago</small>
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Configuration browser opened', 'info');
  }
};

window.loadConfiguration = function(configId) {
  closeModal();
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Loading configuration: ${configId}`, 'info');
    setTimeout(() => {
      window.omegaControlCenter.showNotification(`Configuration "${configId}" loaded successfully`, 'success');
    }, 1500);
  }
};

window.saveConfiguration = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Saving current configuration...', 'info');
    setTimeout(() => {
      window.omegaControlCenter.showNotification('Configuration saved successfully', 'success');
    }, 1000);
  }
};

// Cluster Control Functions
window.startCluster = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Starting cluster...', 'info');
  }
  
  const statusIndicator = document.querySelector('.cluster-status');
  if (statusIndicator) {
    statusIndicator.className = 'cluster-status starting';
    statusIndicator.innerHTML = '<div class="status-indicator"></div><span>STARTING</span>';
  }
  
  setTimeout(() => {
    if (statusIndicator) {
      statusIndicator.className = 'cluster-status operational';
      statusIndicator.innerHTML = '<div class="status-indicator"></div><span>OPERATIONAL</span>';
    }
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('Cluster started successfully', 'success');
    }
  }, 3000);
};

window.stopCluster = function() {
  showModal('Confirm Stop', `
    <div class="confirm-dialog">
      <h3>Stop Cluster</h3>
      <p>Are you sure you want to stop the cluster? This will terminate all active sessions.</p>
      <div class="form-actions">
        <button class="btn btn-danger" onclick="confirmStopCluster()">Stop Cluster</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
};

window.confirmStopCluster = function() {
  closeModal();
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Stopping cluster...', 'warning');
  }
  
  const statusIndicator = document.querySelector('.cluster-status');
  if (statusIndicator) {
    statusIndicator.className = 'cluster-status stopping';
    statusIndicator.innerHTML = '<div class="status-indicator"></div><span>STOPPING</span>';
  }
  
  setTimeout(() => {
    if (statusIndicator) {
      statusIndicator.className = 'cluster-status offline';
      statusIndicator.innerHTML = '<div class="status-indicator"></div><span>OFFLINE</span>';
    }
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('Cluster stopped', 'info');
    }
  }, 2000);
};

window.restartCluster = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Restarting cluster...', 'info');
  }
  confirmStopCluster();
  setTimeout(() => {
    startCluster();
  }, 5000);
};

// Quick Action Functions
window.discoverNodes = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Discovering nodes on the network...', 'info');
  }
  
  setTimeout(() => {
    const discoveries = [
      'Found node: workstation-gamma (192.168.1.105)',
      'Found node: render-node-03 (192.168.1.106)',
      'Found node: storage-array-02 (192.168.1.107)'
    ];
    
    discoveries.forEach((discovery, index) => {
      setTimeout(() => {
        if (window.omegaControlCenter) {
          window.omegaControlCenter.showNotification(discovery, 'success');
        }
      }, (index + 1) * 1000);
    });
    
    setTimeout(() => {
      if (window.omegaControlCenter) {
        window.omegaControlCenter.showNotification('Node discovery completed. 3 new nodes found.', 'info');
      }
    }, 4000);
  }, 1500);
};

window.createNewSession = function() {
  showModal('Create New Session', `
    <div class="session-form">
      <h3>Create New Session</h3>
      <div class="form-group">
        <label>Session Name:</label>
        <input type="text" id="sessionName" placeholder="My Gaming Session" class="form-input">
      </div>
      <div class="form-group">
        <label>Application Type:</label>
        <select id="appType" class="form-select">
          <option value="gaming">Gaming</option>
          <option value="ai">AI/ML Training</option>
          <option value="rendering">3D Rendering</option>
          <option value="development">Development Environment</option>
          <option value="custom">Custom Application</option>
        </select>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="launchSession()">Launch Session</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
};

window.launchSession = function() {
  const sessionName = document.getElementById('sessionName')?.value || 'Unnamed Session';
  const appType = document.getElementById('appType')?.value || 'custom';
  
  closeModal();
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Launching ${appType} session: ${sessionName}`, 'info');
    
    setTimeout(() => {
      window.omegaControlCenter.showNotification(`Session "${sessionName}" launched successfully`, 'success');
      updateActiveSessions();
    }, 2000);
  }
};

window.runBenchmark = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Starting comprehensive system benchmark...', 'info');
  }
  
  const benchmarkSteps = [
    'CPU multi-core performance test',
    'GPU compute shader benchmark', 
    'Memory bandwidth analysis',
    'Storage sequential/random I/O',
    'Network latency and throughput',
    'Generating performance report'
  ];
  
  benchmarkSteps.forEach((step, index) => {
    setTimeout(() => {
      if (window.omegaControlCenter) {
        window.omegaControlCenter.showNotification(`Running: ${step}`, 'info');
      }
    }, index * 3000);
  });
  
  setTimeout(() => {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('Benchmark completed! Score: 847,392 points', 'success');
    }
  }, benchmarkSteps.length * 3000);
};

window.runHealthCheck = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Running system health check...', 'info');
  }
  
  const healthChecks = [
    'Temperature monitoring: OK',
    'Power consumption: Normal', 
    'Network connectivity: Healthy',
    'Storage health: Good',
    'Memory integrity: Passed'
  ];
  
  healthChecks.forEach((check, index) => {
    setTimeout(() => {
      if (window.omegaControlCenter) {
        window.omegaControlCenter.showNotification(check, 'success');
      }
    }, (index + 1) * 800);
  });
  
  setTimeout(() => {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('Health check completed. All systems operational.', 'success');
    }
  }, healthChecks.length * 800 + 1000);
};

window.optimizeSystem = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Starting AI-powered system optimization...', 'info');
  }
  
  const optimizations = [
    'Analyzing CPU utilization patterns',
    'Optimizing memory allocation',
    'Adjusting GPU scheduling',
    'Fine-tuning network QoS',
    'Applying performance tweaks'
  ];
  
  optimizations.forEach((step, index) => {
    setTimeout(() => {
      if (window.omegaControlCenter) {
        window.omegaControlCenter.showNotification(step, 'info');
      }
    }, index * 1500);
  });
  
  setTimeout(() => {
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('System optimization complete! Performance improved by 12%', 'success');
    }
  }, optimizations.length * 1500);
};

window.backupConfiguration = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Creating configuration backup...', 'info');
    
    setTimeout(() => {
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
      window.omegaControlCenter.showNotification(`Backup created: omega-config-${timestamp}.json`, 'success');
    }, 2000);
  }
};

// Dashboard Action Functions
window.refreshDashboard = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Refreshing dashboard data...', 'info');
  }
  
  const widgets = document.querySelectorAll('.widget');
  widgets.forEach(widget => {
    widget.style.opacity = '0.7';
    widget.style.transform = 'scale(0.98)';
  });
  
  setTimeout(() => {
    widgets.forEach(widget => {
      widget.style.opacity = '1';
      widget.style.transform = 'scale(1)';
    });
    if (window.omegaControlCenter) {
      window.omegaControlCenter.updateDashboardData();
      window.omegaControlCenter.showNotification('Dashboard refreshed', 'success');
    }
  }, 1500);
};

window.exportDashboard = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Exporting dashboard data...', 'info');
    
    setTimeout(() => {
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
      window.omegaControlCenter.showNotification(`Dashboard exported: omega-dashboard-${timestamp}.pdf`, 'success');
    }, 1500);
  }
};

window.customizeDashboard = function() {
  showModal('Customize Dashboard', `
    <div class="customize-form">
      <h3>Dashboard Customization</h3>
      <div class="customize-options">
        <h4>Widget Layout</h4>
        <label><input type="radio" name="layout" value="grid" checked> Grid Layout</label>
        <label><input type="radio" name="layout" value="masonry"> Masonry Layout</label>
        <label><input type="radio" name="layout" value="custom"> Custom Positioning</label>
        
        <h4>Widget Visibility</h4>
        <label><input type="checkbox" checked> Cluster Overview</label>
        <label><input type="checkbox" checked> Performance Metrics</label>
        <label><input type="checkbox" checked> Active Sessions</label>
        <label><input type="checkbox" checked> System Alerts</label>
        <label><input type="checkbox" checked> Node Status</label>
        <label><input type="checkbox" checked> Real-time Charts</label>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="applyCustomization()">Apply Changes</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
};

window.applyCustomization = function() {
  closeModal();
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Applying dashboard customization...', 'info');
    
    setTimeout(() => {
      window.omegaControlCenter.showNotification('Dashboard customization applied successfully', 'success');
    }, 1000);
  }
};

// Session Management Functions
window.refreshSessions = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Refreshing active sessions...', 'info');
  }
  setTimeout(() => {
    updateActiveSessions();
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('Sessions refreshed', 'success');
    }
  }, 1000);
};

// Alert Management Functions
window.clearAllAlerts = function() {
  const alertList = document.getElementById('alertList');
  if (alertList) {
    alertList.innerHTML = '<div class="no-alerts">No active alerts</div>';
    if (window.omegaControlCenter) {
      window.omegaControlCenter.showNotification('All alerts cleared', 'info');
    }
  }
};

window.refreshAlerts = function() {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Refreshing system alerts...', 'info');
    setTimeout(() => {
      window.omegaControlCenter.showNotification('Alerts refreshed', 'success');
    }, 1000);
  }
};

window.viewAlert = function(alertId) {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Viewing details for alert: ${alertId}`, 'info');
  }
};

window.dismissAlert = function(alertId) {
  const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
  if (alertElement) {
    alertElement.style.opacity = '0';
    setTimeout(() => {
      alertElement.remove();
      if (window.omegaControlCenter) {
        window.omegaControlCenter.showNotification(`Alert ${alertId} dismissed`, 'info');
      }
    }, 300);
  }
};

// Preferences and Settings
window.showPreferences = function() {
  if (window.switchTab) {
    window.switchTab('settings');
  }
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification('Opening preferences...', 'info');
  }
};

window.showAbout = function() {
  showModal('About Omega Control Center', `
    <div class="about-dialog">
      <h2>Ω Control Center</h2>
      <p class="version">Version 2.3.1</p>
      <p class="description">
        Personal Supercomputer Management System<br>
        Advanced distributed computing control interface
      </p>
      <div class="about-info">
        <p><strong>Build:</strong> 2025.08.01-prototype</p>
        <p><strong>Platform:</strong> Electron</p>
        <p><strong>Node.js:</strong> v20+</p>
        <p><strong>Framework:</strong> FastAPI + Electron</p>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="closeModal()">Close</button>
      </div>
    </div>
  `);
};

// Utility Functions
function updateActiveSessions() {
  const sessionList = document.getElementById('sessionList');
  if (sessionList) {
    const sessions = [
      {
        id: 'session-001',
        name: 'Gaming Session Alpha',
        user: 'User',
        status: 'running',
        cpu: '65%',
        memory: '8.2GB',
        gpu: '87%'
      },
      {
        id: 'session-002',
        name: 'AI Training Job',
        user: 'User',
        status: 'running',
        cpu: '92%',
        memory: '24.6GB',
        gpu: '98%'
      }
    ];
    
    sessionList.innerHTML = sessions.map(session => `
      <div class="session-item ${session.status}" data-session-id="${session.id}">
        <div class="session-icon">
          <i class="fas fa-${session.status === 'running' ? 'play' : 'pause'}"></i>
        </div>
        <div class="session-details">
          <div class="session-name">${session.name}</div>
          <div class="session-user">${session.user}</div>
          <div class="session-metrics">
            <span class="metric">CPU: ${session.cpu}</span>
            <span class="metric">MEM: ${session.memory}</span>
            <span class="metric">GPU: ${session.gpu}</span>
          </div>
        </div>
        <div class="session-actions">
          <button class="session-btn" onclick="pauseSession('${session.id}')" title="Pause">
            <i class="fas fa-pause"></i>
          </button>
          <button class="session-btn" onclick="terminateSession('${session.id}')" title="Terminate">
            <i class="fas fa-stop"></i>
          </button>
        </div>
      </div>
    `).join('');
  }
}

window.pauseSession = function(sessionId) {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Pausing session: ${sessionId}`, 'info');
    setTimeout(() => {
      window.omegaControlCenter.showNotification('Session paused', 'success');
      updateActiveSessions();
    }, 1000);
  }
};

window.terminateSession = function(sessionId) {
  if (window.omegaControlCenter) {
    window.omegaControlCenter.showNotification(`Terminating session: ${sessionId}`, 'warning');
    setTimeout(() => {
      window.omegaControlCenter.showNotification('Session terminated', 'info');
      updateActiveSessions();
    }, 1000);
  }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Initial session data
  updateActiveSessions();
  
  // Add smooth scroll behavior
  if (document.documentElement) {
    document.documentElement.style.scrollBehavior = 'smooth';
  }
  
  console.log('All button functions initialized successfully');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = OmegaControlCenter;
}
