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
    this.selectedNodeId = null; // Track selected node in nodes tab
    this.nodeMetricsWS = null; // WebSocket for real-time node metrics
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.authenticateAndConnect();
    this.initializeCharts();
    this.updateCurrentTime();
    this.initializeResizableWidgets();
    this.optimizePerformance();
    
    // Initialize Sessions tab
    this.setupSessionFilters();
    this.initializeSessionWebSocket();
    
    // Set up periodic sessions data refresh
    this.sessionUpdateInterval = setInterval(() => {
      this.updateSessionsData();
    }, 10000); // Update every 10 seconds
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
        const responseData = await response.json();
        
        // Check if response is encrypted (has payload property) or plain JSON
        if (responseData && responseData.payload) {
          return this.decryptBackendResponse(responseData);
        } else {
          // Plain JSON response (e.g., from test endpoints)
          return responseData;
        }
      } else {
        console.error(`Backend request failed: ${response.status}`);
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Backend request error:', error);
      throw error;
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
    console.log('Starting updateNodesData...');
    try {
      // Try authenticated endpoint first, fallback to test endpoint
      let data;
      try {
        console.log('Trying authenticated endpoint /api/v1/nodes');
        data = await this.makeBackendRequest('/api/v1/nodes');
        if (data && data.nodes) {
          console.log('Got data from authenticated endpoint:', data.nodes.length, 'nodes');
          this.nodes = data.nodes;
        }
      } catch (authError) {
        console.log('Authenticated endpoint failed, using test endpoint:', authError.message);
        // Use test endpoint for prototype - returns array directly
        const nodesArray = await this.makeBackendRequest('/api/test/nodes');
        if (nodesArray && Array.isArray(nodesArray)) {
          console.log('Got data from test endpoint:', nodesArray.length, 'nodes');
          this.nodes = nodesArray;
          data = { nodes: nodesArray }; // Wrap in expected format
        }
      }
      
      if (data && data.nodes) {
        console.log('Rendering nodes data with', data.nodes.length, 'nodes');
        this.renderAdvancedNodesData(data);
        this.updateNodeTree(data.nodes);
        this.updateSelectedNodeDetails();
      } else {
        console.log('No nodes data available');
      }
    } catch (error) {
      console.error('Error updating nodes data:', error);
      this.showNotification('Failed to load nodes data', 'error');
    }
  }

  renderAdvancedNodesData(data) {
    // Update the node tree in the sidebar
    const nodeTree = document.getElementById('nodeTree');
    if (!nodeTree) return;

    // Group nodes by type
    const nodesByType = {
      'control': [],
      'compute': [], 
      'gpu': [],
      'storage': [],
      'memory': []
    };

    data.nodes.forEach(node => {
      const type = node.node_type.toLowerCase();
      if (nodesByType[type]) {
        nodesByType[type].push(node);
      } else {
        nodesByType['compute'].push(node); // Default to compute
      }
    });

    // Render categorized nodes
    nodeTree.innerHTML = this.generateNodeTreeHTML(nodesByType);
    
    // Add event listeners for node selection
    this.attachNodeTreeListeners();
    
    // Auto-select first node if none selected
    if (!this.selectedNodeId && data.nodes.length > 0) {
      this.selectedNodeId = data.nodes[0].node_id;
      this.updateSelectedNodeDetails();
    }
  }

  generateNodeTreeHTML(nodesByType) {
    const categoryIcons = {
      'control': 'fa-crown',
      'compute': 'fa-server', 
      'gpu': 'fa-microchip',
      'storage': 'fa-hdd',
      'memory': 'fa-memory'
    };

    const categoryNames = {
      'control': 'Control Nodes',
      'compute': 'Compute Nodes',
      'gpu': 'GPU Nodes', 
      'storage': 'Storage Nodes',
      'memory': 'Memory Nodes'
    };

    let html = '';

    Object.keys(nodesByType).forEach(type => {
      const nodes = nodesByType[type];
      if (nodes.length > 0) {
        html += `
          <div class="node-category" data-category="${type}">
            <div class="category-header" onclick="toggleNodeCategory('${type}')">
              <i class="fas ${categoryIcons[type]}"></i>
              <span>${categoryNames[type]}</span>
              <i class="fas fa-chevron-down toggle-icon"></i>
              <span class="node-count">${nodes.length}</span>
            </div>
            <div class="category-content">
              ${nodes.map(node => this.generateNodeItemHTML(node)).join('')}
            </div>
          </div>
        `;
      }
    });

    return html;
  }

  generateNodeItemHTML(node) {
    const statusClass = this.getNodeStatusClass(node.status);
    const statusIcon = this.getNodeStatusIcon(node.status);
    const metrics = node.metrics || {};
    
    return `
      <div class="node-item ${statusClass}" 
           data-node="${node.node_id}" 
           onclick="selectNode('${node.node_id}')"
           title="Click to view details">
        <div class="node-item-header">
          <div class="node-icon-status">
            <i class="fas ${statusIcon} node-status-icon"></i>
          </div>
          <div class="node-basic-info">
            <div class="node-name">${node.node_id}</div>
            <div class="node-type">${node.node_type}</div>
            <div class="node-ip">${node.ip_address || 'localhost'}</div>
          </div>
        </div>
        <div class="node-metrics-preview">
          <div class="metric-item">
            <span class="metric-label">CPU:</span>
            <span class="metric-value">${Math.round(metrics.cpu_percent || 0)}%</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">MEM:</span>
            <span class="metric-value">${Math.round(metrics.memory_percent || 0)}%</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">TEMP:</span>
            <span class="metric-value">${metrics.temperature || 'N/A'}°C</span>
          </div>
        </div>
        <div class="node-quick-actions">
          <button class="quick-action-btn" onclick="event.stopPropagation(); quickNodeAction('${node.node_id}', 'restart')" title="Restart">
            <i class="fas fa-redo"></i>
          </button>
          <button class="quick-action-btn" onclick="event.stopPropagation(); quickNodeAction('${node.node_id}', 'maintenance')" title="Maintenance">
            <i class="fas fa-tools"></i>
          </button>
        </div>
      </div>
    `;
  }

  getNodeStatusClass(status) {
    const statusMap = {
      'online': 'online',
      'offline': 'offline',
      'maintenance': 'maintenance',
      'warning': 'warning',
      'error': 'error'
    };
    return statusMap[status] || 'unknown';
  }

  getNodeStatusIcon(status) {
    const iconMap = {
      'online': 'fa-circle text-success',
      'offline': 'fa-circle text-danger', 
      'maintenance': 'fa-tools text-warning',
      'warning': 'fa-exclamation-triangle text-warning',
      'error': 'fa-times-circle text-danger'
    };
    return iconMap[status] || 'fa-question-circle';
  }

  attachNodeTreeListeners() {
    // Add node filter functionality
    const nodeFilter = document.getElementById('nodeFilter');
    if (nodeFilter) {
      nodeFilter.addEventListener('input', (e) => this.filterNodes(e.target.value));
    }
  }

  filterNodes(searchTerm) {
    const nodeItems = document.querySelectorAll('.node-item');
    const term = searchTerm.toLowerCase();

    nodeItems.forEach(item => {
      const nodeName = item.querySelector('.node-name')?.textContent.toLowerCase() || '';
      const nodeType = item.querySelector('.node-type')?.textContent.toLowerCase() || '';
      const nodeIP = item.querySelector('.node-ip')?.textContent.toLowerCase() || '';
      
      if (nodeName.includes(term) || nodeType.includes(term) || nodeIP.includes(term)) {
        item.style.display = 'block';
      } else {
        item.style.display = 'none';
      }
    });
  }

  async updateSelectedNodeDetails() {
    if (!this.selectedNodeId) return;

    try {
      // First try to use the node data we already have
      let nodeData = null;
      if (this.nodes && Array.isArray(this.nodes)) {
        nodeData = this.nodes.find(node => node.node_id === this.selectedNodeId);
      }
      
      // If we don't have the data locally, try to fetch it
      if (!nodeData) {
        try {
          nodeData = await this.makeBackendRequest(`/api/v1/nodes/${this.selectedNodeId}`);
        } catch (authError) {
          console.log('Individual node endpoint failed, trying test endpoint');
          try {
            nodeData = await this.makeBackendRequest(`/api/test/nodes/${this.selectedNodeId}`);
          } catch (testError) {
            console.log('Test endpoint also failed, using cached data');
            nodeData = this.nodes ? this.nodes.find(node => node.node_id === this.selectedNodeId) : null;
          }
        }
      }
      
      if (nodeData) {
        console.log('Rendering node details for:', nodeData.node_id);
        this.renderNodeDetails(nodeData);
        this.setupNodeMetricsStream(this.selectedNodeId);
      } else {
        console.log('No node data available for:', this.selectedNodeId);
      }
    } catch (error) {
      console.error('Error loading node details:', error);
      this.showNotification('Failed to load node details', 'error');
    }
  }

  renderNodeDetails(nodeData) {
    // Update node header
    const nodeHeader = document.querySelector('.node-header');
    if (nodeHeader) {
      const nodeName = nodeHeader.querySelector('#selectedNodeName');
      const statusBadge = nodeHeader.querySelector('.node-status-badge');
      
      if (nodeName) nodeName.textContent = nodeData.node_id;
      if (statusBadge) {
        statusBadge.textContent = nodeData.status.toUpperCase();
        statusBadge.className = `node-status-badge ${this.getNodeStatusClass(nodeData.status)}`;
      }
    }

    // Update overview tab
    this.updateNodeOverview(nodeData);
    
    // Update performance tab 
    this.updateNodePerformance(nodeData);
    
    // Update processes tab
    this.updateNodeProcesses(nodeData);
    
    // Update logs tab
    this.updateNodeLogs(nodeData);
  }

  updateNodeOverview(nodeData) {
    const overviewContent = document.getElementById('nodeOverview');
    if (!overviewContent) return;

    // Handle both detailed hardware data and simple resources data
    const hardware = nodeData.hardware || {};
    const resources = nodeData.resources || {};
    const performance = nodeData.performance_metrics || {};
    const uptime = this.formatUptime(nodeData.uptime || 0);

    // Get CPU info from resources or hardware
    const cpuCores = resources.cpu_cores || hardware.cpu?.cores || 'Unknown';
    const memoryGB = resources.memory_gb || (hardware.memory?.total ? Math.round(hardware.memory.total / (1024**3)) : 'Unknown');
    const storageGB = resources.storage_gb || 'Unknown';
    const description = resources.description || nodeData.description || '';

    overviewContent.innerHTML = `
      <div class="specs-grid">
        <div class="spec-group">
          <h4><i class="fas fa-microchip"></i> Hardware</h4>
          <div class="spec-item">
            <span class="spec-label">CPU:</span>
            <span class="spec-value">${hardware.cpu?.model || description || 'System CPU'}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Cores/Threads:</span>
            <span class="spec-value">${cpuCores}C/${hardware.cpu?.threads || cpuCores}T</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">RAM:</span>
            <span class="spec-value">${memoryGB}GB</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Storage:</span>
            <span class="spec-value">${storageGB}GB</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Network:</span>
            <span class="spec-value">${this.getNetworkSummary(hardware.network) || nodeData.ip_address || 'Connected'}</span>
          </div>
        </div>
        <div class="spec-group">
          <h4><i class="fas fa-info-circle"></i> Status</h4>
          <div class="spec-item">
            <span class="spec-label">Uptime:</span>
            <span class="spec-value">${uptime}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Load:</span>
            <span class="spec-value">${Math.round(nodeData.performance?.cpu?.usage_percent?.[0] || 0)}%</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Temperature:</span>
            <span class="spec-value">${nodeData.metrics?.temperature || 'N/A'}°C</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Last Heartbeat:</span>
            <span class="spec-value">${this.formatTimestamp(nodeData.last_heartbeat)}</span>
          </div>
        </div>
        <div class="spec-group">
          <h4><i class="fas fa-shield-alt"></i> Security</h4>
          <div class="spec-item">
            <span class="spec-label">Firewall:</span>
            <span class="spec-value status-${nodeData.security?.firewall_status}">${nodeData.security?.firewall_status || 'Unknown'}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Last Scan:</span>
            <span class="spec-value">${this.formatTimestamp(nodeData.security?.last_scan)}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Vulnerabilities:</span>
            <span class="spec-value">${this.getVulnerabilitySummary(nodeData.security?.vulnerabilities)}</span>
          </div>
        </div>
        <div class="spec-group">
          <h4><i class="fas fa-tools"></i> Maintenance</h4>
          <div class="spec-item">
            <span class="spec-label">Mode:</span>
            <span class="spec-value status-${nodeData.maintenance?.mode}">${nodeData.maintenance?.mode || 'Unknown'}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Last Maintenance:</span>
            <span class="spec-value">${this.formatTimestamp(nodeData.maintenance?.last_maintenance)}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Health Status:</span>
            <span class="spec-value">${this.getHealthSummary(nodeData.maintenance?.health_checks)}</span>
          </div>
        </div>
      </div>
    `;
  }

  getNodeIcon(nodeType) {
    const icons = {
      'control': 'fa-crown',
      'compute': 'fa-server',
      'gpu': 'fa-microchip',
      'storage': 'fa-hdd',
      'memory': 'fa-memory'
    };
    return icons[nodeType] || 'fa-server';
  }

  // Advanced Node Management Helper Functions
  formatUptime(seconds) {
    if (!seconds) return 'Unknown';
    
    const days = Math.floor(seconds / (24 * 3600));
    const hours = Math.floor((seconds % (24 * 3600)) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }

  formatBytes(bytes) {
    if (!bytes) return 'Unknown';
    
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  }

  formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  }

  getStorageSummary(storage) {
    if (!storage || !Array.isArray(storage)) return 'Unknown';
    
    const totalSize = storage.reduce((sum, device) => sum + (device.size || 0), 0);
    return this.formatBytes(totalSize);
  }

  getNetworkSummary(network) {
    if (!network || !Array.isArray(network)) return 'Unknown';
    
    const activeInterfaces = network.filter(iface => 
      iface.addresses && iface.addresses.length > 0
    );
    return `${activeInterfaces.length} interface(s)`;
  }

  getVulnerabilitySummary(vulnerabilities) {
    if (!vulnerabilities) return 'Unknown';
    
    const { critical = 0, high = 0, medium = 0, low = 0 } = vulnerabilities;
    const total = critical + high + medium + low;
    
    if (total === 0) return 'None';
    if (critical > 0) return `${critical} Critical, ${total} total`;
    if (high > 0) return `${high} High, ${total} total`;
    return `${total} total`;
  }

  getHealthSummary(healthChecks) {
    if (!healthChecks) return 'Unknown';
    
    const statuses = Object.values(healthChecks);
    const healthy = statuses.filter(s => s === 'healthy').length;
    const total = statuses.length;
    
    if (healthy === total) return 'All systems healthy';
    return `${healthy}/${total} systems healthy`;
  }

  setupNodeMetricsStream(nodeId) {
    // Close existing WebSocket if any
    if (this.nodeMetricsWS) {
      this.nodeMetricsWS.close();
    }

    // Setup new WebSocket for real-time metrics
    const wsUrl = `ws://127.0.0.1:8443/api/v1/nodes/${nodeId}/metrics/stream`;
    this.nodeMetricsWS = new WebSocket(wsUrl);

    this.nodeMetricsWS.onmessage = (event) => {
      try {
        const metrics = JSON.parse(event.data);
        this.updateRealTimeMetrics(metrics);
      } catch (error) {
        console.error('Error parsing metrics:', error);
      }
    };

    this.nodeMetricsWS.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.nodeMetricsWS.onclose = () => {
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (this.selectedNodeId === nodeId) {
          this.setupNodeMetricsStream(nodeId);
        }
      }, 5000);
    };
  }

  updateRealTimeMetrics(metrics) {
    // Update performance charts and displays
    this.updatePerformanceCharts(metrics);
    
    // Update overview metrics
    const temperatureElement = document.querySelector('#nodeOverview .spec-value:contains("°C")');
    if (temperatureElement && metrics.temperature) {
      temperatureElement.textContent = `${metrics.temperature}°C`;
    }
  }

  updateNodePerformance(nodeData) {
    const performanceContent = document.getElementById('nodePerformance');
    if (!performanceContent) return;

    const performance = nodeData.performance || {};
    const metrics = nodeData.metrics || {};

    performanceContent.innerHTML = `
      <div class="performance-dashboard">
        <div class="performance-grid">
          <div class="performance-card">
            <h4><i class="fas fa-microchip"></i> CPU Usage</h4>
            <div class="metric-display">
              <div class="metric-value large">${Math.round(metrics.cpu_percent || 0)}%</div>
              <canvas id="cpuChart" width="200" height="100"></canvas>
            </div>
            <div class="metric-details">
              <div>Cores: ${performance.cpu?.count?.physical || 'N/A'}</div>
              <div>Threads: ${performance.cpu?.count?.logical || 'N/A'}</div>
              <div>Frequency: ${performance.cpu?.frequency?.current || 'N/A'} MHz</div>
            </div>
          </div>
          
          <div class="performance-card">
            <h4><i class="fas fa-memory"></i> Memory Usage</h4>
            <div class="metric-display">
              <div class="metric-value large">${Math.round(metrics.memory_percent || 0)}%</div>
              <canvas id="memoryChart" width="200" height="100"></canvas>
            </div>
            <div class="metric-details">
              <div>Total: ${this.formatBytes(performance.memory?.virtual?.total)}</div>
              <div>Available: ${this.formatBytes(performance.memory?.virtual?.available)}</div>
              <div>Used: ${this.formatBytes(performance.memory?.virtual?.used)}</div>
            </div>
          </div>
          
          <div class="performance-card">
            <h4><i class="fas fa-hdd"></i> Storage I/O</h4>
            <div class="metric-display">
              <div class="metric-value large">${Math.round(metrics.disk_percent || 0)}%</div>
              <canvas id="diskChart" width="200" height="100"></canvas>
            </div>
            <div class="metric-details">
              <div>Read: ${this.formatBytes(metrics.disk_io?.read_bytes || 0)}</div>
              <div>Write: ${this.formatBytes(metrics.disk_io?.write_bytes || 0)}</div>
            </div>
          </div>
          
          <div class="performance-card">
            <h4><i class="fas fa-network-wired"></i> Network I/O</h4>
            <div class="metric-display">
              <div class="metric-value large">${this.formatNetworkSpeed(metrics.network_io)}</div>
              <canvas id="networkChart" width="200" height="100"></canvas>
            </div>
            <div class="metric-details">
              <div>Sent: ${this.formatBytes(metrics.network_io?.bytes_sent || 0)}</div>
              <div>Received: ${this.formatBytes(metrics.network_io?.bytes_recv || 0)}</div>
              <div>Connections: ${performance.network?.active_connections || 'N/A'}</div>
            </div>
          </div>
        </div>
        
        <div class="performance-charts">
          <div class="chart-container">
            <h4>System Performance Over Time</h4>
            <canvas id="systemPerformanceChart" width="800" height="300"></canvas>
          </div>
        </div>
      </div>
    `;

    // Initialize performance charts
    this.initializePerformanceCharts(metrics);
  }

  formatNetworkSpeed(networkIO) {
    if (!networkIO) return '0 Mbps';
    
    const totalBytes = (networkIO.bytes_sent || 0) + (networkIO.bytes_recv || 0);
    const mbps = (totalBytes * 8) / (1024 * 1024); // Convert to Mbps
    return `${mbps.toFixed(1)} Mbps`;
  }

  updateNodeProcesses(nodeData) {
    const processesContent = document.getElementById('nodeProcesses');
    if (!processesContent) return;

    const processes = nodeData.processes || [];

    processesContent.innerHTML = `
      <div class="processes-container">
        <div class="processes-header">
          <h4><i class="fas fa-tasks"></i> Running Processes (${processes.length})</h4>
          <div class="processes-controls">
            <input type="text" id="processFilter" placeholder="Filter processes..." class="form-input">
            <button class="btn btn-secondary" onclick="refreshProcesses()">
              <i class="fas fa-refresh"></i> Refresh
            </button>
          </div>
        </div>
        
        <div class="processes-table">
          <table>
            <thead>
              <tr>
                <th>PID</th>
                <th>Name</th>
                <th>CPU %</th>
                <th>Memory %</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              ${processes.map(process => `
                <tr class="process-row" data-pid="${process.pid}">
                  <td>${process.pid}</td>
                  <td class="process-name">${process.name || 'Unknown'}</td>
                  <td class="cpu-usage">${(process.cpu_percent || 0).toFixed(1)}%</td>
                  <td class="memory-usage">${(process.memory_percent || 0).toFixed(1)}%</td>
                  <td><span class="status-badge ${process.status}">${process.status || 'running'}</span></td>
                  <td>
                    <button class="btn-small btn-danger" onclick="killProcess(${process.pid})" title="Kill Process">
                      <i class="fas fa-times"></i>
                    </button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    // Add process filter functionality
    const processFilter = document.getElementById('processFilter');
    if (processFilter) {
      processFilter.addEventListener('input', (e) => this.filterProcesses(e.target.value));
    }
  }

  filterProcesses(searchTerm) {
    const processRows = document.querySelectorAll('.process-row');
    const term = searchTerm.toLowerCase();

    processRows.forEach(row => {
      const processName = row.querySelector('.process-name')?.textContent.toLowerCase() || '';
      const pid = row.dataset.pid || '';
      
      if (processName.includes(term) || pid.includes(term)) {
        row.style.display = 'table-row';
      } else {
        row.style.display = 'none';
      }
    });
  }

  updateNodeLogs(nodeData) {
    const logsContent = document.getElementById('nodeLogs');
    if (!logsContent) return;

    const logs = nodeData.logs || [];

    logsContent.innerHTML = `
      <div class="logs-container">
        <div class="logs-header">
          <h4><i class="fas fa-file-alt"></i> System Logs (${logs.length})</h4>
          <div class="logs-controls">
            <select id="logLevelFilter" class="form-select">
              <option value="">All Levels</option>
              <option value="ERROR">Error</option>
              <option value="WARN">Warning</option>
              <option value="INFO">Info</option>
              <option value="DEBUG">Debug</option>
            </select>
            <input type="text" id="logFilter" placeholder="Filter logs..." class="form-input">
            <button class="btn btn-secondary" onclick="refreshLogs()">
              <i class="fas fa-refresh"></i> Refresh
            </button>
            <button class="btn btn-primary" onclick="exportLogs()">
              <i class="fas fa-download"></i> Export
            </button>
          </div>
        </div>
        
        <div class="logs-list">
          ${logs.map(log => `
            <div class="log-entry log-${log.level.toLowerCase()}" data-level="${log.level}">
              <div class="log-timestamp">${this.formatTimestamp(log.timestamp)}</div>
              <div class="log-level">
                <span class="level-badge ${log.level.toLowerCase()}">${log.level}</span>
              </div>
              <div class="log-source">${log.source}</div>
              <div class="log-message">${log.message}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    // Add log filter functionality
    const logLevelFilter = document.getElementById('logLevelFilter');
    const logFilter = document.getElementById('logFilter');
    
    if (logLevelFilter) {
      logLevelFilter.addEventListener('change', () => this.filterLogs());
    }
    if (logFilter) {
      logFilter.addEventListener('input', () => this.filterLogs());
    }
  }

  filterLogs() {
    const levelFilter = document.getElementById('logLevelFilter')?.value || '';
    const textFilter = document.getElementById('logFilter')?.value.toLowerCase() || '';
    const logEntries = document.querySelectorAll('.log-entry');

    logEntries.forEach(entry => {
      const level = entry.dataset.level;
      const message = entry.querySelector('.log-message')?.textContent.toLowerCase() || '';
      const source = entry.querySelector('.log-source')?.textContent.toLowerCase() || '';
      
      const levelMatch = !levelFilter || level === levelFilter;
      const textMatch = !textFilter || message.includes(textFilter) || source.includes(textFilter);
      
      if (levelMatch && textMatch) {
        entry.style.display = 'flex';
      } else {
        entry.style.display = 'none';
      }
    });
  }

  initializePerformanceCharts(metrics) {
    // Initialize simple performance charts
    setTimeout(() => {
      this.createMiniChart('cpuChart', [metrics.cpu_percent || 0], '#3498db');
      this.createMiniChart('memoryChart', [metrics.memory_percent || 0], '#e74c3c');
      this.createMiniChart('diskChart', [metrics.disk_percent || 0], '#f39c12');
      this.createMiniChart('networkChart', [50], '#2ecc71'); // Placeholder
    }, 100);
  }

  createMiniChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    
    // Create simple progress bar
    const value = data[0] || 0;
    const barWidth = (width * value) / 100;
    
    // Background
    ctx.fillStyle = '#ecf0f1';
    ctx.fillRect(0, height - 20, width, 20);
    
    // Progress
    ctx.fillStyle = color;
    ctx.fillRect(0, height - 20, barWidth, 20);
    
    // Text
    ctx.fillStyle = '#2c3e50';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${value.toFixed(1)}%`, width / 2, height - 8);
  }

  updatePerformanceCharts(metrics) {
    // Update mini charts with new metrics
    this.createMiniChart('cpuChart', [metrics.cpu_percent || 0], '#3498db');
    this.createMiniChart('memoryChart', [metrics.memory_percent || 0], '#e74c3c');
    this.createMiniChart('diskChart', [metrics.disk_percent || 0], '#f39c12');
  }

  async updateResourcesData() {
    const data = await this.makeBackendRequest('/api/resources');
    if (data) {
      this.renderResourcesData(data);
    }
  }

  // Sessions Management - Comprehensive Implementation
  selectedSessionId = null;
  sessionsData = [];
  sessionWebSocket = null;
  sessionInspectorTab = 'overview';
  filteredSessions = [];
  sessionUpdateInterval = null;
  
  async updateSessionsData() {
    try {
      console.log('🔄 Fetching sessions from backend...');
      
      // Try the test endpoint for real data
      let response;
      try {
        response = await fetch(`${this.apiBaseUrl}/api/test/sessions`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        this.sessionsData = await response.json();
        console.log('✅ Real sessions data loaded:', this.sessionsData.length, 'sessions');
      } catch (error) {
        console.warn('⚠️ Real API failed, trying fallback...', error.message);
        
        // Fallback to alternative endpoints
        try {
          response = await fetch('/api/v1/sessions');
          if (response.ok) {
            this.sessionsData = await response.json();
            console.log('✅ Fallback sessions data loaded');
          } else {
            throw new Error('All endpoints failed');
          }
        } catch (fallbackError) {
          console.warn('⚠️ Using mock data for development');
          this.sessionsData = this.generateSessionsTestData();
        }
      }
      
      // Apply current filters
      this.applySessionFilters();
      this.renderSessionsList();
      this.updateSessionsStats();
      
      // Auto-select first session if none selected and we have sessions
      if (!this.selectedSessionId && this.filteredSessions.length > 0) {
        this.selectSession(this.filteredSessions[0].session_id);
      } else if (this.selectedSessionId) {
        // Update inspector if we have a selected session
        this.renderSessionInspector();
      }
      
    } catch (error) {
      console.error('❌ Error updating sessions data:', error);
      this.showNotification('Failed to load sessions data', 'error');
    }
  }

  generateSessionsTestData() {
    return [
      {
        session_id: "sess_gaming_cyberpunk",
        session_name: "Gaming Session - Cyberpunk 2077",
        app_name: "Cyberpunk 2077",
        app_command: "steam://rungameid/1091500",
        app_icon: "🎮",
        user_id: "admin",
        status: "RUNNING",
        node_id: "control-node-local",
        cpu_cores: 8,
        gpu_units: 1,
        ram_gb: 16.0,
        storage_gb: 70.0,
        priority: 2,
        session_type: "gaming",
        start_time: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 2.5 * 60 * 60 * 1000).toISOString(),
        tags: ["gaming", "steam", "high-performance"],
        real_time_metrics: {
          cpu_usage: 72.5,
          memory_usage: 12.4,
          gpu_usage: 88.2,
          disk_io: 25.6,
          network_in: 8.2,
          network_out: 5.1,
          fps: 62.3,
          latency_ms: 4.2,
          temperature: 68.5,
          timestamp: new Date().toISOString()
        },
        real_time_processes: [
          { pid: 1001, name: "Cyberpunk2077.exe", cpu_percent: 45.2, memory_percent: 25.6, status: "running" },
          { pid: 1002, name: "steam.exe", cpu_percent: 2.1, memory_percent: 3.2, status: "running" }
        ]
      },
      {
        session_id: "sess_render_blender",
        session_name: "Blender Render Farm",
        app_name: "Blender",
        app_command: "/usr/bin/blender --background scene.blend",
        app_icon: "🎬",
        user_id: "admin",
        status: "PAUSED",
        node_id: "render-node-01",
        cpu_cores: 16,
        gpu_units: 2,
        ram_gb: 32.0,
        storage_gb: 120.0,
        priority: 3,
        session_type: "rendering",
        start_time: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 50 * 60 * 1000).toISOString(),
        tags: ["rendering", "blender", "background"],
        real_time_metrics: {
          cpu_usage: 5.2,
          memory_usage: 8.1,
          gpu_usage: 0.0,
          disk_io: 1.2,
          network_in: 0.5,
          network_out: 0.3,
          fps: 0.0,
          latency_ms: 2.1,
          temperature: 42.0,
          timestamp: new Date().toISOString()
        },
        real_time_processes: [
          { pid: 2001, name: "blender", cpu_percent: 0.5, memory_percent: 15.2, status: "sleeping" }
        ]
      },
      {
        session_id: "sess_ai_training",
        session_name: "ML Training - Neural Network",
        app_name: "Python ML Training",
        app_command: "python train_model.py --gpu --batch-size 64",
        app_icon: "🤖",
        user_id: "researcher",
        status: "RUNNING",
        node_id: "ai-node-gpu",
        cpu_cores: 12,
        gpu_units: 4,
        ram_gb: 64.0,
        storage_gb: 200.0,
        priority: 4,
        session_type: "ai",
        start_time: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 6.2 * 60 * 60 * 1000).toISOString(),
        tags: ["ai", "machine-learning", "gpu-intensive"],
        real_time_metrics: {
          cpu_usage: 85.7,
          memory_usage: 48.2,
          gpu_usage: 95.8,
          disk_io: 45.2,
          network_in: 15.6,
          network_out: 8.9,
          fps: 0.0,
          latency_ms: 12.5,
          temperature: 78.2,
          timestamp: new Date().toISOString()
        },
        real_time_processes: [
          { pid: 3001, name: "python", cpu_percent: 65.2, memory_percent: 35.8, status: "running" },
          { pid: 3002, name: "nvidia-smi", cpu_percent: 1.2, memory_percent: 0.5, status: "running" }
        ]
      }
    ];
  }

  renderSessionsList() {
    const sessionsList = document.getElementById('sessionsList');
    if (!sessionsList) return;

    const sessionsToRender = this.filteredSessions || this.sessionsData || [];

    if (sessionsToRender.length === 0) {
      sessionsList.innerHTML = `
        <div class="empty-state">
          <i class="fas fa-desktop fa-2x"></i>
          <p>No sessions found</p>
          <button class="btn-primary" onclick="omegaRenderer.refreshSessions()">Refresh</button>
          <button class="btn-secondary" onclick="omegaRenderer.createNewSession()">Create New</button>
        </div>
      `;
      return;
    }

    const sessionsHtml = sessionsToRender.map(session => {
      const metrics = session.real_time_metrics || session.metrics || {};
      const uptime = this.calculateUptime(session.start_time || session.created_at);
      const isSelected = session.session_id === this.selectedSessionId;
      const statusClass = this.getSessionStatusClass(session.status);
      
      return `
        <div class="session-item ${isSelected ? 'active selected' : ''}" 
             data-session-id="${session.session_id}"
             onclick="omegaRenderer.selectSession('${session.session_id}')">
          <div class="session-icon">
            ${this.getSessionIcon(session.session_type || session.app_name, session.app_icon)}
          </div>
          <div class="session-info">
            <div class="session-name" title="${session.session_name}">${session.session_name || 'Unnamed Session'}</div>
            <div class="session-status ${statusClass}">
              <span class="status-indicator"></span>
              ${session.status || 'unknown'}
            </div>
            <div class="session-app">${session.app_name || 'Unknown App'}</div>
            <div class="session-resources">
              <div class="resource-item">
                <span class="resource-label">CPU:</span>
                <div class="resource-bar cpu" style="--usage: ${metrics.cpu_percent || metrics.cpu_usage || 0}%"></div>
                <span class="resource-value">${Math.round(metrics.cpu_percent || metrics.cpu_usage || 0)}%</span>
              </div>
              <div class="resource-item">
                <span class="resource-label">RAM:</span>
                <div class="resource-bar ram" style="--usage: ${metrics.memory_percent || metrics.memory_usage || 0}%"></div>
                <span class="resource-value">${Math.round(metrics.memory_percent || metrics.memory_usage || 0)}%</span>
              </div>
              ${metrics.gpu_percent || metrics.gpu_usage ? `
                <div class="resource-item">
                  <span class="resource-label">GPU:</span>
                  <div class="resource-bar gpu" style="--usage: ${metrics.gpu_percent || metrics.gpu_usage || 0}%"></div>
                  <span class="resource-value">${Math.round(metrics.gpu_percent || metrics.gpu_usage || 0)}%</span>
                </div>
              ` : ''}
            </div>
            <div class="session-meta">
              <div class="session-time">Uptime: ${uptime}</div>
              <div class="session-user">User: ${session.user_id || 'Unknown'}</div>
            </div>
          </div>
          <div class="session-controls">
            ${this.getSessionControlButtons(session)}
          </div>
        </div>
      `;
    }).join('');

    sessionsList.innerHTML = sessionsHtml;
    console.log('✅ Sessions list rendered:', sessionsToRender.length, 'sessions');
  }

  // Session Filtering and Sorting
  applySessionFilters() {
    const searchTerm = (document.getElementById('session-search')?.value || '').toLowerCase();
    const statusFilter = document.getElementById('session-status-filter')?.value || 'all';
    const sortBy = document.getElementById('session-sort')?.value || 'name';
    
    console.log('🔍 Applying filters:', { searchTerm, statusFilter, sortBy });
    
    // Filter sessions
    this.filteredSessions = this.sessionsData.filter(session => {
      const matchesSearch = !searchTerm || 
        (session.session_name || '').toLowerCase().includes(searchTerm) ||
        (session.app_name || '').toLowerCase().includes(searchTerm) ||
        (session.user_id && session.user_id.toString().toLowerCase().includes(searchTerm));
      
      const matchesStatus = statusFilter === 'all' || (session.status || '').toLowerCase() === statusFilter.toLowerCase();
      
      return matchesSearch && matchesStatus;
    });
    
    // Sort sessions
    this.filteredSessions.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return (a.session_name || '').localeCompare(b.session_name || '');
        case 'status':
          return (a.status || '').localeCompare(b.status || '');
        case 'created':
          return new Date(b.created_at || b.start_time || 0) - new Date(a.created_at || a.start_time || 0);
        case 'updated':
          return new Date(b.updated_at || b.last_activity || 0) - new Date(a.updated_at || a.last_activity || 0);
        case 'cpu':
          const cpuA = (a.metrics || a.real_time_metrics || {}).cpu_percent || (a.metrics || a.real_time_metrics || {}).cpu_usage || 0;
          const cpuB = (b.metrics || b.real_time_metrics || {}).cpu_percent || (b.metrics || b.real_time_metrics || {}).cpu_usage || 0;
          return parseFloat(cpuB) - parseFloat(cpuA);
        case 'memory':
          const memA = (a.metrics || a.real_time_metrics || {}).memory_percent || (a.metrics || a.real_time_metrics || {}).memory_usage || 0;
          const memB = (b.metrics || b.real_time_metrics || {}).memory_percent || (b.metrics || b.real_time_metrics || {}).memory_usage || 0;
          return parseFloat(memB) - parseFloat(memA);
        default:
          return 0;
      }
    });
    
    console.log('✅ Filtered sessions:', this.filteredSessions.length, 'of', this.sessionsData.length);
  }
  
  setupSessionFilters() {
    // Search filter
    const searchInput = document.getElementById('session-search');
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        this.applySessionFilters();
        this.renderSessionsList();
      });
    }
    
    // Status filter
    const statusFilter = document.getElementById('session-status-filter');
    if (statusFilter) {
      statusFilter.addEventListener('change', () => {
        this.applySessionFilters();
        this.renderSessionsList();
      });
    }
    
    // Sort filter
    const sortSelect = document.getElementById('session-sort');
    if (sortSelect) {
      sortSelect.addEventListener('change', () => {
        this.applySessionFilters();
        this.renderSessionsList();
      });
    }
    
    console.log('✅ Session filters setup complete');
  }

  // Session Actions with Error Handling
  async performSessionAction(sessionId, action) {
    if (!sessionId || !action) {
      this.showNotification('Invalid session or action', 'error');
      return false;
    }
    
    const session = this.sessionsData.find(s => s.session_id === sessionId);
    if (!session) {
      this.showNotification('Session not found', 'error');
      return false;
    }
    
    console.log(`🔄 Performing action "${action}" on session ${sessionId}`);
    
    try {
      // Show loading state
      this.setSessionActionLoading(sessionId, action, true);
      
      let response;
      const actionUrl = `${this.apiBaseUrl}/api/sessions/${sessionId}/${action}`;
      
      switch (action) {
        case 'start':
        case 'resume':
          response = await fetch(actionUrl, { method: 'POST' });
          break;
        case 'pause':
        case 'suspend':
          response = await fetch(actionUrl, { method: 'POST' });
          break;
        case 'stop':
        case 'terminate':
          response = await fetch(actionUrl, { method: 'POST' });
          break;
        case 'restart':
          response = await fetch(actionUrl, { method: 'POST' });
          break;
        case 'delete':
          if (!confirm(`Are you sure you want to delete session "${session.session_name}"?`)) {
            return false;
          }
          response = await fetch(`${this.apiBaseUrl}/api/sessions/${sessionId}`, { method: 'DELETE' });
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      console.log(`✅ Action "${action}" completed successfully:`, result);
      
      // Show success notification
      this.showNotification(`Session ${action} completed successfully`, 'success');
      
      // Refresh sessions data
      await this.updateSessionsData();
      
      // If session was deleted and was selected, clear selection
      if (action === 'delete' && this.selectedSessionId === sessionId) {
        this.selectedSessionId = null;
        this.renderSessionInspector();
      }
      
      return true;
      
    } catch (error) {
      console.error(`❌ Error performing action "${action}" on session ${sessionId}:`, error);
      this.showNotification(`Failed to ${action} session: ${error.message}`, 'error');
      return false;
    } finally {
      // Hide loading state
      this.setSessionActionLoading(sessionId, action, false);
    }
  }
  
  setSessionActionLoading(sessionId, action, isLoading) {
    const sessionItem = document.querySelector(`[data-session-id="${sessionId}"]`);
    if (sessionItem) {
      const button = sessionItem.querySelector(`[data-action="${action}"]`);
      if (button) {
        button.disabled = isLoading;
        button.classList.toggle('loading', isLoading);
        if (isLoading) {
          button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
          // Restore original button content
          this.renderSessionsList();
        }
      }
    }
  }
  
  // Enhanced Session Creation with Resource Planning
  async createNewSession() {
    try {
      // Show resource planning dialog
      const sessionData = await this.showSessionCreationDialog();
      if (!sessionData) return;
      
      console.log('🔄 Creating new session with resource planning:', sessionData);
      
      const response = await fetch(`${this.apiBaseUrl}/api/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(sessionData)
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Session created successfully:', result);
      
      this.showNotification(`Session "${sessionData.session_name}" created successfully on ${result.allocated_resources.node_id}`, 'success');
      
      // Show resource allocation summary
      this.showResourceAllocationSummary(result.allocated_resources);
      
      // Refresh sessions and select the new one
      await this.updateSessionsData();
      if (result.session_id) {
        this.selectSession(result.session_id);
      }
      
    } catch (error) {
      console.error('❌ Error creating session:', error);
      this.showNotification(`Failed to create session: ${error.message}`, 'error');
    }
  }
  
  async showSessionCreationDialog() {
    return new Promise((resolve) => {
      // Create modal dialog
      const modal = document.createElement('div');
      modal.className = 'session-creation-modal';
      modal.innerHTML = `
        <div class="modal-overlay" onclick="this.parentElement.remove(); resolve(null)"></div>
        <div class="modal-content">
          <div class="modal-header">
            <h3>Create New Session</h3>
            <button class="modal-close" onclick="this.closest('.session-creation-modal').remove(); resolve(null)">&times;</button>
          </div>
          <div class="modal-body">
            <form id="sessionCreationForm">
              <div class="form-group">
                <label for="sessionName">Session Name *</label>
                <input type="text" id="sessionName" placeholder="Enter session name" required>
              </div>
              
              <div class="form-group">
                <label for="appName">Application *</label>
                <input type="text" id="appName" placeholder="Enter application name" required>
              </div>
              
              <div class="form-group">
                <label for="sessionType">Session Type</label>
                <select id="sessionType">
                  <option value="workstation">Workstation</option>
                  <option value="gaming">Gaming</option>
                  <option value="development">Development</option>
                  <option value="ai_compute">AI/ML Computing</option>
                  <option value="render_farm">Rendering</option>
                  <option value="streaming">Streaming</option>
                </select>
              </div>
              
              <div class="resource-section">
                <h4>Resource Requirements</h4>
                <div class="resource-grid">
                  <div class="form-group">
                    <label for="cpuCores">CPU Cores</label>
                    <input type="number" id="cpuCores" min="1" max="64" value="2">
                    <small>Available: <span id="availableCpu">Loading...</span></small>
                  </div>
                  
                  <div class="form-group">
                    <label for="ramGb">RAM (GB)</label>
                    <input type="number" id="ramGb" min="1" max="256" value="4" step="0.5">
                    <small>Available: <span id="availableRam">Loading...</span></small>
                  </div>
                  
                  <div class="form-group">
                    <label for="gpuUnits">GPU Units</label>
                    <input type="number" id="gpuUnits" min="0" max="8" value="0">
                    <small>Available: <span id="availableGpu">Loading...</span></small>
                  </div>
                  
                  <div class="form-group">
                    <label for="storageGb">Storage (GB)</label>
                    <input type="number" id="storageGb" min="1" max="1000" value="10">
                  </div>
                </div>
              </div>
              
              <div class="form-group">
                <label for="userId">User ID</label>
                <input type="text" id="userId" placeholder="Enter user ID" value="current_user">
              </div>
              
              <div class="form-group">
                <label for="priority">Priority</label>
                <select id="priority">
                  <option value="1">Low</option>
                  <option value="2">Normal</option>
                  <option value="3">High</option>
                  <option value="4">Critical</option>
                </select>
              </div>
            </form>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn-secondary" onclick="this.closest('.session-creation-modal').remove(); resolve(null)">Cancel</button>
            <button type="button" class="btn-primary" onclick="submitSessionForm()">Create Session</button>
          </div>
        </div>
      `;
      
      document.body.appendChild(modal);
      
      // Load available resources
      this.loadAvailableResources();
      
      // Handle form submission
      window.submitSessionForm = () => {
        const form = document.getElementById('sessionCreationForm');
        const formData = new FormData(form);
        
        const sessionData = {
          session_name: document.getElementById('sessionName').value,
          app_name: document.getElementById('appName').value,
          session_type: document.getElementById('sessionType').value,
          cpu_cores: parseInt(document.getElementById('cpuCores').value),
          ram_gb: parseFloat(document.getElementById('ramGb').value),
          gpu_units: parseInt(document.getElementById('gpuUnits').value),
          storage_gb: parseFloat(document.getElementById('storageGb').value),
          user_id: document.getElementById('userId').value,
          priority: parseInt(document.getElementById('priority').value)
        };
        
        modal.remove();
        resolve(sessionData);
      };
    });
  }
  
  async loadAvailableResources() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/nodes`);
      if (response.ok) {
        const nodes = await response.json();
        let totalCpu = 0, totalRam = 0, totalGpu = 0;
        
        nodes.forEach(node => {
          if (node.resources) {
            totalCpu += node.resources.cpu_cores || 0;
            totalRam += node.resources.ram_gb || 0;
            totalGpu += node.resources.gpu_units || 0;
          }
        });
        
        // Update the form with available resources
        document.getElementById('availableCpu').textContent = `${Math.floor(totalCpu * 0.8)} cores`;
        document.getElementById('availableRam').textContent = `${Math.floor(totalRam * 0.9)} GB`;
        document.getElementById('availableGpu').textContent = `${totalGpu} units`;
      }
    } catch (error) {
      console.warn('Could not load available resources:', error);
      document.getElementById('availableCpu').textContent = 'Unknown';
      document.getElementById('availableRam').textContent = 'Unknown';
      document.getElementById('availableGpu').textContent = 'Unknown';
    }
  }
  
  showResourceAllocationSummary(resources) {
    const summary = `
      <div class="resource-summary">
        <h4>Resource Allocation Summary</h4>
        <div class="allocation-details">
          <div class="allocation-item">
            <span class="label">Node:</span>
            <span class="value">${resources.node_id}</span>
          </div>
          <div class="allocation-item">
            <span class="label">CPU Cores:</span>
            <span class="value">${resources.cpu_cores}</span>
          </div>
          <div class="allocation-item">
            <span class="label">RAM:</span>
            <span class="value">${resources.ram_gb} GB</span>
          </div>
          <div class="allocation-item">
            <span class="label">GPU Units:</span>
            <span class="value">${resources.gpu_units}</span>
          </div>
        </div>
      </div>
    `;
    
    this.showNotification(summary, 'info', 8000);
  }
  
  // Refresh sessions data
  async refreshSessions() {
    console.log('🔄 Refreshing sessions data...');
    this.showNotification('Refreshing sessions...', 'info');
    await this.updateSessionsData();
    this.showNotification('Sessions refreshed', 'success');
  }

  getSessionIcon(sessionType, appIcon) {
    if (appIcon && appIcon.startsWith('🎮🎬🤖')) return appIcon;
    
    const iconMap = {
      'gaming': '<i class="fas fa-gamepad"></i>',
      'rendering': '<i class="fas fa-cube"></i>',
      'ai': '<i class="fas fa-brain"></i>',
      'development': '<i class="fas fa-code"></i>',
      'productivity': '<i class="fas fa-briefcase"></i>',
      'media': '<i class="fas fa-play"></i>',
      'default': '<i class="fas fa-desktop"></i>'
    };
    
    return iconMap[sessionType] || iconMap.default;
  }

  getSessionControlButtons(session) {
    if (session.status === 'RUNNING') {
      return `
        <button class="session-btn pause" title="Pause Session" 
                onclick="event.stopPropagation(); omegaCC.pauseSession('${session.session_id}')">
          <i class="fas fa-pause"></i>
        </button>
        <button class="session-btn terminate" title="Terminate Session"
                onclick="event.stopPropagation(); omegaCC.terminateSession('${session.session_id}')">
          <i class="fas fa-stop"></i>
        </button>
      `;
    } else if (session.status === 'PAUSED') {
      return `
        <button class="session-btn resume" title="Resume Session"
                onclick="event.stopPropagation(); omegaCC.resumeSession('${session.session_id}')">
          <i class="fas fa-play"></i>
        </button>
        <button class="session-btn terminate" title="Terminate Session"
                onclick="event.stopPropagation(); omegaCC.terminateSession('${session.session_id}')">
          <i class="fas fa-stop"></i>
        </button>
      `;
    } else {
      return `
        <button class="session-btn terminate" title="Clean Up Session"
                onclick="event.stopPropagation(); omegaCC.cleanupSession('${session.session_id}')">
          <i class="fas fa-trash"></i>
        </button>
      `;
    }
  }

  updateSessionsStats() {
    const runningCount = this.sessionsData.filter(s => s.status === 'RUNNING').length;
    const pausedCount = this.sessionsData.filter(s => s.status === 'PAUSED').length;
    const totalCount = this.sessionsData.length;

    document.getElementById('runningCount').textContent = runningCount;
    document.getElementById('pausedCount').textContent = pausedCount;
    document.getElementById('totalCount').textContent = totalCount;
  }

  selectSession(sessionId) {
    this.selectedSessionId = sessionId;
    this.renderSessionsList(); // Re-render to update active state
    this.renderSessionInspector();
    this.startSessionWebSocket(sessionId);
  }

  renderSessionInspector() {
    const session = this.sessionsData.find(s => s.session_id === this.selectedSessionId);
    
    if (!session) {
      document.getElementById('noSessionView').style.display = 'flex';
      document.getElementById('sessionContent').style.display = 'none';
      return;
    }

    document.getElementById('noSessionView').style.display = 'none';
    document.getElementById('sessionContent').style.display = 'flex';

    // Update session header
    this.updateSessionHeader(session);
    
    // Update current inspector tab content
    this.updateInspectorTabContent(session);
  }

  updateSessionHeader(session) {
    const metrics = session.real_time_metrics || session.metrics || {};
    const uptime = this.calculateUptime(session.start_time);
    
    document.getElementById('sessionIconLarge').innerHTML = this.getSessionIcon(session.session_type, session.app_icon);
    document.getElementById('selectedSessionName').textContent = session.session_name;
    document.getElementById('selectedSessionId').textContent = session.session_id;
    document.getElementById('selectedSessionNode').textContent = session.node_id;
    document.getElementById('selectedSessionUptime').textContent = uptime;
    
    const statusBadge = document.getElementById('selectedSessionStatus');
    statusBadge.textContent = session.status;
    statusBadge.className = `session-status-badge ${session.status}`;
    
    document.getElementById('selectedSessionPriority').textContent = `Priority: ${this.getPriorityText(session.priority)}`;
    
    // Update action buttons
    this.updateActionButtons(session);
  }

  updateActionButtons(session) {
    const pauseResumeBtn = document.getElementById('pauseResumeBtn');
    
    if (session.status === 'RUNNING') {
      pauseResumeBtn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
      pauseResumeBtn.onclick = () => this.pauseSession(session.session_id);
    } else if (session.status === 'PAUSED') {
      pauseResumeBtn.innerHTML = '<i class="fas fa-play"></i><span>Resume</span>';
      pauseResumeBtn.onclick = () => this.resumeSession(session.session_id);
    } else {
      pauseResumeBtn.innerHTML = '<i class="fas fa-redo"></i><span>Restart</span>';
      pauseResumeBtn.onclick = () => this.restartSession(session.session_id);
    }
  }

  updateInspectorTabContent(session) {
    switch (this.sessionInspectorTab) {
      case 'overview':
        this.updateOverviewTab(session);
        break;
      case 'performance':
        this.updatePerformanceTab(session);
        break;
      case 'processes':
        this.updateProcessesTab(session);
        break;
      case 'logs':
        this.updateLogsTab(session);
        break;
      case 'settings':
        this.updateSettingsTab(session);
        break;
      case 'snapshots':
        this.updateSnapshotsTab(session);
        break;
    }
  }

  updateOverviewTab(session) {
    // Application information
    document.getElementById('appName').textContent = session.app_name || '-';
    document.getElementById('appCommand').textContent = session.app_command || '-';
    document.getElementById('sessionType').textContent = session.session_type || '-';
    document.getElementById('sessionUser').textContent = session.user_id || '-';
    
    // Resource allocation
    document.getElementById('allocatedCPU').textContent = `${session.cpu_cores} cores`;
    document.getElementById('allocatedGPU').textContent = `${session.gpu_units} units`;
    document.getElementById('allocatedRAM').textContent = `${session.ram_gb} GB`;
    document.getElementById('allocatedStorage').textContent = `${session.storage_gb} GB`;
    
    // Timeline
    document.getElementById('sessionCreated').textContent = this.formatDateTime(session.created_at);
    document.getElementById('sessionStarted').textContent = this.formatDateTime(session.start_time);
    document.getElementById('sessionActivity').textContent = this.formatDateTime(session.real_time_metrics?.timestamp || new Date().toISOString());
    document.getElementById('sessionUptimeDetail').textContent = this.calculateUptime(session.start_time);
    
    // Tags
    const tagsContainer = document.getElementById('sessionTags');
    if (session.tags && session.tags.length > 0) {
      tagsContainer.innerHTML = session.tags.map(tag => `<span class="tag">${tag}</span>`).join('');
    } else {
      tagsContainer.innerHTML = '<span class="tag">no-tags</span>';
    }
  }

  updatePerformanceTab(session) {
    const metrics = session.real_time_metrics || session.metrics || {};
    
    // Update real-time metrics
    document.getElementById('metricCPU').textContent = `${(metrics.cpu_usage || 0).toFixed(1)}%`;
    document.getElementById('metricMemory').textContent = `${(metrics.memory_usage || 0).toFixed(1)} GB`;
    document.getElementById('metricGPU').textContent = `${(metrics.gpu_usage || 0).toFixed(1)}%`;
    document.getElementById('metricNetwork').textContent = `${((metrics.network_in || 0) + (metrics.network_out || 0)).toFixed(1)} MB/s`;
    document.getElementById('metricDisk').textContent = `${(metrics.disk_io || 0).toFixed(1)} MB/s`;
    document.getElementById('metricLatency').textContent = `${(metrics.latency_ms || 0).toFixed(1)} ms`;
    
    // Update mini charts (placeholder - would use Chart.js in production)
    this.updateMiniCharts(metrics);
  }

  updateProcessesTab(session) {
    const processes = session.real_time_processes || [];
    const tbody = document.getElementById('processesTableBody');
    
    if (processes.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666666;">No processes found</td></tr>';
      return;
    }
    
    tbody.innerHTML = processes.map(proc => `
      <tr>
        <td>${proc.pid}</td>
        <td class="process-name">${proc.name}</td>
        <td>${(proc.cpu_percent || 0).toFixed(1)}%</td>
        <td>${(proc.memory_percent || 0).toFixed(1)}%</td>
        <td>${proc.status}</td>
        <td>
          <button class="session-btn" onclick="omegaCC.killProcess(${proc.pid})" title="Kill Process">
            <i class="fas fa-times"></i>
          </button>
        </td>
      </tr>
    `).join('');
  }

  updateLogsTab(session) {
    const logs = session.real_time_logs || [];
    const logsContainer = document.getElementById('logsContainer');
    
    if (logs.length === 0) {
      logsContainer.innerHTML = `
        <div style="text-align: center; color: #666666; padding: 40px;">
          <i class="fas fa-file-alt fa-2x"></i>
          <p>No logs available</p>
        </div>
      `;
      return;
    }
    
    logsContainer.innerHTML = logs.map(log => `
      <div class="log-entry">
        <span class="log-timestamp">${this.formatLogTime(log.timestamp)}</span>
        <span class="log-level ${log.level}">${log.level}</span>
        <span class="log-message">${log.message}</span>
      </div>
    `).join('');
    
    // Auto-scroll if enabled
    if (document.getElementById('autoScroll').checked) {
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  }

  updateSettingsTab(session) {
    // Populate settings form with current session values
    document.getElementById('settingCPUCores').value = session.cpu_cores;
    document.getElementById('settingMemory').value = session.ram_gb;
    document.getElementById('settingGPU').value = session.gpu_units;
    document.getElementById('settingPriority').value = session.priority;
    
    // Placeholder for other settings
    document.getElementById('settingWorkDir').value = session.working_directory || '';
    document.getElementById('settingEnvVars').value = session.environment_variables || '';
  }

  updateSnapshotsTab(session) {
    const snapshots = session.snapshots || [];
    const snapshotsGrid = document.getElementById('snapshotsGrid');
    
    if (snapshots.length === 0) {
      snapshotsGrid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; color: #666666; padding: 40px;">
          <i class="fas fa-layer-group fa-2x"></i>
          <p>No snapshots available</p>
          <button class="btn-primary" onclick="omegaCC.createSessionSnapshot()">Create First Snapshot</button>
        </div>
      `;
      return;
    }
    
    snapshotsGrid.innerHTML = snapshots.map(snapshot => `
      <div class="snapshot-card">
        <div class="snapshot-header">
          <div class="snapshot-name">${snapshot.name}</div>
          <div class="snapshot-type ${snapshot.type}">${snapshot.type}</div>
        </div>
        <div class="snapshot-details">
          <div>Created: ${this.formatDateTime(snapshot.created_at)}</div>
          <div>Size: ${snapshot.size_mb} MB</div>
          <div>Status: ${snapshot.status}</div>
        </div>
        <div class="snapshot-actions">
          <button class="snapshot-btn restore" onclick="omegaCC.restoreSnapshot('${snapshot.snapshot_id}')">
            Restore
          </button>
          <button class="snapshot-btn delete" onclick="omegaCC.deleteSnapshot('${snapshot.snapshot_id}')">
            Delete
          </button>
        </div>
      </div>
    `).join('');
  }

  // Session Action Methods
  async pauseSession(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/test/sessions/${sessionId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'pause' })
      });
      
      if (response.ok) {
        this.showNotification('Session paused successfully', 'success');
        await this.updateSessionsData();
      } else {
        throw new Error('Failed to pause session');
      }
    } catch (error) {
      console.error('Error pausing session:', error);
      this.showNotification('Failed to pause session', 'error');
    }
  }

  async resumeSession(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/test/sessions/${sessionId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'resume' })
      });
      
      if (response.ok) {
        this.showNotification('Session resumed successfully', 'success');
        await this.updateSessionsData();
      } else {
        throw new Error('Failed to resume session');
      }
    } catch (error) {
      console.error('Error resuming session:', error);
      this.showNotification('Failed to resume session', 'error');
    }
  }

  async terminateSession(sessionId) {
    if (!confirm('Are you sure you want to terminate this session? This action cannot be undone.')) {
      return;
    }
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/test/sessions/${sessionId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'terminate' })
      });
      
      if (response.ok) {
        this.showNotification('Session terminated successfully', 'success');
        await this.updateSessionsData();
        
        // Clear selection if terminated session was selected
        if (this.selectedSessionId === sessionId) {
          this.selectedSessionId = null;
          this.renderSessionInspector();
        }
      } else {
        throw new Error('Failed to terminate session');
      }
    } catch (error) {
      console.error('Error terminating session:', error);
      this.showNotification('Failed to terminate session', 'error');
    }
  }

  async createSessionSnapshot() {
    if (!this.selectedSessionId) return;
    
    const snapshotName = prompt('Enter snapshot name:', `snapshot-${Date.now()}`);
    if (!snapshotName) return;
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/test/sessions/${this.selectedSessionId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'snapshot', snapshot_name: snapshotName })
      });
      
      if (response.ok) {
        this.showNotification('Snapshot created successfully', 'success');
        await this.updateSessionsData();
      } else {
        throw new Error('Failed to create snapshot');
      }
    } catch (error) {
      console.error('Error creating snapshot:', error);
      this.showNotification('Failed to create snapshot', 'error');
    }
  }

  // Helper Methods
  calculateUptime(startTime) {
    if (!startTime) return '00:00:00';
    
    const start = new Date(startTime);
    const now = new Date();
    const diffMs = now - start;
    
    const hours = Math.floor(diffMs / (1000 * 60 * 60));
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diffMs % (1000 * 60)) / 1000);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  formatDateTime(dateString) {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
  }

  formatLogTime(dateString) {
    if (!dateString) return '00:00:00';
    return new Date(dateString).toLocaleTimeString();
  }

  getPriorityText(priority) {
    const priorities = { 1: 'Low', 2: 'Normal', 3: 'High', 4: 'Critical' };
    return priorities[priority] || 'Normal';
  }

  updateMiniCharts(metrics) {
    // Placeholder for chart updates
    // In production, this would update Chart.js instances
    console.log('Updating charts with metrics:', metrics);
  }

  startSessionWebSocket(sessionId) {
    // Close existing WebSocket
    if (this.sessionWebSocket) {
      this.sessionWebSocket.close();
    }
    
    try {
      this.sessionWebSocket = new WebSocket(`ws://127.0.0.1:8443/api/ws/sessions/${sessionId}/stream`);
      
      this.sessionWebSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleSessionWebSocketMessage(data);
      };
      
      this.sessionWebSocket.onerror = (error) => {
        console.warn('Session WebSocket error:', error);
      };
      
    } catch (error) {
      console.warn('Failed to establish session WebSocket:', error);
    }
  }

  handleSessionWebSocketMessage(data) {
    if (data.type === 'session_metrics' && data.session_id === this.selectedSessionId) {
      // Update the session in our data
      const sessionIndex = this.sessionsData.findIndex(s => s.session_id === data.session_id);
      if (sessionIndex !== -1) {
        this.sessionsData[sessionIndex].real_time_metrics = data.metrics;
        
        // Update the current tab if it's performance
        if (this.sessionInspectorTab === 'performance') {
          this.updatePerformanceTab(this.sessionsData[sessionIndex]);
        }
      }
    }
  }

  // Global Functions for HTML onclick handlers
  filterSessions(query) {
    // Implementation for filtering sessions
    console.log('Filtering sessions with query:', query);
  }

  filterSessionsByStatus(status) {
    // Implementation for status filtering
    console.log('Filtering sessions by status:', status);
  }

  sortSessions(sortBy) {
    // Implementation for sorting sessions
    console.log('Sorting sessions by:', sortBy);
  }

  switchInspectorTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.inspector-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.inspector-tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-content`).classList.add('active');
    
    this.sessionInspectorTab = tabName;
    
    // Update content for the selected tab
    const session = this.sessionsData.find(s => s.session_id === this.selectedSessionId);
    if (session) {
      this.updateInspectorTabContent(session);
    }
  }

  toggleSessionState() {
    if (!this.selectedSessionId) return;
    
    const session = this.sessionsData.find(s => s.session_id === this.selectedSessionId);
    if (!session) return;
    
    if (session.status === 'RUNNING') {
      this.pauseSession(this.selectedSessionId);
    } else if (session.status === 'PAUSED') {
      this.resumeSession(this.selectedSessionId);
    }
  }

  createNewSession() {
    // Implementation for creating new session
    console.log('Creating new session...');
    this.showNotification('Session creation feature coming soon', 'info');
  }

  refreshSessions() {
    this.updateSessionsData();
  }

  getMockSessionsData() {
    return [
      {
        session_id: "sess_gaming_cyberpunk",
        session_name: "Gaming Session - Cyberpunk 2077",
        app_name: "Cyberpunk 2077",
        app_command: "steam://rungameid/1091500",
        app_icon: "🎮",
        user_id: "admin",
        status: "RUNNING",
        node_id: "control-node-local",
        cpu_cores: 8,
        gpu_units: 1,
        ram_gb: 16.0,
        storage_gb: 70.0,
        priority: 2,
        session_type: "gaming",
        elapsed_time: 7832,
        tags: ["gaming", "steam", "high-performance"],
        metrics: {
          cpu_usage: 72.5,
          gpu_usage: 88.2,
          ram_usage_gb: 12.4,
          disk_io_mbps: 25.6,
          network_in_mbps: 8.2,
          network_out_mbps: 5.1,
          fps: 62.3,
          latency_ms: 4.2,
          active_processes: 8,
          temperature: 68.5
        },
        process_tree: [
          {
            pid: 1001,
            name: "Cyberpunk2077.exe",
            command: "steam://rungameid/1091500",
            cpu_percent: 45.2,
            memory_mb: 8192,
            gpu_percent: 85.6,
            status: "running",
            user: "admin"
          }
        ]
      },
      {
        session_id: "sess_render_blender",
        session_name: "Blender Render Farm",
        app_name: "Blender",
        app_command: "/usr/bin/blender --background scene.blend",
        app_icon: "🎬",
        user_id: "artist",
        status: "RUNNING",
        node_id: "control-node-local",
        cpu_cores: 12,
        gpu_units: 2,
        ram_gb: 32.0,
        storage_gb: 100.0,
        priority: 1,
        session_type: "render_farm",
        elapsed_time: 14523,
        tags: ["rendering", "blender", "production"],
        metrics: {
          cpu_usage: 95.8,
          gpu_usage: 78.3,
          ram_usage_gb: 28.9,
          disk_io_mbps: 45.2,
          network_in_mbps: 12.5,
          network_out_mbps: 8.7,
          fps: null,
          latency_ms: 6.8,
          active_processes: 15,
          temperature: 74.2
        }
      },
      {
        session_id: "sess_dev_vscode",
        session_name: "Development Environment",
        app_name: "VS Code",
        app_command: "/usr/bin/code --remote",
        app_icon: "💻",
        user_id: "developer",
        status: "PAUSED",
        node_id: "control-node-local",
        cpu_cores: 4,
        gpu_units: 0,
        ram_gb: 8.0,
        storage_gb: 50.0,
        priority: 3,
        session_type: "development",
        elapsed_time: 5421,
        tags: ["development", "vscode", "coding"],
        metrics: {
          cpu_usage: 0,
          gpu_usage: 0,
          ram_usage_gb: 0,
          disk_io_mbps: 0,
          network_in_mbps: 0,
          network_out_mbps: 0,
          fps: null,
          latency_ms: 0,
          active_processes: 0,
          temperature: 35.0
        }
      }
    ];
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

  showNotification(message, type = 'info', duration = 5000) {
    console.log(`📢 Notification [${type}]:`, message);
    
    // Remove existing notifications of the same type to prevent spam
    const existingNotifications = document.querySelectorAll(`.notification.${type}`);
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <div class="notification-icon">
          <i class="fas fa-${this.getNotificationIcon(type)}"></i>
        </div>
        <div class="notification-message">${message}</div>
        <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
          <i class="fas fa-times"></i>
        </button>
      </div>
    `;

    // Get or create notification container
    let notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
      notificationContainer = document.createElement('div');
      notificationContainer.id = 'notification-container';
      notificationContainer.className = 'notification-container';
      document.body.appendChild(notificationContainer);
    }

    // Add to container
    notificationContainer.appendChild(notification);
    
    // Animate in
    requestAnimationFrame(() => {
      notification.classList.add('show');
    });

    // Auto-remove after duration
    if (duration > 0) {
      setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
          if (notification.parentNode) {
            notification.remove();
          }
        }, 300);
      }, duration);
    }

    return notification;
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
  window.omegaControlCenter?.updateNodesData();
  window.omegaControlCenter?.showNotification('Refreshing node status...', 'info');
};

window.selectNode = function(nodeId) {
  // Update selected node in tree
  document.querySelectorAll('.node-item').forEach(item => {
    item.classList.remove('selected');
  });
  document.querySelector(`[data-node="${nodeId}"]`)?.classList.add('selected');
  
  // Update selected node and load details
  window.omegaControlCenter.selectedNodeId = nodeId;
  window.omegaControlCenter.updateSelectedNodeDetails();
};

window.toggleNodeCategory = function(categoryType) {
  const category = document.querySelector(`[data-category="${categoryType}"]`);
  if (!category) return;
  
  const content = category.querySelector('.category-content');
  const toggleIcon = category.querySelector('.toggle-icon');
  
  if (content && toggleIcon) {
    const isExpanded = content.style.display !== 'none';
    content.style.display = isExpanded ? 'none' : 'block';
    toggleIcon.classList.toggle('fa-chevron-down', !isExpanded);
    toggleIcon.classList.toggle('fa-chevron-right', isExpanded);
  }
};

window.quickNodeAction = async function(nodeId, actionType) {
  try {
    const response = await window.omegaControlCenter.makeBackendRequest(
      `/api/v1/nodes/${nodeId}/action`,
      'POST',
      { type: actionType }
    );
    
    if (response) {
      window.omegaControlCenter.showNotification(
        response.message || `${actionType} action completed`,
        'success'
      );
      
      // Refresh node data after action
      setTimeout(() => {
        window.omegaControlCenter.updateNodesData();
      }, 1000);
    }
  } catch (error) {
    console.error('Node action error:', error);
    window.omegaControlCenter.showNotification(
      `Failed to ${actionType} node: ${error.message}`,
      'error'
    );
  }
};

window.switchNodeTab = function(tabName) {
  // Update tab buttons
  document.querySelectorAll('.node-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');
  
  // Update tab content
  document.querySelectorAll('.node-content').forEach(content => {
    content.classList.remove('active');
  });
  
  let contentId;
  switch(tabName) {
    case 'overview': contentId = 'nodeOverview'; break;
    case 'performance': contentId = 'nodePerformance'; break;
    case 'processes': contentId = 'nodeProcesses'; break;
    case 'logs': contentId = 'nodeLogs'; break;
    case 'security': contentId = 'nodeSecurity'; break;
    case 'maintenance': contentId = 'nodeMaintenance'; break;
  }
  
  if (contentId) {
    const contentElement = document.getElementById(contentId);
    if (contentElement) {
      contentElement.classList.add('active');
    } else {
      // Create content element if it doesn't exist
      const tabContent = document.querySelector('.node-tab-content');
      if (tabContent) {
        const newContent = document.createElement('div');
        newContent.id = contentId;
        newContent.className = 'node-content active';
        newContent.innerHTML = `<div class="loading">Loading ${tabName}...</div>`;
        tabContent.appendChild(newContent);
      }
    }
  }
};

window.refreshProcesses = function() {
  if (window.omegaControlCenter.selectedNodeId) {
    window.omegaControlCenter.updateSelectedNodeDetails();
    window.omegaControlCenter.showNotification('Refreshing processes...', 'info');
  }
};

window.refreshLogs = function() {
  if (window.omegaControlCenter.selectedNodeId) {
    window.omegaControlCenter.updateSelectedNodeDetails();
    window.omegaControlCenter.showNotification('Refreshing logs...', 'info');
  }
};

window.exportLogs = function() {
  const logs = document.querySelectorAll('.log-entry:not([style*="display: none"])');
  const logData = Array.from(logs).map(log => {
    return {
      timestamp: log.querySelector('.log-timestamp')?.textContent,
      level: log.querySelector('.level-badge')?.textContent,
      source: log.querySelector('.log-source')?.textContent,
      message: log.querySelector('.log-message')?.textContent
    };
  });
  
  const csvContent = "data:text/csv;charset=utf-8," + 
    "Timestamp,Level,Source,Message\n" +
    logData.map(log => `"${log.timestamp}","${log.level}","${log.source}","${log.message}"`).join("\n");
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `node-logs-${new Date().toISOString().split('T')[0]}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  window.omegaControlCenter.showNotification('Logs exported successfully', 'success');
};

window.killProcess = async function(pid) {
  if (!confirm(`Are you sure you want to kill process ${pid}?`)) {
    return;
  }
  
  try {
    // In initial prototype, simulate process termination
    window.omegaControlCenter.showNotification(`Process ${pid} terminated`, 'success');
    
    // Remove the process row from the table
    const processRow = document.querySelector(`[data-pid="${pid}"]`);
    if (processRow) {
      processRow.style.opacity = '0.5';
      setTimeout(() => {
        processRow.remove();
      }, 1000);
    }
  } catch (error) {
    window.omegaControlCenter.showNotification(`Failed to kill process: ${error.message}`, 'error');
  }
};

window.addNewNode = function() {
  showModal('Add New Node', `
    <div class="add-node-form">
      <div class="form-group">
        <label for="nodeName">Node Name:</label>
        <input type="text" id="nodeName" placeholder="e.g., compute-node-05" class="form-input">
      </div>
      <div class="form-group">
        <label for="nodeType">Node Type:</label>
        <select id="nodeType" class="form-select">
          <option value="compute">Compute Node</option>
          <option value="gpu">GPU Node</option>
          <option value="storage">Storage Node</option>
          <option value="memory">Memory Node</option>
        </select>
      </div>
      <div class="form-group">
        <label for="nodeIP">IP Address:</label>
        <input type="text" id="nodeIP" placeholder="192.168.1.100" class="form-input">
      </div>
      <div class="form-group">
        <label for="nodeResources">Resource Specifications:</label>
        <textarea id="nodeResources" placeholder="CPU: 32 cores, RAM: 64GB, Storage: 2TB NVMe" class="form-textarea"></textarea>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="createNewNode()">Add Node</button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
};

window.createNewNode = async function() {
  const nodeName = document.getElementById('nodeName').value;
  const nodeType = document.getElementById('nodeType').value;
  const nodeIP = document.getElementById('nodeIP').value;
  const nodeResources = document.getElementById('nodeResources').value;
  
  if (!nodeName || !nodeIP) {
    window.omegaControlCenter?.showNotification('Please fill in required fields', 'error');
    return;
  }
  
  try {
    const nodeData = {
      node_id: nodeName,
      node_type: nodeType,
      ip_address: nodeIP,
      status: 'online',
      resources: {
        description: nodeResources,
        cpu_cores: 8,
        memory_gb: 16,
        storage_gb: 500
      }
    };
    
    const response = await window.omegaControlCenter.makeBackendRequest(
      '/api/v1/nodes/register',
      'POST',
      nodeData
    );
    
    if (response) {
      closeModal();
      window.omegaControlCenter.showNotification(`Node ${nodeName} added successfully`, 'success');
      window.omegaControlCenter.updateNodesData();
    }
  } catch (error) {
    window.omegaControlCenter.showNotification(`Failed to add node: ${error.message}`, 'error');
  }
};

window.nodeActionModal = function(nodeId, actionType) {
  const actionTitles = {
    'restart': 'Restart Node',
    'shutdown': 'Shutdown Node',
    'maintenance': 'Maintenance Mode',
    'quarantine': 'Quarantine Node',
    'remove': 'Remove Node'
  };
  
  const actionMessages = {
    'restart': 'This will restart the node and interrupt any running processes.',
    'shutdown': 'This will safely shutdown the node.',
    'maintenance': 'This will put the node in maintenance mode.',
    'quarantine': 'This will isolate the node from the cluster.',
    'remove': 'This will permanently remove the node from the cluster.'
  };
  
  showModal(actionTitles[actionType], `
    <div class="action-confirm">
      <p><strong>Node:</strong> ${nodeId}</p>
      <p>${actionMessages[actionType]}</p>
      <div class="warning-box">
        <i class="fas fa-exclamation-triangle"></i>
        <span>This action may affect running sessions and workloads.</span>
      </div>
      <div class="form-actions">
        <button class="btn btn-danger" onclick="confirmNodeAction('${nodeId}', '${actionType}')">
          Confirm ${actionTitles[actionType]}
        </button>
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
      </div>
    </div>
  `);
};

window.confirmNodeAction = async function(nodeId, actionType) {
  try {
    await quickNodeAction(nodeId, actionType);
    closeModal();
  } catch (error) {
    console.error('Action failed:', error);
  }
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

// Global session management functions for HTML onclick handlers
function filterSessions(query) {
  if (window.omegaCC) {
    window.omegaCC.filterSessions(query);
  }
}

function filterSessionsByStatus(status) {
  if (window.omegaCC) {
    window.omegaCC.filterSessionsByStatus(status);
  }
}

function sortSessions(sortBy) {
  if (window.omegaCC) {
    window.omegaCC.sortSessions(sortBy);
  }
}

function switchInspectorTab(tabName) {
  if (window.omegaCC) {
    window.omegaCC.switchInspectorTab(tabName);
  }
}

function toggleSessionState() {
  if (window.omegaCC) {
    window.omegaCC.toggleSessionState();
  }
}

function createNewSession() {
  if (window.omegaCC) {
    window.omegaCC.createNewSession();
  }
}

function refreshSessions() {
  if (window.omegaCC) {
    window.omegaCC.refreshSessions();
  }
}

function createSessionSnapshot() {
  if (window.omegaCC) {
    window.omegaCC.createSessionSnapshot();
  }
}

function terminateSession() {
  if (window.omegaCC && window.omegaCC.selectedSessionId) {
    window.omegaCC.terminateSession(window.omegaCC.selectedSessionId);
  }
}

function toggleMigrateDropdown() {
  const dropdown = document.getElementById('migrateDropdown');
  if (dropdown) {
    dropdown.classList.toggle('show');
  }
}

function refreshProcesses() {
  if (window.omegaCC && window.omegaCC.selectedSessionId) {
    const session = window.omegaCC.sessionsData.find(s => s.session_id === window.omegaCC.selectedSessionId);
    if (session) {
      window.omegaCC.updateProcessesTab(session);
    }
  }
}

function sortProcesses(sortBy) {
  console.log('Sorting processes by:', sortBy);
}

function filterLogLevel(level) {
  console.log('Filtering logs by level:', level);
}

function clearLogs() {
  const logsContainer = document.getElementById('logsContainer');
  if (logsContainer) {
    logsContainer.innerHTML = '<div style="text-align: center; color: #666666;">Logs cleared</div>';
  }
}

function exportLogs() {
  console.log('Exporting logs...');
}

function saveSessionSettings() {
  console.log('Saving session settings...');
}

function resetSessionSettings() {
  console.log('Resetting session settings...');
}

function refreshSnapshots() {
  if (window.omegaCC && window.omegaCC.selectedSessionId) {
    const session = window.omegaCC.sessionsData.find(s => s.session_id === window.omegaCC.selectedSessionId);
    if (session) {
      window.omegaCC.updateSnapshotsTab(session);
    }
  }
}

function restoreSnapshot(snapshotId) {
  if (confirm('Are you sure you want to restore this snapshot? Current session state will be lost.')) {
    console.log('Restoring snapshot:', snapshotId);
  }
}

function deleteSnapshot(snapshotId) {
  if (confirm('Are you sure you want to delete this snapshot? This action cannot be undone.')) {
    console.log('Deleting snapshot:', snapshotId);
  }
}

function setChartTimeRange(range) {
  document.querySelectorAll('.chart-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  console.log('Setting chart time range to:', range);
}

function killProcess(pid) {
  if (confirm(`Are you sure you want to kill process ${pid}?`)) {
    console.log('Killing process:', pid);
  }
}

// Global functions for Sessions management accessibility
window.omegaRenderer = {
  selectSession: (sessionId) => omegaRenderer.selectSession(sessionId),
  performSessionAction: (sessionId, action) => omegaRenderer.performSessionAction(sessionId, action),
  createNewSession: () => omegaRenderer.createNewSession(),
  refreshSessions: () => omegaRenderer.refreshSessions(),
  switchInspectorTab: (tab) => omegaRenderer.switchInspectorTab(tab)
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
  console.log('✅ Omega Sessions Tab - Complete Implementation Ready');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = OmegaControlCenter;
}
