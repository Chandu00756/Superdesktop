export function renderSessions(root, state) {
  // Ensure state structure exists
  if (!state || !state.data || !state.data.sessions) {
    console.log('[Sessions] State structure missing:', { state, data: state?.data, sessions: state?.data?.sessions });
    root.innerHTML = '<div style="padding: 20px; color: #ff4444;">State not initialized properly</div>';
    return;
  }
  
  const sData = state.data.sessions || {};
  const sessions = sData.sessions || [];
  
  // Create advanced sessions interface with virtual desktop integration
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
      <!-- Session Controls -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; align-items: center;">
  <button onclick="openVdCreateDialog()" class="session-create-btn">
          <i class="fas fa-video"></i>
          <div>New WebRTC Session</div>
          <div>Ultra-low latency desktop</div>
        </button>
        <button onclick="createSession('rdp')" class="session-create-btn">
          <i class="fas fa-desktop"></i>
          <div>Connect via RDP</div>
          <div>Remote Desktop Protocol</div>
        </button>
        <button onclick="createSession('vnc')" class="session-create-btn">
          <i class="fas fa-eye"></i>
          <div>Connect via VNC</div>
          <div>Virtual Network Computing</div>
        </button>
        <button onclick="createSession('ssh')" class="session-create-btn">
          <i class="fas fa-terminal"></i>
          <div>SSH Terminal</div>
          <div>Command line access</div>
        </button>
      </div>
      
      <!-- Main Sessions Layout -->
      <div style="display: grid; grid-template-columns: 300px 1fr; gap: 16px; height: 100%;">
        <!-- Session List -->
        <div style="display: flex; flex-direction: column; background: var(--omega-dark-3); border: 1px solid var(--omega-gray-1); border-radius: 4px; overflow: hidden;">
          <div style="padding: 12px; font: 600 12px var(--font-mono); color: var(--omega-cyan); border-bottom: 1px solid var(--omega-gray-1); display: flex; justify-content: space-between; align-items: center;">
            <span><i class="fas fa-desktop"></i> SESSIONS (${sessions.length})</span>
            <button onclick="refreshSessions()" class="btn-secondary" style="font-size: 10px; padding: 4px 8px;">
              <i class="fas fa-sync"></i>
            </button>
          </div>
          <div id="session-list" style="flex: 1; overflow: auto;"></div>
        </div>
        
        <!-- Session Detail/Inspector -->
        <div style="display: flex; flex-direction: column; background: var(--omega-dark-3); border: 1px solid var(--omega-gray-1); border-radius: 4px; overflow: hidden;">
          <div id="session-inspector-header" style="padding: 12px; border-bottom: 1px solid var(--omega-gray-1);">
            <div style="font: 600 14px var(--font-mono); color: var(--omega-white);">Session Inspector</div>
          </div>
          
          <!-- Session Sub-tabs -->
          <div id="session-subtabs" style="display: none; background: var(--omega-dark-2); border-bottom: 1px solid var(--omega-gray-1);">
            <div style="display: flex;">
              <button class="session-subtab active" data-subtab="overview">Overview</button>
              <button class="session-subtab" data-subtab="desktop">Desktop</button>
              <button class="session-subtab" data-subtab="performance">Performance</button>
              <button class="session-subtab" data-subtab="files">Files</button>
              <button class="session-subtab" data-subtab="processes">Processes</button>
              <button class="session-subtab" data-subtab="snapshots">Snapshots</button>
              <button class="session-subtab" data-subtab="settings">Settings</button>
            </div>
          </div>
          
          <!-- Session Content -->
          <div id="session-content" style="flex: 1; overflow: hidden;">
            <div id="session-overview" class="session-panel active">
              <div style="padding: 16px; text-align: center; color: var(--omega-light-1);">
                Select a session to view details
              </div>
            </div>
            <div id="session-desktop" class="session-panel">
              <div id="desktop-canvas-container" style="width: 100%; height: 100%; background: var(--omega-black); position: relative;">
                <!-- Virtual desktop will be rendered here -->
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: var(--omega-light-1);">
                  <i class="fas fa-desktop" style="font-size: 48px; opacity: 0.3; margin-bottom: 16px;"></i>
                  <div>Desktop session not active</div>
                </div>
              </div>
            </div>
            <div id="session-performance" class="session-panel">
              <div style="padding: 16px;">
                <h4>Session Performance Metrics</h4>
                <div id="session-perf-charts"></div>
              </div>
            </div>
            <div id="session-files" class="session-panel">
              <div style="padding: 16px;">
                <h4>Session File Browser</h4>
                <div id="session-file-tree"></div>
              </div>
            </div>
            <div id="session-processes" class="session-panel">
              <div style="padding: 16px;">
                <h4>Session Processes</h4>
                <div id="session-process-list"></div>
              </div>
            </div>
            <div id="session-snapshots" class="session-panel">
              <div style="padding: 16px;">
                <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom: 12px;">
                  <h4>Snapshots</h4>
                  <div style="display:flex; gap:8px;">
                    <button class="btn-secondary" id="snapshot-refresh-btn"><i class="fas fa-sync"></i> Refresh</button>
                    <button class="btn-secondary" id="snapshot-create-btn"><i class="fas fa-camera"></i> New Snapshot</button>
                  </div>
                </div>
                <div id="session-snapshot-list"></div>
              </div>
            </div>
            <div id="session-settings" class="session-panel">
              <div style="padding: 16px;">
                <h4>Session Configuration</h4>
                <div id="session-config-form"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const list = root.querySelector('#session-list');
  let selectedSession = null;
  
  // Render session list
  sessions.forEach(s => {
    const status = s.status || 'unknown';
    const statusColor = status === 'running' ? '--omega-green' : status === 'paused' ? '--omega-yellow' : '--omega-red';
    const lastActivity = new Date((s.last_activity || s.created_at) * 1000);
    
    const sessionItem = document.createElement('div');
    sessionItem.style.cssText = `
      cursor: pointer; padding: 12px; border-bottom: 1px solid var(--omega-gray-2);
      transition: background 0.15s ease;
    `;
    sessionItem.onmouseenter = () => sessionItem.style.background = 'var(--omega-dark-4)';
    sessionItem.onmouseleave = () => sessionItem.style.background = 'transparent';
    sessionItem.onclick = () => selectSession(s);
    
    sessionItem.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div style="font: 600 12px var(--font-mono); color: var(--omega-white);">
          ${s.session_id.substring(0, 8)}...
        </div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background: var(${statusColor});"></div>
      </div>
      <div style="font: 400 10px var(--font-mono); color: var(--omega-light-1); margin-bottom: 4px;">
        ${s.application} • ${s.user_id}
      </div>
      <div style="display: flex; justify-content: space-between; font: 400 9px var(--font-mono); color: var(--omega-light-1);">
        <span>Node: ${s.node_id}</span>
        <span>${lastActivity.toLocaleTimeString()}</span>
      </div>
      <div style="margin-top: 6px; display: flex; gap: 8px; font: 400 9px var(--font-mono);">
        <span>CPU: ${s.cpu_cores}</span>
        <span>GPU: ${s.gpu_units}</span>
        <span>RAM: ${s.memory_gb}GB</span>
      </div>
    `;
    list.appendChild(sessionItem);
  });

  function selectSession(session) {
    selectedSession = session;
  try { localStorage.setItem('omega:lastSession', session.session_id); } catch {}
    const header = root.querySelector('#session-inspector-header');
    const subtabs = root.querySelector('#session-subtabs');
    
    header.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <div style="font: 600 16px var(--font-mono); color: var(--omega-cyan);">
            Session ${session.session_id.substring(0, 12)}
          </div>
          <div style="font: 400 12px var(--font-mono); color: var(--omega-light-1);">
            ${session.application} • ${session.user_id} • ${session.node_id}
          </div>
        </div>
        <div style="display: flex; gap: 8px;">
          <button onclick="pauseSession('${session.session_id}')" class="btn-secondary">
            <i class="fas fa-pause"></i> ${session.status === 'paused' ? 'Resume' : 'Pause'}
          </button>
          <button onclick="terminateSession('${session.session_id}')" class="btn-danger">
            <i class="fas fa-stop"></i> Terminate
          </button>
        </div>
      </div>
    `;
    
    subtabs.style.display = 'block';
    showSessionSubtab('overview');
    
    // Setup subtab handlers
    root.querySelectorAll('.session-subtab').forEach(tab => {
      tab.onclick = () => showSessionSubtab(tab.dataset.subtab);
    });
  }

  function showSessionSubtab(subtab) {
  try { localStorage.setItem('omega:lastSessionTab', subtab); } catch {}
    // Update active tab
    root.querySelectorAll('.session-subtab').forEach(t => {
      t.classList.toggle('active', t.dataset.subtab === subtab);
    });
    
    // Show/hide panels
    root.querySelectorAll('.session-panel').forEach(p => {
      p.classList.toggle('active', p.id === `session-${subtab}`);
    });
    
    // Render content based on subtab
    switch(subtab) {
      case 'overview':
        renderSessionOverview();
        break;
      case 'desktop':
        renderSessionDesktop();
        break;
      case 'performance':
        renderSessionPerformance();
        break;
      case 'files':
        renderSessionFiles();
        break;
      case 'processes':
        renderSessionProcesses();
        break;
      case 'settings':
        renderSessionSettings();
        break;
      case 'snapshots':
        renderSessionSnapshots();
        break;
    }
  }

  function renderSessionOverview() {
    const panel = root.querySelector('#session-overview');
    if (!selectedSession) return;
    
    const s = selectedSession;
    const uptime = ((Date.now() / 1000) - s.created_at) / 3600; // hours
    
    panel.innerHTML = `
      <div style="padding: 16px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
          <div class="session-info-card">
            <h4><i class="fas fa-info-circle"></i> Session Details</h4>
            <div class="info-grid">
              ${infoItem('Session ID', s.session_id)}
              ${infoItem('User', s.user_id)}
              ${infoItem('Application', s.application)}
              ${infoItem('Status', s.status)}
              ${infoItem('Created', new Date(s.created_at * 1000).toLocaleString())}
              ${infoItem('Uptime', uptime.toFixed(1) + ' hours')}
            </div>
          </div>
          
          <div class="session-info-card">
            <h4><i class="fas fa-server"></i> Resource Allocation</h4>
            <div class="info-grid">
              ${infoItem('Node', s.node_id)}
              ${infoItem('CPU Cores', s.cpu_cores)}
              ${infoItem('GPU Units', s.gpu_units)}
              ${infoItem('Memory', s.memory_gb + ' GB')}
              ${infoItem('Storage', '50 GB (allocated)')}
              ${infoItem('Network', 'Gigabit')}
            </div>
          </div>
          
          <div class="session-info-card">
            <h4><i class="fas fa-chart-line"></i> Current Metrics</h4>
            <div class="info-grid">
              ${infoItem('CPU Usage', '45.2%')}
              ${infoItem('Memory Usage', '67.8%')}
              ${infoItem('GPU Usage', '12.1%')}
              ${infoItem('Network I/O', '125 KB/s')}
              ${infoItem('Disk I/O', '2.1 MB/s')}
              ${infoItem('Latency', '8ms')}
            </div>
          </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
          <div class="session-info-card">
            <h4><i class="fas fa-history"></i> Recent Activity</h4>
            <div class="activity-log">
              <div class="activity-item">
                <div class="activity-time">14:32</div>
                <div class="activity-desc">Application started successfully</div>
              </div>
              <div class="activity-item">
                <div class="activity-time">14:28</div>
                <div class="activity-desc">Session created and resources allocated</div>
              </div>
              <div class="activity-item">
                <div class="activity-time">14:27</div>
                <div class="activity-desc">User authentication completed</div>
              </div>
            </div>
          </div>
          
          <div class="session-info-card">
            <h4><i class="fas fa-cogs"></i> Quick Actions</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
              <button onclick="connectDesktop('${s.session_id}')" class="session-action-btn">
                <i class="fas fa-external-link-alt"></i> Connect Desktop
              </button>
              <button onclick="snapshotSession('${s.session_id}')" class="session-action-btn">
                <i class="fas fa-camera"></i> Snapshot
              </button>
              <button onclick="shareSession('${s.session_id}')" class="session-action-btn">
                <i class="fas fa-share"></i> Share
              </button>
              <button onclick="backupSession('${s.session_id}')" class="session-action-btn">
                <i class="fas fa-save"></i> Backup
              </button>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderSessionDesktop() {
    const container = root.querySelector('#desktop-canvas-container');
    if (!selectedSession) return;
    
    // Create virtual desktop interface
    container.innerHTML = `
      <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column;">
        <!-- Desktop Toolbar -->
        <div style="background: var(--omega-dark-2); border-bottom: 1px solid var(--omega-gray-1); padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
          <div style="display: flex; gap: 8px; align-items: center;">
            <button onclick="connectDesktop('${selectedSession.session_id}')" class="desktop-tool-btn">
              <i class="fas fa-play"></i> Connect
            </button>
            <button onclick="disconnectDesktop()" class="desktop-tool-btn">
              <i class="fas fa-stop"></i> Disconnect
            </button>
            <div style="width: 1px; height: 20px; background: var(--omega-gray-1); margin: 0 8px;"></div>
            <button onclick="fullscreenDesktop()" class="desktop-tool-btn">
              <i class="fas fa-expand"></i> Fullscreen
            </button>
            <button onclick="screenshotDesktop()" class="desktop-tool-btn">
              <i class="fas fa-camera"></i> Screenshot
            </button>
          </div>
          <div style="display: flex; gap: 8px; align-items: center; font: 400 11px var(--font-mono); color: var(--omega-light-1);">
            <span>Quality: High</span>
            <span>•</span>
            <span>Latency: 8ms</span>
            <span>•</span>
            <span>FPS: 60</span>
          </div>
        </div>
        
        <!-- Desktop Canvas -->
        <div style="flex: 1; position: relative; background: var(--omega-black);">
          <canvas id="desktop-canvas" style="width: 100%; height: 100%; object-fit: contain;"></canvas>
          <div id="desktop-overlay" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: var(--omega-light-1); pointer-events: none;">
            <i class="fas fa-desktop" style="font-size: 64px; opacity: 0.2; margin-bottom: 16px;"></i>
            <div style="font: 400 14px var(--font-mono);">Click Connect to start desktop session</div>
            <div style="font: 400 12px var(--font-mono); margin-top: 8px; opacity: 0.7;">WebRTC • Ultra-low latency</div>
          </div>
        </div>
        
        <!-- Desktop Status Bar -->
        <div style="background: var(--omega-dark-2); border-top: 1px solid var(--omega-gray-1); padding: 6px 12px; display: flex; justify-content: between; align-items: center; font: 400 10px var(--font-mono);">
          <div style="display: flex; gap: 12px;">
            <span>Resolution: 1920x1080</span>
            <span>Color Depth: 24-bit</span>
            <span>Compression: H.264</span>
          </div>
          <div id="desktop-status" style="color: var(--omega-red);">Disconnected</div>
        </div>
      </div>
    `;
  }

  function renderSessionPerformance() {
    const panel = root.querySelector('#session-performance');
    panel.innerHTML = `
      <div style="padding: 16px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
          <div class="perf-metric-card">
            <div class="metric-header">CPU Usage</div>
            <div class="metric-value">45.2%</div>
            <div class="metric-chart">
              <canvas id="cpu-mini-chart" width="100" height="30"></canvas>
            </div>
          </div>
          <div class="perf-metric-card">
            <div class="metric-header">Memory Usage</div>
            <div class="metric-value">67.8%</div>
            <div class="metric-chart">
              <canvas id="mem-mini-chart" width="100" height="30"></canvas>
            </div>
          </div>
          <div class="perf-metric-card">
            <div class="metric-header">GPU Usage</div>
            <div class="metric-value">12.1%</div>
            <div class="metric-chart">
              <canvas id="gpu-mini-chart" width="100" height="30"></canvas>
            </div>
          </div>
          <div class="perf-metric-card">
            <div class="metric-header">Network I/O</div>
            <div class="metric-value">125 KB/s</div>
            <div class="metric-chart">
              <canvas id="net-mini-chart" width="100" height="30"></canvas>
            </div>
          </div>
        </div>
        
        <div class="session-info-card">
          <h4><i class="fas fa-chart-area"></i> Performance History</h4>
          <canvas id="session-perf-history" style="width: 100%; height: 200px;"></canvas>
        </div>
      </div>
    `;
  }

  function renderSessionFiles() {
    const panel = root.querySelector('#session-files');
    panel.innerHTML = `
      <div style="padding: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h4><i class="fas fa-folder"></i> Session File Browser</h4>
          <div style="display: flex; gap: 8px;">
            <button class="btn-secondary" style="font-size: 11px;">
              <i class="fas fa-upload"></i> Upload
            </button>
            <button class="btn-secondary" style="font-size: 11px;">
              <i class="fas fa-folder-plus"></i> New Folder
            </button>
          </div>
        </div>
        
        <div class="file-browser">
          <div class="file-browser-header">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
              <button class="btn-secondary" style="font-size: 10px; padding: 4px 8px;">
                <i class="fas fa-arrow-left"></i>
              </button>
              <div style="flex: 1; background: var(--omega-dark-4); border: 1px solid var(--omega-gray-1); padding: 6px 10px; border-radius: 3px; font: 400 11px var(--font-mono);">
                /home/${selectedSession?.user_id || 'user'}
              </div>
            </div>
          </div>
          
          <div class="file-list">
            <div class="file-item folder">
              <i class="fas fa-folder"></i>
              <span>Documents</span>
              <span class="file-size">--</span>
              <span class="file-date">2024-08-10 14:30</span>
            </div>
            <div class="file-item folder">
              <i class="fas fa-folder"></i>
              <span>Downloads</span>
              <span class="file-size">--</span>
              <span class="file-date">2024-08-10 14:25</span>
            </div>
            <div class="file-item">
              <i class="fas fa-file-alt"></i>
              <span>readme.txt</span>
              <span class="file-size">2.1 KB</span>
              <span class="file-date">2024-08-10 13:45</span>
            </div>
            <div class="file-item">
              <i class="fas fa-image"></i>
              <span>screenshot.png</span>
              <span class="file-size">1.2 MB</span>
              <span class="file-date">2024-08-10 12:15</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderSessionProcesses() {
    const panel = root.querySelector('#session-processes');
    const processes = state.data.processes?.processes || [];
    
    panel.innerHTML = `
      <div style="padding: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h4><i class="fas fa-tasks"></i> Session Processes (${processes.length})</h4>
          <button onclick="refreshProcesses()" class="btn-secondary">
            <i class="fas fa-sync"></i> Refresh
          </button>
        </div>
        
        <div class="process-table">
          <div class="process-header">
            <div>PID</div>
            <div>Process Name</div>
            <div>CPU %</div>
            <div>Memory</div>
            <div>Status</div>
            <div>Actions</div>
          </div>
          <div class="process-list" style="max-height: 300px; overflow-y: auto;">
            ${processes.slice(0, 10).map(p => `
              <div class="process-row">
                <div>${p.pid}</div>
                <div style="font-family: var(--font-mono);">${p.name}</div>
                <div>${(p.cpu || 0).toFixed(1)}%</div>
                <div>${p.mem_mb} MB</div>
                <div><span class="status-badge running">Running</span></div>
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

  async function renderSessionSnapshots() {
    const panel = root.querySelector('#session-snapshots');
    if (!selectedSession) {
      panel.innerHTML = '<div style="padding: 16px; color: var(--omega-light-1);">No session selected</div>';
      return;
    }
    // Structure already in DOM; attach handlers and load data
    const listEl = panel.querySelector('#session-snapshot-list');
    const refreshBtn = panel.querySelector('#snapshot-refresh-btn');
    const createBtn = panel.querySelector('#snapshot-create-btn');

    refreshBtn.onclick = async () => {
      await loadSnapshots();
    };
    createBtn.onclick = async () => {
      try {
        createBtn.disabled = true; createBtn.textContent = 'Creating...';
        const res = await window.api.snapshotVirtualDesktop(selectedSession.session_id);
        window.notify('success', 'Snapshot', `Created: ${res.snapshot}`);
        await loadSnapshots();
      } catch (e) {
        window.notify('error', 'Snapshot', e.message);
        showInlineError('session-snapshots', 'Snapshot failed', e.message);
      } finally {
        createBtn.disabled = false; createBtn.innerHTML = '<i class="fas fa-camera"></i> New Snapshot';
      }
    };

    async function loadSnapshots() {
      listEl.innerHTML = '<div style="color: var(--omega-light-1);">Loading snapshots...</div>';
      try {
        const resp = await window.api.listSnapshots(selectedSession.session_id);
        const items = resp.snapshots || [];
        if (!items.length) {
          listEl.innerHTML = '<div style="color: var(--omega-light-1);">No snapshots yet</div>';
          return;
        }
        listEl.innerHTML = `
          <div class="snapshot-timeline">
            ${items.map(evt => snapshotItem(evt)).join('')}
          </div>
        `;
      } catch (e) {
        listEl.innerHTML = '<div style="color: var(--omega-red);">Failed to load snapshots</div>';
        showInlineError('session-snapshots', 'Failed to load snapshots', e.message);
      }
    }

    function snapshotItem(evt) {
      const ts = new Date((evt.created_at || Date.now()/1000) * 1000).toLocaleString();
      const tag = evt.tag || (evt.message?.replace('Snapshot created: ','') || 'snapshot');
      return `
        <div class="snapshot-item">
          <div class="snapshot-meta">
            <div class="snapshot-time"><i class="fas fa-clock"></i> ${ts}</div>
            <div class="snapshot-tag">${tag}</div>
          </div>
          <div style="display:flex; gap:6px; margin-top:6px;">
            <button class="btn-secondary" onclick="restoreSnapshot('${selectedSession.session_id}','${tag}')"><i class="fas fa-undo"></i> Restore</button>
            <button class="btn-danger" onclick="deleteSnapshot('${selectedSession.session_id}','${tag}')"><i class="fas fa-trash"></i> Delete</button>
          </div>
        </div>
      `;
    }

    await loadSnapshots();
  }

  function renderSessionSettings() {
    const panel = root.querySelector('#session-settings');
    if (!selectedSession) return;
    
    panel.innerHTML = `
      <div style="padding: 16px;">
        <h4><i class="fas fa-cog"></i> Session Configuration</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px;">
          <div class="settings-section">
            <h5>Resource Limits</h5>
            <div class="setting-item">
              <label>CPU Cores</label>
              <input type="number" value="${selectedSession.cpu_cores}" min="1" max="16">
            </div>
            <div class="setting-item">
              <label>Memory (GB)</label>
              <input type="number" value="${selectedSession.memory_gb}" min="1" max="64">
            </div>
            <div class="setting-item">
              <label>GPU Units</label>
              <input type="number" value="${selectedSession.gpu_units}" min="0" max="4">
            </div>
          </div>
          
          <div class="settings-section">
            <h5>Display Settings</h5>
            <div class="setting-item">
              <label>Resolution</label>
              <select>
                <option>1920x1080</option>
                <option>2560x1440</option>
                <option>3840x2160</option>
              </select>
            </div>
            <div class="setting-item">
              <label>Color Depth</label>
              <select>
                <option>24-bit</option>
                <option>16-bit</option>
                <option>32-bit</option>
              </select>
            </div>
            <div class="setting-item">
              <label>Frame Rate</label>
              <select>
                <option>60 FPS</option>
                <option>30 FPS</option>
                <option>120 FPS</option>
              </select>
            </div>
          </div>
        </div>
        
        <div style="margin-top: 20px; display: flex; gap: 8px;">
          <button onclick="saveSessionSettings('${selectedSession.session_id}')" class="btn-primary">
            <i class="fas fa-save"></i> Save Settings
          </button>
          <button onclick="resetSessionSettings('${selectedSession.session_id}')" class="btn-secondary">
            <i class="fas fa-undo"></i> Reset to Defaults
          </button>
        </div>
      </div>
    `;
  }

  function infoItem(label, value) {
    return `
      <div class="info-item">
        <div class="info-label">${label}</div>
        <div class="info-value">${value}</div>
      </div>
    `;
  }

  // Global functions for session management
  // (OS dropdown removed; OS selection happens in the create dialog. Default image is ubuntu-xfce.)

  // Create dialog for VD creation
  window.openVdCreateDialog = async () => {
    // Load OS catalog and profiles concurrently
    const [cat, profResp] = await Promise.all([
      window.api.getOsCatalog().catch(()=>({ builtin: [], custom: [] })),
      window.api.getVdProfiles().catch(() => ({ profiles: [
        { id: 'browser', label: 'Browser', packages: ['firefox','curl','wget','zip','unzip'] },
        { id: 'developer', label: 'Developer', packages: ['firefox','git','curl','wget','htop','build-essential','vim','python3','python3-pip'] },
        { id: 'office', label: 'Office', packages: ['firefox','libreoffice','curl','zip','unzip'] }
      ] }))
    ]);
    const options = [...(cat.builtin||[]), ...(cat.custom||[])];
    const selOptions = options.map(o => `<option value="${o.id}">${o.id}${o.description?` (${o.description})`:''}</option>`).join('');
    const profiles = profResp?.profiles || [];
    const profOptions = (profiles.length
      ? profiles.map(p => `<option value="${p.id}" ${p.id==='browser'?'selected':''}>${p.label}</option>`).join('')
      : '<option value="browser" selected>Browser</option>');
    const profMap = Object.fromEntries((profiles||[]).map(p => [p.id, p.packages || []]));
    const html = `
      <div class="inline-rdp-dialog" id="vd-create-dialog"><div class="rdp-form" style="width:520px;">
        <div style="font:600 14px var(--font-mono); color: var(--omega-white); margin-bottom:8px;">Create Virtual Desktop</div>
        <div class="rdp-form-row"><label>OS</label><select id="vd-os">${selOptions || '<option value="ubuntu-xfce">ubuntu-xfce (default)</option>'}</select></div>
        <div class="rdp-form-row"><label>Profile</label><select id="vd-profile">${profOptions}</select></div>
        <div class="rdp-form-row"><label>CPU Cores</label><input id="vd-cpu" type="number" min="1" max="16" value="2"/></div>
        <div class="rdp-form-row"><label>Memory (GB)</label><input id="vd-ram" type="number" min="2" max="64" value="4"/></div>
        <div class="rdp-form-row"><label>Resolution</label><input id="vd-res" placeholder="1920x1080"/></div>
        <div class="rdp-form-row"><label>VNC Password</label><input id="vd-pass" type="password" placeholder="auto-generate"/></div>
        <div class="rdp-form-row"><label>Packages</label><input id="vd-pkgs" placeholder="comma separated (e.g., firefox,vlc,libreoffice)"/></div>
        <div class="rdp-actions">
          <button id="vd-cancel" class="btn-secondary">Cancel</button>
          <button id="vd-create" class="btn-primary">Create</button>
        </div>
      </div></div>`;
    const dlg = document.createElement('div'); dlg.innerHTML = html; document.body.appendChild(dlg);
    // Auto-fill packages when profile changes (for transparency)
    const profileSel = dlg.querySelector('#vd-profile');
    const pkgsInput = dlg.querySelector('#vd-pkgs');
    const syncPkgs = () => {
      const pid = profileSel?.value;
      if (pid && profMap[pid]) pkgsInput.value = (profMap[pid] || []).join(',');
    };
    if (profileSel && pkgsInput) {
      syncPkgs();
      profileSel.onchange = syncPkgs;
    }
    dlg.querySelector('#vd-cancel').onclick = () => dlg.remove();
    dlg.querySelector('#vd-create').onclick = async () => {
      try {
        const os_image = dlg.querySelector('#vd-os').value || 'ubuntu-xfce';
        const profile = (dlg.querySelector('#vd-profile')?.value || 'browser');
        const cpu_cores = Math.max(1, parseInt(dlg.querySelector('#vd-cpu').value||'2',10)||2);
        const memory_gb = Math.max(1, parseInt(dlg.querySelector('#vd-ram').value||'4',10)||4);
        const resolution = (dlg.querySelector('#vd-res').value||'').trim() || undefined;
        const vnc_password = (dlg.querySelector('#vd-pass').value||'').trim() || undefined;
        const pkgsRaw = (dlg.querySelector('#vd-pkgs').value||'').trim();
        const packages = pkgsRaw ? pkgsRaw.split(',').map(s=>s.trim()).filter(Boolean) : undefined;
        window.notify('info', 'Session Creation', 'Launching desktop...');
        const pkt = await window.api.createVirtualDesktop({ user_id: 'admin', os_image, cpu_cores, memory_gb, gpu_units: 0, packages, resolution, vnc_password, profile });
        dlg.remove();
        const sessionsPkt = await window.api.getSessions();
        window.state.setState('sessions', sessionsPkt);
        // auto-select and connect
        const newId = pkt.session_id;
        const newSession = (sessionsPkt.sessions||[]).find(s=>s.session_id===newId);
        if (newSession) {
          selectSession(newSession);
          showSessionSubtab('desktop');
          setTimeout(() => window.connectDesktop(newId), 300);
        }
        window.notify('success', 'Virtual Desktop', 'Desktop created');
      } catch (e) {
        window.notify('error', 'Virtual Desktop', e.message||String(e));
      }
    };
  };

  window.createSession = async (type) => {
    try {
      window.notify('info', 'Session Creation', `Creating ${type} session...`);
      // Disable create buttons to prevent duplicate calls
      const btns = Array.from(document.querySelectorAll('.session-create-btn'));
      btns.forEach(b => { b.disabled = true; b.classList.add('is-disabled'); });
      if (type === 'webrtc') {
        // Fallback path (primary flow uses openVdCreateDialog). Default OS image.
        const os_image = 'ubuntu-xfce';
        const pkt = await window.api.createVirtualDesktop({ user_id: 'admin', os_image, profile: 'browser' });
        const payload = pkt;
        // Refresh sessions list from backend
        const sessionsPkt = await window.api.getSessions();
        window.state.setState('sessions', sessionsPkt);
        window.notify('success', 'Virtual Desktop', 'Desktop created. Select it to connect.');
        // Auto-select new session if visible
        const newId = payload.session_id;
        const newSession = (sessionsPkt.sessions || []).find(s => s.session_id === newId);
        if (newSession) {
          selectSession(newSession);
          showSessionSubtab('desktop');
          // auto-connect
          setTimeout(() => window.connectDesktop(newId), 300);
        }
      } else if (type === 'vnc') {
        const os_image = 'ubuntu-xfce';
        const pkt = await window.api.createVirtualDesktop({ user_id: 'admin', os_image, profile: 'browser' });
        const sessionsPkt = await window.api.getSessions();
        window.state.setState('sessions', sessionsPkt);
        window.notify('success', 'Virtual Desktop', 'VNC desktop created');
      } else if (type === 'rdp') {
        // Show inline RDP dialog
        const dialog = document.createElement('div');
        dialog.className = 'inline-rdp-dialog';
        dialog.innerHTML = `
          <div class="rdp-form">
            <div class="rdp-form-row"><label>Host</label><input id="rdp-host" placeholder="ip or hostname"/></div>
            <div class="rdp-form-row"><label>Port</label><input id="rdp-port" type="number" value="3389"/></div>
            <div class="rdp-form-row"><label>Username</label><input id="rdp-user" placeholder="domain\\user or user"/></div>
            <div class="rdp-form-row"><label>Password</label><input id="rdp-pass" type="password" placeholder="password"/></div>
            <div class="rdp-form-row"><label>Domain</label><input id="rdp-domain" placeholder="optional"/></div>
            <div class="rdp-actions">
              <button id="rdp-cancel" class="btn-secondary">Cancel</button>
              <button id="rdp-create" class="btn-primary">Create</button>
            </div>
          </div>`;
        document.body.appendChild(dialog);
        await new Promise((resolve) => {
          dialog.querySelector('#rdp-cancel').onclick = () => { dialog.remove(); resolve(); };
          dialog.querySelector('#rdp-create').onclick = async () => {
            try {
              const host = dialog.querySelector('#rdp-host').value.trim();
              const port = parseInt(dialog.querySelector('#rdp-port').value || '3389', 10) || 3389;
              const username = dialog.querySelector('#rdp-user').value.trim();
              const password = dialog.querySelector('#rdp-pass').value;
              const domain = dialog.querySelector('#rdp-domain').value.trim();
              if (!host || !username || !password) throw new Error('Host, username, password required');
              await window.api.createRdpSession({ user_id: 'admin', host, port, username, password, domain });
              dialog.remove();
              resolve();
            } catch (e) {
              window.notify('error', 'RDP', e.message);
            }
          };
        });
        const sessionsPkt = await window.api.getSessions();
        window.state.setState('sessions', sessionsPkt);
        window.notify('success', 'RDP', 'RDP session created. Select it to open.');
      } else if (type === 'ssh') {
        window.notify('warning', 'SSH', 'SSH terminal will be added after VD core.');
      }
    } catch (e) {
      window.notify('error', 'Session Creation', String(e?.message || e));
      showInlineError('session-list', 'Failed to create session', e.message);
    } finally {
      const btns = Array.from(document.querySelectorAll('.session-create-btn'));
      btns.forEach(b => { b.disabled = false; b.classList.remove('is-disabled'); });
    }
  };

  window.connectDesktop = async (sessionId) => {
    try {
      window.notify('info', 'Desktop Connection', `Connecting to session ${sessionId}...`);
      // Detect session type
      const curSessions = window.state.data.sessions?.sessions || [];
      const s = curSessions.find(x => x.session_id === sessionId);
      if (s && s.application === 'rdp-desktop') {
        const info = await window.api.getRdpUrl(sessionId);
        const url = info.connect_url;
        window.notify('info', 'RDP', 'Opening RDP client...');
        window.open(url, '_blank');
        return;
      }
      // Wait for viewer readiness: poll health for up to ~10s
      const url = await (async () => {
        const start = Date.now();
        const timeoutMs = 10000;
        const delay = (ms) => new Promise(r => setTimeout(r, ms));
        // Helper to check health
        const healthy = async () => {
          try {
            const pkt = await window.api.getVirtualDesktopHealth(sessionId);
            return !!pkt.ready;
          } catch {
            return false;
          }
        };
        // First ask for URL (so we can embed immediately if already ready)
        let currentUrl = await window.api.getVirtualDesktopUrl(sessionId).then(p=>p.connect_url);
        if (await healthy()) return currentUrl;
        // Poll a few times with backoff
        let attempt = 0; const steps = [250, 400, 600, 800, 1200, 1600, 2000, 2000];
        while (Date.now() - start < timeoutMs && attempt < steps.length) {
          await delay(steps[attempt++]);
          currentUrl = await window.api.getVirtualDesktopUrl(sessionId).then(p=>p.connect_url);
          if (await healthy()) return currentUrl;
        }
        // Return last known URL anyway
        return currentUrl;
      })();
      const canvas = document.getElementById('desktop-canvas');
      const overlay = document.getElementById('desktop-overlay');
      const container = canvas?.parentElement;
      // Replace canvas with iframe of noVNC page for now
      if (container) {
        container.innerHTML = `<iframe id="novnc-frame" src="${url}" style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;background:#000;"></iframe>`;
      }
      const status = document.getElementById('desktop-status');
      if (status) { status.textContent = 'Connected'; status.style.color = 'var(--omega-green)'; }
      if (overlay) overlay.style.display = 'none';
    } catch (e) {
      const msg = String(e?.message || e);
      window.notify('error', 'Desktop Connection', msg);
      showInlineError('session-desktop', 'Failed to connect desktop', msg);
    }
  };

  window.pauseSession = async (sessionId) => {
    try {
      // Toggle based on current selected state
      if (selectedSession && selectedSession.session_id === sessionId && selectedSession.status === 'paused') {
        await window.api.resumeVirtualDesktop(sessionId);
        window.notify('success', 'Session Control', `Session ${sessionId} resumed`);
      } else {
        await window.api.pauseVirtualDesktop(sessionId);
        window.notify('success', 'Session Control', `Session ${sessionId} paused`);
      }
      const sessionsPkt = await window.api.getSessions();
      window.state.setState('sessions', sessionsPkt);
  // Re-render to update button labels and statuses
  renderSessions(root, window.state);
    } catch (e) {
      window.notify('error', 'Session Control', e.message);
  showInlineError('session-inspector-header', 'Failed to change session state', e.message);
    }
  };

  window.terminateSession = async (sessionId) => {
    try {
  // delete VD and remove from list
  await window.api.deleteVirtualDesktop(sessionId);
  const sessionsPkt = await window.api.getSessions();
  window.state.setState('sessions', sessionsPkt);
  window.notify('success', 'Session Control', `Session ${sessionId} terminated`);
  // Auto-refresh list UI
  renderSessions(root, window.state);
    } catch (e) {
      window.notify('error', 'Session Control', e.message);
      showInlineError('session-inspector-header', 'Failed to terminate session', e.message);
    }
  };

  // Extra actions wired to backend
  window.snapshotSession = async (sessionId) => {
    try {
      const res = await window.api.snapshotVirtualDesktop(sessionId);
      window.notify('success', 'Snapshot', `Snapshot created: ${res.snapshot}`);
    } catch (e) {
      window.notify('error', 'Snapshot', e.message);
  showInlineError('session-overview', 'Snapshot failed', e.message);
    }
  };

  window.killProcess = async (pid) => {
    try {
      await window.api.killProcess(pid);
      window.notify('success', 'Process', `PID ${pid} terminated`);
      const procs = await window.api.getProcesses();
      window.state.setState('processes', procs);
      renderSessionProcesses();
    } catch (e) {
      window.notify('error', 'Process', e.message);
      showInlineError('session-processes', `Failed to kill PID ${pid}`, e.message);
    }
  };

  window.deleteSnapshot = async (sessionId, tag) => {
    try {
      await window.api.deleteSnapshot(sessionId, tag);
      window.notify('success', 'Snapshot', `Deleted ${tag}`);
      const panel = root.querySelector('#session-snapshots');
      if (panel) {
        await renderSessionSnapshots();
      }
    } catch (e) {
      window.notify('error', 'Snapshot', e.message);
    }
  };

  window.restoreSnapshot = async (sessionId, tag) => {
    try {
      const res = await window.api.restoreSnapshot(sessionId, tag);
      window.notify('success', 'Snapshot', 'Restored snapshot to new container');
      // Optionally open restored URL in new window
      if (res.connect_url) {
        window.open(res.connect_url, '_blank');
      }
    } catch (e) {
      window.notify('error', 'Snapshot', e.message);
    }
  };

  // Helper to show inline error banners near panels
  function showInlineError(containerId, title, message) {
    const container = root.querySelector(`#${containerId}`) || document.getElementById(containerId);
    if (!container) return;
    const banner = document.createElement('div');
    banner.className = 'inline-error';
    banner.innerHTML = `
      <div class="inline-error-content">
        <i class="fas fa-exclamation-circle"></i>
        <div>
          <div class="inline-error-title">${title}</div>
          <div class="inline-error-message">${message}</div>
        </div>
        <button class="inline-error-close" aria-label="Dismiss">×</button>
      </div>`;
    const close = () => banner.remove();
    banner.querySelector('.inline-error-close').onclick = close;
    setTimeout(close, 6000);
    container.parentElement?.insertBefore(banner, container.nextSibling);
  }

  // Refresh helpers
  window.refreshSessions = async () => {
    try {
      const sessionsPkt = await window.api.getSessions();
      window.state.setState('sessions', sessionsPkt);
      // Re-render sessions tab to reflect new list
      renderSessions(root, window.state);
    } catch (e) {
      window.notify('error', 'Sessions', e.message);
    }
  };

  window.refreshProcesses = async () => {
    try {
      const procs = await window.api.getProcesses();
      window.state.setState('processes', procs);
      // Re-render the processes panel if present
      renderSessionProcesses();
    } catch (e) {
      window.notify('error', 'Processes', e.message);
    }
  };

  // UI stubs to avoid undefined handler errors
  window.disconnectDesktop = () => {
  if (window.vd) window.vd.disconnect();
    const frame = document.getElementById('novnc-frame');
    if (frame && frame.parentElement) {
      frame.parentElement.innerHTML = '<canvas id="desktop-canvas" style="width: 100%; height: 100%; object-fit: contain;"></canvas>';
    }
    const status = document.getElementById('desktop-status');
    if (status) { status.textContent = 'Disconnected'; status.style.color = 'var(--omega-red)'; }
    window.notify('info', 'Desktop', 'Disconnected');
  };
  window.fullscreenDesktop = () => {
    const el = document.getElementById('novnc-frame') || document.getElementById('desktop-canvas');
    if (el && el.requestFullscreen) el.requestFullscreen();
  };
  window.screenshotDesktop = () => {
    window.notify('warning', 'Desktop', 'Screenshot not implemented yet.');
  };
  window.shareSession = (sessionId) => {
    window.notify('info', 'Share', `Share link for ${sessionId} coming soon.`);
  };
  window.backupSession = (sessionId) => {
    window.notify('info', 'Backup', `Backup for ${sessionId} queued (stub).`);
  };

  // Add CSS for sessions
  if (!document.getElementById('sessions-styles')) {
    const style = document.createElement('style');
    style.id = 'sessions-styles';
    style.textContent = `
      .session-create-btn {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 6px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: var(--omega-white);
        text-align: center;
      }
      
      .session-create-btn:hover {
        border-color: var(--omega-cyan);
        background: var(--omega-dark-3);
        transform: translateY(-2px);
      }
      
      .session-create-btn i {
        font-size: 24px;
        color: var(--omega-cyan);
        margin-bottom: 8px;
        display: block;
      }
      
      .session-create-btn > div:first-of-type {
        font: 600 13px var(--font-mono);
        margin-bottom: 4px;
      }
      
      .session-create-btn > div:last-of-type {
        font: 400 10px var(--font-mono);
        opacity: 0.7;
      }
      
      .session-subtab {
        background: transparent;
        border: none;
        color: var(--omega-light-1);
        padding: 8px 16px;
        cursor: pointer;
        font: 400 11px var(--font-mono);
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
      }
      
      .session-subtab:hover {
        background: var(--omega-dark-3);
        color: var(--omega-cyan);
      }
      
      .session-subtab.active {
        background: var(--omega-dark-3);
        color: var(--omega-cyan);
        border-bottom-color: var(--omega-cyan);
      }
      
      .session-panel {
        display: none;
        height: 100%;
        overflow: auto;
      }
      
      .session-panel.active {
        display: block;
      }
      
      .session-info-card {
        background: var(--omega-dark-4);
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        padding: 16px;
      }
      
      .session-info-card h4 {
        color: var(--omega-cyan);
        font: 600 12px var(--font-mono);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      
      .info-grid {
        display: grid;
        gap: 8px;
      }
      
      .info-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid var(--omega-gray-2);
        font: 400 11px var(--font-mono);
      }
      
      .info-label {
        color: var(--omega-light-1);
      }
      
      .info-value {
        color: var(--omega-white);
        font-weight: 600;
      }
      
      .desktop-tool-btn {
        background: var(--omega-dark-3);
        border: 1px solid var(--omega-gray-1);
        color: var(--omega-white);
        padding: 6px 10px;
        border-radius: 3px;
        cursor: pointer;
        font: 400 10px var(--font-mono);
        transition: all 0.15s ease;
      }
      
      .desktop-tool-btn:hover {
        border-color: var(--omega-cyan);
        background: var(--omega-dark-4);
      }
      
      .session-action-btn {
        background: var(--omega-dark-3);
        border: 1px solid var(--omega-gray-1);
        color: var(--omega-white);
        padding: 8px 12px;
        border-radius: 3px;
        cursor: pointer;
        font: 400 10px var(--font-mono);
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      
      .session-action-btn:hover {
        border-color: var(--omega-cyan);
        background: var(--omega-dark-4);
      }
      
      .file-browser {
        border: 1px solid var(--omega-gray-1);
        border-radius: 4px;
        overflow: hidden;
      }
      
      .file-list {
        background: var(--omega-dark-4);
      }
      
      .file-item {
        display: grid;
        grid-template-columns: 20px 1fr auto auto;
        gap: 12px;
        padding: 8px 12px;
        border-bottom: 1px solid var(--omega-gray-2);
        font: 400 11px var(--font-mono);
        cursor: pointer;
        transition: background 0.15s ease;
      }
      
      .file-item:hover {
        background: var(--omega-dark-3);
      }
      
      .file-item.folder {
        color: var(--omega-cyan);
      }
      
      .file-size, .file-date {
        color: var(--omega-light-1);
        font-size: 10px;
      }
      
      .settings-section h5 {
        color: var(--omega-white);
        font: 600 12px var(--font-mono);
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--omega-gray-2);
      }
      
      .setting-item {
        margin-bottom: 12px;
      }
      
      .setting-item label {
        display: block;
        font: 400 11px var(--font-mono);
        color: var(--omega-light-1);
        margin-bottom: 4px;
      }
      
      .setting-item input,
      .setting-item select {
        width: 100%;
        background: var(--omega-dark-3);
        border: 1px solid var(--omega-gray-1);
        color: var(--omega-white);
        padding: 6px 8px;
        border-radius: 3px;
        font: 400 11px var(--font-mono);
      }
      
      .setting-item input:focus,
      .setting-item select:focus {
        outline: none;
        border-color: var(--omega-cyan);
      }
    `;
    document.head.appendChild(style);
  }

  // Extra styles for disabled buttons and inline errors (separate tag to avoid editing large block above)
  if (!document.getElementById('sessions-styles-extra')) {
    const style2 = document.createElement('style');
    style2.id = 'sessions-styles-extra';
    style2.textContent = `
      .session-create-btn.is-disabled { opacity: 0.6; cursor: not-allowed; }
      .inline-error { background: rgba(255, 77, 77, 0.08); border: 1px solid rgba(255, 77, 77, 0.45); color: #ff7777; border-radius: 4px; margin: 8px 12px; }
      .inline-error-content { display: flex; gap: 10px; align-items: center; padding: 10px 12px; }
      .inline-error-title { font: 600 12px var(--font-mono); }
      .inline-error-message { font: 400 11px var(--font-mono); opacity: 0.9; }
      .inline-error-close { margin-left: auto; background: transparent; border: none; color: inherit; font-size: 16px; cursor: pointer; }
  .snapshot-timeline { display: grid; gap: 8px; }
  .snapshot-item { border: 1px solid var(--omega-gray-1); background: var(--omega-dark-4); border-radius: 4px; padding: 8px 10px; }
  .snapshot-meta { display: flex; justify-content: space-between; font: 400 11px var(--font-mono); color: var(--omega-light-1); }
  .snapshot-tag { color: var(--omega-cyan); font-weight: 600; }
  .inline-rdp-dialog { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: grid; place-items: center; z-index: 9999; }
  .rdp-form { width: 360px; background: var(--omega-dark-3); border: 1px solid var(--omega-gray-1); border-radius: 4px; padding: 12px; }
  .rdp-form-row { display: grid; grid-template-columns: 90px 1fr; gap: 8px; align-items: center; margin-bottom: 8px; }
  .rdp-form-row label { color: var(--omega-light-1); font: 600 11px var(--font-mono); }
  .rdp-form-row input { background: var(--omega-dark-2); border: 1px solid var(--omega-gray-1); color: var(--omega-white); padding: 6px 8px; border-radius: 3px; font: 400 11px var(--font-mono); }
  /* Keep selects within the dialog and visually consistent with inputs */
  .rdp-form-row select { 
    background: var(--omega-dark-2);
    border: 1px solid var(--omega-gray-1);
    color: var(--omega-white);
    padding: 6px 8px;
    border-radius: 3px;
    font: 400 11px var(--font-mono);
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
  }
  /* Specific fix for the VD create dialog where long OS labels caused overflow */
  #vd-create-dialog .rdp-form { max-width: 560px; width: 520px; }
  #vd-create-dialog select { width: 100%; max-width: 100%; }
  @media (max-width: 600px) {
    #vd-create-dialog .rdp-form { width: 92vw; }
  }
  .rdp-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 6px; }
    `;
    document.head.appendChild(style2);
  }

  // Auto-restore last selected session and subtab after render
  try {
    const savedId = localStorage.getItem('omega:lastSession');
    if (savedId) {
      const s = sessions.find(x => x.session_id === savedId);
      if (s) {
        selectSession(s);
        const tab = localStorage.getItem('omega:lastSessionTab') || 'desktop';
        showSessionSubtab(tab);
        if (tab === 'desktop') {
          setTimeout(() => window.connectDesktop(savedId), 300);
        }
      }
    }
  } catch {}
}
