const { app, BrowserWindow, ipcMain, Menu, dialog, shell } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

// Keep a global reference of the window object
let mainWindow;
let isServerReady = false;

// Configure app
app.setName('Ω Control Center');

// Enhanced window creation with state management
function createWindow() {
  // Manage window state
  let mainWindowState = windowStateKeeper({
    defaultWidth: 1600,
    defaultHeight: 900
  });

  mainWindow = new BrowserWindow({
    x: mainWindowState.x,
    y: mainWindowState.y,
    width: mainWindowState.width,
    height: mainWindowState.height,
    minWidth: 1200,
    minHeight: 700,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: path.join(__dirname, 'assets/icon.png'),
    show: false // Don't show until ready
  });

  // Manage window state
  mainWindowState.manage(mainWindow);

  // Load the app
  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Create application menu
  createMenu();
  
  // Setup auto-updater in initial prototype
  if (!isDev) {
    autoUpdater.checkForUpdatesAndNotify();
  }
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Session',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow.webContents.send('menu-new-session')
        },
        {
          label: 'Open Logs',
          click: () => shell.openPath(path.join(__dirname, 'logs'))
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => app.quit()
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Cluster',
      submenu: [
        {
          label: 'Add Node',
          accelerator: 'CmdOrCtrl+Shift+A',
          click: () => mainWindow.webContents.send('menu-add-node')
        },
        {
          label: 'System Diagnostics',
          click: () => mainWindow.webContents.send('menu-diagnostics')
        },
        {
          label: 'Performance Tuning',
          click: () => mainWindow.webContents.send('menu-performance')
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://omega-docs.example.com')
        },
        {
          label: 'Report Issue',
          click: () => shell.openExternal('https://github.com/omega/issues')
        },
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Omega Super Desktop Console',
              message: 'Omega Super Desktop Console v1.0.0',
              detail: 'Distributed computing console that aggregates multiple PC resources.'
            });
          }
        }
      ]
    }
  ];

  // macOS menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App event handlers
app.whenReady().then(() => {
  createWindow();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers for backend communication

// Authentication
ipcMain.handle('auth-login', async (event, credentials) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(credentials)
    });
    
    if (response.ok) {
      const result = await response.json();
      authToken = result.access_token;
      return { success: true, token: authToken };
    } else {
      return { success: false, error: 'Invalid credentials' };
    }
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Node management
ipcMain.handle('get-nodes', async () => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/nodes`, {
      headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {}
    });
    
    if (response.ok) {
      const result = await response.json();
      return result.nodes || [];
    }
    return [];
  } catch (error) {
    console.error('Error fetching nodes:', error);
    return [];
  }
});

ipcMain.handle('register-node', async (event, nodeInfo) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/nodes/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(nodeInfo)
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error registering node:', error);
    return false;
  }
});

// Session management
ipcMain.handle('get-sessions', async () => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/sessions`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    if (response.ok) {
      const result = await response.json();
      return result.sessions || [];
    }
    return [];
  } catch (error) {
    console.error('Error fetching sessions:', error);
    return [];
  }
});

ipcMain.handle('create-session', async (event, sessionRequest) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/sessions/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(sessionRequest)
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    console.error('Error creating session:', error);
    return null;
  }
});

ipcMain.handle('terminate-session', async (event, sessionId) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error terminating session:', error);
    return false;
  }
});

// Metrics and monitoring
ipcMain.handle('get-dashboard-metrics', async () => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/metrics/dashboard`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return null;
  }
});

ipcMain.handle('report-latency', async (event, latencyData) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/metrics/latency`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(latencyData)
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error reporting latency:', error);
    return false;
  }
});

// Task execution
ipcMain.handle('execute-task', async (event, taskRequest) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/tasks/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(taskRequest)
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    console.error('Error executing task:', error);
    return null;
  }
});

ipcMain.handle('get-task-status', async (event, taskId) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/tasks/${taskId}`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    console.error('Error fetching task status:', error);
    return null;
  }
});

// AI prediction
ipcMain.handle('predict-input', async (event, inputData) => {
  try {
    const response = await fetch(`${CONTROL_NODE_URL}/api/v1/ai/predict_input`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(inputData)
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    console.error('Error predicting input:', error);
    return null;
  }
});

// System utilities
ipcMain.handle('open-external', async (event, url) => {
  shell.openExternal(url);
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

// Auto-updater events
autoUpdater.on('checking-for-update', () => {
  console.log('Checking for update...');
});

autoUpdater.on('update-available', (info) => {
  console.log('Update available.');
});

autoUpdater.on('update-not-available', (info) => {
  console.log('Update not available.');
});

autoUpdater.on('error', (err) => {
  console.log('Error in auto-updater. ' + err);
});

autoUpdater.on('download-progress', (progressObj) => {
  let log_message = "Download speed: " + progressObj.bytesPerSecond;
  log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
  log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
  console.log(log_message);
});

autoUpdater.on('update-downloaded', (info) => {
  console.log('Update downloaded');
  autoUpdater.quitAndInstall();
});
