const { app, BrowserWindow, ipcMain, Menu, dialog, shell } = require('electron');
const path = require('path');
const fetch = require('node-fetch'); // For API calls
const isDev = process.env.NODE_ENV === 'development';

// Keep a global reference of the window object
let mainWindow;
let isServerReady = false;

// Configure app
app.setName('Ω Control Center');

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    frame: true, // Use system default frame
    titleBarStyle: 'default', // Use system default title bar
    titleBarOverlay: false, // No custom title bar overlay
    transparent: false, // Not transparent
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: !isDev
    },
    icon: path.join(__dirname, 'assets', 'omega-icon.png'),
    show: false, // Don't show until ready
    backgroundColor: '#000000'
  });

  // Load the app
  const startUrl = isDev 
    ? 'http://localhost:8001' 
    : `file://${path.join(__dirname, 'omega-control-center.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Show window when ready to prevent visual flash
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

  // Handle window maximize/unmaximize
  mainWindow.on('maximize', () => {
    mainWindow.webContents.send('window-maximized');
  });

  mainWindow.on('unmaximize', () => {
    mainWindow.webContents.send('window-unmaximized');
  });

  // Configure menu
  createMenu();
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Configuration',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu-action', 'new-config');
          }
        },
        {
          label: 'Open Configuration',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile'],
              filters: [
                { name: 'Omega Config', extensions: ['omega', 'json'] },
                { name: 'All Files', extensions: ['*'] }
              ]
            });
            
            if (!result.canceled) {
              mainWindow.webContents.send('menu-action', 'open-config', result.filePaths[0]);
            }
          }
        },
        {
          label: 'Save Configuration',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            mainWindow.webContents.send('menu-action', 'save-config');
          }
        },
        { type: 'separator' },
        {
          label: 'Import Settings',
          click: () => {
            mainWindow.webContents.send('menu-action', 'import-settings');
          }
        },
        {
          label: 'Export Settings',
          click: () => {
            mainWindow.webContents.send('menu-action', 'export-settings');
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Cluster',
      submenu: [
        {
          label: 'Discover Nodes',
          accelerator: 'CmdOrCtrl+D',
          click: () => {
            mainWindow.webContents.send('menu-action', 'discover-nodes');
          }
        },
        {
          label: 'Add Node',
          accelerator: 'CmdOrCtrl+Shift+A',
          click: () => {
            mainWindow.webContents.send('menu-action', 'add-node');
          }
        },
        { type: 'separator' },
        {
          label: 'Start Cluster',
          click: () => {
            mainWindow.webContents.send('menu-action', 'start-cluster');
          }
        },
        {
          label: 'Stop Cluster',
          click: () => {
            mainWindow.webContents.send('menu-action', 'stop-cluster');
          }
        },
        {
          label: 'Restart Cluster',
          click: () => {
            mainWindow.webContents.send('menu-action', 'restart-cluster');
          }
        },
        { type: 'separator' },
        {
          label: 'Health Check',
          accelerator: 'CmdOrCtrl+H',
          click: () => {
            mainWindow.webContents.send('menu-action', 'health-check');
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        {
          label: 'Dashboard',
          accelerator: 'CmdOrCtrl+1',
          click: () => {
            mainWindow.webContents.send('switch-tab', 'dashboard');
          }
        },
        {
          label: 'Nodes',
          accelerator: 'CmdOrCtrl+2',
          click: () => {
            mainWindow.webContents.send('switch-tab', 'nodes');
          }
        },
        {
          label: 'Performance',
          accelerator: 'CmdOrCtrl+3',
          click: () => {
            mainWindow.webContents.send('switch-tab', 'performance');
          }
        },
        { type: 'separator' },
        {
          label: 'Reload',
          accelerator: 'CmdOrCtrl+R',
          click: () => {
            mainWindow.reload();
          }
        },
        {
          label: 'Force Reload',
          accelerator: 'CmdOrCtrl+Shift+R',
          click: () => {
            mainWindow.webContents.reloadIgnoringCache();
          }
        },
        {
          label: 'Toggle Developer Tools',
          accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
          click: () => {
            mainWindow.webContents.toggleDevTools();
          }
        },
        { type: 'separator' },
        {
          label: 'Actual Size',
          accelerator: 'CmdOrCtrl+0',
          click: () => {
            mainWindow.webContents.setZoomLevel(0);
          }
        },
        {
          label: 'Zoom In',
          accelerator: 'CmdOrCtrl+Plus',
          click: () => {
            const currentZoom = mainWindow.webContents.getZoomLevel();
            mainWindow.webContents.setZoomLevel(currentZoom + 1);
          }
        },
        {
          label: 'Zoom Out',
          accelerator: 'CmdOrCtrl+-',
          click: () => {
            const currentZoom = mainWindow.webContents.getZoomLevel();
            mainWindow.webContents.setZoomLevel(currentZoom - 1);
          }
        },
        { type: 'separator' },
        {
          label: 'Toggle Fullscreen',
          accelerator: process.platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11',
          click: () => {
            mainWindow.setFullScreen(!mainWindow.isFullScreen());
          }
        }
      ]
    },
    {
      label: 'Tools',
      submenu: [
        {
          label: 'Benchmark Suite',
          click: () => {
            mainWindow.webContents.send('menu-action', 'benchmark');
          }
        },
        {
          label: 'Latency Analyzer',
          click: () => {
            mainWindow.webContents.send('menu-action', 'latency-analyzer');
          }
        },
        {
          label: 'Network Diagnostics',
          click: () => {
            mainWindow.webContents.send('menu-action', 'network-diagnostics');
          }
        },
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: 'CmdOrCtrl+,',
          click: () => {
            mainWindow.webContents.send('menu-action', 'preferences');
          }
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Getting Started',
          click: () => {
            shell.openExternal('https://omega-docs.example.com/getting-started');
          }
        },
        {
          label: 'Documentation',
          click: () => {
            shell.openExternal('https://omega-docs.example.com');
          }
        },
        {
          label: 'Community Discord',
          click: () => {
            shell.openExternal('https://discord.gg/omega');
          }
        },
        { type: 'separator' },
        {
          label: 'Report Issue',
          click: () => {
            shell.openExternal('https://github.com/omega/omega-desktop/issues');
          }
        },
        {
          label: 'Check for Updates',
          click: () => {
            mainWindow.webContents.send('menu-action', 'check-updates');
          }
        },
        { type: 'separator' },
        {
          label: 'About Omega Control Center',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Omega Control Center',
              message: 'Ω Control Center - Personal Supercomputer',
              detail: 'Version 1.0.0-rc.2\nDistributed Computing Platform\n\n© 2025 Omega Technologies',
              buttons: ['OK']
            });
          }
        }
      ]
    }
  ];

  // macOS specific menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        {
          label: 'About ' + app.getName(),
          role: 'about'
        },
        { type: 'separator' },
        {
          label: 'Services',
          role: 'services',
          submenu: []
        },
        { type: 'separator' },
        {
          label: 'Hide ' + app.getName(),
          accelerator: 'Command+H',
          role: 'hide'
        },
        {
          label: 'Hide Others',
          accelerator: 'Command+Shift+H',
          role: 'hideothers'
        },
        {
          label: 'Show All',
          role: 'unhide'
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: 'Command+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC handlers
ipcMain.handle('window-minimize', () => {
  mainWindow.minimize();
});

ipcMain.handle('window-maximize', () => {
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});

ipcMain.handle('window-close', () => {
  mainWindow.close();
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-message-box', async (event, options) => {
  const result = await dialog.showMessageBox(mainWindow, options);
  return result;
});

// API handlers for backend communication
ipcMain.handle('get-dashboard-metrics', async () => {
  try {
    const https = require('https');
    const agent = new https.Agent({
      rejectUnauthorized: false // Accept self-signed certificates
    });
    
    const response = await fetch('https://localhost:8443/api/v1/dashboard', { agent });
    if (response.ok) {
      return await response.json();
    } else {
      console.error('Dashboard API error:', response.status);
      return null;
    }
  } catch (error) {
    console.error('Error fetching dashboard metrics:', error);
    return null;
  }
});

ipcMain.handle('auth-login', async (event, credentials) => {
  try {
    // In a real app, this would authenticate with the backend
    // For now, we'll simulate authentication
    if (credentials.username && credentials.password) {
      return {
        success: true,
        token: 'omega-session-token',
        user: { username: credentials.username, role: 'admin' }
      };
    } else {
      return { success: false, error: 'Invalid credentials' };
    }
  } catch (error) {
    console.error('Authentication error:', error);
    return { success: false, error: 'Authentication failed' };
  }
});

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

app.on('before-quit', (event) => {
  // Cleanup logic here if needed
  console.log('Omega Control Center shutting down...');
});

// Handle certificate errors in dev mode
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  if (isDev) {
    event.preventDefault();
    callback(true);
  } else {
    callback(false);
  }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (navigationEvent, url) => {
    navigationEvent.preventDefault();
    shell.openExternal(url);
  });
});

console.log('Omega Control Center - Electron Main Process Started');
console.log(`Environment: ${isDev ? 'Development' : 'Initial Prototype'}`);
console.log(`Platform: ${process.platform}`);
