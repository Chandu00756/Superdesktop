// Omega SuperDesktop v2.0 - Main Bootstrap Entry Point
import { renderDashboard } from './tabs/dashboard.js';
import { renderNodes } from './tabs/nodes.js';
import { renderSessions } from './tabs/sessions.js';
import { renderNetwork } from './tabs/network.js';
import { renderPerformance } from './tabs/performance.js';
import { renderSecurity } from './tabs/security.js';
import { renderPlugins } from './tabs/plugins.js';
import { renderAIHub } from './tabs/ai-hub.js';
import { renderSettings } from './tabs/settings.js';
import StateStore from './core/StateStore.js';
import { ApiClient } from './core/apiClient.js';
import { notify } from './core/notify.js';
import { SidebarManager } from './core/sidebarManager.js';
import { WidgetManager } from './core/widgetManager.js';

// Initialize global state and API
const initialState = {
  dashboard: {
    cluster: {},
    performance: {},
    alerts: []
  },
  nodes: {
    nodes: []
  },
  sessions: {},
  network: {
    statistics: {
      interfaces: []
    }
  },
  performance: {
    benchmark: {
      history: []
    }
  },
  processes: {
    processes: []
  },
  logs: {
    events: []
  },
  security: {
    users: [],
    certificates: []
  },
  plugins: {
    installed: []
  },
  aihub: {}
};

console.log('[Omega] Initializing state with:', initialState);
window.state = new StateStore(initialState);
console.log('[Omega] State initialized:', window.state);
console.log('[Omega] State.data:', window.state.data);
window.api = new ApiClient();
window.notify = notify;

// Tab registry
const tabs = {
  'dashboard': renderDashboard,
  'nodes': renderNodes, 
  'sessions': renderSessions,
  'network': renderNetwork,
  'performance': renderPerformance,
  'security': renderSecurity,
  'plugins': renderPlugins,
  'ai-hub': renderAIHub,
  'settings': renderSettings
};

// Global tab switching function
window.switchTab = function(tabName) {
  console.log(`[Omega] Switching to tab: ${tabName}`);
  console.log(`[Omega] Current state:`, window.state);
  console.log(`[Omega] State data:`, window.state?.data);
  
  // Update tab navigation
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  const tabBtn = document.querySelector(`[data-tab="${tabName}"]`);
  const tabContent = document.getElementById(tabName);
  
  if (tabBtn) tabBtn.classList.add('active');
  if (tabContent) tabContent.classList.add('active');
  
  // Render tab content
  const tabRoot = document.getElementById('tab-root');
  if (tabRoot && tabs[tabName]) {
    try {
      tabRoot.innerHTML = '';
      console.log(`[Omega] About to render ${tabName} with state:`, window.state);
      tabs[tabName](tabRoot, window.state);
      console.log(`[Omega] Successfully rendered ${tabName} tab`);
    } catch (error) {
      console.error(`[Omega] Error rendering ${tabName} tab:`, error);
      tabRoot.innerHTML = `<div style="padding: 20px; color: #ff4444;">Error loading ${tabName}: ${error.message}</div>`;
    }
  }
};

// Initialize the application
async function initializeOmega() {
  console.log('[Omega] Initializing SuperDesktop v2.0...');
  
  // Initialize sidebar
  const sidebarContainer = document.getElementById('omega-sidebar');
  if (sidebarContainer) {
    window.sidebarManager = new SidebarManager(sidebarContainer, window.state, window.api);
  }
  
  // Initialize widgets
  const widgetsContainer = document.getElementById('floating-widgets');
  if (widgetsContainer) {
    window.widgetManager = new WidgetManager(widgetsContainer, window.state, window.api);
  }
  
  // Load initial data
  try {
    console.log('[Omega] Loading dashboard data...');
    const dashboardData = await window.api.getDashboard();
    window.state.setState('dashboard', dashboardData);
    console.log('[Omega] Dashboard data loaded:', dashboardData);
  } catch (error) {
    console.error('[Omega] Failed to load dashboard data:', error);
    // Data structure is already initialized, no need to set empty data
  }
  
  // Set up initial tab
  setTimeout(() => {
    window.switchTab('dashboard');
  }, 100);
  
  // Start data refresh intervals
  setInterval(() => {
    if (window.state && typeof window.state.refreshData === 'function') {
      window.state.refreshData();
    }
  }, 5000);
  
  console.log('[Omega] Initialization complete');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeOmega);
} else {
  initializeOmega();
}

export { initializeOmega };
