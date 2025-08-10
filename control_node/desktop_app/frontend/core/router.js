import { renderDashboard } from '../tabs/dashboard.js';
import { renderNodes } from '../tabs/nodes.js';
import { renderResources } from '../tabs/resources.js';
import { renderSessions } from '../tabs/sessions.js';
import { renderNetwork } from '../tabs/network.js';
import { renderPerformance } from '../tabs/performance.js';
import { renderSecurity } from '../tabs/security.js';
import { renderPlugins } from '../tabs/plugins.js';
import { renderAIHub } from '../tabs/ai-hub.js';
import { renderSettings } from '../tabs/settings.js';

export const Tabs = ['dashboard','nodes','resources','sessions','network','performance','security','plugins','ai-hub','settings'];

export function mountTab(name, state){
  const container = document.getElementById('tab-root');
  if(!container) return;
  container.innerHTML = '';
  const map = { 'dashboard':renderDashboard, 'nodes':renderNodes, 'resources':renderResources, 'sessions':renderSessions, 'network':renderNetwork, 'performance':renderPerformance, 'security':renderSecurity, 'plugins':renderPlugins, 'ai-hub':renderAIHub, 'settings':renderSettings };
  (map[name]||(()=>container.textContent='Unknown Tab')).call(null, container, state);
}
