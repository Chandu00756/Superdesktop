// Omega SuperDesktop v2.0 - Widget Manager
import { formatBytes, formatNumber } from './utils.js';

export class WidgetManager {
    constructor(container, state, api) {
        this.container = container;
        this.state = state;
        this.api = api;
        this.widgets = new Map();
        this.updateInterval = null;
        this.init();
    }

    init() {
        console.log('[WidgetManager] Initializing widgets...');
        this.render();
        this.startUpdateLoop();
        console.log('[WidgetManager] Widgets initialized');
    }

    render() {
        console.log('[WidgetManager] Rendering widgets...');
        const widgetConfigs = [
            { id: 'active-desktop', title: 'Active Desktop', icon: 'display', type: 'active-desktop' },
            { id: 'quick-actions', title: 'Quick Actions', icon: 'bolt', type: 'actions' },
            { id: 'resource-usage', title: 'Resource Usage', icon: 'chart-pie', type: 'resources' },
            { id: 'recent-activity', title: 'Recent Activity', icon: 'history', type: 'activity' },
            { id: 'network-status', title: 'Network Status', icon: 'network-wired', type: 'network' },
            { id: 'security-status', title: 'Security Status', icon: 'shield-alt', type: 'security' }
        ];

        const widgetHTML = widgetConfigs.map(config => this.renderWidget(config)).join('');
        
        this.container.innerHTML = widgetHTML;
        this.attachEventListeners();
    }

    renderWidget(config) {
        return `
            <div class="widget" id="widget-${config.id}" data-widget-type="${config.type}">
                <div class="widget-header">
                    <span>${config.title}</span>
                    <i class="fas fa-${config.icon}"></i>
                </div>
                <div class="widget-content" id="widget-content-${config.id}">
                    ${this.renderWidgetContent(config.type)}
                </div>
            </div>
        `;
    }

    renderWidgetContent(type) {
        switch (type) {
            case 'active-desktop':
                return this.renderActiveDesktop();
            case 'actions':
                return this.renderQuickActions();
            case 'resources':
                return this.renderResourceUsage();
            case 'activity':
                return this.renderRecentActivity();
            case 'network':
                return this.renderNetworkStatus();
            case 'security':
                return this.renderSecurityStatus();
            default:
                return '<div class="loading"><div class="spinner"></div>Loading...</div>';
        }
    }

    renderActiveDesktop() {
        const vd = window.vd?.getCurrent?.();
        if (!vd) {
            return `
                <div style="font-size: var(--font-size-xs); color: var(--omega-light-1);">
                    No active desktop
                </div>
                <div style="display:flex; gap:6px; margin-top:8px;">
                    <button class="widget-action-btn" data-action="open-sessions"><i class="fas fa-desktop"></i> Open Sessions</button>
                </div>
            `;
        }
        const shortId = vd.sessionId ? (vd.sessionId.slice(0, 10) + '…') : '--';
        return `
            <div style="display:flex; flex-direction:column; gap:6px; font-size: var(--font-size-xs);">
                <div><b>Session:</b> ${shortId}</div>
                <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"><b>URL:</b> ${vd.url}</div>
            </div>
            <div style="display:flex; gap:6px; margin-top:8px;">
                <button class="widget-action-btn" data-action="reopen-desktop"><i class="fas fa-external-link-alt"></i> Reopen</button>
                <button class="widget-action-btn" data-action="disconnect-desktop"><i class="fas fa-stop"></i> Disconnect</button>
            </div>
        `;
    }

    renderQuickActions() {
        const actions = [
            { id: 'deploy', label: 'Deploy Service', icon: 'rocket', color: 'var(--omega-cyan)' },
            { id: 'backup', label: 'Backup System', icon: 'save', color: 'var(--omega-green)' },
            { id: 'optimize', label: 'System Check', icon: 'magic', color: 'var(--omega-yellow)' },
            { id: 'restart', label: 'Restart Services', icon: 'redo', color: 'var(--omega-red)' }
        ];

        return actions.map(action => `
            <button class="widget-action-btn" data-action="${action.id}" style="
                width: 100%; 
                padding: 8px; 
                margin-bottom: 4px; 
                background: ${action.id === 'deploy' ? action.color : 'var(--omega-dark-4)'}; 
                color: ${action.id === 'deploy' ? 'var(--omega-black)' : 'var(--omega-white)'}; 
                border: ${action.id === 'deploy' ? 'none' : '1px solid var(--omega-gray-1)'}; 
                border-radius: 2px; 
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: var(--font-mono);
                font-size: var(--font-size-xs);
                transition: var(--transition-fast);
            ">
                <i class="fas fa-${action.icon}"></i>
                ${action.label}
            </button>
        `).join('');
    }

    renderResourceUsage() {
        const resources = this.state.getResourceUsage();
        
        const items = [
            { label: 'CPU Cores', value: resources.cpuCores || '8', unit: '' },
            { label: 'Memory', value: resources.memoryTotal || '16', unit: 'GB' },
            { label: 'Storage', value: resources.storageTotal || '1', unit: 'TB' },
            { label: 'GPU', value: resources.gpu || 'RTX 4090', unit: '' }
        ];

        return items.map(item => `
            <div class="metric-item">
                <span>${item.label}</span>
                <span class="metric-value">${item.value}${item.unit}</span>
            </div>
        `).join('');
    }

    renderRecentActivity() {
        const activities = this.state.getRecentActivity();
        
        if (!activities || activities.length === 0) {
            return `
                <div style="text-align: center; color: var(--omega-light-1); font-size: var(--font-size-xs); padding: 20px;">
                    No recent activity
                </div>
            `;
        }

        return activities.slice(0, 3).map(activity => `
            <div style="font-size: 10px; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--omega-gray-2);">
                <div style="color: ${this.getActivityColor(activity.type)}; display: flex; align-items: center; gap: 4px;">
                    <i class="fas fa-${this.getActivityIcon(activity.type)}"></i>
                    <span>● ${activity.message}</span>
                </div>
                <div style="opacity: 0.7; margin-top: 2px;">
                    ${this.formatTimeAgo(activity.timestamp)}
                </div>
            </div>
        `).join('');
    }

    renderNetworkStatus() {
        const network = this.state.getNetworkStatus();
        
        const items = [
            { label: 'Status', value: network.status || 'Connected', type: 'status' },
            { label: 'Latency', value: `${network.latency || '--'}ms`, type: 'metric' },
            { label: 'Bandwidth', value: `${network.bandwidth || '--'} Mbps`, type: 'metric' },
            { label: 'Active Connections', value: network.connections || 0, type: 'metric' }
        ];

        return items.map(item => `
            <div class="metric-item">
                <span>${item.label}</span>
                <span class="metric-value ${item.type === 'status' && item.value === 'Connected' ? 'status-good' : ''}">${item.value}</span>
            </div>
        `).join('') + `
            <style>
                .status-good { color: var(--omega-green) !important; }
            </style>
        `;
    }

    renderSecurityStatus() {
        const security = this.state.getSecurityStatus();
        
        const items = [
            { label: 'Firewall', value: security.firewall || 'Active', type: 'status' },
            { label: 'Encryption', value: security.encryption || 'AES-256', type: 'info' },
            { label: 'Threats Blocked', value: security.threatsBlocked || 0, type: 'metric' },
            { label: 'Last Scan', value: security.lastScan || 'Never', type: 'info' }
        ];

        return items.map(item => `
            <div class="metric-item">
                <span>${item.label}</span>
                <span class="metric-value ${item.type === 'status' && item.value === 'Active' ? 'status-good' : ''}">${item.value}</span>
            </div>
        `).join('');
    }

    attachEventListeners() {
        // Quick action buttons
        this.container.querySelectorAll('.widget-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                // Route desktop actions
                if (action === 'reopen-desktop') {
                    const vd = window.vd?.getCurrent?.();
                    if (vd) {
                        window.switchTab?.('sessions');
                        // try reconnect in current session view
                        setTimeout(() => window.connectDesktop?.(vd.sessionId), 150);
                    } else {
                        window.notify?.('No active desktop to reopen', 'warning');
                    }
                    return;
                }
                if (action === 'disconnect-desktop') {
                    window.disconnectDesktop?.();
                    return;
                }
                if (action === 'open-sessions') {
                    window.switchTab?.('sessions');
                    return;
                }
                this.handleQuickAction(action, e.currentTarget);
            });
        });

        // Widget hover effects
        this.container.querySelectorAll('.widget').forEach(widget => {
            widget.addEventListener('mouseenter', () => {
                widget.style.transform = 'translateY(-2px)';
            });
            
            widget.addEventListener('mouseleave', () => {
                widget.style.transform = 'translateY(0)';
            });
        });
    }

    handleQuickAction(action, button) {
        console.log(`[Widgets] Quick action: ${action}`);
        
        // Visual feedback
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        button.disabled = true;
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);

        switch (action) {
            case 'deploy':
                this.handleDeploy();
                break;
            case 'backup':
                this.handleBackup();
                break;
            case 'optimize':
                this.handleOptimize();
                break;
            case 'restart':
                this.handleRestart();
                break;
        }
    }

    handleDeploy() {
        window.notify?.('Deployment initiated...', 'info');
        window.switchTab?.('sessions');
    }

    handleBackup() {
        window.notify?.('Starting system backup...', 'info');
        // Simulate backup process
        setTimeout(() => {
            window.notify?.('Backup completed successfully', 'success');
        }, 3000);
    }

    handleOptimize() {
        window.notify?.('Running system optimization...', 'info');
        // Simulate optimization
        setTimeout(() => {
            window.notify?.('System optimized', 'success');
            this.state.refreshData?.();
        }, 2500);
    }

    handleRestart() {
        if (confirm('Are you sure you want to restart all services?')) {
            window.notify?.('Restarting services...', 'warning');
            // Simulate restart
            setTimeout(() => {
                window.notify?.('Services restarted', 'success');
            }, 4000);
        }
    }

    getActivityColor(type) {
        switch (type) {
            case 'success': return 'var(--omega-green)';
            case 'warning': return 'var(--omega-yellow)';
            case 'error': return 'var(--omega-red)';
            case 'info': return 'var(--omega-cyan)';
            default: return 'var(--omega-light-1)';
        }
    }

    getActivityIcon(type) {
        switch (type) {
            case 'session': return 'desktop';
            case 'plugin': return 'puzzle-piece';
            case 'network': return 'network-wired';
            case 'security': return 'shield-alt';
            case 'system': return 'cog';
            default: return 'circle';
        }
    }

    formatTimeAgo(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    startUpdateLoop() {
        this.updateInterval = setInterval(() => {
            this.updateWidgets();
        }, 3000);
        // Subscribe to active desktop changes to update widget immediately
        if (window.vd?.onChange) {
            window.vd.onChange(() => this.updateActiveDesktopWidget());
        }
    }

    updateWidgets() {
        // Update specific widget contents without full re-render
        this.updateResourceUsage();
        this.updateNetworkStatus();
        this.updateRecentActivity();
    this.updateActiveDesktopWidget();
    }

    updateResourceUsage() {
        const content = document.getElementById('widget-content-resource-usage');
        if (content) {
            content.innerHTML = this.renderResourceUsage();
        }
    }

    updateNetworkStatus() {
        const content = document.getElementById('widget-content-network-status');
        if (content) {
            content.innerHTML = this.renderNetworkStatus();
        }
    }

    updateRecentActivity() {
        const content = document.getElementById('widget-content-recent-activity');
        if (content) {
            content.innerHTML = this.renderRecentActivity();
        }
    }

    updateActiveDesktopWidget() {
        const content = document.getElementById('widget-content-active-desktop');
        if (content) {
            content.innerHTML = this.renderActiveDesktop();
            this.attachEventListeners();
        }
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        this.widgets.clear();
    }
}
