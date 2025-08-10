// Omega SuperDesktop v2.0 - Sidebar Manager
import { formatBytes, formatNumber } from './utils.js';

export class SidebarManager {
    constructor(container, state, api) {
        this.container = container;
        this.state = state;
        this.api = api;
        this.updateInterval = null;
        this.init();
    }

    init() {
        this.render();
        this.startUpdateLoop();
    }

    render() {
        const sections = [
            this.renderSystemMonitor(),
            this.renderAIRecommendations(), 
            this.renderAlerts(),
            this.renderQuickActions()
        ];

        this.container.innerHTML = `
            <div class="sidebar-content">
                ${sections.join('')}
            </div>
        `;

        this.attachEventListeners();
    }

    renderSystemMonitor() {
        const metrics = this.state.getSystemMetrics();
        
        return `
            <div class="sidebar-section">
                <div class="sidebar-title">
                    <i class="fas fa-chart-line"></i>
                    System Monitor
                </div>
                <div class="sidebar-item">
                    <div class="metric-item">
                        <span>CPU</span>
                        <span class="metric-value">${metrics.cpu || '--'}%</span>
                    </div>
                </div>
                <div class="sidebar-item">
                    <div class="metric-item">
                        <span>Memory</span>
                        <span class="metric-value">${metrics.memory || '--'}%</span>
                    </div>
                </div>
                <div class="sidebar-item">
                    <div class="metric-item">
                        <span>Network</span>
                        <span class="metric-value">${metrics.network || '--'}ms</span>
                    </div>
                </div>
                <div class="sidebar-item">
                    <div class="metric-item">
                        <span>Sessions</span>
                        <span class="metric-value">${metrics.sessions || 0}</span>
                    </div>
                </div>
                <div class="sidebar-item">
                    <div class="metric-item">
                        <span>Nodes</span>
                        <span class="metric-value">${metrics.activeNodes || 0}</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderAIRecommendations() {
        const recommendations = this.state.getAIRecommendations();
        
        return `
            <div class="sidebar-section">
                <div class="sidebar-title">
                    <i class="fas fa-lightbulb"></i>
                    AI Recommendations
                </div>
                ${recommendations.length > 0 ? 
                    recommendations.map(rec => `
                        <div class="sidebar-item ai-recommendation" data-rec-id="${rec.id}">
                            <div style="font-size: 10px; color: var(--omega-cyan); margin-bottom: 4px;">
                                ${rec.type.toUpperCase()}
                            </div>
                            <div style="font-size: 11px; margin-bottom: 4px;">
                                ${rec.title}
                            </div>
                            <div style="font-size: 10px; color: var(--omega-light-1);">
                                ${rec.impact}
                            </div>
                        </div>
                    `).join('') :
                    `<div class="sidebar-item">
                        <div style="font-size: 11px; color: var(--omega-light-1); text-align: center;">
                            Analyzing system...
                        </div>
                    </div>`
                }
            </div>
        `;
    }

    renderAlerts() {
        const alerts = this.state.getAlerts();
        
        return `
            <div class="sidebar-section">
                <div class="sidebar-title">
                    <i class="fas fa-exclamation-triangle"></i>
                    Alerts
                    ${alerts.length > 0 ? `<span class="alert-count">${alerts.length}</span>` : ''}
                </div>
                ${alerts.length > 0 ? 
                    alerts.map(alert => `
                        <div class="sidebar-item alert-item alert-${alert.severity}" data-alert-id="${alert.id}">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                <i class="fas fa-${this.getAlertIcon(alert.severity)}"></i>
                                <span style="font-size: 11px; font-weight: bold;">${alert.title}</span>
                            </div>
                            <div style="font-size: 10px; color: var(--omega-light-1);">
                                ${alert.message}
                            </div>
                            <div style="font-size: 9px; color: var(--omega-light-1); margin-top: 4px;">
                                ${this.formatTimeAgo(alert.timestamp)}
                            </div>
                        </div>
                    `).join('') :
                    `<div class="sidebar-item">
                        <div style="font-size: 11px; color: var(--omega-light-1); text-align: center;">
                            <i class="fas fa-check-circle" style="color: var(--omega-green); margin-right: 4px;"></i>
                            All systems operational
                        </div>
                    </div>`
                }
            </div>
        `;
    }

    renderQuickActions() {
        return `
            <div class="sidebar-section">
                <div class="sidebar-title">
                    <i class="fas fa-bolt"></i>
                    Quick Actions
                </div>
                <div class="sidebar-item quick-action" data-action="refresh">
                    <i class="fas fa-sync-alt"></i>
                    <span>Refresh All</span>
                </div>
                <div class="sidebar-item quick-action" data-action="deploy">
                    <i class="fas fa-rocket"></i>
                    <span>Deploy Service</span>
                </div>
                <div class="sidebar-item quick-action" data-action="backup">
                    <i class="fas fa-save"></i>
                    <span>Backup System</span>
                </div>
                <div class="sidebar-item quick-action" data-action="optimize">
                    <i class="fas fa-magic"></i>
                    <span>Auto Optimize</span>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Quick actions
        this.container.querySelectorAll('.quick-action').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleQuickAction(action);
            });
        });

        // AI recommendations
        this.container.querySelectorAll('.ai-recommendation').forEach(item => {
            item.addEventListener('click', (e) => {
                const recId = e.currentTarget.dataset.recId;
                this.handleRecommendationClick(recId);
            });
        });

        // Alerts
        this.container.querySelectorAll('.alert-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const alertId = e.currentTarget.dataset.alertId;
                this.handleAlertClick(alertId);
            });
        });
    }

    handleQuickAction(action) {
        console.log(`[Sidebar] Quick action: ${action}`);
        
        switch (action) {
            case 'refresh':
                this.state.refreshAll();
                window.notify?.('System refreshed', 'success');
                break;
            case 'deploy':
                window.switchTab?.('sessions');
                break;
            case 'backup':
                this.triggerBackup();
                break;
            case 'optimize':
                this.triggerOptimization();
                break;
        }
    }

    handleRecommendationClick(recId) {
        const rec = this.state.getAIRecommendations().find(r => r.id === recId);
        if (rec) {
            console.log(`[Sidebar] Applying recommendation: ${rec.title}`);
            // Implementation would depend on recommendation type
            window.notify?.(`Applied: ${rec.title}`, 'success');
        }
    }

    handleAlertClick(alertId) {
        const alert = this.state.getAlerts().find(a => a.id === alertId);
        if (alert) {
            console.log(`[Sidebar] Viewing alert: ${alert.title}`);
            // Navigate to relevant section
            switch (alert.type) {
                case 'security':
                    window.switchTab?.('security');
                    break;
                case 'performance':
                    window.switchTab?.('performance');
                    break;
                case 'network':
                    window.switchTab?.('network');
                    break;
                default:
                    window.switchTab?.('dashboard');
            }
        }
    }

    triggerBackup() {
        window.notify?.('Starting system backup...', 'info');
        // Implementation would call backup service
    }

    triggerOptimization() {
        window.notify?.('Running auto-optimization...', 'info');
        // Implementation would call optimization service
    }

    getAlertIcon(severity) {
        switch (severity) {
            case 'critical': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            case 'info': return 'info-circle';
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
            this.update();
        }, 2000);
    }

    update() {
        // Update only specific sections to avoid full re-render
        const systemSection = this.container.querySelector('.sidebar-section:first-child');
        if (systemSection) {
            const newSystemHTML = this.renderSystemMonitor();
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = newSystemHTML;
            systemSection.replaceWith(tempDiv.firstElementChild);
        }
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}
