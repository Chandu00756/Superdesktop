/**
 * Omega SuperDesktop v2.0 - Performance Widget Manager Module
 * Extracted from omega-control-center.html - Handles performance monitoring widgets and charts
 */

class PerformanceWidgetManager extends EventTarget {
    constructor() {
        super();
        this.charts = new Map();
        this.expandedWidgets = new Set();
        this.chartData = new Map();
        this.updateInterval = null;
        this.widgets = ['cpu', 'gpu', 'memory', 'storage', 'network', 'latency'];
        this.isInitialized = false;
    }

    initialize() {
        console.log('📊 Initializing Performance Widget Manager...');
        this.initializeCharts();
        this.startRealTimeUpdates();
        this.setupEventListeners();
        this.isInitialized = true;
        console.log('✅ Performance Widget Manager initialized');
        this.dispatchEvent(new CustomEvent('performanceWidgetManagerInitialized'));
    }

    initializeCharts() {
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded, using fallback visualization');
            this.setupFallbackVisualization();
            return;
        }

        this.widgets.forEach(widget => {
            const canvas = document.getElementById(`${widget}Chart`);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                const chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: this.generateTimeLabels(),
                        datasets: [{
                            label: `${widget.toUpperCase()} Usage`,
                            data: this.generateInitialData(widget),
                            borderColor: this.getChartColor(widget),
                            backgroundColor: this.getChartColor(widget, 0.1),
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 3
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {
                            duration: 0
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#00ff7f',
                                bodyColor: '#ffffff',
                                borderColor: '#00ff7f',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                display: false,
                                grid: {
                                    display: false
                                }
                            },
                            y: {
                                display: false,
                                min: 0,
                                max: widget === 'latency' ? 5 : 100,
                                grid: {
                                    display: false
                                }
                            }
                        },
                        elements: {
                            point: {
                                radius: 0
                            }
                        },
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        }
                    }
                });
                this.charts.set(widget, chart);
                
                // Initialize chart data storage
                this.chartData.set(widget, {
                    current: this.generateRealisticValue(widget),
                    history: this.generateInitialData(widget),
                    lastUpdate: new Date()
                });
            }
        });
    }

    setupFallbackVisualization() {
        this.widgets.forEach(widget => {
            const canvas = document.getElementById(`${widget}Chart`);
            if (canvas) {
                this.drawFallbackChart(canvas, widget);
            }
        });
    }

    drawFallbackChart(canvas, widget) {
        const ctx = canvas.getContext('2d');
        const data = this.generateInitialData(widget);
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = this.getChartColor(widget);
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = (index / (data.length - 1)) * width;
            const y = height - (value / 100) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }

    generateTimeLabels() {
        const labels = [];
        for (let i = 59; i >= 0; i--) {
            labels.push(`${i}s`);
        }
        return labels;
    }

    generateInitialData(widget) {
        const data = [];
        const baseValue = this.getBaseValue(widget);
        
        for (let i = 0; i < 60; i++) {
            data.push(this.generateRealisticValue(widget, baseValue));
        }
        return data;
    }

    getBaseValue(widget) {
        const baseValues = {
            cpu: 45,
            gpu: 38,
            memory: 55,
            storage: 25,
            network: 30,
            latency: 1.2
        };
        return baseValues[widget] || 50;
    }

    getChartColor(widget, alpha = 1) {
        const colors = {
            cpu: `rgba(0, 255, 127, ${alpha})`,      // Superdesktop primary
            gpu: `rgba(34, 197, 94, ${alpha})`,      // Green
            memory: `rgba(59, 130, 246, ${alpha})`,  // Blue
            storage: `rgba(168, 85, 247, ${alpha})`, // Purple
            network: `rgba(249, 115, 22, ${alpha})`, // Orange
            latency: `rgba(236, 72, 153, ${alpha})`  // Pink
        };
        return colors[widget] || `rgba(0, 255, 127, ${alpha})`;
    }

    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(() => {
            this.updateAllCharts();
            this.updateMetrics();
            this.updateWidgetHeaders();
        }, 2000);
    }

    updateAllCharts() {
        this.widgets.forEach(widget => {
            const newValue = this.generateRealisticValue(widget);
            const chartData = this.chartData.get(widget);
            
            if (chartData) {
                chartData.current = newValue;
                chartData.history.shift();
                chartData.history.push(newValue);
                chartData.lastUpdate = new Date();
            }

            const chart = this.charts.get(widget);
            if (chart) {
                chart.data.datasets[0].data = [...chartData.history];
                chart.update('none');
            } else {
                // Fallback update
                const canvas = document.getElementById(`${widget}Chart`);
                if (canvas) {
                    this.drawFallbackChart(canvas, widget);
                }
            }
            
            // Update overlay values
            this.updateOverlayValue(widget, newValue);
        });
    }

    generateRealisticValue(widget, baseValue = null) {
        const base = baseValue || this.getBaseValue(widget);
        const currentData = this.chartData.get(widget);
        
        // Add some trending behavior
        let trend = 0;
        if (currentData && currentData.history.length > 5) {
            const recent = currentData.history.slice(-5);
            const older = currentData.history.slice(-10, -5);
            const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
            const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
            trend = (recentAvg - olderAvg) * 0.1; // Small trend influence
        }

        const variations = {
            cpu: 15,
            gpu: 20,
            memory: 10,
            storage: 25,
            network: 30,
            latency: 0.5
        };
        
        const variation = variations[widget] || 15;
        const randomChange = (Math.random() - 0.5) * variation;
        const newValue = base + randomChange + trend;
        
        if (widget === 'latency') {
            return Math.max(0.1, Math.min(5, newValue));
        } else {
            return Math.max(0, Math.min(100, newValue));
        }
    }

    updateOverlayValue(widget, value) {
        const valueElement = document.getElementById(`${widget}Value`);
        if (valueElement) {
            if (widget === 'latency') {
                valueElement.textContent = `${value.toFixed(1)}ms`;
            } else {
                valueElement.textContent = `${Math.round(value)}%`;
            }
            
            // Add status color based on value
            this.updateValueStatus(valueElement, widget, value);
        }
    }

    updateValueStatus(element, widget, value) {
        element.classList.remove('status-good', 'status-warning', 'status-critical');
        
        const thresholds = {
            cpu: { warning: 70, critical: 90 },
            gpu: { warning: 80, critical: 95 },
            memory: { warning: 75, critical: 90 },
            storage: { warning: 80, critical: 95 },
            network: { warning: 60, critical: 80 },
            latency: { warning: 2, critical: 4 }
        };
        
        const threshold = thresholds[widget];
        if (threshold) {
            if (value >= threshold.critical) {
                element.classList.add('status-critical');
            } else if (value >= threshold.warning) {
                element.classList.add('status-warning');
            } else {
                element.classList.add('status-good');
            }
        }
    }

    updateWidgetHeaders() {
        this.widgets.forEach(widget => {
            const header = document.querySelector(`#${widget}PerformanceWidget .widget-title-value`);
            if (header) {
                const data = this.chartData.get(widget);
                if (data) {
                    if (widget === 'latency') {
                        header.textContent = `${data.current.toFixed(1)}ms`;
                    } else {
                        header.textContent = `${Math.round(data.current)}%`;
                    }
                }
            }
        });
    }

    updateMetrics() {
        // Update detailed metric values with realistic data
        const metrics = this.generateDetailedMetrics();

        Object.entries(metrics).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    generateDetailedMetrics() {
        return {
            // CPU Metrics
            cpuCores: 384,
            cpuFreq: (3.2 + (Math.random() - 0.5) * 0.2).toFixed(1) + ' GHz',
            cpuTemp: Math.round(68 + (Math.random() - 0.5) * 10) + '°C',
            cpuPower: Math.round(185 + (Math.random() - 0.5) * 30) + 'W',
            
            // GPU Metrics
            gpuCount: 24,
            gpuVram: '768GB',
            gpuTemp: Math.round(72 + (Math.random() - 0.5) * 8) + '°C',
            gpuPower: (4.2 + (Math.random() - 0.5) * 0.5).toFixed(1) + 'kW',
            
            // Memory Metrics
            memoryUsed: Math.round(720 + (Math.random() - 0.5) * 50) + 'GB',
            memoryTotal: '1.6TB',
            memoryCached: Math.round(120 + (Math.random() - 0.5) * 20) + 'GB',
            memorySwap: Math.round(5 + (Math.random() - 0.5) * 3) + 'GB',
            
            // Storage Metrics
            storageRead: (1.2 + (Math.random() - 0.5) * 0.3).toFixed(1) + 'GB/s',
            storageWrite: (0.8 + (Math.random() - 0.5) * 0.2).toFixed(1) + 'GB/s',
            storageIops: Math.round(45 + (Math.random() - 0.5) * 10) + 'K',
            storageFree: '4.2TB',
            
            // Network Metrics
            networkUp: (2.4 + (Math.random() - 0.5) * 0.5).toFixed(1) + 'GB/s',
            networkDown: (8.7 + (Math.random() - 0.5) * 1.0).toFixed(1) + 'GB/s',
            networkLatency: (0.8 + (Math.random() - 0.5) * 0.2).toFixed(1) + 'ms',
            networkPackets: Math.round(2500 + (Math.random() - 0.5) * 200) + '/s',
            
            // Latency Metrics
            latencyMin: (0.2 + (Math.random() - 0.5) * 0.1).toFixed(1) + 'ms',
            latencyMax: (2.1 + (Math.random() - 0.5) * 0.3).toFixed(1) + 'ms',
            latencyP99: (1.8 + (Math.random() - 0.5) * 0.2).toFixed(1) + 'ms',
            latencyP95: (1.5 + (Math.random() - 0.5) * 0.2).toFixed(1) + 'ms'
        };
    }

    setupEventListeners() {
        // Widget expansion/collapse
        this.setupWidgetExpansion();
        
        // Widget controls
        this.setupWidgetControls();
        
        // Chart type changes
        this.setupChartTypeControls();
        
        // Feature tabs
        this.setupFeatureTabs();
    }

    setupWidgetExpansion() {
        window.toggleWidgetExpansion = (widgetId) => {
            const widget = document.getElementById(widgetId);
            if (widget) {
                const isExpanded = widget.classList.contains('expanded');
                if (isExpanded) {
                    widget.classList.remove('expanded');
                    this.expandedWidgets.delete(widgetId);
                } else {
                    widget.classList.add('expanded');
                    this.expandedWidgets.add(widgetId);
                }
                
                // Resize chart after animation
                setTimeout(() => {
                    const widgetType = widgetId.replace('PerformanceWidget', '').toLowerCase();
                    const chart = this.charts.get(widgetType);
                    if (chart) {
                        chart.resize();
                    }
                }, 300);
                
                this.dispatchEvent(new CustomEvent('widgetExpansionToggled', {
                    detail: { widgetId, expanded: !isExpanded }
                }));
            }
        };
    }

    setupWidgetControls() {
        window.refreshWidget = (widgetType) => {
            const chart = this.charts.get(widgetType);
            if (chart) {
                chart.data.datasets[0].data = this.generateInitialData(widgetType);
                chart.update();
            }
            
            this.dispatchEvent(new CustomEvent('widgetRefreshed', {
                detail: { widgetType }
            }));
        };

        window.exportWidgetData = (widgetType) => {
            const chartData = this.chartData.get(widgetType);
            if (chartData) {
                const data = {
                    type: widgetType,
                    timestamp: new Date().toISOString(),
                    current: chartData.current,
                    history: chartData.history,
                    metadata: {
                        sampleInterval: '2s',
                        sampleCount: chartData.history.length
                    }
                };
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {
                    type: 'application/json'
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${widgetType}-performance-${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
                
                if (window.menuBarManager) {
                    window.menuBarManager.showNotification(
                        'Data Exported',
                        `${widgetType} performance data exported successfully`,
                        'success'
                    );
                }
            }
        };
    }

    setupChartTypeControls() {
        window.changeChartType = (widgetType, chartType) => {
            const chart = this.charts.get(widgetType);
            if (chart) {
                chart.config.type = chartType;
                chart.update();
                
                // Update button states
                const widget = document.getElementById(`${widgetType}PerformanceWidget`);
                if (widget) {
                    const buttons = widget.querySelectorAll('.chart-type-btn');
                    buttons.forEach(btn => btn.classList.remove('active'));
                    
                    const activeBtn = widget.querySelector(`[onclick*="${chartType}"]`);
                    if (activeBtn) {
                        activeBtn.classList.add('active');
                    }
                }
                
                this.dispatchEvent(new CustomEvent('chartTypeChanged', {
                    detail: { widgetType, chartType }
                }));
            }
        };
    }

    setupFeatureTabs() {
        window.switchFeatureTab = (widgetType, tabName) => {
            const widget = document.getElementById(`${widgetType}PerformanceWidget`);
            if (widget) {
                // Update tab buttons
                const tabs = widget.querySelectorAll('.feature-tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                const activeTab = widget.querySelector(`[onclick*="${tabName}"]`);
                if (activeTab) {
                    activeTab.classList.add('active');
                }
                
                // Update content
                const contents = widget.querySelectorAll('.feature-content');
                contents.forEach(content => content.classList.remove('active'));
                
                const targetContent = widget.querySelector(`#${widgetType}${tabName.charAt(0).toUpperCase() + tabName.slice(1)}Content`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
                
                this.dispatchEvent(new CustomEvent('featureTabSwitched', {
                    detail: { widgetType, tabName }
                }));
            }
        };
    }

    // Public API methods
    getWidgetData(widgetType) {
        return this.chartData.get(widgetType);
    }

    getAllWidgetData() {
        const data = {};
        this.chartData.forEach((value, key) => {
            data[key] = value;
        });
        return data;
    }

    setUpdateInterval(interval) {
        clearInterval(this.updateInterval);
        this.updateInterval = setInterval(() => {
            this.updateAllCharts();
            this.updateMetrics();
            this.updateWidgetHeaders();
        }, interval);
    }

    pauseUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    resumeUpdates() {
        if (!this.updateInterval) {
            this.startRealTimeUpdates();
        }
    }

    resizeAllCharts() {
        this.charts.forEach(chart => {
            chart.resize();
        });
    }

    dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.charts.forEach(chart => {
            chart.destroy();
        });
        
        this.charts.clear();
        this.chartData.clear();
        this.expandedWidgets.clear();
        
        console.log('🧹 Performance Widget Manager disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.PerformanceWidgetManager = PerformanceWidgetManager;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceWidgetManager;
}
