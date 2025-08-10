/**
 * Omega SuperDesktop v2.0 - NLP Search Engine Module
 * Extracted from omega-control-center.html - Handles natural language processing and intelligent search
 */

class NLPSearchEngine extends EventTarget {
    constructor() {
        super();
        this.nodeData = new Map();
        this.jobData = new Map();
        this.searchHistory = [];
        this.patterns = new Map();
        this.isInitialized = false;
        this.initializePatterns();
    }

    initialize() {
        console.log('🔍 Initializing NLP Search Engine...');
        this.initializeNodeData();
        this.initializeJobData();
        this.setupEventListeners();
        this.isInitialized = true;
        console.log('✅ NLP Search Engine initialized');
        this.dispatchEvent(new CustomEvent('nlpSearchEngineInitialized'));
    }

    initializeNodeData() {
        // Initialize sample node data for search
        const sampleNodes = [
            { id: 'gpu-node-01', type: 'gpu', usage: 42, memory: '32GB', memoryUsage: 68, temp: 67, rack: 1, status: 'online' },
            { id: 'gpu-node-02', type: 'gpu', usage: 84, memory: '32GB', memoryUsage: 89, temp: 72, rack: 2, status: 'online' },
            { id: 'gpu-node-03', type: 'gpu', usage: 97, memory: '32GB', memoryUsage: 95, temp: 78, rack: 3, status: 'warning' },
            { id: 'cpu-node-01', type: 'cpu', usage: 35, memory: '64GB', memoryUsage: 45, temp: 45, rack: 1, status: 'online' },
            { id: 'cpu-node-02', type: 'cpu', usage: 78, memory: '64GB', memoryUsage: 89, temp: 52, rack: 2, status: 'online' },
            { id: 'cpu-node-06', type: 'cpu', usage: 45, memory: '64GB', memoryUsage: 82, temp: 48, rack: 3, status: 'maintenance' },
            { id: 'storage-node-01', type: 'storage', usage: 23, memory: '16GB', memoryUsage: 34, temp: 35, rack: 1, status: 'online' },
            { id: 'edge-node-01', type: 'edge', usage: 67, memory: '8GB', memoryUsage: 76, temp: 58, rack: 2, status: 'online' }
        ];

        sampleNodes.forEach(node => {
            this.nodeData.set(node.id, node);
        });
    }

    initializeJobData() {
        // Initialize sample job data for search
        const sampleJobs = [
            { id: 'ml-training-047', type: 'ml', status: 'stuck', node: 'cpu-node-06', duration: '2h 15m', progress: 45 },
            { id: 'data-processing-12', type: 'processing', status: 'failed', node: 'gpu-node-02', duration: '45m', progress: 0 },
            { id: 'render-job-23', type: 'render', status: 'running', node: 'gpu-node-01', duration: '1h 32m', progress: 78 },
            { id: 'backup-task-8', type: 'backup', status: 'completed', node: 'storage-node-01', duration: '3h 45m', progress: 100 },
            { id: 'ai-inference-15', type: 'ai', status: 'running', node: 'gpu-node-03', duration: '25m', progress: 92 },
            { id: 'simulation-run-4', type: 'simulation', status: 'queued', node: null, duration: '0m', progress: 0 }
        ];

        sampleJobs.forEach(job => {
            this.jobData.set(job.id, job);
        });
    }

    initializePatterns() {
        // Define NLP patterns for different query types
        this.patterns.set('high_usage', {
            keywords: ['usage', 'high', 'over', 'above', 'busy', 'loaded'],
            entities: ['gpu', 'cpu', 'memory', 'storage'],
            action: 'findHighUsageNodes'
        });

        this.patterns.set('rack_query', {
            keywords: ['rack', 'in', 'location'],
            entities: ['rack', 'r1', 'r2', 'r3'],
            action: 'findNodesInRack'
        });

        this.patterns.set('job_issues', {
            keywords: ['stuck', 'failed', 'error', 'problem', 'issue'],
            entities: ['job', 'task', 'process'],
            action: 'findProblematicJobs'
        });

        this.patterns.set('status_query', {
            keywords: ['status', 'state', 'health', 'condition'],
            entities: ['node', 'system', 'cluster'],
            action: 'findNodesByStatus'
        });

        this.patterns.set('temperature_query', {
            keywords: ['temperature', 'temp', 'hot', 'cold', 'thermal'],
            entities: ['node', 'gpu', 'cpu'],
            action: 'findTemperatureIssues'
        });

        this.patterns.set('performance_query', {
            keywords: ['performance', 'slow', 'fast', 'optimize', 'bottleneck'],
            entities: ['node', 'job', 'task'],
            action: 'findPerformanceIssues'
        });
    }

    setupEventListeners() {
        // Global search input handling
        window.performNLPSearch = (query) => {
            return this.search(query || '');
        };

        // Search suggestions
        window.getSearchSuggestions = (partial) => {
            return this.generateSuggestions(partial);
        };
    }

    search(query) {
        if (!query.trim()) {
            return this.generateHelpResults();
        }

        // Add to search history
        this.searchHistory.unshift({
            query,
            timestamp: new Date(),
            results: null
        });

        if (this.searchHistory.length > 50) {
            this.searchHistory.pop();
        }

        const lowerQuery = query.toLowerCase();
        const analysis = this.analyzeQuery(lowerQuery);
        let results = [];

        try {
            // Execute search based on pattern analysis
            switch (analysis.action) {
                case 'findHighUsageNodes':
                    results = this.findHighUsageNodes(analysis.threshold, analysis.entity);
                    break;
                case 'findNodesInRack':
                    results = this.findNodesInRack(analysis.rackNumber);
                    break;
                case 'findProblematicJobs':
                    results = this.findProblematicJobs();
                    break;
                case 'findNodesByStatus':
                    results = this.findNodesByStatus(analysis.status);
                    break;
                case 'findTemperatureIssues':
                    results = this.findTemperatureIssues(analysis.threshold);
                    break;
                case 'findPerformanceIssues':
                    results = this.findPerformanceIssues();
                    break;
                default:
                    results = this.performFuzzySearch(lowerQuery);
            }

            // Update search history with results
            this.searchHistory[0].results = results.length;

            this.dispatchEvent(new CustomEvent('searchPerformed', {
                detail: { query, results: results.length, analysis }
            }));

        } catch (error) {
            console.error('Search error:', error);
            results = [this.createErrorResult(error.message)];
        }

        return this.formatSearchResults(query, results);
    }

    analyzeQuery(query) {
        const analysis = {
            action: 'performFuzzySearch',
            entity: null,
            threshold: null,
            rackNumber: null,
            status: null,
            confidence: 0
        };

        // Extract numbers
        const numberMatch = query.match(/\d+/);
        if (numberMatch) {
            analysis.threshold = parseInt(numberMatch[0]);
            analysis.rackNumber = parseInt(numberMatch[0]);
        }

        // Pattern matching
        for (const [patternName, pattern] of this.patterns) {
            let keywordMatches = 0;
            let entityMatches = 0;

            // Check keyword matches
            pattern.keywords.forEach(keyword => {
                if (query.includes(keyword)) {
                    keywordMatches++;
                }
            });

            // Check entity matches
            pattern.entities.forEach(entity => {
                if (query.includes(entity)) {
                    entityMatches++;
                    analysis.entity = entity;
                }
            });

            const confidence = (keywordMatches + entityMatches) / (pattern.keywords.length + pattern.entities.length);
            
            if (confidence > analysis.confidence) {
                analysis.action = pattern.action;
                analysis.confidence = confidence;
            }
        }

        // Special status detection
        const statusKeywords = ['online', 'offline', 'maintenance', 'warning', 'error', 'critical'];
        statusKeywords.forEach(status => {
            if (query.includes(status)) {
                analysis.status = status;
                analysis.action = 'findNodesByStatus';
            }
        });

        return analysis;
    }

    findHighUsageNodes(threshold = 80, entityType = null) {
        const results = [];
        const targetThreshold = threshold || 80;

        this.nodeData.forEach((node, nodeId) => {
            let shouldInclude = false;
            let usageType = '';
            let usageValue = 0;

            if (!entityType || entityType === 'cpu' || entityType === 'gpu') {
                if (node.usage >= targetThreshold) {
                    shouldInclude = true;
                    usageType = 'CPU/GPU';
                    usageValue = node.usage;
                }
            }

            if (!entityType || entityType === 'memory') {
                if (node.memoryUsage >= targetThreshold) {
                    shouldInclude = true;
                    usageType = usageType ? usageType + '/Memory' : 'Memory';
                    usageValue = Math.max(usageValue, node.memoryUsage);
                }
            }

            if (shouldInclude) {
                results.push(this.createNodeResult(node, {
                    type: 'HIGH_USAGE',
                    message: `${usageType} usage: ${usageValue}%`,
                    severity: usageValue > 90 ? 'critical' : 'warning',
                    actions: ['investigate', 'reduce-load', 'migrate-jobs']
                }));
            }
        });

        return results;
    }

    findNodesInRack(rackNumber) {
        const results = [];

        this.nodeData.forEach((node, nodeId) => {
            if (node.rack === rackNumber) {
                const status = this.getNodeHealthStatus(node);
                results.push(this.createNodeResult(node, {
                    type: 'RACK_NODE',
                    message: `Located in Rack ${rackNumber}`,
                    severity: status.severity,
                    actions: ['view-details', 'monitor', 'configure']
                }));
            }
        });

        return results;
    }

    findProblematicJobs() {
        const results = [];

        this.jobData.forEach((job, jobId) => {
            if (['stuck', 'failed', 'error'].includes(job.status)) {
                results.push(this.createJobResult(job, {
                    type: 'PROBLEMATIC_JOB',
                    message: this.getJobStatusMessage(job),
                    severity: job.status === 'failed' ? 'critical' : 'warning',
                    actions: ['restart', 'migrate', 'view-logs', 'cancel']
                }));
            }
        });

        return results;
    }

    findNodesByStatus(targetStatus) {
        const results = [];

        this.nodeData.forEach((node, nodeId) => {
            if (node.status === targetStatus) {
                results.push(this.createNodeResult(node, {
                    type: 'STATUS_MATCH',
                    message: `Status: ${targetStatus}`,
                    severity: this.getStatusSeverity(targetStatus),
                    actions: this.getStatusActions(targetStatus)
                }));
            }
        });

        return results;
    }

    findTemperatureIssues(threshold = 70) {
        const results = [];
        const targetThreshold = threshold || 70;

        this.nodeData.forEach((node, nodeId) => {
            if (node.temp >= targetThreshold) {
                results.push(this.createNodeResult(node, {
                    type: 'TEMPERATURE_ISSUE',
                    message: `Temperature: ${node.temp}°C`,
                    severity: node.temp > 80 ? 'critical' : 'warning',
                    actions: ['check-cooling', 'reduce-load', 'maintenance']
                }));
            }
        });

        return results;
    }

    findPerformanceIssues() {
        const results = [];

        // Find nodes with high usage and temperature
        this.nodeData.forEach((node, nodeId) => {
            if (node.usage > 85 || node.temp > 75 || node.memoryUsage > 90) {
                results.push(this.createNodeResult(node, {
                    type: 'PERFORMANCE_ISSUE',
                    message: `Performance bottleneck detected`,
                    severity: 'warning',
                    actions: ['optimize', 'scale', 'rebalance']
                }));
            }
        });

        // Find stuck or slow jobs
        this.jobData.forEach((job, jobId) => {
            if (job.status === 'stuck' || (job.status === 'running' && job.progress < 10)) {
                results.push(this.createJobResult(job, {
                    type: 'PERFORMANCE_ISSUE',
                    message: `Job performance issue`,
                    severity: 'warning',
                    actions: ['restart', 'migrate', 'optimize']
                }));
            }
        });

        return results;
    }

    performFuzzySearch(query) {
        const results = [];
        
        // Search through node IDs and types
        this.nodeData.forEach((node, nodeId) => {
            if (nodeId.includes(query) || node.type.includes(query) || node.status.includes(query)) {
                results.push(this.createNodeResult(node, {
                    type: 'FUZZY_MATCH',
                    message: `Matched: ${nodeId}`,
                    severity: 'info',
                    actions: ['view-details']
                }));
            }
        });

        // Search through jobs
        this.jobData.forEach((job, jobId) => {
            if (jobId.includes(query) || job.type.includes(query) || job.status.includes(query)) {
                results.push(this.createJobResult(job, {
                    type: 'FUZZY_MATCH',
                    message: `Matched: ${jobId}`,
                    severity: 'info',
                    actions: ['view-details']
                }));
            }
        });

        return results;
    }

    createNodeResult(node, resultInfo) {
        const health = this.getNodeHealthStatus(node);
        
        return {
            id: `node-${node.id}`,
            type: 'node',
            title: node.id.toUpperCase(),
            subtitle: `${node.type.toUpperCase()} Node`,
            message: resultInfo.message,
            severity: resultInfo.severity,
            icon: this.getNodeIcon(node.type),
            details: {
                usage: `${node.usage}%`,
                memory: `${node.memory} (${node.memoryUsage}%)`,
                temperature: `${node.temp}°C`,
                rack: `Rack ${node.rack}`,
                status: node.status
            },
            actions: resultInfo.actions,
            timestamp: new Date(),
            data: node
        };
    }

    createJobResult(job, resultInfo) {
        return {
            id: `job-${job.id}`,
            type: 'job',
            title: job.id.toUpperCase(),
            subtitle: `${job.type.toUpperCase()} Job`,
            message: resultInfo.message,
            severity: resultInfo.severity,
            icon: this.getJobIcon(job.type),
            details: {
                status: job.status,
                node: job.node || 'Unassigned',
                duration: job.duration,
                progress: `${job.progress}%`
            },
            actions: resultInfo.actions,
            timestamp: new Date(),
            data: job
        };
    }

    createErrorResult(message) {
        return {
            id: `error-${Date.now()}`,
            type: 'error',
            title: 'Search Error',
            subtitle: 'Unable to complete search',
            message: message,
            severity: 'error',
            icon: 'fas fa-exclamation-triangle',
            details: {},
            actions: [],
            timestamp: new Date()
        };
    }

    getNodeHealthStatus(node) {
        if (node.status === 'offline' || node.status === 'error') {
            return { severity: 'critical', status: 'unhealthy' };
        } else if (node.status === 'warning' || node.usage > 90 || node.temp > 80) {
            return { severity: 'warning', status: 'degraded' };
        } else if (node.status === 'maintenance') {
            return { severity: 'info', status: 'maintenance' };
        } else {
            return { severity: 'success', status: 'healthy' };
        }
    }

    getJobStatusMessage(job) {
        switch (job.status) {
            case 'stuck':
                return `Stuck for ${job.duration} - No progress`;
            case 'failed':
                return `Failed after ${job.duration}`;
            case 'error':
                return `Error encountered during execution`;
            default:
                return `Status: ${job.status}`;
        }
    }

    getStatusSeverity(status) {
        const severityMap = {
            'online': 'success',
            'offline': 'critical',
            'maintenance': 'info',
            'warning': 'warning',
            'error': 'critical',
            'critical': 'critical'
        };
        return severityMap[status] || 'info';
    }

    getStatusActions(status) {
        const actionMap = {
            'online': ['monitor', 'configure', 'maintain'],
            'offline': ['restart', 'diagnose', 'replace'],
            'maintenance': ['complete', 'extend', 'cancel'],
            'warning': ['investigate', 'monitor', 'fix'],
            'error': ['restart', 'diagnose', 'repair'],
            'critical': ['emergency-stop', 'isolate', 'replace']
        };
        return actionMap[status] || ['view-details'];
    }

    getNodeIcon(nodeType) {
        const iconMap = {
            'gpu': 'fas fa-microchip',
            'cpu': 'fas fa-server',
            'storage': 'fas fa-hdd',
            'edge': 'fas fa-network-wired',
            'memory': 'fas fa-memory'
        };
        return iconMap[nodeType] || 'fas fa-server';
    }

    getJobIcon(jobType) {
        const iconMap = {
            'ml': 'fas fa-brain',
            'processing': 'fas fa-cogs',
            'render': 'fas fa-paint-brush',
            'backup': 'fas fa-archive',
            'ai': 'fas fa-robot',
            'simulation': 'fas fa-calculator'
        };
        return iconMap[jobType] || 'fas fa-tasks';
    }

    formatSearchResults(query, results) {
        if (results.length === 0) {
            return this.generateNoResultsHTML(query);
        }

        const resultsHTML = results.map(result => this.formatResultItem(result)).join('');
        
        return `
            <div class="search-results-container">
                <div class="search-results-header">
                    <h4>Search Results for: "${query}"</h4>
                    <span class="results-count">${results.length} result${results.length !== 1 ? 's' : ''} found</span>
                </div>
                <div class="search-results-list">
                    ${resultsHTML}
                </div>
                <div class="search-results-footer">
                    <button type="button" class="btn btn-secondary" onclick="this.exportSearchResults()">
                        <i class="fas fa-download"></i> Export Results
                    </button>
                    <button type="button" class="btn btn-secondary" onclick="this.saveSearch()">
                        <i class="fas fa-bookmark"></i> Save Search
                    </button>
                </div>
            </div>
        `;
    }

    formatResultItem(result) {
        const actionsHTML = result.actions.map(action => 
            `<button type="button" class="result-action-btn" onclick="this.handleSearchAction('${action}', '${result.id}')" data-action="${action}">
                ${this.getActionLabel(action)}
            </button>`
        ).join('');

        const detailsHTML = Object.entries(result.details).map(([key, value]) =>
            `<span class="detail-item">${key}: ${value}</span>`
        ).join('');

        return `
            <div class="search-result-item" data-result-id="${result.id}" data-type="${result.type}">
                <div class="result-header">
                    <i class="${result.icon}"></i>
                    <div class="result-title-section">
                        <strong class="result-title">${result.title}</strong>
                        <span class="result-subtitle">${result.subtitle}</span>
                    </div>
                    <span class="result-status ${result.severity}">${result.severity.toUpperCase()}</span>
                </div>
                <div class="result-message">${result.message}</div>
                ${detailsHTML ? `<div class="result-details">${detailsHTML}</div>` : ''}
                ${actionsHTML ? `<div class="result-actions">${actionsHTML}</div>` : ''}
                <div class="result-timestamp">${result.timestamp.toLocaleTimeString()}</div>
            </div>
        `;
    }

    getActionLabel(action) {
        const labelMap = {
            'investigate': 'Investigate',
            'reduce-load': 'Reduce Load',
            'migrate-jobs': 'Migrate Jobs',
            'restart': 'Restart',
            'migrate': 'Migrate',
            'view-logs': 'View Logs',
            'cancel': 'Cancel',
            'view-details': 'Details',
            'monitor': 'Monitor',
            'configure': 'Configure',
            'check-cooling': 'Check Cooling',
            'maintenance': 'Maintenance',
            'optimize': 'Optimize',
            'scale': 'Scale',
            'rebalance': 'Rebalance',
            'emergency-stop': 'Emergency Stop',
            'isolate': 'Isolate',
            'repair': 'Repair'
        };
        return labelMap[action] || action;
    }

    generateNoResultsHTML(query) {
        return `
            <div class="search-no-results">
                <div class="no-results-icon">
                    <i class="fas fa-search"></i>
                </div>
                <h4>No results found for "${query}"</h4>
                <p>Try these example queries:</p>
                <ul class="search-suggestions">
                    <li onclick="this.searchFromSuggestion('GPU nodes with usage over 80%')">GPU nodes with usage over 80%</li>
                    <li onclick="this.searchFromSuggestion('All nodes in rack 3')">All nodes in rack 3</li>
                    <li onclick="this.searchFromSuggestion('Stuck jobs last 24 hours')">Stuck jobs last 24 hours</li>
                    <li onclick="this.searchFromSuggestion('Memory usage over 70%')">Memory usage over 70%</li>
                    <li onclick="this.searchFromSuggestion('Temperature issues')">Temperature issues</li>
                    <li onclick="this.searchFromSuggestion('Performance bottlenecks')">Performance bottlenecks</li>
                </ul>
            </div>
        `;
    }

    generateHelpResults() {
        return `
            <div class="search-help">
                <h4>Search Help</h4>
                <p>Use natural language to search for nodes, jobs, and system information:</p>
                <div class="help-categories">
                    <div class="help-category">
                        <h5>Node Queries</h5>
                        <ul>
                            <li>GPU nodes with usage over 80%</li>
                            <li>All nodes in rack 2</li>
                            <li>Offline nodes</li>
                            <li>Temperature over 70 degrees</li>
                        </ul>
                    </div>
                    <div class="help-category">
                        <h5>Job Queries</h5>
                        <ul>
                            <li>Stuck jobs</li>
                            <li>Failed tasks last hour</li>
                            <li>Running ML jobs</li>
                            <li>Performance issues</li>
                        </ul>
                    </div>
                    <div class="help-category">
                        <h5>System Queries</h5>
                        <ul>
                            <li>Memory usage over 90%</li>
                            <li>High temperature alerts</li>
                            <li>Maintenance status</li>
                            <li>Performance bottlenecks</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    generateSuggestions(partial) {
        const suggestions = [
            'GPU nodes with usage over',
            'All nodes in rack',
            'Stuck jobs',
            'Failed tasks',
            'Memory usage over',
            'Temperature over',
            'Performance issues',
            'Offline nodes',
            'Maintenance status',
            'High usage alerts'
        ];

        return suggestions.filter(suggestion => 
            suggestion.toLowerCase().includes(partial.toLowerCase())
        ).slice(0, 5);
    }

    handleSearchAction(action, resultId) {
        console.log(`Executing action: ${action} on ${resultId}`);
        
        // Dispatch action event
        this.dispatchEvent(new CustomEvent('searchActionRequested', {
            detail: { action, resultId }
        }));

        // Show notification
        if (window.menuBarManager) {
            window.menuBarManager.showNotification(
                'Action Executed',
                `${this.getActionLabel(action)} action initiated for ${resultId}`,
                'info'
            );
        }
    }

    exportSearchResults() {
        if (this.searchHistory.length === 0) return;

        const latestSearch = this.searchHistory[0];
        const exportData = {
            timestamp: new Date().toISOString(),
            query: latestSearch.query,
            results: latestSearch.results,
            searchHistory: this.searchHistory.slice(0, 10) // Last 10 searches
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search-results-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    saveSearch() {
        if (this.searchHistory.length === 0) return;

        const savedSearches = JSON.parse(localStorage.getItem('nlp-saved-searches') || '[]');
        savedSearches.unshift(this.searchHistory[0]);
        
        if (savedSearches.length > 20) {
            savedSearches.pop();
        }
        
        localStorage.setItem('nlp-saved-searches', JSON.stringify(savedSearches));
        
        if (window.menuBarManager) {
            window.menuBarManager.showNotification(
                'Search Saved',
                'Search has been saved to your history',
                'success'
            );
        }
    }

    searchFromSuggestion(suggestion) {
        const searchInput = document.getElementById('nlpSearchInput');
        if (searchInput) {
            searchInput.value = suggestion;
            return this.search(suggestion);
        }
    }

    // Public API
    getSearchHistory() {
        return [...this.searchHistory];
    }

    clearSearchHistory() {
        this.searchHistory = [];
        localStorage.removeItem('nlp-saved-searches');
    }

    updateNodeData(nodeId, updates) {
        const node = this.nodeData.get(nodeId);
        if (node) {
            Object.assign(node, updates);
        }
    }

    updateJobData(jobId, updates) {
        const job = this.jobData.get(jobId);
        if (job) {
            Object.assign(job, updates);
        }
    }

    dispose() {
        this.nodeData.clear();
        this.jobData.clear();
        this.patterns.clear();
        this.searchHistory = [];
        console.log('🧹 NLP Search Engine disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.NLPSearchEngine = NLPSearchEngine;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NLPSearchEngine;
}
