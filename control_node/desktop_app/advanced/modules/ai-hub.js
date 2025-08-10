/**
 * Omega SuperDesktop v2.0 - Complete AI Hub System
 * Extracted and enhanced from omega-control-center.html reference implementation
 * Includes ML models, predictive analytics, anomaly detection, and training management
 */

class AIHub extends EventTarget {
    constructor() {
        super();
        this.models = new Map();
        this.predictiveModels = new Map();
        this.trainingJobs = new Map();
        this.initialized = false;
        this.tfReady = false;
        this.analysisQueue = [];
        this.processingAnalysis = false;
        this.workers = new Map();
        this.recommendations = [];
        this.anomalies = [];
        this.predictions = new Map();
        this.eventBus = new EventTarget();
        this.capabilities = {
            tensorflowjs: !!window.tf,
            webgl: this.checkWebGLSupport(),
            webworkers: !!window.Worker,
            wasm: this.checkWASMSupport()
        };
        this.init();
    }

    async init() {
        console.log('🤖 AI Hub initializing...');
        
        try {
            // Check if TensorFlow.js is available
            if (typeof tf !== 'undefined') {
                await tf.ready();
                this.tfReady = true;
                console.log('TensorFlow.js backend:', tf.getBackend());
            } else {
                console.warn('TensorFlow.js not available, using mock AI services');
            }
            
            // Load pre-trained models
            await this.loadPredictiveModels();
            
            // Initialize analysis engine
            this.startAnalysisEngine();
            
            this.initialized = true;
            this.dispatchEvent(new CustomEvent('aiHubInitialized'));
            console.log('🤖 AI Hub ready');
        } catch (error) {
            console.error('Failed to initialize AI Hub:', error);
            this.dispatchEvent(new CustomEvent('aiHubError', { detail: { error } }));
        }
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && canvas.getContext('webgl'));
        } catch (e) {
            return false;
        }
    }

    checkWASMSupport() {
        return typeof WebAssembly === 'object';
    }

    async loadPredictiveModels() {
        try {
            if (this.tfReady) {
                // Try to load actual models, fallback to mock if not available
                try {
                    const resourceModel = await tf.loadLayersModel('/models/resource-prediction/model.json');
                    this.predictiveModels.set('resource-prediction', resourceModel);
                } catch (e) {
                    console.warn('Resource prediction model not found, using mock model');
                    this.predictiveModels.set('resource-prediction', this.createMockModel('resource'));
                }

                try {
                    const anomalyModel = await tf.loadLayersModel('/models/anomaly-detection/model.json');
                    this.predictiveModels.set('anomaly-detection', anomalyModel);
                } catch (e) {
                    console.warn('Anomaly detection model not found, using mock model');
                    this.predictiveModels.set('anomaly-detection', this.createMockModel('anomaly'));
                }

                try {
                    const loadBalanceModel = await tf.loadLayersModel('/models/load-balancing/model.json');
                    this.predictiveModels.set('load-balancing', loadBalanceModel);
                } catch (e) {
                    console.warn('Load balancing model not found, using mock model');
                    this.predictiveModels.set('load-balancing', this.createMockModel('loadbalance'));
                }
            } else {
                // Create mock models
                this.predictiveModels.set('resource-prediction', this.createMockModel('resource'));
                this.predictiveModels.set('anomaly-detection', this.createMockModel('anomaly'));
                this.predictiveModels.set('load-balancing', this.createMockModel('loadbalance'));
            }
            
            console.log('Predictive models loaded successfully');
        } catch (error) {
            console.error('Failed to load predictive models:', error);
        }
    }

    createMockModel(type) {
        return {
            type: 'mock',
            modelType: type,
            predict: (input) => {
                // Return mock predictions based on type
                switch (type) {
                    case 'resource':
                        return {
                            data: () => Promise.resolve(new Float32Array([
                                Math.random() * 0.8 + 0.1, // CPU
                                Math.random() * 0.8 + 0.1, // Memory
                                Math.random() * 0.8 + 0.1, // Disk
                                Math.random() * 0.8 + 0.1, // Network
                                Math.random() * 0.4 + 0.6  // Confidence
                            ])),
                            dispose: () => {}
                        };
                    case 'anomaly':
                        return {
                            data: () => Promise.resolve(new Float32Array([
                                Math.random() * 0.3, // Anomaly probability
                                Math.random() * 0.2, // CPU spike
                                Math.random() * 0.2, // Memory leak
                                Math.random() * 0.2, // Disk full
                                Math.random() * 0.2  // Network congestion
                            ])),
                            dispose: () => {}
                        };
                    case 'loadbalance':
                        return {
                            data: () => Promise.resolve(new Float32Array(
                                Array.from({ length: 6 }, () => Math.random() * 0.8 + 0.1)
                            )),
                            dispose: () => {}
                        };
                    default:
                        return { data: () => Promise.resolve(new Float32Array([0.5])), dispose: () => {} };
                }
            }
        };
    }

    // Predictive Analytics
    async predictResourceUsage(nodeId, timeHorizon = 3600) {
        const model = this.predictiveModels.get('resource-prediction');
        if (!model) return null;

        try {
            // Get historical data
            const historicalData = await this.getHistoricalMetrics(nodeId, 24 * 60 * 60);
            
            let prediction;
            if (this.tfReady && model.type !== 'mock') {
                // Use real TensorFlow model
                const inputTensor = tf.tensor2d([historicalData], [1, historicalData.length]);
                prediction = model.predict(inputTensor);
                inputTensor.dispose();
            } else {
                // Use mock model
                prediction = model.predict(historicalData);
            }
            
            const predictionData = await prediction.data();
            if (prediction.dispose) prediction.dispose();
            
            const result = {
                nodeId,
                cpu: predictionData[0] * 100,
                memory: predictionData[1] * 100,
                disk: predictionData[2] * 100,
                network: predictionData[3] * 100,
                confidence: predictionData[4] || 0.8,
                timeHorizon: timeHorizon,
                timestamp: new Date().toISOString()
            };

            this.predictions.set(nodeId, result);
            this.dispatchEvent(new CustomEvent('resourcePredictionComplete', {
                detail: { nodeId, prediction: result }
            }));

            return result;
        } catch (error) {
            console.error('Resource prediction failed:', error);
            return null;
        }
    }

    async detectAnomalies(metrics, nodeId = null) {
        const model = this.predictiveModels.get('anomaly-detection');
        if (!model) return { isAnomaly: false, confidence: 0 };

        try {
            let prediction;
            if (this.tfReady && model.type !== 'mock') {
                const inputTensor = tf.tensor2d([metrics], [1, metrics.length]);
                prediction = model.predict(inputTensor);
                inputTensor.dispose();
            } else {
                prediction = model.predict(metrics);
            }
            
            const predictionData = await prediction.data();
            if (prediction.dispose) prediction.dispose();
            
            const result = {
                nodeId,
                isAnomaly: predictionData[0] > 0.7,
                confidence: predictionData[0],
                anomalyType: this.classifyAnomaly(predictionData),
                severity: this.calculateSeverity(predictionData[0]),
                timestamp: new Date().toISOString(),
                details: {
                    cpuSpikeProbability: predictionData[1] || 0,
                    memoryLeakProbability: predictionData[2] || 0,
                    diskFullProbability: predictionData[3] || 0,
                    networkCongestionProbability: predictionData[4] || 0
                },
                metrics
            };

            if (result.isAnomaly) {
                this.anomalies.push(result);
                this.dispatchEvent(new CustomEvent('anomalyDetected', {
                    detail: { anomaly: result, metrics }
                }));
            }
            
            return result;
        } catch (error) {
            console.error('Anomaly detection failed:', error);
            return { isAnomaly: false, confidence: 0 };
        }
    }

    async optimizeLoadBalancing(nodeMetrics) {
        const model = this.predictiveModels.get('load-balancing');
        if (!model) return nodeMetrics;

        try {
            let prediction;
            if (this.tfReady && model.type !== 'mock') {
                const inputTensor = tf.tensor2d([nodeMetrics.flat()], [1, nodeMetrics.flat().length]);
                prediction = model.predict(inputTensor);
                inputTensor.dispose();
            } else {
                prediction = model.predict(nodeMetrics.flat());
            }
            
            const predictionData = await prediction.data();
            if (prediction.dispose) prediction.dispose();
            
            const optimizedMetrics = this.interpretLoadBalancingResults(predictionData, nodeMetrics);

            this.dispatchEvent(new CustomEvent('loadBalancingOptimized', {
                detail: { original: nodeMetrics, optimized: optimizedMetrics }
            }));

            return optimizedMetrics;
        } catch (error) {
            console.error('Load balancing optimization failed:', error);
            return nodeMetrics;
        }
    }

    classifyAnomaly(predictionData) {
        const types = ['cpu_spike', 'memory_leak', 'disk_full', 'network_congestion', 'security_breach'];
        let maxIndex = 1; // Start from index 1 to skip the main anomaly score
        for (let i = 2; i < Math.min(predictionData.length, types.length + 1); i++) {
            if (predictionData[i] > predictionData[maxIndex]) {
                maxIndex = i;
            }
        }
        return types[maxIndex - 1] || 'unknown';
    }

    calculateSeverity(confidence) {
        if (confidence >= 0.9) return 'critical';
        if (confidence >= 0.7) return 'high';
        if (confidence >= 0.5) return 'medium';
        return 'low';
    }

    interpretLoadBalancingResults(predictionData, nodeMetrics) {
        return nodeMetrics.map((metrics, index) => ({
            ...metrics,
            recommendedLoad: predictionData[index] * 100,
            priority: predictionData[index] > 0.8 ? 'high' : predictionData[index] > 0.5 ? 'medium' : 'low',
            loadShift: predictionData[index] - (metrics.currentLoad || 0.5),
            efficiency: Math.min(1, predictionData[index] + 0.2)
        }));
    }

    // Training Management
    startTraining(modelType, trainingData, config = {}) {
        const jobId = `training-${Date.now()}`;
        
        const trainingJob = {
            id: jobId,
            modelType: modelType,
            status: 'initializing',
            progress: 0,
            epoch: 0,
            maxEpochs: config.epochs || 100,
            batchSize: config.batchSize || 32,
            learningRate: config.learningRate || 0.001,
            startTime: Date.now(),
            loss: null,
            accuracy: null,
            validationLoss: null,
            validationAccuracy: null
        };
        
        this.trainingJobs.set(jobId, trainingJob);
        
        this.dispatchEvent(new CustomEvent('trainingStarted', {
            detail: { job: trainingJob }
        }));
        
        // Start training in background
        this.performTraining(jobId, trainingData, config);
        
        return jobId;
    }

    async performTraining(jobId, trainingData, config) {
        const job = this.trainingJobs.get(jobId);
        if (!job) return;

        try {
            job.status = 'training';
            
            if (this.tfReady && trainingData && trainingData.x && trainingData.y) {
                // Real TensorFlow training
                await this.performRealTraining(jobId, trainingData, config);
            } else {
                // Mock training for demo
                await this.performMockTraining(jobId, config);
            }
        } catch (error) {
            job.status = 'failed';
            job.error = error.message;
            job.endTime = Date.now();
            
            this.dispatchEvent(new CustomEvent('trainingFailed', {
                detail: { job, error }
            }));
            
            console.error('Training failed:', error);
        }
    }

    async performRealTraining(jobId, trainingData, config) {
        const job = this.trainingJobs.get(jobId);
        
        // Create model architecture
        const model = tf.sequential({
            layers: [
                tf.layers.dense({ 
                    inputShape: [trainingData.inputSize || 10], 
                    units: 128, 
                    activation: 'relu' 
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ 
                    units: trainingData.outputSize || 1, 
                    activation: 'sigmoid' 
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(job.learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        // Training callback
        const trainingCallback = {
            onEpochEnd: (epoch, logs) => {
                job.epoch = epoch + 1;
                job.progress = ((epoch + 1) / job.maxEpochs) * 100;
                job.loss = logs.loss;
                job.accuracy = logs.acc;
                job.validationLoss = logs.val_loss;
                job.validationAccuracy = logs.val_acc;
                
                this.dispatchEvent(new CustomEvent('trainingProgress', {
                    detail: { job }
                }));
            }
        };

        // Train the model
        await model.fit(trainingData.x, trainingData.y, {
            epochs: job.maxEpochs,
            batchSize: job.batchSize,
            validationSplit: 0.2,
            callbacks: [trainingCallback]
        });

        job.status = 'completed';
        job.progress = 100;
        job.endTime = Date.now();
        
        // Save trained model
        const modelName = `${job.modelType}-${jobId}`;
        this.models.set(modelName, model);
        
        this.dispatchEvent(new CustomEvent('trainingCompleted', {
            detail: { job, modelName }
        }));
    }

    async performMockTraining(jobId, config) {
        const job = this.trainingJobs.get(jobId);
        
        // Simulate training progress
        for (let epoch = 0; epoch < job.maxEpochs; epoch++) {
            await new Promise(resolve => setTimeout(resolve, 50)); // 50ms per epoch
            
            job.epoch = epoch + 1;
            job.progress = ((epoch + 1) / job.maxEpochs) * 100;
            job.loss = Math.max(0.1, 2.0 * Math.exp(-epoch / 20) + Math.random() * 0.1);
            job.accuracy = Math.min(0.95, 0.5 + 0.4 * (1 - Math.exp(-epoch / 30)) + Math.random() * 0.05);
            job.validationLoss = job.loss + Math.random() * 0.05;
            job.validationAccuracy = job.accuracy - Math.random() * 0.02;
            
            this.dispatchEvent(new CustomEvent('trainingProgress', {
                detail: { job }
            }));
        }

        job.status = 'completed';
        job.progress = 100;
        job.endTime = Date.now();
        
        this.dispatchEvent(new CustomEvent('trainingCompleted', {
            detail: { job }
        }));
    }

    // Analysis Engine
    startAnalysisEngine() {
        setInterval(() => {
            this.processAnalysisQueue();
        }, 5000); // Process queue every 5 seconds
    }

    async processAnalysisQueue() {
        if (this.processingAnalysis || this.analysisQueue.length === 0) return;
        
        this.processingAnalysis = true;
        
        try {
            const analysisItem = this.analysisQueue.shift();
            await this.performAnalysis(analysisItem);
        } catch (error) {
            console.error('Analysis processing failed:', error);
        } finally {
            this.processingAnalysis = false;
        }
    }

    async performAnalysis(analysisItem) {
        const { type, data, nodeId } = analysisItem;
        
        switch (type) {
            case 'resource-prediction':
                await this.predictResourceUsage(nodeId);
                break;
            case 'anomaly-detection':
                await this.detectAnomalies(data, nodeId);
                break;
            case 'load-balancing':
                await this.optimizeLoadBalancing(data);
                break;
            case 'system-health':
                await this.analyzeSystemHealth(data);
                break;
        }
    }

    queueAnalysis(type, data, nodeId = null) {
        this.analysisQueue.push({ type, data, nodeId, timestamp: Date.now() });
    }

    async analyzeSystemHealth(systemData) {
        try {
            const healthScore = this.calculateHealthScore(systemData);
            const recommendations = this.generateRecommendations(systemData, healthScore);
            
            const analysis = {
                overallHealth: healthScore,
                status: this.getHealthStatus(healthScore),
                recommendations: recommendations,
                timestamp: new Date().toISOString(),
                details: {
                    cpuHealth: this.calculateCpuHealth(systemData),
                    memoryHealth: this.calculateMemoryHealth(systemData),
                    diskHealth: this.calculateDiskHealth(systemData),
                    networkHealth: this.calculateNetworkHealth(systemData)
                }
            };

            this.dispatchEvent(new CustomEvent('systemHealthAnalyzed', {
                detail: { analysis, systemData }
            }));

            return analysis;
        } catch (error) {
            console.error('System health analysis failed:', error);
            return null;
        }
    }

    calculateHealthScore(systemData) {
        const weights = { cpu: 0.3, memory: 0.3, disk: 0.2, network: 0.2 };
        
        const cpuScore = Math.max(0, 100 - (systemData.cpu || 0));
        const memoryScore = Math.max(0, 100 - (systemData.memory || 0));
        const diskScore = Math.max(0, 100 - (systemData.disk || 0));
        const networkScore = Math.max(0, 100 - (systemData.network || 50));
        
        return (
            cpuScore * weights.cpu +
            memoryScore * weights.memory +
            diskScore * weights.disk +
            networkScore * weights.network
        );
    }

    calculateCpuHealth(systemData) {
        const cpuUsage = systemData.cpu || 0;
        if (cpuUsage > 90) return 'critical';
        if (cpuUsage > 80) return 'warning';
        if (cpuUsage > 70) return 'caution';
        return 'good';
    }

    calculateMemoryHealth(systemData) {
        const memoryUsage = systemData.memory || 0;
        if (memoryUsage > 95) return 'critical';
        if (memoryUsage > 85) return 'warning';
        if (memoryUsage > 75) return 'caution';
        return 'good';
    }

    calculateDiskHealth(systemData) {
        const diskUsage = systemData.disk || 0;
        if (diskUsage > 95) return 'critical';
        if (diskUsage > 90) return 'warning';
        if (diskUsage > 80) return 'caution';
        return 'good';
    }

    calculateNetworkHealth(systemData) {
        const networkLatency = systemData.networkLatency || 0;
        if (networkLatency > 1000) return 'critical';
        if (networkLatency > 500) return 'warning';
        if (networkLatency > 200) return 'caution';
        return 'good';
    }

    getHealthStatus(score) {
        if (score >= 90) return 'excellent';
        if (score >= 80) return 'good';
        if (score >= 70) return 'fair';
        if (score >= 60) return 'poor';
        return 'critical';
    }

    generateRecommendations(systemData, healthScore) {
        const recommendations = [];
        
        if (systemData.cpu > 80) {
            recommendations.push({
                type: 'performance',
                priority: 'high',
                message: 'High CPU usage detected. Consider scaling up or optimizing workloads.',
                action: 'scale-cpu'
            });
        }
        
        if (systemData.memory > 85) {
            recommendations.push({
                type: 'performance',
                priority: 'high',
                message: 'High memory usage detected. Consider adding more memory or optimizing applications.',
                action: 'scale-memory'
            });
        }
        
        if (systemData.disk > 90) {
            recommendations.push({
                type: 'storage',
                priority: 'critical',
                message: 'Disk space is running low. Immediate action required.',
                action: 'cleanup-disk'
            });
        }
        
        if (healthScore < 70) {
            recommendations.push({
                type: 'system',
                priority: 'medium',
                message: 'Overall system health is below optimal. Review system performance.',
                action: 'system-review'
            });
        }
        
        return recommendations;
    }

    // Historical data
    async getHistoricalMetrics(nodeId, duration) {
        try {
            // Try to fetch real data first
            const response = await fetch(`http://127.0.0.1:8443/api/nodes/${nodeId}/metrics/historical?duration=${duration}`);
            if (response.ok) {
                const data = await response.json();
                return data.metrics || this.generateMockHistoricalData(duration);
            }
        } catch (error) {
            console.warn('Failed to fetch real historical data, using mock data');
        }
        
        return this.generateMockHistoricalData(duration);
    }

    generateMockHistoricalData(duration) {
        const dataPoints = Math.min(100, Math.floor(duration / 60)); // One point per minute, max 100 points
        const data = [];
        
        for (let i = 0; i < dataPoints; i++) {
            data.push(
                Math.sin(i / 10) * 0.2 + 0.5 + Math.random() * 0.1, // CPU
                Math.sin(i / 15) * 0.3 + 0.6 + Math.random() * 0.1, // Memory
                Math.sin(i / 20) * 0.1 + 0.3 + Math.random() * 0.05, // Disk
                Math.sin(i / 8) * 0.2 + 0.4 + Math.random() * 0.1   // Network
            );
        }
        
        return data;
    }

    // Smart Recommendations Engine
    generateSmartRecommendations() {
        const recommendations = [];
        
        // Analyze recent anomalies
        const recentAnomalies = this.anomalies.filter(a => 
            Date.now() - new Date(a.timestamp).getTime() < 3600000 // Last hour
        );
        
        if (recentAnomalies.length > 3) {
            recommendations.push({
                type: 'alert',
                priority: 'high',
                title: 'Multiple Anomalies Detected',
                message: `${recentAnomalies.length} anomalies detected in the last hour. System may be under stress.`,
                action: 'investigate-anomalies'
            });
        }
        
        // Analyze predictions for resource shortages
        this.predictions.forEach((prediction, nodeId) => {
            if (prediction.confidence > 0.8) {
                if (prediction.cpu > 90) {
                    recommendations.push({
                        type: 'resource',
                        priority: 'medium',
                        title: 'CPU Shortage Predicted',
                        message: `Node ${nodeId} predicted to reach ${prediction.cpu.toFixed(1)}% CPU usage`,
                        action: 'scale-cpu'
                    });
                }
                
                if (prediction.memory > 90) {
                    recommendations.push({
                        type: 'resource',
                        priority: 'medium',
                        title: 'Memory Shortage Predicted',
                        message: `Node ${nodeId} predicted to reach ${prediction.memory.toFixed(1)}% memory usage`,
                        action: 'scale-memory'
                    });
                }
            }
        });
        
        return recommendations;
    }

    // Public API methods
    getTrainingJobs() {
        return Array.from(this.trainingJobs.values());
    }

    getActiveTrainingJobs() {
        return Array.from(this.trainingJobs.values()).filter(job => 
            job.status === 'training' || job.status === 'initializing'
        );
    }

    getModels() {
        return Array.from(this.models.keys());
    }

    getTrainingJob(jobId) {
        return this.trainingJobs.get(jobId);
    }

    cancelTraining(jobId) {
        const job = this.trainingJobs.get(jobId);
        if (job && (job.status === 'training' || job.status === 'initializing')) {
            job.status = 'cancelled';
            job.endTime = Date.now();
            
            this.dispatchEvent(new CustomEvent('trainingCancelled', {
                detail: { job }
            }));
        }
    }

    deleteTrainingJob(jobId) {
        const job = this.trainingJobs.get(jobId);
        if (job) {
            this.trainingJobs.delete(jobId);
            
            this.dispatchEvent(new CustomEvent('trainingJobDeleted', {
                detail: { jobId, job }
            }));
        }
    }

    getPredictions() {
        return this.predictions;
    }

    getAnomalies() {
        return this.anomalies;
    }

    getRecentAnomalies(timeframe = 3600000) { // Default 1 hour
        const cutoff = Date.now() - timeframe;
        return this.anomalies.filter(a => 
            new Date(a.timestamp).getTime() > cutoff
        );
    }

    getInsights() {
        const recentAnomalies = this.getRecentAnomalies();
        
        return {
            predictedIssues: Array.from(this.predictions.values()).filter(p => 
                p.confidence > 0.7 && (p.cpu > 80 || p.memory > 80)
            ).map(p => ({
                type: p.cpu > 80 ? 'cpu_shortage' : 'memory_shortage',
                probability: p.confidence,
                timeframe: `${Math.floor(p.timeHorizon / 3600)} hours`,
                description: `${p.cpu > 80 ? 'CPU' : 'Memory'} usage trending upward on node ${p.nodeId}`
            })),
            optimizationOpportunities: [
                {
                    type: 'load_redistribution',
                    potential_savings: '15%',
                    description: 'Rebalance workloads across nodes'
                }
            ],
            anomaliesDetected: recentAnomalies.length,
            systemHealth: 'good', // Would be calculated from recent data
            recommendations: this.generateSmartRecommendations()
        };
    }

    getCapabilities() {
        return this.capabilities;
    }

    isInitialized() {
        return this.initialized;
    }

    // Event handling
    addEventListener(type, callback) {
        this.eventBus.addEventListener(type, callback);
    }

    removeEventListener(type, callback) {
        this.eventBus.removeEventListener(type, callback);
    }

    // Cleanup
    dispose() {
        // Dispose of TensorFlow models
        this.models.forEach(model => {
            if (model.dispose) model.dispose();
        });
        this.predictiveModels.forEach(model => {
            if (model.dispose && model.type !== 'mock') model.dispose();
        });
        
        this.models.clear();
        this.predictiveModels.clear();
        this.trainingJobs.clear();
        this.analysisQueue.length = 0;
        this.predictions.clear();
        this.anomalies.length = 0;
        this.recommendations.length = 0;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.AIHub = AIHub;
}

// Also support CommonJS/ES modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIHub;
}
