export function renderAIHub(root, state) {
  // Advanced AI Hub with model management and training pipelines
  root.innerHTML = `
    <div style="display: grid; grid-template-rows: auto auto 1fr; gap: 12px; height: 100%; padding: 8px;">
      <!-- AI Hub Status Header -->
      <div class="ai-hub-status-header">
        <div class="ai-hub-overview">
          <div class="ai-metric">
            <i class="fas fa-robot"></i>
            <span>Models: <strong id="models-count">8</strong></span>
          </div>
          <div class="ai-metric">
            <i class="fas fa-play"></i>
            <span>Running: <strong id="running-count">3</strong></span>
          </div>
          <div class="ai-metric">
            <i class="fas fa-cloud"></i>
            <span>Training: <strong id="training-count">1</strong></span>
          </div>
          <div class="ai-metric">
            <i class="fas fa-chart-line"></i>
            <span>Inferences: <strong id="inferences-count">15.2K</strong></span>
          </div>
        </div>
        <div class="ai-hub-actions">
          <button onclick="refreshAIModels()" class="ai-btn">
            <i class="fas fa-sync"></i>
            Refresh
          </button>
          <button onclick="deployModel()" class="ai-btn">
            <i class="fas fa-rocket"></i>
            Deploy
          </button>
          <button onclick="createTrainingPipeline()" class="ai-btn primary">
            <i class="fas fa-plus"></i>
            New Training
          </button>
        </div>
      </div>

      <!-- AI Hub Tabs -->
      <div class="ai-hub-tabs">
        <button class="ai-tab active" onclick="switchAITab('models')">
          <i class="fas fa-brain"></i>
          <span>Models</span>
        </button>
        <button class="ai-tab" onclick="switchAITab('training')">
          <i class="fas fa-graduation-cap"></i>
          <span>Training</span>
        </button>
        <button class="ai-tab" onclick="switchAITab('inference')">
          <i class="fas fa-bolt"></i>
          <span>Inference</span>
        </button>
        <button class="ai-tab" onclick="switchAITab('marketplace')">
          <i class="fas fa-store"></i>
          <span>Marketplace</span>
        </button>
        <button class="ai-tab" onclick="switchAITab('analytics')">
          <i class="fas fa-chart-bar"></i>
          <span>Analytics</span>
        </button>
        <button class="ai-tab" onclick="switchAITab('deployment')">
          <i class="fas fa-cloud-upload-alt"></i>
          <span>Deployment</span>
        </button>
      </div>

      <!-- AI Hub Content -->
      <div class="ai-hub-content">
        <!-- Models View -->
        <div id="ai-models" class="ai-view active">
          <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; height: 100%;">
            <div class="models-main">
              <div class="models-toolbar">
                <div class="models-filters">
                  <select id="model-type-filter" class="ai-select">
                    <option value="all">All Types</option>
                    <option value="llm">Language Models</option>
                    <option value="vision">Computer Vision</option>
                    <option value="audio">Audio Processing</option>
                    <option value="multimodal">Multimodal</option>
                  </select>
                  <select id="model-status-filter" class="ai-select">
                    <option value="all">All Status</option>
                    <option value="running">Running</option>
                    <option value="stopped">Stopped</option>
                    <option value="training">Training</option>
                    <option value="error">Error</option>
                  </select>
                </div>
                <div class="models-search">
                  <input type="text" id="models-search" placeholder="Search models..." class="ai-search">
                </div>
              </div>
              <div class="models-grid" id="models-grid">
                <!-- Populated by renderModelsGrid() -->
              </div>
            </div>
            <div class="model-details-panel">
              <div id="model-details" class="model-details">
                <div class="no-selection">
                  <i class="fas fa-brain"></i>
                  <p>Select a model to view details</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Training View -->
        <div id="ai-training" class="ai-view">
          <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
            <div class="training-controls">
              <div class="training-stats">
                <div class="training-stat">
                  <span class="stat-label">Active Jobs</span>
                  <span class="stat-value">2</span>
                </div>
                <div class="training-stat">
                  <span class="stat-label">Queue</span>
                  <span class="stat-value">1</span>
                </div>
                <div class="training-stat">
                  <span class="stat-label">Completed</span>
                  <span class="stat-value">15</span>
                </div>
                <div class="training-stat">
                  <span class="stat-label">GPU Utilization</span>
                  <span class="stat-value">78%</span>
                </div>
              </div>
              <div class="training-actions">
                <button onclick="createTrainingJob()" class="training-btn primary">
                  <i class="fas fa-plus"></i>
                  New Training Job
                </button>
                <button onclick="pauseAllTraining()" class="training-btn">
                  <i class="fas fa-pause"></i>
                  Pause All
                </button>
              </div>
            </div>
            <div class="training-jobs" id="training-jobs">
              <!-- Populated by renderTrainingJobs() -->
            </div>
          </div>
        </div>

        <!-- Inference View -->
        <div id="ai-inference" class="ai-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="inference-endpoints">
              <h5>Active Endpoints</h5>
              <div id="inference-endpoints-list">
                <!-- Populated by renderInferenceEndpoints() -->
              </div>
            </div>
            <div class="inference-playground">
              <h5>Model Playground</h5>
              <div class="playground-interface">
                <div class="playground-model-select">
                  <select id="playground-model" class="ai-select">
                    <option value="">Select Model...</option>
                    <option value="gpt-4o">GPT-4O</option>
                    <option value="claude-3">Claude 3</option>
                    <option value="vision-model">Vision Model</option>
                  </select>
                </div>
                <div class="playground-input">
                  <textarea id="playground-prompt" placeholder="Enter your prompt..." class="ai-textarea"></textarea>
                </div>
                <div class="playground-controls">
                  <button onclick="runInference()" class="playground-btn primary">
                    <i class="fas fa-play"></i>
                    Run Inference
                  </button>
                  <button onclick="clearPlayground()" class="playground-btn">
                    <i class="fas fa-trash"></i>
                    Clear
                  </button>
                </div>
                <div class="playground-output">
                  <div id="playground-result" class="playground-result">
                    Results will appear here...
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Marketplace View -->
        <div id="ai-marketplace" class="ai-view">
          <div class="marketplace-header">
            <h5>AI Model Marketplace</h5>
            <div class="marketplace-filters">
              <select id="marketplace-category" class="ai-select">
                <option value="all">All Categories</option>
                <option value="llm">Language Models</option>
                <option value="vision">Computer Vision</option>
                <option value="audio">Audio/Speech</option>
                <option value="embedding">Embeddings</option>
              </select>
              <select id="marketplace-size" class="ai-select">
                <option value="all">All Sizes</option>
                <option value="small">Small (&lt;1B)</option>
                <option value="medium">Medium (1B-10B)</option>
                <option value="large">Large (&gt;10B)</option>
              </select>
            </div>
          </div>
          <div class="marketplace-content">
            <div class="marketplace-featured">
              <h6>Featured Models</h6>
              <div id="featured-models">
                <!-- Populated by renderFeaturedModels() -->
              </div>
            </div>
            <div class="marketplace-browse">
              <h6>Browse Models</h6>
              <div id="marketplace-models">
                <!-- Populated by renderMarketplaceModels() -->
              </div>
            </div>
          </div>
        </div>

        <!-- Analytics View -->
        <div id="ai-analytics" class="ai-view">
          <div style="display: grid; grid-template-rows: auto 1fr; gap: 16px; height: 100%;">
            <div class="analytics-summary">
              <div class="analytics-card">
                <div class="analytics-title">Model Performance</div>
                <div class="analytics-chart" id="model-performance-chart">
                  <!-- Chart placeholder -->
                  <div class="chart-placeholder">Performance metrics chart</div>
                </div>
              </div>
              <div class="analytics-card">
                <div class="analytics-title">Resource Usage</div>
                <div class="analytics-chart" id="resource-usage-chart">
                  <!-- Chart placeholder -->
                  <div class="chart-placeholder">Resource usage chart</div>
                </div>
              </div>
            </div>
            <div class="analytics-details">
              <div class="analytics-tabs">
                <button class="analytics-tab active" onclick="switchAnalyticsTab('performance')">Performance</button>
                <button class="analytics-tab" onclick="switchAnalyticsTab('usage')">Usage</button>
                <button class="analytics-tab" onclick="switchAnalyticsTab('costs')">Costs</button>
                <button class="analytics-tab" onclick="switchAnalyticsTab('errors')">Errors</button>
              </div>
              <div id="analytics-content" class="analytics-tab-content">
                <!-- Populated by renderAnalyticsContent() -->
              </div>
            </div>
          </div>
        </div>

        <!-- Deployment View -->
        <div id="ai-deployment" class="ai-view">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%;">
            <div class="deployment-manager">
              <h5>Deployment Manager</h5>
              <div class="deployment-environments">
                <div class="environment-item" onclick="selectEnvironment('production')">
                  <div class="env-icon production"></div>
                  <div class="env-info">
                    <div class="env-name">Production</div>
                    <div class="env-status">8 models deployed</div>
                  </div>
                  <div class="env-indicator active"></div>
                </div>
                <div class="environment-item" onclick="selectEnvironment('staging')">
                  <div class="env-icon staging"></div>
                  <div class="env-info">
                    <div class="env-name">Staging</div>
                    <div class="env-status">3 models deployed</div>
                  </div>
                  <div class="env-indicator"></div>
                </div>
                <div class="environment-item" onclick="selectEnvironment('development')">
                  <div class="env-icon development"></div>
                  <div class="env-info">
                    <div class="env-name">Development</div>
                    <div class="env-status">5 models deployed</div>
                  </div>
                  <div class="env-indicator"></div>
                </div>
              </div>
            </div>
            <div class="deployment-pipeline">
              <h5>Deployment Pipeline</h5>
              <div id="deployment-pipeline-view">
                <!-- Populated by renderDeploymentPipeline() -->
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initialize AI Hub data
  renderModelsGrid(state);
  renderTrainingJobs(state);
  renderInferenceEndpoints(state);
  renderFeaturedModels(state);
  renderMarketplaceModels(state);
  renderAnalyticsContent('performance');
  renderDeploymentPipeline(state);
}

function renderModelsGrid(state) {
  const container = document.getElementById('models-grid');
  
  // Sample AI models
  const models = [
    { name: 'GPT-4O-Local', type: 'llm', status: 'running', size: '8B', accuracy: 0.94, latency: '120ms', memory: '16GB' },
    { name: 'Claude-3-Fine', type: 'llm', status: 'stopped', size: '70B', accuracy: 0.96, latency: '250ms', memory: '40GB' },
    { name: 'Vision-Pro-v2', type: 'vision', status: 'running', size: '2B', accuracy: 0.89, latency: '80ms', memory: '8GB' },
    { name: 'Audio-Transcriber', type: 'audio', status: 'running', size: '1B', accuracy: 0.91, latency: '200ms', memory: '4GB' },
    { name: 'MultiModal-Alpha', type: 'multimodal', status: 'training', size: '12B', accuracy: 0.87, latency: '300ms', memory: '24GB' },
    { name: 'Code-Assistant-v3', type: 'llm', status: 'stopped', size: '3B', accuracy: 0.88, latency: '90ms', memory: '6GB' }
  ];
  
  container.innerHTML = `
    <div class="models-list">
      ${models.map(model => `
        <div class="model-card" onclick="selectModel('${model.name}')">
          <div class="model-header">
            <div class="model-info">
              <div class="model-name">${model.name}</div>
              <div class="model-type">${model.type.toUpperCase()}</div>
            </div>
            <div class="model-status ${model.status}">${model.status.toUpperCase()}</div>
          </div>
          <div class="model-metrics">
            <div class="model-metric">
              <span class="metric-label">Size:</span>
              <span class="metric-value">${model.size}</span>
            </div>
            <div class="model-metric">
              <span class="metric-label">Accuracy:</span>
              <span class="metric-value">${(model.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="model-metric">
              <span class="metric-label">Latency:</span>
              <span class="metric-value">${model.latency}</span>
            </div>
            <div class="model-metric">
              <span class="metric-label">Memory:</span>
              <span class="metric-value">${model.memory}</span>
            </div>
          </div>
          <div class="model-actions">
            ${model.status === 'running' ? 
              `<button onclick="stopModel('${model.name}')" class="model-action-btn danger">
                <i class="fas fa-stop"></i>
                Stop
              </button>` :
              `<button onclick="startModel('${model.name}')" class="model-action-btn primary">
                <i class="fas fa-play"></i>
                Start
              </button>`
            }
            <button onclick="configureModel('${model.name}')" class="model-action-btn">
              <i class="fas fa-cog"></i>
              Configure
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function selectModel(modelName) {
  const detailsPanel = document.getElementById('model-details');
  
  // Remove selection from other cards
  document.querySelectorAll('.model-card').forEach(card => card.classList.remove('selected'));
  
  // Add selection to clicked card
  event.currentTarget.classList.add('selected');
  
  // Sample model details
  detailsPanel.innerHTML = `
    <div class="model-detail-header">
      <div class="detail-icon">
        <i class="fas fa-brain"></i>
      </div>
      <div class="detail-title">
        <h6>${modelName}</h6>
        <span class="detail-type">Large Language Model</span>
      </div>
    </div>
    <div class="model-detail-content">
      <div class="detail-section">
        <h7>Performance Metrics</h7>
        <div class="metrics-grid">
          <div class="metric-item">
            <span class="metric-name">Accuracy</span>
            <span class="metric-val">94.2%</span>
          </div>
          <div class="metric-item">
            <span class="metric-name">Latency</span>
            <span class="metric-val">120ms</span>
          </div>
          <div class="metric-item">
            <span class="metric-name">Throughput</span>
            <span class="metric-val">450 tok/s</span>
          </div>
          <div class="metric-item">
            <span class="metric-name">Memory</span>
            <span class="metric-val">16GB</span>
          </div>
        </div>
      </div>
      <div class="detail-section">
        <h7>Model Configuration</h7>
        <div class="config-item">
          <span class="config-label">Parameters:</span>
          <span class="config-value">8.1B</span>
        </div>
        <div class="config-item">
          <span class="config-label">Context Length:</span>
          <span class="config-value">32K tokens</span>
        </div>
        <div class="config-item">
          <span class="config-label">Quantization:</span>
          <span class="config-value">FP16</span>
        </div>
      </div>
      <div class="detail-actions">
        <button onclick="testModel('${modelName}')" class="detail-action-btn primary">
          <i class="fas fa-vial"></i>
          Test Model
        </button>
        <button onclick="exportModel('${modelName}')" class="detail-action-btn">
          <i class="fas fa-download"></i>
          Export
        </button>
      </div>
    </div>
  `;
}

function renderTrainingJobs(state) {
  const container = document.getElementById('training-jobs');
  
  const jobs = [
    { name: 'Custom LLM Fine-tune', status: 'running', progress: 65, eta: '2h 15m', gpu: 'A100-80GB' },
    { name: 'Vision Model Training', status: 'queued', progress: 0, eta: '4h 30m', gpu: 'Pending' },
    { name: 'Audio Classification', status: 'completed', progress: 100, eta: 'Completed', gpu: 'V100-32GB' }
  ];
  
  container.innerHTML = `
    <div class="training-jobs-list">
      ${jobs.map(job => `
        <div class="training-job-item">
          <div class="job-header">
            <div class="job-name">${job.name}</div>
            <div class="job-status ${job.status}">${job.status.toUpperCase()}</div>
          </div>
          <div class="job-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${job.progress}%;"></div>
            </div>
            <span class="progress-text">${job.progress}%</span>
          </div>
          <div class="job-details">
            <div class="job-detail">
              <span class="detail-label">ETA:</span>
              <span class="detail-value">${job.eta}</span>
            </div>
            <div class="job-detail">
              <span class="detail-label">GPU:</span>
              <span class="detail-value">${job.gpu}</span>
            </div>
          </div>
          <div class="job-actions">
            ${job.status === 'running' ? 
              `<button onclick="pauseTraining('${job.name}')" class="job-action-btn">
                <i class="fas fa-pause"></i>
                Pause
              </button>` : 
              job.status === 'queued' ?
              `<button onclick="startTraining('${job.name}')" class="job-action-btn primary">
                <i class="fas fa-play"></i>
                Start
              </button>` : ''
            }
            <button onclick="viewTrainingLogs('${job.name}')" class="job-action-btn">
              <i class="fas fa-file-alt"></i>
              Logs
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderInferenceEndpoints(state) {
  const container = document.getElementById('inference-endpoints-list');
  
  const endpoints = [
    { name: 'GPT-4O API', url: '/api/v1/gpt-4o', status: 'active', requests: '1.2K/h', latency: '120ms' },
    { name: 'Vision API', url: '/api/v1/vision', status: 'active', requests: '450/h', latency: '80ms' },
    { name: 'Audio API', url: '/api/v1/audio', status: 'inactive', requests: '0/h', latency: 'N/A' }
  ];
  
  container.innerHTML = `
    <div class="endpoints-list">
      ${endpoints.map(endpoint => `
        <div class="endpoint-item">
          <div class="endpoint-header">
            <div class="endpoint-name">${endpoint.name}</div>
            <div class="endpoint-status ${endpoint.status}">${endpoint.status.toUpperCase()}</div>
          </div>
          <div class="endpoint-url">${endpoint.url}</div>
          <div class="endpoint-metrics">
            <div class="endpoint-metric">
              <span class="metric-label">Requests:</span>
              <span class="metric-value">${endpoint.requests}</span>
            </div>
            <div class="endpoint-metric">
              <span class="metric-label">Latency:</span>
              <span class="metric-value">${endpoint.latency}</span>
            </div>
          </div>
          <div class="endpoint-actions">
            <button onclick="testEndpoint('${endpoint.name}')" class="endpoint-action-btn primary">
              <i class="fas fa-play"></i>
              Test
            </button>
            <button onclick="configureEndpoint('${endpoint.name}')" class="endpoint-action-btn">
              <i class="fas fa-cog"></i>
              Configure
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderFeaturedModels(state) {
  const container = document.getElementById('featured-models');
  
  const featured = [
    { name: 'Llama 3.1 405B', description: 'State-of-the-art open-source language model', downloads: '2.1M', rating: 4.9 },
    { name: 'CLIP Vision', description: 'Advanced image understanding and classification', downloads: '890K', rating: 4.8 },
    { name: 'Whisper v3', description: 'Robust speech recognition across languages', downloads: '1.5M', rating: 4.7 }
  ];
  
  container.innerHTML = `
    <div class="featured-models-grid">
      ${featured.map(model => `
        <div class="featured-model-card">
          <div class="featured-badge">FEATURED</div>
          <div class="featured-content">
            <h6>${model.name}</h6>
            <p>${model.description}</p>
            <div class="featured-stats">
              <div class="featured-stat">
                <i class="fas fa-download"></i>
                <span>${model.downloads}</span>
              </div>
              <div class="featured-stat">
                <i class="fas fa-star"></i>
                <span>${model.rating}</span>
              </div>
            </div>
          </div>
          <div class="featured-actions">
            <button onclick="downloadModel('${model.name}')" class="featured-action-btn primary">
              Download
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderMarketplaceModels(state) {
  const container = document.getElementById('marketplace-models');
  
  const models = [
    { name: 'GPT-Neo 2.7B', type: 'llm', size: '2.7B', license: 'MIT', rating: 4.6 },
    { name: 'ResNet-50', type: 'vision', size: '98MB', license: 'Apache 2.0', rating: 4.8 },
    { name: 'BERT Base', type: 'embedding', size: '440MB', license: 'Apache 2.0', rating: 4.7 },
    { name: 'WaveNet TTS', type: 'audio', size: '256MB', license: 'MIT', rating: 4.5 }
  ];
  
  container.innerHTML = `
    <div class="marketplace-models-grid">
      ${models.map(model => `
        <div class="marketplace-model-card">
          <div class="marketplace-model-header">
            <div class="marketplace-model-name">${model.name}</div>
            <div class="marketplace-model-type">${model.type.toUpperCase()}</div>
          </div>
          <div class="marketplace-model-info">
            <div class="marketplace-model-size">Size: ${model.size}</div>
            <div class="marketplace-model-license">License: ${model.license}</div>
            <div class="marketplace-model-rating">
              <i class="fas fa-star"></i>
              <span>${model.rating}</span>
            </div>
          </div>
          <div class="marketplace-model-actions">
            <button onclick="downloadMarketplaceModel('${model.name}')" class="marketplace-action-btn primary">
              <i class="fas fa-download"></i>
              Download
            </button>
            <button onclick="viewMarketplaceDetails('${model.name}')" class="marketplace-action-btn">
              <i class="fas fa-info"></i>
              Details
            </button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderAnalyticsContent(tab) {
  const container = document.getElementById('analytics-content');
  
  if (tab === 'performance') {
    container.innerHTML = `
      <div class="analytics-performance">
        <div class="performance-table">
          <table class="ai-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Latency</th>
                <th>Throughput</th>
                <th>Memory Usage</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>GPT-4O-Local</td>
                <td>94.2%</td>
                <td>120ms</td>
                <td>450 tok/s</td>
                <td>16GB</td>
              </tr>
              <tr>
                <td>Vision-Pro-v2</td>
                <td>89.1%</td>
                <td>80ms</td>
                <td>1200 img/s</td>
                <td>8GB</td>
              </tr>
              <tr>
                <td>Audio-Transcriber</td>
                <td>91.5%</td>
                <td>200ms</td>
                <td>15 min/s</td>
                <td>4GB</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    `;
  }
}

function renderDeploymentPipeline(state) {
  const container = document.getElementById('deployment-pipeline-view');
  
  container.innerHTML = `
    <div class="deployment-pipeline">
      <div class="pipeline-stage">
        <div class="stage-icon development"></div>
        <div class="stage-info">
          <div class="stage-name">Development</div>
          <div class="stage-status">5 models</div>
        </div>
        <div class="stage-arrow">→</div>
      </div>
      <div class="pipeline-stage">
        <div class="stage-icon staging"></div>
        <div class="stage-info">
          <div class="stage-name">Staging</div>
          <div class="stage-status">3 models</div>
        </div>
        <div class="stage-arrow">→</div>
      </div>
      <div class="pipeline-stage">
        <div class="stage-icon production"></div>
        <div class="stage-info">
          <div class="stage-name">Production</div>
          <div class="stage-status">8 models</div>
        </div>
      </div>
    </div>
    <div class="pipeline-actions">
      <button onclick="promoteModel()" class="pipeline-action-btn primary">
        <i class="fas fa-arrow-up"></i>
        Promote Model
      </button>
      <button onclick="rollbackDeployment()" class="pipeline-action-btn danger">
        <i class="fas fa-undo"></i>
        Rollback
      </button>
    </div>
  `;
}

// Global action functions
window.switchAITab = (tab) => {
  // Remove active class from all tabs and views
  document.querySelectorAll('.ai-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.ai-view').forEach(view => view.classList.remove('active'));
  
  // Add active class to selected tab
  document.querySelector(`[onclick="switchAITab('${tab}')"]`).classList.add('active');
  
  // Show appropriate view
  document.getElementById(`ai-${tab}`).classList.add('active');
};

window.switchAnalyticsTab = (tab) => {
  document.querySelectorAll('.analytics-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[onclick="switchAnalyticsTab('${tab}')"]`).classList.add('active');
  renderAnalyticsContent(tab);
};

window.startModel = async (modelName) => {
  try {
    window.notify('info', 'AI Hub', `Starting model ${modelName}...`);
    // Model start logic would go here
  } catch (e) {
    window.notify('error', 'AI Hub', e.message);
  }
};

window.createTrainingPipeline = async () => {
  try {
    window.notify('info', 'AI Hub', 'Opening training pipeline creator...');
    // Training pipeline creation would go here
  } catch (e) {
    window.notify('error', 'AI Hub', e.message);
  }
};

// Add AI Hub specific CSS
if (!document.getElementById('ai-hub-styles')) {
  const style = document.createElement('style');
  style.id = 'ai-hub-styles';
  style.textContent = `
    .ai-hub-status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 12px 16px;
    }
    
    .ai-hub-overview {
      display: flex;
      gap: 24px;
    }
    
    .ai-metric {
      display: flex;
      align-items: center;
      gap: 8px;
      font: 400 11px var(--font-mono);
    }
    
    .ai-metric i {
      color: var(--omega-cyan);
      width: 14px;
    }
    
    .ai-hub-actions {
      display: flex;
      gap: 8px;
    }
    
    .ai-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 10px var(--font-mono);
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .ai-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .ai-btn:hover {
      border-color: var(--omega-cyan);
    }
    
    .ai-hub-tabs {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--omega-gray-1);
    }
    
    .ai-tab {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-bottom: none;
      color: var(--omega-light-1);
      padding: 8px 16px;
      cursor: pointer;
      transition: all 0.15s ease;
      font: 400 11px var(--font-mono);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .ai-tab:hover {
      background: var(--omega-dark-2);
      color: var(--omega-white);
    }
    
    .ai-tab.active {
      background: var(--omega-dark-1);
      color: var(--omega-cyan);
      border-color: var(--omega-cyan);
    }
    
    .ai-hub-content {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    
    .ai-view {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: var(--omega-dark-2);
      padding: 16px;
      opacity: 0;
      transform: translateX(20px);
      transition: all 0.2s ease;
      pointer-events: none;
      overflow: auto;
    }
    
    .ai-view.active {
      opacity: 1;
      transform: translateX(0);
      pointer-events: all;
    }
    
    .model-card {
      background: var(--omega-dark-3);
      border: 1px solid var(--omega-gray-1);
      border-radius: 4px;
      padding: 16px;
      cursor: pointer;
      transition: all 0.15s ease;
      margin-bottom: 12px;
    }
    
    .model-card:hover {
      border-color: var(--omega-cyan);
      transform: translateY(-1px);
    }
    
    .model-card.selected {
      border-color: var(--omega-cyan);
      background: var(--omega-dark-2);
    }
    
    .model-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
    }
    
    .model-name {
      font: 600 14px var(--font-mono);
      color: var(--omega-white);
    }
    
    .model-type {
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .model-status {
      padding: 2px 6px;
      border-radius: 2px;
      font: 600 8px var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .model-status.running {
      background: var(--omega-green);
      color: var(--omega-black);
    }
    
    .model-status.stopped {
      background: var(--omega-gray-1);
      color: var(--omega-white);
    }
    
    .model-status.training {
      background: var(--omega-yellow);
      color: var(--omega-black);
    }
    
    .model-metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 12px;
    }
    
    .model-metric {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    
    .metric-label {
      font: 400 9px var(--font-mono);
      color: var(--omega-light-1);
      text-transform: uppercase;
    }
    
    .metric-value {
      font: 600 11px var(--font-mono);
      color: var(--omega-white);
    }
    
    .model-actions {
      display: flex;
      gap: 8px;
    }
    
    .model-action-btn {
      background: var(--omega-dark-4);
      border: 1px solid var(--omega-gray-1);
      color: var(--omega-white);
      padding: 6px 12px;
      border-radius: 3px;
      cursor: pointer;
      font: 400 9px var(--font-mono);
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .model-action-btn.primary {
      background: var(--omega-cyan);
      color: var(--omega-black);
    }
    
    .model-action-btn.danger {
      background: var(--omega-red);
      color: var(--omega-white);
    }
    
    .model-action-btn:hover {
      transform: translateY(-1px);
    }
  `;
  document.head.appendChild(style);
}
