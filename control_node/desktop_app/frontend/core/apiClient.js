// Simplified API client for testing
export class ApiClient {
  constructor(base='http://127.0.0.1:8443') { this.base=base; }
  
  async getDashboard() { 
    // Return mock data for now to test the UI
    return {
      cluster: {
        name: "Local-Cluster",
        status: "OPERATIONAL", 
        nodes: 3,
        active_sessions: 5,
        total_memory: "32GB",
        used_memory: "18GB",
        cpu_usage: 45,
        network_bandwidth: "1.2GB/s"
      },
      performance: {
        cpu_usage: 45,
        memory_usage: 56,
        network_usage: 78,
        disk_usage: 34
      },
      alerts: [
        { type: "info", message: "System running normally", timestamp: Date.now() },
        { type: "warning", message: "High memory usage detected", timestamp: Date.now() - 60000 }
      ]
    };
  }
  
  async getNodes() { return []; }
  async getResources() { return []; }
  async getSessions() { return []; }
  async getProcesses() { return []; }
}
