// Component Loader System
class ComponentLoader {
    constructor() {
        this.loadedComponents = new Set();
        this.componentInstances = {};
        this.activeTab = 'dashboard';
    }

    async loadComponent(componentName) {
        if (this.loadedComponents.has(componentName)) {
            return this.componentInstances[componentName];
        }

        try {
            // Load the component script
            await this.loadScript(`components/${componentName}.js`);
            
            // Wait for component to be available
            await this.waitForComponent(componentName);
            
            this.loadedComponents.add(componentName);
            console.log(`Component ${componentName} loaded successfully`);
            
            return this.componentInstances[componentName];
        } catch (error) {
            console.error(`Failed to load component ${componentName}:`, error);
            return null;
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            // Check if script is already loaded
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    waitForComponent(componentName) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Component ${componentName} timeout`));
            }, 5000);

            const checkComponent = () => {
                const instanceName = `${componentName}Component`;
                if (window[instanceName]) {
                    this.componentInstances[componentName] = window[instanceName];
                    clearTimeout(timeout);
                    resolve();
                } else {
                    setTimeout(checkComponent, 100);
                }
            };

            checkComponent();
        });
    }

    async showTab(tabName) {
        try {
            // Show loading state
            this.showLoadingState(tabName);
            
            // Load component if needed
            await this.loadComponent(tabName);
            
            // Get component instance
            const component = this.componentInstances[tabName];
            if (!component) {
                throw new Error(`Component ${tabName} not found`);
            }

            // Render component
            const tabContent = document.getElementById('tab-content');
            tabContent.innerHTML = component.render();
            
            // Update active tab
            this.updateActiveTab(tabName);
            
            console.log(`Tab ${tabName} loaded and rendered`);
            
        } catch (error) {
            console.error(`Failed to show tab ${tabName}:`, error);
            this.showErrorState(tabName, error.message);
        }
    }

    showLoadingState(tabName) {
        const tabContent = document.getElementById('tab-content');
        tabContent.innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <h3>Loading ${tabName.charAt(0).toUpperCase() + tabName.slice(1)}...</h3>
                <p>Please wait while we prepare the ${tabName} interface.</p>
            </div>
        `;
    }

    showErrorState(tabName, errorMessage) {
        const tabContent = document.getElementById('tab-content');
        tabContent.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Failed to Load ${tabName.charAt(0).toUpperCase() + tabName.slice(1)}</h3>
                <p>Error: ${errorMessage}</p>
                <button class="btn-primary" onclick="componentLoader.showTab('${tabName}')">
                    <i class="fas fa-redo"></i> Retry
                </button>
            </div>
        `;
    }

    updateActiveTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-item').forEach(tab => {
            tab.classList.remove('active');
        });
        
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
        
        this.activeTab = tabName;
        
        // Update URL hash
        window.location.hash = tabName;
    }

    // Preload critical components
    async preloadComponents() {
        const criticalComponents = ['dashboard', 'nodes', 'resources'];
        
        for (const component of criticalComponents) {
            try {
                await this.loadComponent(component);
                console.log(`Preloaded component: ${component}`);
            } catch (error) {
                console.warn(`Failed to preload component ${component}:`, error);
            }
        }
    }

    // Initialize the component loader
    async init() {
        // Handle browser back/forward navigation
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.substring(1);
            if (hash) {
                this.showTab(hash);
            }
        });

        // Load initial tab from URL hash or default to dashboard
        const initialTab = window.location.hash.substring(1) || 'dashboard';
        await this.showTab(initialTab);

        // Preload other components in background
        setTimeout(() => {
            this.preloadComponents();
        }, 1000);
    }
}

// Global instance
window.componentLoader = new ComponentLoader();

// Tab switching function for global use
window.showTab = function(tabName) {
    componentLoader.showTab(tabName);
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    componentLoader.init();
});
