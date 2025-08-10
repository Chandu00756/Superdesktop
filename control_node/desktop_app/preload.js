/**
 * Omega SuperDesktop v2.0 - Preload Script
 * Essential initialization and polyfills loaded before main application
 */

// Prevent default drag and drop
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

// Global configuration
window.OMEGA_CONFIG = {
    VERSION: '2.0',
    API_BASE_URL: window.location.origin,
    FALLBACK_API_URL: 'http://localhost:8443',
    DEBUG: true,
    MODULES_PATH: './advanced/modules/',
    REFRESH_INTERVAL: 5000,
    THEME: 'dark'
};

// Essential polyfills and feature detection
(function() {
    'use strict';
    
    console.log('🔧 Omega SuperDesktop v2.0 - Preload initialization');
    
    // Check for required features
    const requiredFeatures = {
        fetch: typeof fetch !== 'undefined',
        Promise: typeof Promise !== 'undefined',
        localStorage: typeof localStorage !== 'undefined',
        WebGL: checkWebGLSupport(),
        ES6: checkES6Support()
    };
    
    // Log feature support
    Object.entries(requiredFeatures).forEach(([feature, supported]) => {
        console.log(`  ${supported ? '✅' : '❌'} ${feature}: ${supported ? 'Supported' : 'Not supported'}`);
    });
    
    // WebGL support check
    function checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && 
                     (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }
    
    // ES6 support check
    function checkES6Support() {
        try {
            eval('class TestClass {}');
            eval('const testArrow = () => {};');
            return true;
        } catch (e) {
            return false;
        }
    }
    
    // Polyfill for requestAnimationFrame
    if (!window.requestAnimationFrame) {
        window.requestAnimationFrame = function(callback) {
            return setTimeout(callback, 1000 / 60);
        };
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
    }
    
    // Polyfill for performance.now
    if (!window.performance || !window.performance.now) {
        window.performance = window.performance || {};
        window.performance.now = function() {
            return Date.now();
        };
    }
    
    // Setup global error handling
    window.addEventListener('error', function(event) {
        console.error('Global error caught in preload:', {
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            error: event.error
        });
    });
    
    // Setup unhandled promise rejection handling
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection caught in preload:', event.reason);
    });
    
})();

// CSS loading utilities
function loadCSS(href) {
    return new Promise((resolve, reject) => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = href;
        link.onload = resolve;
        link.onerror = reject;
        document.head.appendChild(link);
    });
}

// JavaScript loading utilities
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Theme initialization
function initializeTheme() {
    const savedTheme = localStorage.getItem('omega-theme') || window.OMEGA_CONFIG.THEME;
    document.body.setAttribute('data-theme', savedTheme);
    document.documentElement.style.setProperty('--theme', savedTheme);
}

// Initialize theme as early as possible
initializeTheme();

// Utility functions for modules
window.omegaUtils = {
    loadCSS,
    loadScript,
    
    // Debounce utility
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Throttle utility
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Format bytes
    formatBytes: function(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    // Generate unique ID
    generateId: function() {
        return 'omega-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    },
    
    // Safe JSON parse
    safeJSONParse: function(str, fallback = null) {
        try {
            return JSON.parse(str);
        } catch (e) {
            return fallback;
        }
    },
    
    // Create element with attributes
    createElement: function(tag, attributes = {}, innerHTML = '') {
        const element = document.createElement(tag);
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'dataset') {
                Object.entries(value).forEach(([dataKey, dataValue]) => {
                    element.dataset[dataKey] = dataValue;
                });
            } else {
                element.setAttribute(key, value);
            }
        });
        if (innerHTML) {
            element.innerHTML = innerHTML;
        }
        return element;
    }
};

// Module loading system
window.omegaModuleLoader = {
    loaded: new Set(),
    loading: new Map(),
    
    async loadModule(name, path) {
        if (this.loaded.has(name)) {
            return Promise.resolve();
        }
        
        if (this.loading.has(name)) {
            return this.loading.get(name);
        }
        
        const loadPromise = loadScript(path).then(() => {
            this.loaded.add(name);
            this.loading.delete(name);
            console.log(`📦 Module loaded: ${name}`);
        }).catch(error => {
            this.loading.delete(name);
            console.error(`❌ Failed to load module ${name}:`, error);
            throw error;
        });
        
        this.loading.set(name, loadPromise);
        return loadPromise;
    },
    
    async loadMultiple(modules) {
        const promises = modules.map(({ name, path }) => this.loadModule(name, path));
        return Promise.all(promises);
    }
};

// Initialize console styling
const consoleCss = 'color: #4ade80; font-weight: bold; font-size: 14px;';
console.log('%c🚀 Omega SuperDesktop v2.0 - Preload Complete', consoleCss);

// Early DOM ready state check
if (document.readyState === 'loading') {
    console.log('📄 DOM is loading...');
} else {
    console.log('📄 DOM already loaded');
}

// Preload complete flag
window.OMEGA_PRELOAD_COMPLETE = true;

// Export configuration for other scripts
window.OMEGA_PRELOAD_VERSION = '2.0';
window.OMEGA_PRELOAD_TIMESTAMP = Date.now();
