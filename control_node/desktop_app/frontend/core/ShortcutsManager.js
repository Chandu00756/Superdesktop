/**
 * ShortcutsManager - Centralized keyboard shortcut management
 */
class ShortcutsManager {
    constructor() {
        this.shortcuts = new Map();
        this.globalShortcuts = new Map();
        this.contexts = new Map();
        this.activeContext = 'global';
        this.enabled = true;
        this.settings = {
            enabled: true,
            customShortcuts: new Map()
        };
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.registerDefaults();
        this.loadSettings();
    }

    /**
     * Register a keyboard shortcut
     */
    register(key, callback, options = {}) {
        const {
            context = 'global',
            description = '',
            preventDefault = true,
            allowInInputs = false
        } = options;

        const shortcut = {
            key: this.normalizeKey(key),
            callback,
            context,
            description,
            preventDefault,
            allowInInputs,
            id: Date.now() + Math.random()
        };

        if (context === 'global') {
            this.globalShortcuts.set(shortcut.key, shortcut);
        } else {
            if (!this.contexts.has(context)) {
                this.contexts.set(context, new Map());
            }
            this.contexts.get(context).set(shortcut.key, shortcut);
        }

        return () => this.unregister(shortcut.id);
    }

    /**
     * Unregister a shortcut
     */
    unregister(id) {
        // Remove from global shortcuts
        for (const [key, shortcut] of this.globalShortcuts) {
            if (shortcut.id === id) {
                this.globalShortcuts.delete(key);
                return true;
            }
        }

        // Remove from context shortcuts
        for (const [contextName, contextShortcuts] of this.contexts) {
            for (const [key, shortcut] of contextShortcuts) {
                if (shortcut.id === id) {
                    contextShortcuts.delete(key);
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Set active context
     */
    setContext(context) {
        this.activeContext = context;
        
        if (window.EventBus) {
            window.EventBus.emit('shortcutContextChanged', { context });
        }
    }

    /**
     * Enable/disable shortcuts
     */
    setEnabled(enabled) {
        this.enabled = enabled;
        this.settings.enabled = enabled;
        this.saveSettings();
    }

    /**
     * Check if shortcuts are enabled
     */
    isEnabled() {
        return this.enabled && this.settings.enabled;
    }

    /**
     * Get all registered shortcuts
     */
    getShortcuts(context = null) {
        const result = {};

        if (!context || context === 'global') {
            result.global = Array.from(this.globalShortcuts.values()).map(s => ({
                key: s.key,
                description: s.description,
                context: s.context
            }));
        }

        if (context) {
            const contextShortcuts = this.contexts.get(context);
            if (contextShortcuts) {
                result[context] = Array.from(contextShortcuts.values()).map(s => ({
                    key: s.key,
                    description: s.description,
                    context: s.context
                }));
            }
        } else {
            for (const [contextName, contextShortcuts] of this.contexts) {
                result[contextName] = Array.from(contextShortcuts.values()).map(s => ({
                    key: s.key,
                    description: s.description,
                    context: s.context
                }));
            }
        }

        return result;
    }

    /**
     * Update shortcut key
     */
    updateShortcut(oldKey, newKey, context = 'global') {
        const normalizedOld = this.normalizeKey(oldKey);
        const normalizedNew = this.normalizeKey(newKey);

        if (context === 'global') {
            const shortcut = this.globalShortcuts.get(normalizedOld);
            if (shortcut) {
                this.globalShortcuts.delete(normalizedOld);
                shortcut.key = normalizedNew;
                this.globalShortcuts.set(normalizedNew, shortcut);
                return true;
            }
        } else {
            const contextShortcuts = this.contexts.get(context);
            if (contextShortcuts) {
                const shortcut = contextShortcuts.get(normalizedOld);
                if (shortcut) {
                    contextShortcuts.delete(normalizedOld);
                    shortcut.key = normalizedNew;
                    contextShortcuts.set(normalizedNew, shortcut);
                    return true;
                }
            }
        }

        return false;
    }

    // Private methods
    bindEvents() {
        document.addEventListener('keydown', (e) => {
            if (!this.isEnabled()) return;

            const key = this.getKeyFromEvent(e);
            if (!key) return;

            // Check if we should ignore this event
            if (this.shouldIgnoreEvent(e)) return;

            // Try context shortcuts first
            const contextShortcuts = this.contexts.get(this.activeContext);
            if (contextShortcuts) {
                const shortcut = contextShortcuts.get(key);
                if (shortcut && this.canExecuteShortcut(shortcut, e)) {
                    this.executeShortcut(shortcut, e);
                    return;
                }
            }

            // Try global shortcuts
            const globalShortcut = this.globalShortcuts.get(key);
            if (globalShortcut && this.canExecuteShortcut(globalShortcut, e)) {
                this.executeShortcut(globalShortcut, e);
            }
        });
    }

    shouldIgnoreEvent(e) {
        const target = e.target;
        const tagName = target.tagName.toLowerCase();
        
        // Ignore when typing in inputs, textareas, or contenteditable elements
        if (tagName === 'input' || tagName === 'textarea' || target.contentEditable === 'true') {
            return true;
        }

        // Ignore when modals are open and shortcut is not modal-specific
        const hasModal = document.querySelector('.modal-overlay.active');
        if (hasModal && this.activeContext !== 'modal') {
            return true;
        }

        return false;
    }

    canExecuteShortcut(shortcut, e) {
        if (!shortcut.allowInInputs && this.isInInput(e.target)) {
            return false;
        }
        return true;
    }

    isInInput(element) {
        const tagName = element.tagName.toLowerCase();
        return tagName === 'input' || tagName === 'textarea' || element.contentEditable === 'true';
    }

    executeShortcut(shortcut, e) {
        try {
            if (shortcut.preventDefault) {
                e.preventDefault();
                e.stopPropagation();
            }

            shortcut.callback(e);

            if (window.EventBus) {
                window.EventBus.emit('shortcutExecuted', {
                    key: shortcut.key,
                    context: shortcut.context,
                    description: shortcut.description
                });
            }
        } catch (error) {
            console.error('Shortcut execution error:', error);
        }
    }

    getKeyFromEvent(e) {
        const parts = [];

        if (e.ctrlKey) parts.push('ctrl');
        if (e.altKey) parts.push('alt');
        if (e.shiftKey) parts.push('shift');
        if (e.metaKey) parts.push('meta');

        const key = e.key.toLowerCase();
        
        // Don't include modifier keys as the main key
        if (['control', 'alt', 'shift', 'meta'].includes(key)) {
            return null;
        }

        parts.push(key);
        return parts.join('+');
    }

    normalizeKey(key) {
        return key.toLowerCase().replace(/\s+/g, '');
    }

    registerDefaults() {
        // Global shortcuts
        this.register('ctrl+shift+p', () => {
            window.omegaDesktop.openCommandPalette();
        }, { description: 'Open command palette' });

        this.register('ctrl+shift+s', () => {
            window.omegaDesktop.openSettings();
        }, { description: 'Open settings' });

        this.register('ctrl+shift+d', () => {
            window.omegaDesktop.switchTab('dashboard');
        }, { description: 'Switch to dashboard' });

        this.register('ctrl+shift+v', () => {
            window.omegaDesktop.switchTab('virtual-desktop');
        }, { description: 'Switch to virtual desktop' });

        this.register('ctrl+shift+n', () => {
            window.omegaDesktop.createNewSession();
        }, { description: 'Create new session' });

        this.register('ctrl+shift+w', () => {
            window.omegaDesktop.closeActiveSession();
        }, { description: 'Close active session' });

        this.register('f11', () => {
            window.omegaDesktop.toggleFullscreen();
        }, { description: 'Toggle fullscreen' });

        this.register('escape', () => {
            window.omegaDesktop.handleEscape();
        }, { description: 'Close modals/panels' });

        // Tab navigation
        this.register('ctrl+tab', () => {
            window.omegaDesktop.nextTab();
        }, { description: 'Next tab' });

        this.register('ctrl+shift+tab', () => {
            window.omegaDesktop.prevTab();
        }, { description: 'Previous tab' });

        // Session shortcuts
        this.register('ctrl+1', () => {
            window.omegaDesktop.switchToSession(1);
        }, { context: 'sessions', description: 'Switch to session 1' });

        this.register('ctrl+2', () => {
            window.omegaDesktop.switchToSession(2);
        }, { context: 'sessions', description: 'Switch to session 2' });

        this.register('ctrl+3', () => {
            window.omegaDesktop.switchToSession(3);
        }, { context: 'sessions', description: 'Switch to session 3' });

        // Modal shortcuts
        this.register('escape', () => {
            window.omegaDesktop.closeModal();
        }, { context: 'modal', description: 'Close modal' });

        this.register('enter', () => {
            window.omegaDesktop.confirmModal();
        }, { context: 'modal', description: 'Confirm modal action' });
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('omega-shortcuts-settings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.settings = { ...this.settings, ...settings };
                this.enabled = this.settings.enabled;
            }
        } catch (error) {
            console.warn('Failed to load shortcut settings:', error);
        }
    }

    saveSettings() {
        try {
            localStorage.setItem('omega-shortcuts-settings', JSON.stringify(this.settings));
        } catch (error) {
            console.warn('Failed to save shortcut settings:', error);
        }
    }

    /**
     * Get help text for all shortcuts
     */
    getHelpText() {
        const shortcuts = this.getShortcuts();
        const sections = [];

        for (const [context, contextShortcuts] of Object.entries(shortcuts)) {
            sections.push({
                context,
                shortcuts: contextShortcuts.map(s => ({
                    key: s.key.replace(/\+/g, ' + ').toUpperCase(),
                    description: s.description
                }))
            });
        }

        return sections;
    }
}

// Global instance
window.ShortcutsManager = new ShortcutsManager();

export default ShortcutsManager;
