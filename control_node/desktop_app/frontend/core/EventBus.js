/**
 * EventBus - Lightweight event system for component communication
 */
class EventBus {
    constructor() {
        this.events = new Map();
        this.wildcardListeners = [];
    }

    /**
     * Subscribe to an event
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     * @param {Object} options - Options {once: boolean}
     * @return {Function} Unsubscribe function
     */
    on(event, callback, options = {}) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        
        const listener = { callback, once: options.once || false, id: Date.now() + Math.random() };
        this.events.get(event).push(listener);
        
        return () => this.off(event, listener.id);
    }

    /**
     * Subscribe to an event once
     */
    once(event, callback) {
        return this.on(event, callback, { once: true });
    }

    /**
     * Unsubscribe from an event
     */
    off(event, listenerId) {
        const listeners = this.events.get(event);
        if (listeners) {
            const index = listeners.findIndex(l => l.id === listenerId);
            if (index >= 0) {
                listeners.splice(index, 1);
            }
        }
    }

    /**
     * Emit an event
     */
    emit(event, data) {
        // Emit to specific listeners
        const listeners = this.events.get(event);
        if (listeners) {
            const toRemove = [];
            listeners.forEach((listener, index) => {
                try {
                    listener.callback(data, event);
                    if (listener.once) {
                        toRemove.push(index);
                    }
                } catch (error) {
                    console.error(`EventBus error in listener for ${event}:`, error);
                }
            });
            
            // Remove one-time listeners
            toRemove.reverse().forEach(index => listeners.splice(index, 1));
        }

        // Emit to wildcard listeners
        this.wildcardListeners.forEach(listener => {
            try {
                listener.callback(data, event);
            } catch (error) {
                console.error(`EventBus error in wildcard listener:`, error);
            }
        });
    }

    /**
     * Subscribe to all events (wildcard)
     */
    onAny(callback) {
        const listener = { callback, id: Date.now() + Math.random() };
        this.wildcardListeners.push(listener);
        
        return () => {
            const index = this.wildcardListeners.findIndex(l => l.id === listener.id);
            if (index >= 0) {
                this.wildcardListeners.splice(index, 1);
            }
        };
    }

    /**
     * Clear all listeners
     */
    clear() {
        this.events.clear();
        this.wildcardListeners = [];
    }

    /**
     * Get debug info
     */
    getDebugInfo() {
        return {
            events: Array.from(this.events.keys()),
            listenerCounts: Array.from(this.events.entries()).map(([event, listeners]) => 
                ({ event, count: listeners.length })
            ),
            wildcardCount: this.wildcardListeners.length
        };
    }
}

// Global instance
window.EventBus = new EventBus();

export default EventBus;
