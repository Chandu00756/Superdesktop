/**
 * NotificationManager - Toast notification system
 */
class NotificationManager {
    constructor() {
        this.container = null;
        this.notifications = new Map();
        this.maxVisible = 5;
        this.defaultTimeout = 5000;
        this.init();
    }

    init() {
        this.createContainer();
        this.bindEvents();
    }

    createContainer() {
        this.container = document.createElement('div');
        this.container.className = 'notification-container';
        this.container.setAttribute('aria-live', 'polite');
        this.container.setAttribute('aria-label', 'Notifications');
        document.body.appendChild(this.container);
    }

    /**
     * Show notification
     */
    show(type, title, message, options = {}) {
        const notification = this.createNotification(type, title, message, options);
        const id = notification.id;
        
        this.notifications.set(id, {
            element: notification,
            timeout: options.timeout !== 0 ? (options.timeout || this.defaultTimeout) : 0,
            type,
            title,
            message
        });

        this.container.appendChild(notification);
        
        // Trigger animation
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });

        // Auto-dismiss
        if (options.timeout !== 0) {
            const timeout = options.timeout || this.defaultTimeout;
            setTimeout(() => {
                this.dismiss(id);
            }, timeout);
        }

        // Manage visible count
        this.enforceMaxVisible();

        // Emit event
        if (window.EventBus) {
            window.EventBus.emit('notificationShown', { id, type, title, message });
        }

        return id;
    }

    /**
     * Convenience methods
     */
    info(title, message, options = {}) {
        return this.show('info', title, message, options);
    }

    success(title, message, options = {}) {
        return this.show('success', title, message, options);
    }

    warning(title, message, options = {}) {
        return this.show('warning', title, message, options);
    }

    error(title, message, options = {}) {
        return this.show('error', title, message, { ...options, timeout: 0 });
    }

    /**
     * Dismiss notification
     */
    dismiss(id) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        const element = notification.element;
        element.classList.remove('show');
        element.classList.add('hiding');

        setTimeout(() => {
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
            this.notifications.delete(id);
            
            if (window.EventBus) {
                window.EventBus.emit('notificationDismissed', { id });
            }
        }, 300);
    }

    /**
     * Show progress notification
     */
    showProgress(title, message, options = {}) {
        const notification = this.createNotification('info', title, message, {
            ...options,
            progress: true,
            timeout: 0
        });
        
        const id = notification.id;
        this.notifications.set(id, {
            element: notification,
            timeout: 0,
            type: 'progress',
            title,
            message,
            progress: 0
        });

        this.container.appendChild(notification);
        
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });

        this.enforceMaxVisible();

        return {
            id,
            updateProgress: (percent, newMessage) => {
                this.updateProgress(id, percent, newMessage);
            },
            complete: (successMessage) => {
                this.completeProgress(id, successMessage);
            },
            fail: (errorMessage) => {
                this.failProgress(id, errorMessage);
            }
        };
    }

    /**
     * Update progress notification
     */
    updateProgress(id, percent, message) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        notification.progress = Math.max(0, Math.min(100, percent));
        
        const progressBar = notification.element.querySelector('.notification-progress-bar');
        if (progressBar) {
            progressBar.style.transform = `translateX(-${100 - notification.progress}%)`;
        }

        if (message) {
            const messageEl = notification.element.querySelector('.notification-message');
            if (messageEl) {
                messageEl.textContent = message;
            }
        }
    }

    /**
     * Complete progress notification
     */
    completeProgress(id, successMessage) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        this.updateProgress(id, 100, successMessage || 'Completed successfully');
        
        notification.element.classList.remove('info');
        notification.element.classList.add('success');
        
        setTimeout(() => {
            this.dismiss(id);
        }, 2000);
    }

    /**
     * Fail progress notification
     */
    failProgress(id, errorMessage) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        notification.element.classList.remove('info');
        notification.element.classList.add('error');
        
        const messageEl = notification.element.querySelector('.notification-message');
        if (messageEl) {
            messageEl.textContent = errorMessage || 'Operation failed';
        }

        const progressEl = notification.element.querySelector('.notification-progress');
        if (progressEl) {
            progressEl.style.display = 'none';
        }
    }

    /**
     * Clear all notifications
     */
    clear() {
        this.notifications.forEach((_, id) => {
            this.dismiss(id);
        });
    }

    /**
     * Get active notifications
     */
    getActive() {
        return Array.from(this.notifications.values()).map(n => ({
            id: n.element.id,
            type: n.type,
            title: n.title,
            message: n.message
        }));
    }

    // Private methods
    createNotification(type, title, message, options = {}) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        const hasProgress = options.progress;
        
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">${this.escapeHtml(title)}</span>
                <button class="notification-close" aria-label="Close notification">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="notification-message">${this.escapeHtml(message)}</div>
            ${hasProgress ? `
                <div class="notification-progress">
                    <div class="notification-progress-bar"></div>
                </div>
            ` : ''}
        `;

        // Bind close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.dismiss(notification.id);
        });

        return notification;
    }

    enforceMaxVisible() {
        const visibleNotifications = Array.from(this.notifications.values())
            .filter(n => n.element.classList.contains('show'));

        if (visibleNotifications.length > this.maxVisible) {
            const oldest = visibleNotifications[0];
            this.dismiss(oldest.element.id);
        }
    }

    bindEvents() {
        // Listen for window events
        if (window.EventBus) {
            window.EventBus.on('sessionCreated', (data) => {
                this.success('Session Created', `Session "${data.name}" has been created successfully`);
            });

            window.EventBus.on('sessionError', (data) => {
                this.error('Session Error', data.message || 'An error occurred with the session');
            });

            window.EventBus.on('pluginInstalled', (data) => {
                this.success('Plugin Installed', `${data.name} has been installed successfully`);
            });

            window.EventBus.on('networkIssue', (data) => {
                this.warning('Network Issue', data.message || 'Network connectivity issue detected');
            });
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    bindEvents() {
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.notifications.size > 0) {
                const latest = Array.from(this.notifications.keys()).pop();
                this.dismiss(latest);
            }
        });
    }
}

// Global instance
window.NotificationManager = new NotificationManager();

export default NotificationManager;
