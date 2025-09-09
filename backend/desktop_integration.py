"""
Omega Super Desktop Console v2.0 - Desktop App Integration Engine
Enterprise-grade desktop application integration with native OS features and seamless UX
"""

import asyncio
import json
import logging
import time
import uuid
import subprocess
import platform
import os
import threading
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class AppType(Enum):
    NATIVE = "native"
    WEB = "web"
    ELECTRON = "electron"
    PWA = "pwa"
    BRIDGE = "bridge"
    VIRTUAL = "virtual"

class IntegrationType(Enum):
    SYSTEM_TRAY = "system_tray"
    DOCK = "dock"
    TASKBAR = "taskbar"
    MENU_BAR = "menu_bar"
    CONTEXT_MENU = "context_menu"
    FILE_ASSOCIATION = "file_association"
    PROTOCOL_HANDLER = "protocol_handler"
    NOTIFICATION = "notification"
    GLOBAL_HOTKEY = "global_hotkey"

class LaunchMode(Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    MINIMIZED = "minimized"
    FULLSCREEN = "fullscreen"
    KIOSK = "kiosk"

@dataclass
class DesktopApp:
    """Desktop application definition"""
    app_id: str
    name: str
    description: str
    app_type: AppType
    executable_path: str
    working_directory: str
    command_line_args: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    icon_path: Optional[str] = None
    launch_mode: LaunchMode = LaunchMode.FOREGROUND
    auto_start: bool = False
    single_instance: bool = True
    integrations: Set[IntegrationType] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class AppInstance:
    """Running application instance"""
    instance_id: str
    app_id: str
    process_id: Optional[int]
    window_handle: Optional[int]
    started_at: float
    state: str = "running"  # running, paused, minimized, focused
    memory_usage: int = 0
    cpu_usage: float = 0.0
    window_title: Optional[str] = None
    window_geometry: Optional[Dict[str, int]] = None
    last_activity: float = field(default_factory=time.time)

@dataclass
class SystemIntegration:
    """System integration configuration"""
    integration_id: str
    app_id: str
    integration_type: IntegrationType
    configuration: Dict[str, Any]
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

class WindowManager:
    """Cross-platform window management"""
    
    def __init__(self):
        self.platform = platform.system()
        self.windows: Dict[str, Dict[str, Any]] = {}
        
    async def create_window(self, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create a new application window"""
        try:
            window_id = str(uuid.uuid4())
            
            if self.platform == "Darwin":  # macOS
                return await self._create_macos_window(window_id, app_id, config)
            elif self.platform == "Windows":
                return await self._create_windows_window(window_id, app_id, config)
            elif self.platform == "Linux":
                return await self._create_linux_window(window_id, app_id, config)
            else:
                logger.warning(f"Unsupported platform: {self.platform}")
                return None
                
        except Exception as e:
            logger.error(f"Window creation failed: {e}")
            return None
            
    async def _create_macos_window(self, window_id: str, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create macOS window using Cocoa"""
        try:
            # In production, this would use PyObjC to create native Cocoa windows
            # For demo, we'll simulate window creation
            
            window_info = {
                'window_id': window_id,
                'app_id': app_id,
                'title': config.get('title', 'Omega Application'),
                'width': config.get('width', 800),
                'height': config.get('height', 600),
                'x': config.get('x', 100),
                'y': config.get('y', 100),
                'resizable': config.get('resizable', True),
                'minimizable': config.get('minimizable', True),
                'closable': config.get('closable', True),
                'level': config.get('level', 'normal'),  # normal, floating, modal
                'style': config.get('style', 'titled')   # titled, borderless, utility
            }
            
            self.windows[window_id] = window_info
            logger.info(f"Created macOS window {window_id} for app {app_id}")
            return window_id
            
        except Exception as e:
            logger.error(f"macOS window creation failed: {e}")
            return None
            
    async def _create_windows_window(self, window_id: str, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create Windows window using Win32 API"""
        try:
            # In production, this would use win32gui and win32api
            # For demo, we'll simulate window creation
            
            window_info = {
                'window_id': window_id,
                'app_id': app_id,
                'title': config.get('title', 'Omega Application'),
                'width': config.get('width', 800),
                'height': config.get('height', 600),
                'x': config.get('x', 100),
                'y': config.get('y', 100),
                'style': config.get('style', 'overlapped'),  # overlapped, popup, child
                'extended_style': config.get('extended_style', 'default')
            }
            
            self.windows[window_id] = window_info
            logger.info(f"Created Windows window {window_id} for app {app_id}")
            return window_id
            
        except Exception as e:
            logger.error(f"Windows window creation failed: {e}")
            return None
            
    async def _create_linux_window(self, window_id: str, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create Linux window using X11/Wayland"""
        try:
            # In production, this would use Xlib or wayland bindings
            # For demo, we'll simulate window creation
            
            window_info = {
                'window_id': window_id,
                'app_id': app_id,
                'title': config.get('title', 'Omega Application'),
                'width': config.get('width', 800),
                'height': config.get('height', 600),
                'x': config.get('x', 100),
                'y': config.get('y', 100),
                'class': config.get('class', 'OmegaApp'),
                'type': config.get('type', 'normal')  # normal, dialog, splash, dock
            }
            
            self.windows[window_id] = window_info
            logger.info(f"Created Linux window {window_id} for app {app_id}")
            return window_id
            
        except Exception as e:
            logger.error(f"Linux window creation failed: {e}")
            return None
            
    async def get_window_info(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a window"""
        return self.windows.get(window_id)
        
    async def update_window(self, window_id: str, updates: Dict[str, Any]) -> bool:
        """Update window properties"""
        try:
            if window_id not in self.windows:
                return False
                
            self.windows[window_id].update(updates)
            logger.debug(f"Updated window {window_id}")
            return True
            
        except Exception as e:
            logger.error(f"Window update failed: {e}")
            return False
            
    async def close_window(self, window_id: str) -> bool:
        """Close a window"""
        try:
            if window_id in self.windows:
                del self.windows[window_id]
                logger.info(f"Closed window {window_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Window close failed: {e}")
            return False

class SystemTrayManager:
    """Cross-platform system tray integration"""
    
    def __init__(self):
        self.platform = platform.system()
        self.tray_items: Dict[str, Dict[str, Any]] = {}
        
    async def create_tray_icon(self, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create system tray icon"""
        try:
            tray_id = str(uuid.uuid4())
            
            tray_config = {
                'tray_id': tray_id,
                'app_id': app_id,
                'icon_path': config.get('icon_path'),
                'tooltip': config.get('tooltip', 'Omega Application'),
                'menu_items': config.get('menu_items', []),
                'click_action': config.get('click_action', 'show_window'),
                'notification_enabled': config.get('notification_enabled', True)
            }
            
            # Platform-specific implementation
            if self.platform == "Darwin":
                success = await self._create_macos_tray_icon(tray_config)
            elif self.platform == "Windows":
                success = await self._create_windows_tray_icon(tray_config)
            elif self.platform == "Linux":
                success = await self._create_linux_tray_icon(tray_config)
            else:
                success = False
                
            if success:
                self.tray_items[tray_id] = tray_config
                return tray_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Tray icon creation failed: {e}")
            return None
            
    async def _create_macos_tray_icon(self, config: Dict[str, Any]) -> bool:
        """Create macOS menu bar icon using NSStatusBar"""
        try:
            # In production, use PyObjC with NSStatusBar
            logger.info(f"Created macOS tray icon for {config['app_id']}")
            return True
        except Exception as e:
            logger.error(f"macOS tray icon creation failed: {e}")
            return False
            
    async def _create_windows_tray_icon(self, config: Dict[str, Any]) -> bool:
        """Create Windows system tray icon using Shell_NotifyIcon"""
        try:
            # In production, use win32gui with Shell_NotifyIcon
            logger.info(f"Created Windows tray icon for {config['app_id']}")
            return True
        except Exception as e:
            logger.error(f"Windows tray icon creation failed: {e}")
            return False
            
    async def _create_linux_tray_icon(self, config: Dict[str, Any]) -> bool:
        """Create Linux system tray icon using StatusNotifierItem"""
        try:
            # In production, use dbus with StatusNotifierItem
            logger.info(f"Created Linux tray icon for {config['app_id']}")
            return True
        except Exception as e:
            logger.error(f"Linux tray icon creation failed: {e}")
            return False
            
    async def update_tray_icon(self, tray_id: str, updates: Dict[str, Any]) -> bool:
        """Update tray icon properties"""
        try:
            if tray_id not in self.tray_items:
                return False
                
            self.tray_items[tray_id].update(updates)
            logger.debug(f"Updated tray icon {tray_id}")
            return True
            
        except Exception as e:
            logger.error(f"Tray icon update failed: {e}")
            return False
            
    async def remove_tray_icon(self, tray_id: str) -> bool:
        """Remove tray icon"""
        try:
            if tray_id in self.tray_items:
                del self.tray_items[tray_id]
                logger.info(f"Removed tray icon {tray_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Tray icon removal failed: {e}")
            return False

class NotificationManager:
    """Cross-platform native notifications"""
    
    def __init__(self):
        self.platform = platform.system()
        self.notification_history: List[Dict[str, Any]] = []
        
    async def send_notification(self, config: Dict[str, Any]) -> bool:
        """Send native notification"""
        try:
            notification_id = str(uuid.uuid4())
            
            notification = {
                'notification_id': notification_id,
                'title': config.get('title', 'Omega Notification'),
                'message': config.get('message', ''),
                'app_id': config.get('app_id'),
                'icon_path': config.get('icon_path'),
                'sound': config.get('sound', True),
                'persistent': config.get('persistent', False),
                'actions': config.get('actions', []),
                'timestamp': time.time()
            }
            
            # Platform-specific implementation
            if self.platform == "Darwin":
                success = await self._send_macos_notification(notification)
            elif self.platform == "Windows":
                success = await self._send_windows_notification(notification)
            elif self.platform == "Linux":
                success = await self._send_linux_notification(notification)
            else:
                success = False
                
            if success:
                self.notification_history.append(notification)
                # Keep only recent notifications
                if len(self.notification_history) > 100:
                    self.notification_history = self.notification_history[-50:]
                    
            return success
            
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
            return False
            
    async def _send_macos_notification(self, notification: Dict[str, Any]) -> bool:
        """Send macOS notification using NSUserNotification"""
        try:
            # In production, use PyObjC with NSUserNotificationCenter
            logger.info(f"Sent macOS notification: {notification['title']}")
            return True
        except Exception as e:
            logger.error(f"macOS notification failed: {e}")
            return False
            
    async def _send_windows_notification(self, notification: Dict[str, Any]) -> bool:
        """Send Windows notification using WinRT Toast"""
        try:
            # In production, use win10toast or plyer
            logger.info(f"Sent Windows notification: {notification['title']}")
            return True
        except Exception as e:
            logger.error(f"Windows notification failed: {e}")
            return False
            
    async def _send_linux_notification(self, notification: Dict[str, Any]) -> bool:
        """Send Linux notification using libnotify"""
        try:
            # In production, use plyer or direct libnotify calls
            logger.info(f"Sent Linux notification: {notification['title']}")
            return True
        except Exception as e:
            logger.error(f"Linux notification failed: {e}")
            return False

class FileAssociationManager:
    """Manage file type associations"""
    
    def __init__(self):
        self.platform = platform.system()
        self.associations: Dict[str, Dict[str, Any]] = {}
        
    async def register_file_association(self, config: Dict[str, Any]) -> bool:
        """Register file type association"""
        try:
            association_id = str(uuid.uuid4())
            
            association = {
                'association_id': association_id,
                'app_id': config['app_id'],
                'file_extensions': config['file_extensions'],
                'mime_types': config.get('mime_types', []),
                'description': config.get('description', ''),
                'icon_path': config.get('icon_path'),
                'executable_path': config['executable_path'],
                'command_template': config.get('command_template', '{executable} "{file}"')
            }
            
            # Platform-specific registration
            if self.platform == "Darwin":
                success = await self._register_macos_association(association)
            elif self.platform == "Windows":
                success = await self._register_windows_association(association)
            elif self.platform == "Linux":
                success = await self._register_linux_association(association)
            else:
                success = False
                
            if success:
                self.associations[association_id] = association
                
            return success
            
        except Exception as e:
            logger.error(f"File association registration failed: {e}")
            return False
            
    async def _register_macos_association(self, association: Dict[str, Any]) -> bool:
        """Register macOS file association using Launch Services"""
        try:
            # In production, use LaunchServices framework
            logger.info(f"Registered macOS file association for {association['file_extensions']}")
            return True
        except Exception as e:
            logger.error(f"macOS file association failed: {e}")
            return False
            
    async def _register_windows_association(self, association: Dict[str, Any]) -> bool:
        """Register Windows file association using registry"""
        try:
            # In production, use winreg to modify registry
            logger.info(f"Registered Windows file association for {association['file_extensions']}")
            return True
        except Exception as e:
            logger.error(f"Windows file association failed: {e}")
            return False
            
    async def _register_linux_association(self, association: Dict[str, Any]) -> bool:
        """Register Linux file association using desktop files"""
        try:
            # In production, create .desktop files and update MIME database
            logger.info(f"Registered Linux file association for {association['file_extensions']}")
            return True
        except Exception as e:
            logger.error(f"Linux file association failed: {e}")
            return False

class ProtocolHandlerManager:
    """Manage custom URL protocol handlers"""
    
    def __init__(self):
        self.platform = platform.system()
        self.handlers: Dict[str, Dict[str, Any]] = {}
        
    async def register_protocol_handler(self, config: Dict[str, Any]) -> bool:
        """Register custom protocol handler"""
        try:
            handler_id = str(uuid.uuid4())
            
            handler = {
                'handler_id': handler_id,
                'app_id': config['app_id'],
                'protocol': config['protocol'],  # e.g., "omega"
                'description': config.get('description', ''),
                'executable_path': config['executable_path'],
                'command_template': config.get('command_template', '{executable} --url "{url}"')
            }
            
            # Platform-specific registration
            if self.platform == "Darwin":
                success = await self._register_macos_protocol(handler)
            elif self.platform == "Windows":
                success = await self._register_windows_protocol(handler)
            elif self.platform == "Linux":
                success = await self._register_linux_protocol(handler)
            else:
                success = False
                
            if success:
                self.handlers[handler_id] = handler
                
            return success
            
        except Exception as e:
            logger.error(f"Protocol handler registration failed: {e}")
            return False
            
    async def _register_macos_protocol(self, handler: Dict[str, Any]) -> bool:
        """Register macOS protocol handler using Info.plist"""
        try:
            # In production, modify app bundle's Info.plist
            logger.info(f"Registered macOS protocol handler for {handler['protocol']}://")
            return True
        except Exception as e:
            logger.error(f"macOS protocol handler failed: {e}")
            return False
            
    async def _register_windows_protocol(self, handler: Dict[str, Any]) -> bool:
        """Register Windows protocol handler using registry"""
        try:
            # In production, use winreg to add registry entries
            logger.info(f"Registered Windows protocol handler for {handler['protocol']}://")
            return True
        except Exception as e:
            logger.error(f"Windows protocol handler failed: {e}")
            return False
            
    async def _register_linux_protocol(self, handler: Dict[str, Any]) -> bool:
        """Register Linux protocol handler using desktop files"""
        try:
            # In production, create desktop file with MimeType entry
            logger.info(f"Registered Linux protocol handler for {handler['protocol']}://")
            return True
        except Exception as e:
            logger.error(f"Linux protocol handler failed: {e}")
            return False

class DesktopAppIntegration:
    """Main desktop application integration system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/desktop_integration.db')
        
        # Core components
        self.apps: Dict[str, DesktopApp] = {}
        self.instances: Dict[str, AppInstance] = {}
        self.integrations: Dict[str, SystemIntegration] = {}
        
        # Managers
        self.window_manager = WindowManager()
        self.tray_manager = SystemTrayManager()
        self.notification_manager = NotificationManager()
        self.file_association_manager = FileAssociationManager()
        self.protocol_handler_manager = ProtocolHandlerManager()
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Monitoring
        self.running = False
        
        # Metrics
        self.metrics = {
            'registered_apps': 0,
            'running_instances': 0,
            'system_integrations': 0,
            'notifications_sent': 0,
            'window_operations': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the desktop integration system"""
        try:
            logger.info("Initializing Desktop App Integration...")
            
            # Setup database
            await self._setup_database()
            
            # Load registered apps
            await self._load_registered_apps()
            
            # Start monitoring
            self.running = True
            asyncio.create_task(self._instance_monitor())
            
            logger.info("Desktop App Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Desktop integration initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the desktop integration system"""
        try:
            logger.info("Shutting down Desktop App Integration...")
            self.running = False
            
            # Stop all running instances
            for instance_id in list(self.instances.keys()):
                await self.stop_app_instance(instance_id)
                
            logger.info("Desktop App Integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Desktop integration shutdown error: {e}")
            
    async def register_app(self, app: DesktopApp) -> bool:
        """Register a desktop application"""
        try:
            # Validate app configuration
            if not os.path.exists(app.executable_path):
                raise ValueError(f"Executable not found: {app.executable_path}")
                
            self.apps[app.app_id] = app
            self.metrics['registered_apps'] += 1
            
            # Setup system integrations
            for integration_type in app.integrations:
                await self._setup_integration(app, integration_type)
                
            # Save to database
            await self._save_app(app)
            
            await self._emit_event('app_registered', {
                'app_id': app.app_id,
                'name': app.name,
                'integrations': [i.value for i in app.integrations]
            })
            
            logger.info(f"Registered desktop app: {app.name}")
            return True
            
        except Exception as e:
            logger.error(f"App registration failed: {e}")
            return False
            
    async def launch_app(self, app_id: str, args: Optional[List[str]] = None) -> Optional[str]:
        """Launch a registered application"""
        try:
            if app_id not in self.apps:
                raise ValueError(f"App {app_id} not registered")
                
            app = self.apps[app_id]
            
            # Check if single instance and already running
            if app.single_instance:
                for instance in self.instances.values():
                    if instance.app_id == app_id and instance.state == "running":
                        logger.info(f"App {app_id} already running, bringing to front")
                        await self._bring_to_front(instance.instance_id)
                        return instance.instance_id
                        
            # Prepare command
            command = [app.executable_path]
            command.extend(app.command_line_args)
            if args:
                command.extend(args)
                
            # Launch process
            env = os.environ.copy()
            env.update(app.environment_vars)
            
            process = subprocess.Popen(
                command,
                cwd=app.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Create instance
            instance_id = str(uuid.uuid4())
            instance = AppInstance(
                instance_id=instance_id,
                app_id=app_id,
                process_id=process.pid,
                started_at=time.time()
            )
            
            self.instances[instance_id] = instance
            self.metrics['running_instances'] += 1
            
            await self._emit_event('app_launched', {
                'app_id': app_id,
                'instance_id': instance_id,
                'process_id': process.pid
            })
            
            logger.info(f"Launched app {app.name} (PID: {process.pid})")
            return instance_id
            
        except Exception as e:
            logger.error(f"App launch failed: {e}")
            return None
            
    async def stop_app_instance(self, instance_id: str) -> bool:
        """Stop a running application instance"""
        try:
            if instance_id not in self.instances:
                return False
                
            instance = self.instances[instance_id]
            
            # Terminate process
            if instance.process_id:
                try:
                    if platform.system() == "Windows":
                        subprocess.run(['taskkill', '/F', '/PID', str(instance.process_id)], 
                                     check=False, capture_output=True)
                    else:
                        subprocess.run(['kill', '-TERM', str(instance.process_id)], 
                                     check=False, capture_output=True)
                except Exception as e:
                    logger.warning(f"Failed to terminate process {instance.process_id}: {e}")
                    
            # Clean up instance
            del self.instances[instance_id]
            self.metrics['running_instances'] -= 1
            
            await self._emit_event('app_stopped', {
                'instance_id': instance_id,
                'app_id': instance.app_id
            })
            
            logger.info(f"Stopped app instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"App stop failed: {e}")
            return False
            
    async def create_window(self, app_id: str, config: Dict[str, Any]) -> Optional[str]:
        """Create a new window for an application"""
        try:
            window_id = await self.window_manager.create_window(app_id, config)
            if window_id:
                self.metrics['window_operations'] += 1
                
                await self._emit_event('window_created', {
                    'window_id': window_id,
                    'app_id': app_id,
                    'title': config.get('title', '')
                })
                
            return window_id
            
        except Exception as e:
            logger.error(f"Window creation failed: {e}")
            return None
            
    async def send_notification(self, config: Dict[str, Any]) -> bool:
        """Send a native notification"""
        try:
            success = await self.notification_manager.send_notification(config)
            if success:
                self.metrics['notifications_sent'] += 1
                
                await self._emit_event('notification_sent', {
                    'title': config.get('title', ''),
                    'app_id': config.get('app_id', '')
                })
                
            return success
            
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return False
            
    async def get_app_instances(self, app_id: Optional[str] = None) -> List[AppInstance]:
        """Get running application instances"""
        if app_id:
            return [instance for instance in self.instances.values() if instance.app_id == app_id]
        else:
            return list(self.instances.values())
            
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics"""
        try:
            app_types = {}
            integration_types = {}
            
            for app in self.apps.values():
                app_types[app.app_type.value] = app_types.get(app.app_type.value, 0) + 1
                
            for integration in self.integrations.values():
                int_type = integration.integration_type.value
                integration_types[int_type] = integration_types.get(int_type, 0) + 1
                
            running_states = {}
            for instance in self.instances.values():
                running_states[instance.state] = running_states.get(instance.state, 0) + 1
                
            return {
                'integration_status': {
                    'running': self.running,
                    'platform': platform.system(),
                    'total_registered_apps': len(self.apps),
                    'total_integrations': len(self.integrations)
                },
                'application_statistics': {
                    'registered_apps': self.metrics['registered_apps'],
                    'running_instances': len(self.instances),
                    'app_types': app_types,
                    'instance_states': running_states
                },
                'integration_statistics': {
                    'system_integrations': len(self.integrations),
                    'integration_types': integration_types,
                    'tray_icons': len(self.tray_manager.tray_items),
                    'file_associations': len(self.file_association_manager.associations),
                    'protocol_handlers': len(self.protocol_handler_manager.handlers)
                },
                'activity_metrics': {
                    'notifications_sent': self.metrics['notifications_sent'],
                    'window_operations': self.metrics['window_operations'],
                    'notification_history_size': len(self.notification_manager.notification_history)
                },
                'resource_usage': {
                    'total_memory_usage': sum(instance.memory_usage for instance in self.instances.values()),
                    'avg_cpu_usage': sum(instance.cpu_usage for instance in self.instances.values()) / len(self.instances) if self.instances else 0,
                    'active_windows': len(self.window_manager.windows)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get integration metrics: {e}")
            return {}
            
    async def _setup_integration(self, app: DesktopApp, integration_type: IntegrationType):
        """Setup a specific integration type for an app"""
        try:
            integration_id = str(uuid.uuid4())
            
            if integration_type == IntegrationType.SYSTEM_TRAY:
                tray_config = {
                    'icon_path': app.icon_path,
                    'tooltip': app.name,
                    'menu_items': [
                        {'label': 'Show', 'action': 'show_window'},
                        {'label': 'Hide', 'action': 'hide_window'},
                        {'type': 'separator'},
                        {'label': 'Exit', 'action': 'exit_app'}
                    ]
                }
                tray_id = await self.tray_manager.create_tray_icon(app.app_id, tray_config)
                if tray_id:
                    self._store_integration(integration_id, app.app_id, integration_type, {'tray_id': tray_id})
                    
            elif integration_type == IntegrationType.FILE_ASSOCIATION:
                if 'file_extensions' in app.metadata:
                    association_config = {
                        'app_id': app.app_id,
                        'file_extensions': app.metadata['file_extensions'],
                        'mime_types': app.metadata.get('mime_types', []),
                        'description': f"{app.name} Document",
                        'icon_path': app.icon_path,
                        'executable_path': app.executable_path
                    }
                    success = await self.file_association_manager.register_file_association(association_config)
                    if success:
                        self._store_integration(integration_id, app.app_id, integration_type, association_config)
                        
            elif integration_type == IntegrationType.PROTOCOL_HANDLER:
                if 'protocol' in app.metadata:
                    protocol_config = {
                        'app_id': app.app_id,
                        'protocol': app.metadata['protocol'],
                        'description': f"{app.name} Protocol Handler",
                        'executable_path': app.executable_path
                    }
                    success = await self.protocol_handler_manager.register_protocol_handler(protocol_config)
                    if success:
                        self._store_integration(integration_id, app.app_id, integration_type, protocol_config)
                        
        except Exception as e:
            logger.error(f"Integration setup failed: {e}")
            
    def _store_integration(self, integration_id: str, app_id: str, integration_type: IntegrationType, config: Dict[str, Any]):
        """Store integration configuration"""
        integration = SystemIntegration(
            integration_id=integration_id,
            app_id=app_id,
            integration_type=integration_type,
            configuration=config
        )
        self.integrations[integration_id] = integration
        self.metrics['system_integrations'] += 1
        
    async def _bring_to_front(self, instance_id: str):
        """Bring application instance to front"""
        # Platform-specific window activation would go here
        logger.debug(f"Bringing instance {instance_id} to front")
        
    async def _instance_monitor(self):
        """Monitor running application instances"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check instance health and update metrics
                for instance_id, instance in list(self.instances.items()):
                    if instance.process_id:
                        # Check if process is still running
                        try:
                            if platform.system() == "Windows":
                                result = subprocess.run(['tasklist', '/FI', f'PID eq {instance.process_id}'], 
                                                      capture_output=True, text=True, check=False)
                                if str(instance.process_id) not in result.stdout:
                                    instance.state = "stopped"
                            else:
                                result = subprocess.run(['ps', '-p', str(instance.process_id)], 
                                                      capture_output=True, check=False)
                                if result.returncode != 0:
                                    instance.state = "stopped"
                                    
                        except Exception:
                            instance.state = "unknown"
                            
                        # Remove stopped instances
                        if instance.state == "stopped":
                            await self.stop_app_instance(instance_id)
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Instance monitor error: {e}")
                await asyncio.sleep(30)
                
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit integration event"""
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(data)
        except Exception as e:
            logger.error(f"Event emission error: {e}")
            
    async def _setup_database(self):
        """Setup SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS desktop_apps (
                app_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                app_type TEXT NOT NULL,
                executable_path TEXT NOT NULL,
                working_directory TEXT NOT NULL,
                command_line_args TEXT,
                environment_vars TEXT,
                icon_path TEXT,
                launch_mode TEXT,
                auto_start BOOLEAN DEFAULT FALSE,
                single_instance BOOLEAN DEFAULT TRUE,
                integrations TEXT,
                permissions TEXT,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE TABLE IF NOT EXISTS system_integrations (
                integration_id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL,
                integration_type TEXT NOT NULL,
                configuration TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
        """)
        
        conn.commit()
        conn.close()
        
    async def _load_registered_apps(self):
        """Load registered apps from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM desktop_apps")
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    app = DesktopApp(
                        app_id=row[0],
                        name=row[1],
                        description=row[2] or "",
                        app_type=AppType(row[3]),
                        executable_path=row[4],
                        working_directory=row[5],
                        command_line_args=json.loads(row[6] or "[]"),
                        environment_vars=json.loads(row[7] or "{}"),
                        icon_path=row[8],
                        launch_mode=LaunchMode(row[9]) if row[9] else LaunchMode.FOREGROUND,
                        auto_start=bool(row[10]),
                        single_instance=bool(row[11]),
                        integrations=set(IntegrationType(i) for i in json.loads(row[12] or "[]")),
                        permissions=set(json.loads(row[13] or "[]")),
                        metadata=json.loads(row[14] or "{}"),
                        created_at=row[15]
                    )
                    
                    self.apps[app.app_id] = app
                    
                    # Setup integrations
                    for integration_type in app.integrations:
                        await self._setup_integration(app, integration_type)
                        
                except Exception as e:
                    logger.error(f"Failed to load app {row[0]}: {e}")
                    
            conn.close()
            logger.info(f"Loaded {len(self.apps)} registered desktop apps")
            
        except Exception as e:
            logger.error(f"Failed to load registered apps: {e}")
            
    async def _save_app(self, app: DesktopApp):
        """Save app to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO desktop_apps 
                (app_id, name, description, app_type, executable_path, working_directory,
                 command_line_args, environment_vars, icon_path, launch_mode, auto_start,
                 single_instance, integrations, permissions, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                app.app_id, app.name, app.description, app.app_type.value,
                app.executable_path, app.working_directory,
                json.dumps(app.command_line_args), json.dumps(app.environment_vars),
                app.icon_path, app.launch_mode.value, app.auto_start,
                app.single_instance, json.dumps([i.value for i in app.integrations]),
                json.dumps(list(app.permissions)), json.dumps(app.metadata),
                app.created_at
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save app: {e}")

# Global instance
_desktop_integration: Optional[DesktopAppIntegration] = None

async def initialize_desktop_integration(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global desktop integration system"""
    global _desktop_integration
    try:
        _desktop_integration = DesktopAppIntegration(config)
        return await _desktop_integration.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize desktop integration: {e}")
        return False

def get_desktop_integration() -> DesktopAppIntegration:
    """Get the global desktop integration instance"""
    global _desktop_integration
    if _desktop_integration is None:
        raise RuntimeError("Desktop integration not initialized. Call initialize_desktop_integration() first.")
    return _desktop_integration

async def shutdown_desktop_integration():
    """Shutdown the global desktop integration system"""
    global _desktop_integration
    if _desktop_integration:
        await _desktop_integration.shutdown()
        _desktop_integration = None
