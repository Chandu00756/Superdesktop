"""
Omega Super Desktop Console v2.0 - Plugin Framework
Enterprise-grade plugin system with dynamic loading, sandboxing, and lifecycle management
"""

import asyncio
import json
import logging
import time
import uuid
import importlib
import inspect
import sys
import os
import subprocess
import threading
from typing import Dict, List, Optional, Set, Any, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import hashlib
import tempfile
import shutil
from pathlib import Path
import zipfile
import yaml

logger = logging.getLogger(__name__)

class PluginState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    QUARANTINED = "quarantined"

class PluginType(Enum):
    CORE = "core"                   # Core system plugins
    UI_COMPONENT = "ui_component"   # UI widgets/components
    SERVICE = "service"             # Background services
    PROCESSOR = "processor"         # Data processors
    CONNECTOR = "connector"         # External system connectors
    WORKFLOW = "workflow"           # Workflow automation
    SECURITY = "security"           # Security extensions
    ANALYTICS = "analytics"         # Analytics and reporting

class SecurityLevel(Enum):
    TRUSTED = "trusted"             # Full system access
    SANDBOXED = "sandboxed"         # Limited sandbox
    RESTRICTED = "restricted"       # Minimal permissions
    QUARANTINED = "quarantined"     # No execution

class PluginCapability(Enum):
    FILE_ACCESS = "file_access"
    NETWORK_ACCESS = "network_access"
    SYSTEM_CALLS = "system_calls"
    DATABASE_ACCESS = "database_access"
    MEMORY_ACCESS = "memory_access"
    UI_MODIFICATION = "ui_modification"
    SERVICE_DISCOVERY = "service_discovery"
    INTER_PLUGIN_COMM = "inter_plugin_comm"

@dataclass
class PluginManifest:
    """Plugin manifest describing plugin metadata and requirements"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    capabilities: Set[PluginCapability] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.SANDBOXED
    min_platform_version: str = "2.0.0"
    max_platform_version: Optional[str] = None
    configuration_schema: Optional[Dict[str, Any]] = None
    ui_components: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    signed: bool = False
    signature: Optional[str] = None

@dataclass
class PluginInstance:
    """Runtime instance of a loaded plugin"""
    plugin_id: str
    manifest: PluginManifest
    state: PluginState
    module: Optional[Any] = None
    instance: Optional[Any] = None
    load_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    sandbox_path: Optional[str] = None
    process_id: Optional[int] = None
    memory_usage: int = 0
    cpu_usage: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PluginEvent:
    """Plugin system event"""
    event_id: str
    event_type: str
    plugin_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "plugin_framework"

class PluginSandbox:
    """Sandbox environment for plugin execution"""
    
    def __init__(self, plugin_id: str, security_level: SecurityLevel):
        self.plugin_id = plugin_id
        self.security_level = security_level
        self.sandbox_dir = None
        self.allowed_modules = set()
        self.blocked_modules = set()
        self.resource_limits = {}
        
    async def setup(self) -> bool:
        """Setup sandbox environment"""
        try:
            # Create temporary sandbox directory
            self.sandbox_dir = tempfile.mkdtemp(prefix=f"plugin_{self.plugin_id}_")
            
            # Set up module restrictions based on security level
            if self.security_level == SecurityLevel.RESTRICTED:
                self.allowed_modules = {
                    'builtins', 'json', 'time', 'datetime', 'uuid', 're', 'math',
                    'collections', 'itertools', 'functools', 'operator'
                }
                self.blocked_modules = {
                    'os', 'sys', 'subprocess', 'importlib', 'exec', 'eval',
                    'open', '__import__', 'compile', 'globals', 'locals'
                }
            elif self.security_level == SecurityLevel.SANDBOXED:
                self.blocked_modules = {
                    'subprocess', 'os.system', 'eval', 'exec', 'compile'
                }
            # TRUSTED level has no restrictions
            
            # Set resource limits
            self.resource_limits = {
                'max_memory': 100 * 1024 * 1024,  # 100MB
                'max_cpu_time': 60,                # 60 seconds
                'max_file_descriptors': 50,
                'max_threads': 10
            }
            
            logger.info(f"Sandbox setup complete for plugin {self.plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Sandbox setup failed: {e}")
            return False
            
    async def cleanup(self):
        """Cleanup sandbox environment"""
        try:
            if self.sandbox_dir and os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
                logger.info(f"Sandbox cleaned up for plugin {self.plugin_id}")
        except Exception as e:
            logger.error(f"Sandbox cleanup failed: {e}")
            
    def validate_import(self, module_name: str) -> bool:
        """Validate if module import is allowed"""
        if self.security_level == SecurityLevel.TRUSTED:
            return True
        elif self.security_level == SecurityLevel.RESTRICTED:
            return module_name in self.allowed_modules
        else:  # SANDBOXED
            return module_name not in self.blocked_modules

class PluginValidator:
    """Validates plugin security and compliance"""
    
    def __init__(self):
        self.dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(',
            r'file\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'compile\s*\('
        ]
        
    async def validate_manifest(self, manifest: PluginManifest) -> Tuple[bool, List[str]]:
        """Validate plugin manifest"""
        issues = []
        
        try:
            # Check required fields
            if not manifest.plugin_id or not manifest.name or not manifest.version:
                issues.append("Missing required manifest fields")
                
            # Validate version format
            if not self._is_valid_version(manifest.version):
                issues.append("Invalid version format")
                
            # Check entry point
            if not manifest.entry_point:
                issues.append("Missing entry point")
                
            # Validate capabilities vs security level
            if (manifest.security_level == SecurityLevel.RESTRICTED and 
                len(manifest.capabilities) > 2):
                issues.append("Too many capabilities for restricted security level")
                
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Manifest validation error: {e}")
            return False, [f"Validation error: {str(e)}"]
            
    async def scan_code(self, code_path: str, security_level: SecurityLevel) -> Tuple[bool, List[str]]:
        """Scan plugin code for security issues"""
        issues = []
        
        try:
            # Read and parse code
            with open(code_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # AST analysis
            try:
                tree = ast.parse(code)
                issues.extend(await self._analyze_ast(tree, security_level))
            except SyntaxError as e:
                issues.append(f"Syntax error: {str(e)}")
                
            # Pattern matching for dangerous code
            if security_level != SecurityLevel.TRUSTED:
                import re
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, code):
                        issues.append(f"Dangerous pattern detected: {pattern}")
                        
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Code scanning error: {e}")
            return False, [f"Scanning error: {str(e)}"]
            
    async def _analyze_ast(self, tree: ast.AST, security_level: SecurityLevel) -> List[str]:
        """Analyze AST for security issues"""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile']:
                        if security_level != SecurityLevel.TRUSTED:
                            issues.append(f"Dangerous function call: {node.func.id}")
                            
            # Check imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys']:
                        if security_level == SecurityLevel.RESTRICTED:
                            issues.append(f"Restricted import: {alias.name}")
                            
        return issues
        
    def _is_valid_version(self, version: str) -> bool:
        """Validate version format (semver)"""
        import re
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        return bool(re.match(pattern, version))

class PluginRegistry:
    """Registry for plugin discovery and management"""
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.plugins: Dict[str, PluginManifest] = {}
        self.repositories: List[str] = []
        
    async def register_plugin(self, manifest: PluginManifest) -> bool:
        """Register a plugin in the registry"""
        try:
            self.plugins[manifest.plugin_id] = manifest
            await self._save_registry()
            logger.info(f"Plugin {manifest.plugin_id} registered")
            return True
        except Exception as e:
            logger.error(f"Plugin registration failed: {e}")
            return False
            
    async def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin"""
        try:
            if plugin_id in self.plugins:
                del self.plugins[plugin_id]
                await self._save_registry()
                logger.info(f"Plugin {plugin_id} unregistered")
                return True
            return False
        except Exception as e:
            logger.error(f"Plugin unregistration failed: {e}")
            return False
            
    async def find_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginManifest]:
        """Find plugins by type"""
        if plugin_type:
            return [p for p in self.plugins.values() if p.plugin_type == plugin_type]
        return list(self.plugins.values())
        
    async def _save_registry(self):
        """Save registry to disk"""
        try:
            registry_data = {
                'plugins': {
                    pid: {
                        'plugin_id': p.plugin_id,
                        'name': p.name,
                        'version': p.version,
                        'description': p.description,
                        'author': p.author,
                        'plugin_type': p.plugin_type.value,
                        'entry_point': p.entry_point,
                        'dependencies': p.dependencies,
                        'capabilities': [c.value for c in p.capabilities],
                        'security_level': p.security_level.value,
                        'checksum': p.checksum
                    }
                    for pid, p in self.plugins.items()
                }
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Registry save failed: {e}")

class PluginFramework:
    """Main plugin framework managing the entire plugin lifecycle"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.plugins: Dict[str, PluginInstance] = {}
        self.validator = PluginValidator()
        self.registry = PluginRegistry(
            self.config.get('registry_path', 'plugins/registry.json')
        )
        self.plugin_directories = self.config.get('plugin_directories', ['plugins/'])
        self.running = False
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.events_queue: asyncio.Queue = asyncio.Queue()
        
        # Metrics
        self.metrics = {
            'total_plugins': 0,
            'active_plugins': 0,
            'failed_loads': 0,
            'security_violations': 0,
            'total_events': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the plugin framework"""
        try:
            logger.info("Initializing Plugin Framework...")
            
            # Create plugin directories
            for plugin_dir in self.plugin_directories:
                os.makedirs(plugin_dir, exist_ok=True)
                
            # Start background tasks
            self.running = True
            asyncio.create_task(self._event_processor())
            asyncio.create_task(self._plugin_monitor())
            
            # Discover and load plugins
            await self.discover_plugins()
            
            logger.info("Plugin Framework initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin framework initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the plugin framework"""
        try:
            logger.info("Shutting down Plugin Framework...")
            self.running = False
            
            # Stop all active plugins
            for plugin_id in list(self.plugins.keys()):
                await self.stop_plugin(plugin_id)
                
            logger.info("Plugin Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Plugin framework shutdown error: {e}")
            
    async def discover_plugins(self) -> List[str]:
        """Discover plugins in configured directories"""
        discovered = []
        
        try:
            for plugin_dir in self.plugin_directories:
                if not os.path.exists(plugin_dir):
                    continue
                    
                for item in os.listdir(plugin_dir):
                    item_path = os.path.join(plugin_dir, item)
                    
                    # Check for plugin directory
                    if os.path.isdir(item_path):
                        manifest_path = os.path.join(item_path, 'manifest.json')
                        if os.path.exists(manifest_path):
                            plugin_id = await self._load_plugin_manifest(manifest_path)
                            if plugin_id:
                                discovered.append(plugin_id)
                                
                    # Check for plugin archive
                    elif item.endswith(('.zip', '.tar.gz')):
                        plugin_id = await self._extract_and_load_plugin(item_path)
                        if plugin_id:
                            discovered.append(plugin_id)
                            
            logger.info(f"Discovered {len(discovered)} plugins")
            return discovered
            
        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")
            return []
            
    async def load_plugin(self, plugin_id: str) -> bool:
        """Load a specific plugin"""
        try:
            if plugin_id not in self.registry.plugins:
                raise ValueError(f"Plugin {plugin_id} not found in registry")
                
            if plugin_id in self.plugins:
                logger.warning(f"Plugin {plugin_id} already loaded")
                return True
                
            manifest = self.registry.plugins[plugin_id]
            
            # Create plugin instance
            instance = PluginInstance(
                plugin_id=plugin_id,
                manifest=manifest,
                state=PluginState.LOADING
            )
            
            # Setup sandbox
            sandbox = PluginSandbox(plugin_id, manifest.security_level)
            success = await sandbox.setup()
            if not success:
                raise Exception("Sandbox setup failed")
                
            instance.sandbox_path = sandbox.sandbox_dir
            
            # Validate plugin code
            plugin_path = await self._find_plugin_path(plugin_id)
            if not plugin_path:
                raise Exception("Plugin files not found")
                
            code_valid, issues = await self.validator.scan_code(
                os.path.join(plugin_path, manifest.entry_point),
                manifest.security_level
            )
            
            if not code_valid:
                instance.state = PluginState.QUARANTINED
                logger.error(f"Plugin {plugin_id} quarantined: {issues}")
                self.metrics['security_violations'] += 1
                return False
                
            # Load plugin module
            try:
                spec = importlib.util.spec_from_file_location(
                    plugin_id,
                    os.path.join(plugin_path, manifest.entry_point)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                instance.module = module
                instance.state = PluginState.LOADED
                
                # Create plugin instance if it has a main class
                if hasattr(module, 'Plugin'):
                    plugin_class = getattr(module, 'Plugin')
                    instance.instance = plugin_class()
                    
            except Exception as e:
                instance.state = PluginState.ERROR
                instance.last_error = str(e)
                logger.error(f"Plugin {plugin_id} load failed: {e}")
                self.metrics['failed_loads'] += 1
                return False
                
            # Store plugin instance
            self.plugins[plugin_id] = instance
            self.metrics['total_plugins'] += 1
            
            # Emit event
            await self._emit_event('plugin_loaded', {
                'plugin_id': plugin_id,
                'name': manifest.name,
                'version': manifest.version
            })
            
            logger.info(f"Plugin {plugin_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin load failed: {e}")
            return False
            
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a loaded plugin"""
        try:
            if plugin_id not in self.plugins:
                await self.load_plugin(plugin_id)
                
            instance = self.plugins[plugin_id]
            
            if instance.state not in [PluginState.LOADED, PluginState.STOPPED]:
                logger.warning(f"Plugin {plugin_id} not in startable state: {instance.state}")
                return False
                
            instance.state = PluginState.INITIALIZING
            
            # Initialize plugin
            if instance.instance and hasattr(instance.instance, 'initialize'):
                try:
                    await instance.instance.initialize(instance.configuration)
                    instance.state = PluginState.ACTIVE
                    instance.start_time = time.time()
                    self.metrics['active_plugins'] += 1
                    
                    await self._emit_event('plugin_started', {
                        'plugin_id': plugin_id,
                        'start_time': instance.start_time
                    })
                    
                    logger.info(f"Plugin {plugin_id} started successfully")
                    return True
                    
                except Exception as e:
                    instance.state = PluginState.ERROR
                    instance.last_error = str(e)
                    instance.error_count += 1
                    logger.error(f"Plugin {plugin_id} initialization failed: {e}")
                    return False
            else:
                # Plugin doesn't have initialize method, mark as active
                instance.state = PluginState.ACTIVE
                instance.start_time = time.time()
                self.metrics['active_plugins'] += 1
                return True
                
        except Exception as e:
            logger.error(f"Plugin start failed: {e}")
            return False
            
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop an active plugin"""
        try:
            if plugin_id not in self.plugins:
                return False
                
            instance = self.plugins[plugin_id]
            
            if instance.state != PluginState.ACTIVE:
                return True
                
            instance.state = PluginState.STOPPING
            
            # Stop plugin
            if instance.instance and hasattr(instance.instance, 'shutdown'):
                try:
                    await instance.instance.shutdown()
                except Exception as e:
                    logger.error(f"Plugin {plugin_id} shutdown error: {e}")
                    
            instance.state = PluginState.STOPPED
            instance.stop_time = time.time()
            self.metrics['active_plugins'] -= 1
            
            await self._emit_event('plugin_stopped', {
                'plugin_id': plugin_id,
                'stop_time': instance.stop_time
            })
            
            logger.info(f"Plugin {plugin_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Plugin stop failed: {e}")
            return False
            
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_id not in self.plugins:
                return True
                
            # Stop plugin first
            await self.stop_plugin(plugin_id)
            
            instance = self.plugins[plugin_id]
            
            # Cleanup sandbox
            if instance.sandbox_path:
                sandbox = PluginSandbox(plugin_id, instance.manifest.security_level)
                sandbox.sandbox_dir = instance.sandbox_path
                await sandbox.cleanup()
                
            # Remove from memory
            del self.plugins[plugin_id]
            self.metrics['total_plugins'] -= 1
            
            await self._emit_event('plugin_unloaded', {
                'plugin_id': plugin_id
            })
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Plugin unload failed: {e}")
            return False
            
    async def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin"""
        try:
            if plugin_id not in self.plugins:
                return None
                
            instance = self.plugins[plugin_id]
            manifest = instance.manifest
            
            return {
                'plugin_id': plugin_id,
                'name': manifest.name,
                'version': manifest.version,
                'description': manifest.description,
                'author': manifest.author,
                'plugin_type': manifest.plugin_type.value,
                'state': instance.state.value,
                'security_level': manifest.security_level.value,
                'capabilities': [c.value for c in manifest.capabilities],
                'load_time': instance.load_time,
                'start_time': instance.start_time,
                'memory_usage': instance.memory_usage,
                'cpu_usage': instance.cpu_usage,
                'error_count': instance.error_count,
                'last_error': instance.last_error
            }
            
        except Exception as e:
            logger.error(f"Failed to get plugin info: {e}")
            return None
            
    async def get_framework_metrics(self) -> Dict[str, Any]:
        """Get comprehensive framework metrics"""
        try:
            active_count = sum(1 for p in self.plugins.values() if p.state == PluginState.ACTIVE)
            error_count = sum(1 for p in self.plugins.values() if p.state == PluginState.ERROR)
            
            return {
                'framework_status': {
                    'running': self.running,
                    'total_plugins': len(self.plugins),
                    'registry_plugins': len(self.registry.plugins)
                },
                'plugin_states': {
                    'active': active_count,
                    'loaded': sum(1 for p in self.plugins.values() if p.state == PluginState.LOADED),
                    'stopped': sum(1 for p in self.plugins.values() if p.state == PluginState.STOPPED),
                    'error': error_count,
                    'quarantined': sum(1 for p in self.plugins.values() if p.state == PluginState.QUARANTINED)
                },
                'performance': {
                    'total_memory_usage': sum(p.memory_usage for p in self.plugins.values()),
                    'avg_cpu_usage': sum(p.cpu_usage for p in self.plugins.values()) / len(self.plugins) if self.plugins else 0,
                    'total_errors': sum(p.error_count for p in self.plugins.values())
                },
                'security': {
                    'security_violations': self.metrics['security_violations'],
                    'quarantined_plugins': sum(1 for p in self.plugins.values() if p.state == PluginState.QUARANTINED)
                },
                'events': {
                    'total_events': self.metrics['total_events'],
                    'queue_size': self.events_queue.qsize()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get framework metrics: {e}")
            return {}
            
    async def _load_plugin_manifest(self, manifest_path: str) -> Optional[str]:
        """Load plugin manifest from file"""
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
                
            manifest = PluginManifest(
                plugin_id=manifest_data['plugin_id'],
                name=manifest_data['name'],
                version=manifest_data['version'],
                description=manifest_data.get('description', ''),
                author=manifest_data.get('author', ''),
                plugin_type=PluginType(manifest_data['plugin_type']),
                entry_point=manifest_data['entry_point'],
                dependencies=manifest_data.get('dependencies', []),
                capabilities=set(PluginCapability(c) for c in manifest_data.get('capabilities', [])),
                security_level=SecurityLevel(manifest_data.get('security_level', 'sandboxed'))
            )
            
            # Validate manifest
            valid, issues = await self.validator.validate_manifest(manifest)
            if not valid:
                logger.error(f"Invalid manifest {manifest_path}: {issues}")
                return None
                
            # Register in registry
            await self.registry.register_plugin(manifest)
            return manifest.plugin_id
            
        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            return None
            
    async def _find_plugin_path(self, plugin_id: str) -> Optional[str]:
        """Find plugin files on disk"""
        for plugin_dir in self.plugin_directories:
            plugin_path = os.path.join(plugin_dir, plugin_id)
            if os.path.exists(plugin_path):
                return plugin_path
        return None
        
    async def _extract_and_load_plugin(self, archive_path: str) -> Optional[str]:
        """Extract and load plugin from archive"""
        # Implementation would extract zip/tar files and load manifest
        # For demo purposes, we'll skip this
        return None
        
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a plugin framework event"""
        event = PluginEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            plugin_id=data.get('plugin_id', ''),
            timestamp=time.time(),
            data=data
        )
        
        await self.events_queue.put(event)
        self.metrics['total_events'] += 1
        
    async def _event_processor(self):
        """Process plugin framework events"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.events_queue.get(), timeout=1.0)
                
                # Process event handlers
                if event.event_type in self.event_handlers:
                    for handler in self.event_handlers[event.event_type]:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Event handler error: {e}")
                            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processor error: {e}")
                
    async def _plugin_monitor(self):
        """Monitor plugin health and resources"""
        while self.running:
            try:
                for plugin_id, instance in self.plugins.items():
                    if instance.state == PluginState.ACTIVE:
                        # Monitor resource usage (simplified)
                        # In production, this would use psutil or similar
                        instance.memory_usage = 0  # Placeholder
                        instance.cpu_usage = 0.0   # Placeholder
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Plugin monitor error: {e}")
                await asyncio.sleep(30)

# Global instance
_plugin_framework: Optional[PluginFramework] = None

async def initialize_plugin_framework(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global plugin framework"""
    global _plugin_framework
    try:
        _plugin_framework = PluginFramework(config)
        return await _plugin_framework.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize plugin framework: {e}")
        return False

def get_plugin_framework() -> PluginFramework:
    """Get the global plugin framework instance"""
    global _plugin_framework
    if _plugin_framework is None:
        raise RuntimeError("Plugin framework not initialized. Call initialize_plugin_framework() first.")
    return _plugin_framework

async def shutdown_plugin_framework():
    """Shutdown the global plugin framework"""
    global _plugin_framework
    if _plugin_framework:
        await _plugin_framework.shutdown()
        _plugin_framework = None
