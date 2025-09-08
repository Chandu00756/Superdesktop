"""Lightweight plugin framework scaffold.

Loads plugin manifests from a directory (default 'plugins'). Each plugin
directory must contain manifest.json with keys:
  name, version, description, entrypoint (module:function)
Optional: manifest.sig containing sha256 hex digest of manifest.json for
basic integrity verification. (Future: replace with RSA signature.)

State is kept in-memory; backend endpoint merges DB table data (for
enable/disable flags) with live loaded registry. Plugins can be reloaded
at runtime.
"""
from __future__ import annotations
import os, json, hashlib, importlib, logging, time
from typing import Dict, Any, List

log = logging.getLogger(__name__)

class PluginRecord:
    def __init__(self, meta: Dict[str, Any]):
        self.name = meta.get('name')
        self.version = meta.get('version')
        self.description = meta.get('description','')
        self.entrypoint = meta.get('entrypoint')
        self.loaded_at = time.time()
        self.status = 'pending'
        self.error: str | None = None
        self.module = None

    def to_dict(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'entrypoint': self.entrypoint,
            'loaded_at': self.loaded_at,
            'status': self.status,
            'error': self.error
        }

class PluginManager:
    def __init__(self, directory: str | None = None):
        self.directory = directory or os.getenv('OMEGA_PLUGIN_DIR', 'plugins')
        self._plugins: Dict[str, PluginRecord] = {}

    def _verify_manifest(self, path: str) -> bool:
        manifest_path = os.path.join(path, 'manifest.json')
        sig_path = os.path.join(path, 'manifest.sig')
        if not os.path.exists(manifest_path):
            return False
        if not os.path.exists(sig_path):
            return True  # unsigned accepted for now
        try:
            raw = open(manifest_path,'rb').read()
            want = open(sig_path,'r').read().strip()
            have = hashlib.sha256(raw).hexdigest()
            return want == have
        except Exception as e:
            log.warning(f"Plugin manifest verification failed for {path}: {e}")
            return False

    def load_all(self):
        if not os.path.isdir(self.directory):
            return
        for entry in os.listdir(self.directory):
            p = os.path.join(self.directory, entry)
            if not os.path.isdir(p):
                continue
            if not self._verify_manifest(p):
                log.warning(f"Skipping plugin {entry}: signature/manifest invalid")
                continue
            try:
                meta = json.load(open(os.path.join(p,'manifest.json'),'r'))
            except Exception as e:
                log.error(f"Failed reading manifest for {entry}: {e}")
                continue
            rec = PluginRecord(meta)
            self._plugins[rec.name] = rec
            self._load_plugin(rec)

    def _load_plugin(self, rec: PluginRecord):
        if not rec.entrypoint or ':' not in rec.entrypoint:
            rec.status = 'invalid'
            rec.error = 'missing entrypoint'
            return
        mod_name, func_name = rec.entrypoint.split(':',1)
        try:
            mod = importlib.import_module(mod_name)
            rec.module = mod
            func = getattr(mod, func_name, None)
            if callable(func):
                try:
                    func()  # plugin initialization hook
                except Exception as e:
                    rec.status = 'error'
                    rec.error = f'hook error {e}'
                    return
            rec.status = 'loaded'
        except Exception as e:
            rec.status = 'error'
            rec.error = str(e)

    def reload(self):
        self._plugins.clear()
        self.load_all()

    def list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._plugins.values()]


_manager: PluginManager | None = None

def get_plugin_manager() -> PluginManager:
    global _manager
    if _manager is None:
        _manager = PluginManager()
        _manager.load_all()
    return _manager

__all__ = ['get_plugin_manager','PluginManager']
