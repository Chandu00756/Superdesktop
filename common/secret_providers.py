"""Secret provider abstraction layer.

Supports pluggable backends:
 - EnvProvider: reads from environment variables
 - FileProvider: reads a directory of secret files (each filename = key)
 - VaultProvider (stub): interface placeholder for HashiCorp Vault

Selection hierarchy: explicit provider list or fall back chain.
Thread-safe caching with reload TTL.
"""
from __future__ import annotations
import os, threading, time, json
from typing import Optional, Dict, Any, List

class SecretProviderError(Exception):
    pass

class BaseSecretProvider:
    def get(self, key: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError
    def bulk(self) -> Dict[str,str]:  # pragma: no cover - interface
        raise NotImplementedError

class EnvProvider(BaseSecretProvider):
    def get(self, key: str) -> Optional[str]:
        return os.environ.get(key)
    def bulk(self) -> Dict[str,str]:
        return {k:v for k,v in os.environ.items() if k.startswith('OMEGA_')}

class FileProvider(BaseSecretProvider):
    def __init__(self, directory: str):
        self.directory = directory
    def get(self, key: str) -> Optional[str]:
        path = os.path.join(self.directory, key)
        try:
            with open(path,'r') as f:
                return f.read().strip()
        except Exception:
            return None
    def bulk(self) -> Dict[str,str]:
        out = {}
        if not os.path.isdir(self.directory):
            return out
        for name in os.listdir(self.directory):
            try:
                p = os.path.join(self.directory, name)
                if os.path.isfile(p):
                    with open(p,'r') as f:
                        out[name] = f.read().strip()
            except Exception:
                continue
        return out

class VaultProvider(BaseSecretProvider):
    def __init__(self, url: str, token: str, prefix: str='secret/data'):
        self.url = url; self.token = token; self.prefix = prefix
    def get(self, key: str) -> Optional[str]:  # stub
        return None
    def bulk(self) -> Dict[str,str]:
        return {}

class SecretManager:
    def __init__(self, providers: List[BaseSecretProvider], ttl: int = 30):
        self.providers = providers
        self.ttl = ttl
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
    def get(self, key: str, default: Optional[str]=None) -> Optional[str]:
        with self._lock:
            now = time.time()
            if key in self._cache and self._expiry.get(key,0) > now:
                return self._cache[key]
        val = None
        for p in self.providers:
            val = p.get(key)
            if val is not None:
                break
        if val is None:
            val = default
        with self._lock:
            self._cache[key] = val
            self._expiry[key] = time.time() + self.ttl
        return val
    def snapshot(self) -> Dict[str,Any]:
        snap = {}
        for p in self.providers:
            try:
                snap.update(p.bulk())
            except Exception:
                continue
        return snap

def default_manager() -> SecretManager:
    providers: List[BaseSecretProvider] = []
    secrets_dir = os.environ.get('OMEGA_SECRETS_DIR')
    if secrets_dir:
        providers.append(FileProvider(secrets_dir))
    providers.append(EnvProvider())
    # Vault optional (enabled if VAULT_ADDR + VAULT_TOKEN set)
    if os.environ.get('VAULT_ADDR') and os.environ.get('VAULT_TOKEN'):
        providers.insert(0, VaultProvider(os.environ['VAULT_ADDR'], os.environ['VAULT_TOKEN']))
    return SecretManager(providers)

__all__ = ['SecretManager','EnvProvider','FileProvider','VaultProvider','default_manager']