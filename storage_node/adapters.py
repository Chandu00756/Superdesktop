"""Pluggable storage backend adapters.

Adapters expose a minimal unified interface used by the advanced storage node
for object persistence. Local filesystem is default; MinIO/S3 and in‑memory
variants are optional. All methods are async for future concurrency even if
currently wrapping sync code.
"""
from __future__ import annotations
import os, asyncio, io, json, time, hashlib
from typing import Optional, Dict, Any, AsyncIterator

try:
    from minio import Minio  # type: ignore
    _HAS_MINIO = True
except Exception:  # pragma: no cover
    _HAS_MINIO = False

class StorageError(Exception):
    pass

class BaseObjectStore:
    async def put(self, key: str, data: bytes, content_type: str="application/octet-stream") -> Dict[str,Any]:  # pragma: no cover - interface
        raise NotImplementedError
    async def get(self, key: str) -> Optional[bytes]:  # pragma: no cover
        raise NotImplementedError
    async def delete(self, key: str) -> bool:  # pragma: no cover
        raise NotImplementedError
    async def list(self, prefix: str="") -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError

class MemoryStore(BaseObjectStore):
    def __init__(self):
        self._d: Dict[str, bytes] = {}
    async def put(self, key: str, data: bytes, content_type: str="application/octet-stream"):
        self._d[key] = data
        return {"key": key, "size": len(data)}
    async def get(self, key: str):
        return self._d.get(key)
    async def delete(self, key: str):
        return self._d.pop(key, None) is not None
    async def list(self, prefix: str=""):
        for k in list(self._d.keys()):
            if k.startswith(prefix):
                yield k

class FileSystemStore(BaseObjectStore):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
    def _path(self, key: str) -> str:
        return os.path.join(self.root, key.replace('..','_'))
    async def put(self, key: str, data: bytes, content_type: str="application/octet-stream"):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: open(path,'wb').write(data))
        return {"key": key, "size": len(data)}
    async def get(self, key: str):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        loop = asyncio.get_event_loop()
        with open(path,'rb') as f:
            return await loop.run_in_executor(None, f.read)
    async def delete(self, key: str):
        path = self._path(key)
        try:
            os.remove(path); return True
        except Exception:
            return False
    async def list(self, prefix: str=""):
        base_len = len(self.root.rstrip('/') + '/')
        for root, _, files in os.walk(self.root):
            for name in files:
                rel = os.path.join(root, name)[base_len:]
                if rel.startswith(prefix):
                    yield rel

class MinioStore(BaseObjectStore):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool=True):
        if not _HAS_MINIO:
            raise RuntimeError("minio library not installed")
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
    async def put(self, key: str, data: bytes, content_type: str="application/octet-stream"):
        loop = asyncio.get_event_loop()
        size = len(data)
        def _upload():
            import io
            self.client.put_object(self.bucket, key, io.BytesIO(data), length=size, content_type=content_type)
        await loop.run_in_executor(None, _upload)
        return {"key": key, "size": size}
    async def get(self, key: str):
        loop = asyncio.get_event_loop()
        def _get():
            try:
                resp = self.client.get_object(self.bucket, key)
                try:
                    return resp.read()
                finally:
                    resp.close(); resp.release_conn()
            except Exception:
                return None
        return await loop.run_in_executor(None, _get)
    async def delete(self, key: str):
        loop = asyncio.get_event_loop()
        def _del():
            try:
                self.client.remove_object(self.bucket, key); return True
            except Exception: return False
        return await loop.run_in_executor(None, _del)
    async def list(self, prefix: str=""):
        loop = asyncio.get_event_loop()
        def _list():
            for obj in self.client.list_objects(self.bucket, prefix=prefix, recursive=True):
                yield obj.object_name
        # bridge sync generator to async
        for name in await loop.run_in_executor(None, lambda: list(_list())):
            yield name

def build_default_store() -> BaseObjectStore:
    backend = os.getenv('OMEGA_OBJECT_STORE','filesystem').lower()
    if backend == 'memory':
        return MemoryStore()
    if backend == 'minio':
        try:
            return MinioStore(
                endpoint=os.getenv('MINIO_ENDPOINT','localhost:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY','minioadmin'),
                secret_key=os.getenv('MINIO_SECRET_KEY','minioadmin'),
                bucket=os.getenv('MINIO_BUCKET','omega'),
                secure=bool(int(os.getenv('MINIO_SECURE','0')))
            )
        except Exception:
            # Fallback to filesystem if minio unreachable
            return FileSystemStore(os.getenv('OMEGA_FS_STORE','data/object_storage/objects'))
    # default FS
    return FileSystemStore(os.getenv('OMEGA_FS_STORE','data/object_storage/objects'))

__all__ = ['BaseObjectStore','FileSystemStore','MemoryStore','MinioStore','build_default_store']
