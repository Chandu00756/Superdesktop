"""MinIO-backed async key-value store using object storage semantics.

Keys are mapped to object names under a prefix. Values are raw bytes.
This is intended for small metadata/session blobs where object storage
is acceptable. A lightweight "ping" is provided for health checks.

If the minio package is not available, importing this module will raise
ImportError; the factory is responsible for falling back to memory.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

try:
    from minio import Minio  # type: ignore
    from minio.error import S3Error  # type: ignore
except Exception as e:  # pragma: no cover - optional dep
    raise


class MinioKVStore:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool | None = None,
        prefix: str = "kv/",
    ) -> None:
        endpoint = endpoint or os.getenv("OMEGA_MINIO_ENDPOINT", "localhost:9000")
        access_key = access_key or os.getenv("OMEGA_MINIO_ACCESS", "minioadmin")
        secret_key = secret_key or os.getenv("OMEGA_MINIO_SECRET", "minioadmin")
        bucket = bucket or os.getenv("OMEGA_MINIO_BUCKET", "omega-kv")
        secure = secure if secure is not None else (os.getenv("OMEGA_MINIO_SECURE", "0").lower() in ("1","true","yes","on"))

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'

        # Ensure bucket exists (best effort)
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception:
            # Defer errors to ping/get/set usage to allow fallback by factory
            pass

    def _obj(self, key: str) -> str:
        # Avoid path traversal
        key = key.replace('..', '').lstrip('/')
        return f"{self.prefix}{key}"

    async def ping(self) -> bool:
        loop = asyncio.get_running_loop()

        def _head() -> bool:
            try:
                # List with limit 1 as a cheap health check
                self.client.list_objects(self.bucket, prefix=self.prefix, max_keys=1)
                return True
            except Exception:
                return False

        return await loop.run_in_executor(None, _head)

    async def get(self, key: str) -> Optional[bytes]:
        obj = self._obj(key)
        loop = asyncio.get_running_loop()

        def _get() -> Optional[bytes]:
            try:
                resp = self.client.get_object(self.bucket, obj)
                data = resp.read()
                resp.close()
                resp.release_conn()
                return data
            except S3Error as e:
                if getattr(e, 'code', '') in ("NoSuchKey", "NoSuchObject"):
                    return None
                return None
            except Exception:
                return None

        return await loop.run_in_executor(None, _get)

    async def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> None:  # ttl ignored
        obj = self._obj(key)
        loop = asyncio.get_running_loop()

        def _put() -> None:
            import io
            self.client.put_object(self.bucket, obj, io.BytesIO(value), len(value))

        await loop.run_in_executor(None, _put)

    async def delete(self, key: str) -> None:
        obj = self._obj(key)
        loop = asyncio.get_running_loop()

        def _del() -> None:
            try:
                self.client.remove_object(self.bucket, obj)
            except Exception:
                pass

        await loop.run_in_executor(None, _del)
