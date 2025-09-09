"""Pluggable storage adapters for lightweight KV and object-like usage.

Exports:
- MemoryStore: in-memory async KV with TTL/LRU.
- MinioKVStore: optional MinIO-backed KV using object prefixes (if minio installed).
- build_default_store: factory returning a StoreHandle wrapper around the selected backend.
"""

from .memory_adapter import MemoryStore  # noqa: F401
try:
    from .minio_adapter import MinioKVStore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    MinioKVStore = None  # type: ignore

from .factory import build_default_store, StoreHandle  # noqa: F401

__all__ = [
    "MemoryStore",
    "MinioKVStore",
    "build_default_store",
    "StoreHandle",
]
