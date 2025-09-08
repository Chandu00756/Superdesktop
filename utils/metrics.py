"""Utilities for safe Prometheus metric creation with de-duplication.

This module provides helper factories that:
1. Avoid ValueError exceptions when a metric is registered multiple times
2. Return the already-registered metric instance (cache + registry lookup)
3. Preserve backward compatible function signatures used elsewhere

Use create_counter/create_gauge/create_histogram instead of direct constructors.
"""
import os
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    REGISTRY as DEFAULT_REGISTRY,
    CollectorRegistry,
    PROCESS_COLLECTOR,
    PLATFORM_COLLECTOR,
    GC_COLLECTOR,
)
import logging

log = logging.getLogger(__name__)

# In-process cache to short‑circuit second attempts before hitting the registry
_METRIC_CACHE = {}

# Optional isolated registry support (prevents duplicate metric warnings across forks or multi-service runs)
_ISOLATED = bool(int(os.environ.get('OMEGA_METRICS_ISOLATE', '1')))
if _ISOLATED:
    try:
        _REGISTRY = CollectorRegistry(auto_describe=True)
        # Best-effort registration; ignore if collector objects differ by version
        for collector in (PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR):  # type: ignore
            try:
                _REGISTRY.register(collector)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        _REGISTRY = DEFAULT_REGISTRY
else:
    _REGISTRY = DEFAULT_REGISTRY


def _key(kind: str, name: str, labelnames, buckets=None):
    return (
        kind,
        name,
        tuple(labelnames) if labelnames else tuple(),
        tuple(buckets) if buckets else tuple(),
    )


def _find_existing(registry, name: str):
    """Attempt to locate an already registered collector by metric name."""
    try:
        for collector in list(registry._collector_to_names.keys()):  # type: ignore[attr-defined]
            try:
                names = registry._collector_to_names.get(collector, [])  # type: ignore[attr-defined]
            except Exception:
                continue
            if name in names:
                return collector
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def create_counter(name: str, documentation: str, labelnames=None, registry=None):
    if registry is None:
        registry = _REGISTRY
    key = _key("counter", name, labelnames)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]
    try:
        metric = (
            Counter(name, documentation, labelnames=labelnames, registry=registry)
            if labelnames
            else Counter(name, documentation, registry=registry)
        )
        _METRIC_CACHE[key] = metric
        return metric
    except ValueError:
        existing = _find_existing(registry, name)
        if existing:
            log.debug("Counter %s already registered; returning existing", name)
            _METRIC_CACHE[key] = existing
            return existing
        raise


def create_gauge(name: str, documentation: str, labelnames=None, registry=None):
    if registry is None:
        registry = _REGISTRY
    key = _key("gauge", name, labelnames)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]
    try:
        metric = (
            Gauge(name, documentation, labelnames=labelnames, registry=registry)
            if labelnames
            else Gauge(name, documentation, registry=registry)
        )
        _METRIC_CACHE[key] = metric
        return metric
    except ValueError:
        existing = _find_existing(registry, name)
        if existing:
            log.debug("Gauge %s already registered; returning existing", name)
            _METRIC_CACHE[key] = existing
            return existing
        raise


def create_histogram(name: str, documentation: str, labelnames=None, registry=None, buckets=None):
    if registry is None:
        registry = _REGISTRY
    key = _key("histogram", name, labelnames, buckets)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]
    try:
        metric = (
            Histogram(
                name,
                documentation,
                labelnames=labelnames,
                buckets=buckets,
                registry=registry,
            )
            if labelnames
            else Histogram(name, documentation, buckets=buckets, registry=registry)
        )
        _METRIC_CACHE[key] = metric
        return metric
    except ValueError:
        existing = _find_existing(registry, name)
        if existing:
            log.debug("Histogram %s already registered; returning existing", name)
            _METRIC_CACHE[key] = existing
            return existing
        raise


__all__ = [
    "create_counter",
    "create_gauge",
    "create_histogram",
]
