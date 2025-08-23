"""
Safe Prometheus metrics helper.

Provides a create_counter function that avoids duplicate registration errors
and handles slight API differences between prometheus_client versions.
"""
from prometheus_client import CollectorRegistry, Counter
from prometheus_client import REGISTRY as DEFAULT_REGISTRY
import logging

log = logging.getLogger(__name__)


def create_counter(name: str, documentation: str, labelnames=None, registry=None):
    """Create or return an existing Counter safely.

    - name: metric name
    - documentation: help text
    - labelnames: list of label names, optional
    - registry: CollectorRegistry instance (defaults to global)
    """
    if registry is None:
        registry = DEFAULT_REGISTRY

    labelnames = labelnames or []

    # If a metric with the same name already exists, return it instead of creating a new one
    try:
        # prometheus_client keeps collectors in registry._collector_to_names (internal),
        # but we can catch ValueError on duplicate registration instead of introspecting.
        if labelnames:
            return Counter(name, documentation, labelnames, registry=registry)
        else:
            return Counter(name, documentation, registry=registry)
    except ValueError as ve:
        # Duplicate registered metric, try to find and return it
        log.warning("Metric %s already registered, returning existing instance", name)
        # Search registry for matching collector
        for collector in list(registry._collector_to_names.keys()):
            try:
                names = registry._collector_to_names.get(collector, [])
            except Exception:
                names = []
            if name in names:
                # collector might be a Counter instance
                try:
                    return collector
                except Exception:
                    continue

        # If we couldn't find it, re-raise the original error
        raise


def create_gauge(name: str, documentation: str, labelnames=None, registry=None):
    """Create or return an existing Gauge safely."""
    from prometheus_client import Gauge
    if registry is None:
        registry = DEFAULT_REGISTRY
    labelnames = labelnames or []
    try:
        if labelnames:
            return Gauge(name, documentation, labelnames, registry=registry)
        else:
            return Gauge(name, documentation, registry=registry)
    except ValueError:
        log.warning("Metric %s already registered, returning existing instance", name)
        for collector in list(registry._collector_to_names.keys()):
            try:
                names = registry._collector_to_names.get(collector, [])
            except Exception:
                names = []
            if name in names:
                return collector
        raise


def create_histogram(name: str, documentation: str, labelnames=None, registry=None, buckets=None):
    """Create or return an existing Histogram safely."""
    from prometheus_client import Histogram
    if registry is None:
        registry = DEFAULT_REGISTRY
    labelnames = labelnames or []
    try:
        if labelnames:
            return Histogram(name, documentation, labelnames, registry=registry, buckets=buckets) if buckets else Histogram(name, documentation, labelnames, registry=registry)
        else:
            return Histogram(name, documentation, registry=registry, buckets=buckets) if buckets else Histogram(name, documentation, registry=registry)
    except ValueError:
        log.warning("Metric %s already registered, returning existing instance", name)
        for collector in list(registry._collector_to_names.keys()):
            try:
                names = registry._collector_to_names.get(collector, [])
            except Exception:
                names = []
            if name in names:
                return collector
        raise
