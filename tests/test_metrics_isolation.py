import os


def test_metrics_isolation_registry_flag_and_dedup():
    # Ensure isolation is on by default (as per utils.metrics)
    os.environ.pop('OMEGA_METRICS_ISOLATE', None)

    import importlib
    m = importlib.import_module('utils.metrics')

    # Create metrics with same name twice; should not raise and should return same object
    c1 = m.create_counter('omega_test_counter', 'doc', labelnames=['a'])
    c2 = m.create_counter('omega_test_counter', 'doc', labelnames=['a'])
    assert c1 is c2

    # Gauge duplicate
    g1 = m.create_gauge('omega_test_gauge', 'doc')
    g2 = m.create_gauge('omega_test_gauge', 'doc')
    assert g1 is g2

    # Histogram duplicate with buckets
    h1 = m.create_histogram('omega_test_hist', 'doc', buckets=[0.1, 0.5, 1.0])
    h2 = m.create_histogram('omega_test_hist', 'doc', buckets=[0.1, 0.5, 1.0])
    assert h1 is h2

    # Now force global registry and ensure still dedupes via lookup
    os.environ['OMEGA_METRICS_ISOLATE'] = '0'
    importlib.reload(m)
    c3 = m.create_counter('omega_test_counter2', 'doc')
    c4 = m.create_counter('omega_test_counter2', 'doc')
    assert c3 is c4
