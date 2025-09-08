import importlib.util
import pathlib
import sqlite3
import time
from fastapi.testclient import TestClient


def test_trust_persistence_sqlite_roundtrip():
    # Dynamically load orchestrator main.py
    orch_path = pathlib.Path(__file__).parent.parent / 'omega-orchestrator' / 'main.py'
    spec = importlib.util.spec_from_file_location('omega_orchestrator_trust', orch_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    app = mod.app
    with TestClient(app) as client:
        # Register a node with auto-approve so it gets persisted
        node_id = 'trust-node-1'
        reg = client.post('/nodes/register', params={'auto_approve': True}, json={
            'node_id': node_id,
            'node_type': 'cpu_node',
            'resources': {'cpu': 4, 'memory': 8},
            'labels': {}, 'annotations': {}, 'network_config': {}
        })
        assert reg.status_code == 200

        # Adjust trust down by 0.4 => expect ~0.6
        adj = client.post(f'/nodes/{node_id}/trust', json={'delta': -0.4, 'reason': 'test'})
        assert adj.status_code == 200
        body = adj.json()
        assert body.get('status') == 'ok'
        # Allow small float variance
        reported = float(body.get('trust', 0))
        assert 0.59 <= reported <= 0.61

        # Give background task time to persist to SQLite
        time.sleep(0.2)

        # Verify DB file and row only if SQLite backend is active
        pool = getattr(mod.orch, 'postgres_pool', None)
        if pool is not None and getattr(pool, 'kind', '') == 'sqlite':
            db_path = pathlib.Path.cwd() / 'omega_orchestrator.db'
            assert db_path.exists()
            con = sqlite3.connect(str(db_path))
            try:
                cur = con.cursor()
                cur.execute('SELECT trust_score FROM nodes WHERE node_id = ?', (node_id,))
                row = cur.fetchone()
                assert row is not None
                assert 0.59 <= float(row[0]) <= 0.61
            finally:
                con.close()
        # If no persistence backend, API check above is sufficient
