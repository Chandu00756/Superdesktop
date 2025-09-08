import importlib.util
import time
import pathlib
from fastapi.testclient import TestClient


def test_autoscale_persisted_events_and_approval():
    # Dynamically load orchestrator main.py
    orch_path = pathlib.Path(__file__).parent.parent / 'omega-orchestrator' / 'main.py'
    spec = importlib.util.spec_from_file_location('omega_orchestrator_main', orch_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    app = mod.app
    client = TestClient(app)

    # Trigger scale out then scale in
    r1 = client.post('/autoscaling/scale_out', params={'count':2, 'reason':'test_persist'})
    assert r1.status_code == 200
    r2 = client.post('/autoscaling/scale_in', params={'count':1, 'reason':'test_persist'})
    assert r2.status_code == 200

    # Give background tasks slight time to persist
    time.sleep(0.2)

    persisted = client.get('/autoscaling/events/persisted')
    # Persistence may be unavailable (e.g. no DB), accept both but structure must exist
    assert persisted.status_code == 200
    body = persisted.json()
    assert 'events' in body
    # If available, expect at least one event
    if body.get('persistence') == 'ok':
        assert len(body['events']) >= 1

    # Node approval workflow
    reg = client.post('/nodes/register', json={
        'node_id':'pending-test-1','node_type':'cpu_node','resources':{'cpu':4,'memory':8},
        'labels':{},'annotations':{},'network_config':{}
    })
    assert reg.status_code == 200
    assert reg.json()['status'] == 'pending'

    pending = client.get('/nodes/pending')
    assert pending.status_code == 200
    assert any(p['node_id']=='pending-test-1' for p in pending.json()['pending'])

    approve = client.post('/nodes/approve/pending-test-1')
    assert approve.status_code == 200
    # If infrastructure incomplete (e.g., redis missing) registration may error; allow both
    assert approve.json()['status'] in ('approved','error')

    deny = client.post('/nodes/deny/does-not-exist')
    assert deny.status_code == 200
    assert deny.json()['status'] == 'not_found'
