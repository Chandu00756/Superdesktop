import os, time, json, base64, hashlib
import pytest
from fastapi.testclient import TestClient

from backend.api_server import app, SESSION_META

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_sessions():
    # ensure clean slate each test
    yield
    # prune test sessions
    for k in list(SESSION_META.keys()):
        if k.startswith('tsec-'):
            SESSION_META.pop(k, None)


def _make_session(session_id: str, user: str, roles: list[str]):
    key = base64.b64encode(b'1'*32).decode()
    SESSION_META[session_id] = {
        'key': key,
        'user': user,
        'roles': roles,
        'expires_at': time.time()+3600,
        'counter': 0
    }
    return {'Authorization':'Bearer test','X-Session-ID': session_id}


def test_policy_deny_precedence():
    # admin creates a deny policy for nodes:view on user 'alice'
    admin_headers = _make_session('tsec-admin', 'admin', ['admin'])
    user_headers = _make_session('tsec-alice', 'alice', ['user'])

    policy_raw = json.dumps({
        'version': '1',
        'statements': [
            {'effect':'deny','actions':['nodes:view'],'resources':['*']}
        ]
    })
    r = client.post('/secure/policy/upsert', json={
        'policy_id':'deny-nodes-view','name':'deny nodes','kind':'json','spec':{},'raw':policy_raw
    }, headers=admin_headers)
    assert r.status_code == 200, r.text
    r2 = client.post('/secure/policy/assign', json={'policy_id':'deny-nodes-view','target_type':'user','target_id':'alice'}, headers=admin_headers)
    assert r2.status_code == 200, r2.text
    # Access should now be denied even though role 'user' grants nodes:view
    r3 = client.get('/api/secure/nodes', headers=user_headers)
    assert r3.status_code == 403, r3.text
    body = r3.json()
    assert body.get('detail',{}).get('error') == 'policy_denied'


def test_invalid_policy_schema():
    admin_headers = _make_session('tsec-admin2', 'admin', ['admin'])
    bad_raw = json.dumps({'statements': []})  # missing version and empty statements
    r = client.post('/secure/policy/upsert', json={
        'policy_id':'bad1','name':'bad','kind':'json','spec':{},'raw':bad_raw
    }, headers=admin_headers)
    assert r.status_code == 400


def test_missing_permission_denied():
    # user without any roles tries to list nodes
    headers = _make_session('tsec-norole','bob', [])
    r = client.get('/api/secure/nodes', headers=headers)
    assert r.status_code == 403
    body = r.json()
    assert body.get('detail',{}).get('error') in ('permission_denied','policy_denied')


def test_node_join_nonce_replay():
    import uuid
    admin_headers = _make_session('tsec-admin3','admin',['admin'])
    os.environ['ENROLLMENT_SHARED_SECRET'] = 'join-secret'
    nonce = uuid.uuid4().hex
    payload = {
        'node_id':'join-node-1',
        'capabilities': {'cpu':4},
        'nonce': nonce,
        'timestamp': time.time(),
    }
    canonical = json.dumps({'node_id':payload['node_id'],'nonce':payload['nonce'],'timestamp':payload['timestamp'],'capabilities':payload['capabilities']}, sort_keys=True, separators=(',',':'))
    payload['signature'] = hashlib.sha256((canonical+'join-secret').encode()).hexdigest()
    r1 = client.post('/secure/nodes/join', json=payload, headers=admin_headers)
    assert r1.status_code == 200, r1.text
    # replay attempt using the same nonce and identical payload
    r2 = client.post('/secure/nodes/join', json=payload, headers=admin_headers)
    assert r2.status_code == 400
    assert 'replay' in r2.json()['detail']

