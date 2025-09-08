import json, os, time, sqlite3
from fastapi.testclient import TestClient
from backend.api_server import app, api_server, generate_session_key, encrypt_aes_gcm, register_session_meta, SESSION_META

def bootstrap_session():
    # Create a session meta with admin role + rbac:manage permission via role mapping (admin has all)
    session_id = 'test-session-rbac'
    key = generate_session_key()
    import base64
    register_session_meta(session_id, base64.b64encode(key).decode())
    # attach roles and user
    SESSION_META[session_id]['user'] = 'admin'
    SESSION_META[session_id]['roles'] = ['admin']
    return session_id, key

client = TestClient(app)

def test_rbac_matrix_endpoint():
    session_id, key = bootstrap_session()
    token = 'Bearer placeholder'  # Authorization required for validate_secure
    # ensure permissions exist (bootstrap already done in db init)
    resp = client.get('/secure/rbac/matrix', headers={'X-Session-ID': session_id, 'Authorization': token})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 'roles' in data and 'permissions' in data
    assert 'admin' in data['roles']
    assert any(p.startswith('dashboard:') for p in data['permissions'].keys())

def test_create_update_delete_role():
    session_id, key = bootstrap_session()
    token = 'Bearer placeholder'
    # create role
    resp = client.post('/secure/rbac/roles', json={'role':'qa','description':'QA role','permissions':['dashboard:view']}, headers={'X-Session-ID': session_id, 'Authorization': token})
    assert resp.status_code == 200, resp.text
    # update role
    resp = client.put('/secure/rbac/roles/qa', json={'description':'QA updated','permissions':['dashboard:view','resources:view']}, headers={'X-Session-ID': session_id, 'Authorization': token})
    assert resp.status_code == 200, resp.text
    # delete role
    resp = client.delete('/secure/rbac/roles/qa', headers={'X-Session-ID': session_id, 'Authorization': token})
    assert resp.status_code == 200, resp.text
