import time
import base64
import os
from fastapi.testclient import TestClient
from backend.api_server import app, SESSION_META, api_server
import sqlite3


client = TestClient(app)


def _headers(user: str, roles: list[str], sid: str):
    key = base64.b64encode(b'5'*32).decode()
    SESSION_META[sid] = {
        'key': key,
        'user': user,
        'roles': roles,
        'expires_at': time.time() + 3600,
        'counter': 0
    }
    return {'Authorization': 'Bearer x', 'X-Session-ID': sid}


def _grant_role(username: str, role: str):
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS roles (role TEXT PRIMARY KEY, description TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS user_roles (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, UNIQUE(username, role))')
        conn.execute('INSERT OR IGNORE INTO roles (role, description) VALUES (?,?)', (role, role))
        conn.execute('INSERT OR IGNORE INTO user_roles (username, role) VALUES (?,?)', (username, role))
        conn.commit()
    # Force RBAC cache reload
    import backend.api_server as srv
    srv.RBAC_LAST_LOAD = 0


def _ensure_permission_for_role(role: str, perm: str):
    with sqlite3.connect(api_server.database.db_path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS permissions (code TEXT PRIMARY KEY, description TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS role_permissions (role TEXT NOT NULL, permission_code TEXT NOT NULL, PRIMARY KEY(role, permission_code))')
        conn.execute('INSERT OR IGNORE INTO permissions (code, description) VALUES (?,?)', (perm, perm))
        conn.execute('INSERT OR IGNORE INTO role_permissions (role, permission_code) VALUES (?,?)', (role, perm))
        conn.commit()
    import backend.api_server as srv
    srv.RBAC_LAST_LOAD = 0


def test_latency_measure_requires_nodes_view():
    _grant_role('v1', 'viewer')
    _ensure_permission_for_role('viewer', 'nodes:view')
    h_viewer = _headers('v1', ['viewer'], 'rbac-v1')
    # viewer has nodes:view, should pass auth but still needs session headers
    r1 = client.get('/api/secure/nodes/latency/measure', params={'node_id': 'n1'}, headers=h_viewer)
    assert r1.status_code == 200

    # Create a user without nodes:view by giving no roles
    # user without any roles/permissions
    h_none = _headers('u2', [], 'rbac-u2')
    r2 = client.get('/api/secure/nodes/latency/measure', params={'node_id': 'n1'}, headers=h_none)
    assert r2.status_code == 403


def test_backup_permissions():
    # viewer should be able to list snapshots but not create
    _grant_role('v2', 'viewer')
    _ensure_permission_for_role('viewer', 'backup:view')
    h_viewer = _headers('v2', ['viewer'], 'rbac-v2')
    r_list = client.get('/secure/backup/snapshots', headers=h_viewer)
    assert r_list.status_code == 200
    r_create = client.post('/secure/backup/snapshot', json={'label': 't'}, headers=h_viewer)
    assert r_create.status_code == 403

    # admin can create
    h_admin = _headers('admin', ['admin'], 'rbac-admin')
    r_create2 = client.post('/secure/backup/snapshot', json={'label': 't2'}, headers=h_admin)
    assert r_create2.status_code == 200


def test_nodes_view_required_for_secure_actions_and_gets():
    # grant viewer nodes:view
    _grant_role('v3', 'viewer')
    _ensure_permission_for_role('viewer', 'nodes:view')
    h = _headers('v3', ['viewer'], 'rbac-v3')

    # endpoints should succeed with nodes:view
    r1 = client.post('/api/secure/discover', headers=h, json={})
    assert r1.status_code == 200

    r2 = client.get('/api/secure/nodes/test-node/protocol/health', headers=h)
    assert r2.status_code == 200

    r3 = client.post('/api/secure/nodes/latency/start', headers=h, json={'node_id': 'n1'})
    assert r3.status_code == 200

    r4 = client.get('/api/secure/nodes/test-node/telemetry', headers=h)
    assert r4.status_code == 200

    r5 = client.get('/api/secure/nodes/test-node/policies', headers=h)
    assert r5.status_code == 200

    r6 = client.post('/api/secure/nodes/test-node/policies', headers=h, json={'key': 'k', 'value': 'v'})
    assert r6.status_code == 200

    r7 = client.delete('/api/secure/nodes/test-node/policies/k', headers=h)
    assert r7.status_code == 200

    r8 = client.get('/api/secure/nodes/test-node/trust', headers=h)
    assert r8.status_code == 200

    r9 = client.post('/api/secure/nodes/test-node/diagnostics', headers=h, json={'level': 'basic'})
    assert r9.status_code == 200

    r10 = client.post('/api/secure/nodes/test-node/ota', headers=h, json={'action': 'update', 'version': '1.0'})
    assert r10.status_code == 200

    # user with no roles should get 403 for these endpoints
    h2 = _headers('u3', [], 'rbac-u3')
    for method, url, payload in [
        ('POST','/api/secure/discover', {}),
        ('GET','/api/secure/nodes/test-node/protocol/health', None),
        ('POST','/api/secure/nodes/latency/start', {'node_id':'n1'}),
        ('GET','/api/secure/nodes/test-node/telemetry', None),
        ('GET','/api/secure/nodes/test-node/policies', None),
        ('POST','/api/secure/nodes/test-node/policies', {'key':'k','value':'v'}),
        ('DELETE','/api/secure/nodes/test-node/policies/k', None),
        ('GET','/api/secure/nodes/test-node/trust', None),
        ('POST','/api/secure/nodes/test-node/diagnostics', {'level':'basic'}),
        ('POST','/api/secure/nodes/test-node/ota', {'action':'update'})
    ]:
        if method == 'GET':
            resp = client.get(url, headers=h2)
        elif method == 'POST':
            resp = client.post(url, headers=h2, json=payload)
        elif method == 'DELETE':
            resp = client.delete(url, headers=h2)
        else:
            raise AssertionError('unsupported method in test')
        assert resp.status_code == 403


def test_rbac_for_dashboards_and_system_views():
    # Prepare viewer role with specific permissions and verify access
    endpoints_perms = [
        ('/api/secure/dashboard', 'dashboard:view', 'GET', None),
        ('/api/secure/resources', 'resources:view', 'GET', None),
        ('/api/secure/network', 'network:view', 'GET', None),
        ('/api/secure/performance', 'performance:view', 'GET', None),
        ('/api/secure/plugins', 'plugins:view', 'GET', None),
        ('/api/secure/security', 'security:view', 'GET', None),
        ('/api/secure/nodes', 'nodes:view', 'GET', None),
        ('/api/secure/logs', 'logs:view', 'GET', None),
        ('/api/secure/processes', 'processes:view', 'GET', None),
    ]

    # Single role with accumulating permissions
    _grant_role('viewer_all', 'viewer')
    h_ok = _headers('viewer_all', ['viewer'], 'rbac-v4')

    for url, perm, method, payload in endpoints_perms:
        _ensure_permission_for_role('viewer', perm)
        if method == 'GET':
            r = client.get(url, headers=h_ok)
        else:
            r = client.post(url, headers=h_ok, json=payload or {})
        assert r.status_code == 200, f"expected 200 for {url} with perm {perm}, got {r.status_code}"

    # User without permissions should get 403 for each
    h_none = _headers('no_roles', [], 'rbac-noroles')
    for url, perm, method, payload in endpoints_perms:
        if method == 'GET':
            r = client.get(url, headers=h_none)
        else:
            r = client.post(url, headers=h_none, json=payload or {})
        assert r.status_code == 403, f"expected 403 for {url} without perm {perm}, got {r.status_code}"
