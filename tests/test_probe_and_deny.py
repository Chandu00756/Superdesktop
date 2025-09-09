import os
import time
import base64
import sqlite3
import hashlib
from fastapi.testclient import TestClient

from backend.api_server import app, SESSION_META, api_server


client = TestClient(app)


def _admin_headers(session_id: str = 'tprobe-admin'):
    key = base64.b64encode(b'3'*32).decode()
    SESSION_META[session_id] = {
        'key': key,
        'user': 'admin',
        'roles': ['admin'],
        'expires_at': time.time() + 3600,
        'counter': 0
    }
    return {'Authorization': 'Bearer x', 'X-Session-ID': session_id}


def _db_cred(node_id: str):
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT token, issued_at, expires_at FROM node_credentials WHERE node_id=?', (node_id,))
        return cur.fetchone()


def _db_revoked_contains(fingerprint: str) -> bool:
    with sqlite3.connect(api_server.database.db_path) as conn:
        cur = conn.execute('SELECT 1 FROM revoked_keys WHERE key_id=?', (fingerprint,))
        return cur.fetchone() is not None


def _reset_db():
    db_path = api_server.database.db_path
    for suffix in ('', '-wal', '-shm'):
        try:
            os.remove(db_path + suffix)
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            pass
    api_server.database.init_database()


def test_probe_requires_node_token_and_succeeds_with_valid():
    _reset_db()
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    headers = _admin_headers('tprobe-1')

    node_id = 'probe-node-1'
    # Register (pending)
    r = client.post('/api/secure/nodes/register', json={
        'node_id': node_id,
        'node_type': 'compute',
        'hostname': 'hn',
        'ip_address': '127.0.0.1',
        'port': 8443,
        'resources': {},
        'device_fingerprint': 'fp',
        'public_key_pem': 'pk',
        'signed_challenge': 'sc'
    }, headers={**headers, 'X-Register-Token': 'secret'})
    assert r.status_code == 200

    # Approve -> issues credential
    r2 = client.post('/api/secure/nodes/approve', json={'node_id': node_id}, headers=headers)
    assert r2.status_code == 200
    tok1, _, _ = _db_cred(node_id)
    assert tok1

    # Probe without token -> 401
    r3 = client.post(f'/api/secure/nodes/{node_id}/probe', headers=headers)
    assert r3.status_code == 401
    # Probe with wrong token -> 401
    r4 = client.post(f'/api/secure/nodes/{node_id}/probe', headers={**headers, 'X-Node-Token': 'wrong'})
    assert r4.status_code == 401
    # Probe with valid token -> 200
    r5 = client.post(f'/api/secure/nodes/{node_id}/probe', headers={**headers, 'X-Node-Token': tok1})
    assert r5.status_code == 200


def test_deny_revokes_credential_and_adds_revoked_fingerprint():
    _reset_db()
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    headers = _admin_headers('tdeny-1')

    node_id = 'deny-node-1'
    public_key_pem = 'pk-test'
    # Register and approve
    r = client.post('/api/secure/nodes/register', json={
        'node_id': node_id,
        'node_type': 'compute',
        'hostname': 'hn',
        'ip_address': '127.0.0.1',
        'port': 8443,
        'resources': {},
        'device_fingerprint': 'fp',
        'public_key_pem': public_key_pem,
        'signed_challenge': 'sc'
    }, headers={**headers, 'X-Register-Token': 'secret'})
    assert r.status_code == 200

    r2 = client.post('/api/secure/nodes/approve', json={'node_id': node_id}, headers=headers)
    assert r2.status_code == 200
    tok, _, _ = _db_cred(node_id)
    assert tok
    # Heartbeat succeeds pre-deny
    ok = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers={**headers, 'X-Node-Token': tok})
    assert ok.status_code == 200

    # Deny the node
    r3 = client.post('/api/secure/nodes/deny', json={'node_id': node_id}, headers=headers)
    assert r3.status_code == 200
    # Token should be revoked and heartbeat should now fail (401 missing/invalid)
    r4 = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers={**headers, 'X-Node-Token': tok})
    assert r4.status_code == 401

    # Fingerprint should be recorded in revoked_keys
    fp = hashlib.sha256(public_key_pem.encode()).hexdigest()
    assert _db_revoked_contains(fp)


def test_get_credential_requires_keys_view_and_quarantine_blocks_probe():
    _reset_db()
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    os.environ['OMEGA_NODE_CRED_MAINT_INTERVAL'] = '1'
    os.environ['OMEGA_NODE_CRED_EXPIRE_QUARANTINE'] = '1'
    headers = _admin_headers('tcred-view-1')

    node_id = 'cred-view-node-1'
    # Register + approve to get credential
    r = client.post('/api/secure/nodes/register', json={
        'node_id': node_id,
        'node_type': 'compute',
        'hostname': 'hn',
        'ip_address': '127.0.0.1',
        'port': 8443,
        'resources': {},
        'device_fingerprint': 'fp',
        'public_key_pem': 'pk',
        'signed_challenge': 'sc'
    }, headers={**headers, 'X-Register-Token': 'secret'})
    assert r.status_code == 200
    r2 = client.post('/api/secure/nodes/approve', json={'node_id': node_id}, headers=headers)
    assert r2.status_code == 200
    tok, _, _ = _db_cred(node_id)
    assert tok

    # Heartbeat OK with token
    ok = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers={**headers, 'X-Node-Token': tok})
    assert ok.status_code == 200

    # GET credential should pass for admin (has keys:view via role)
    g1 = client.get(f'/api/secure/nodes/{node_id}/credential', headers=headers)
    assert g1.status_code == 200

    # Rotate with short TTL to force expiry and background quarantine
    rot = client.post('/api/secure/nodes/credential/rotate', json={'node_id': node_id, 'ttl_seconds': 1}, headers=headers)
    assert rot.status_code == 200
    time.sleep(2)
    # Probe should now fail due to expired credential and quarantine
    rbad = client.post(f'/api/secure/nodes/{node_id}/probe', headers={**headers, 'X-Node-Token': tok})
    assert rbad.status_code in (401, 403)


def test_get_credential_forbidden_without_keys_view():
    _reset_db()
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    headers_admin = _admin_headers('tcred-view-2')

    node_id = 'cred-view-node-2'
    # Register only (no need to approve for permission check on GET credential; but approving makes sure a record exists)
    r = client.post('/api/secure/nodes/register', json={
        'node_id': node_id,
        'node_type': 'compute',
        'hostname': 'hn',
        'ip_address': '127.0.0.1',
        'port': 8443,
        'resources': {},
        'device_fingerprint': 'fp',
        'public_key_pem': 'pk',
        'signed_challenge': 'sc'
    }, headers={**headers_admin, 'X-Register-Token': 'secret'})
    assert r.status_code == 200
    r2 = client.post('/api/secure/nodes/approve', json={'node_id': node_id}, headers=headers_admin)
    assert r2.status_code == 200

    # Create a session for a non-privileged user without keys:view
    sid = 'noview'
    key = base64.b64encode(b'4'*32).decode()
    SESSION_META[sid] = {
        'key': key,
        'user': 'viewerUser',
        'roles': ['viewer'],
        'expires_at': time.time() + 3600,
        'counter': 0
    }
    headers_viewer = {'Authorization': 'Bearer x', 'X-Session-ID': sid}
    g = client.get(f'/api/secure/nodes/{node_id}/credential', headers=headers_viewer)
    assert g.status_code == 403
