import os
import time
import base64
import sqlite3
from fastapi.testclient import TestClient

from backend.api_server import app, SESSION_META, api_server


client = TestClient(app)


def _admin_headers(session_id: str = 'tcred-admin'):
    key = base64.b64encode(b'2'*32).decode()
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


def test_credential_enforced_on_heartbeat_and_rotate_and_expire():
    # Start with a clean DB to avoid corruption from previous runs
    db_path = api_server.database.db_path
    for suffix in ('', '-wal', '-shm'):
        try:
            os.remove(db_path + suffix)
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            pass
    # Recreate schema
    api_server.database.init_database()

    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    os.environ['OMEGA_NODE_CRED_MAINT_INTERVAL'] = '1'
    os.environ['OMEGA_NODE_CRED_EXPIRE_QUARANTINE'] = '1'
    headers = _admin_headers('tcred-1')

    # Register node (pending)
    node_id = 'cred-node-1'
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
    # Fetch token directly from DB
    tok1, issued_at, exp1 = _db_cred(node_id)
    assert tok1 is not None

    # Heartbeat without token should fail
    r3 = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers=headers)
    assert r3.status_code == 401

    # Heartbeat with wrong token should fail
    bad_headers = {**headers, 'X-Node-Token': 'bad'}
    r4 = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers=bad_headers)
    assert r4.status_code == 401

    # Heartbeat with correct token passes
    good_headers = {**headers, 'X-Node-Token': tok1}
    r5 = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers=good_headers)
    assert r5.status_code == 200

    # Rotate credential -> token should change
    r6 = client.post('/api/secure/nodes/credential/rotate', json={'node_id': node_id}, headers=headers)
    assert r6.status_code == 200
    tok2, _, _ = _db_cred(node_id)
    assert tok2 != tok1

    # Expire quickly and verify 401
    r7 = client.post('/api/secure/nodes/credential/rotate', json={'node_id': node_id, 'ttl_seconds': 1}, headers=headers)
    assert r7.status_code == 200
    time.sleep(2)
    tok3, _, exp3 = _db_cred(node_id) or (None, None, None)
    # background maint may have removed credential after expiry; in either case heartbeat should be 401
    cur_headers = {**headers}
    if tok3:
        cur_headers['X-Node-Token'] = tok3
    r8 = client.post('/api/secure/nodes/heartbeat', json={'node_id': node_id, 'status': 'online'}, headers=cur_headers)
    assert r8.status_code == 401
