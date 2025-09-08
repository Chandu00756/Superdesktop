import os
import time
import base64
import json
import pytest
from fastapi.testclient import TestClient
from backend.api_server import app, SESSION_META

CLIENT = TestClient(app)

@pytest.fixture(autouse=True)
def setup_session():
    # Create a fake session meta with admin user/roles
    sid = 'testsession'
    key = base64.b64encode(b'0'*32).decode()
    SESSION_META[sid] = {'key': key, 'user': 'admin', 'roles': ['admin'], 'expires_at': time.time() + 3600, 'counter':0}
    yield
    SESSION_META.pop(sid, None)

def test_secure_register_missing_token():
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    r = CLIENT.post('/api/secure/nodes/register', json={
        'node_id':'n1','node_type':'compute','hostname':'h','ip_address':'127.0.0.1','port':8443,
        'resources':{},'device_fingerprint':'df','public_key_pem':'pk','signed_challenge':'sc'} , headers={'Authorization':'Bearer x','X-Session-ID':'testsession'})
    assert r.status_code == 401

def test_secure_register_success():
    os.environ['NODE_REGISTRATION_TOKEN'] = 'secret'
    r = CLIENT.post('/api/secure/nodes/register', json={
        'node_id':'n2','node_type':'compute','hostname':'h2','ip_address':'127.0.0.1','port':8443,
        'resources':{},'device_fingerprint':'df2','public_key_pem':'pk2','signed_challenge':'sc2'}, headers={'Authorization':'Bearer x','X-Session-ID':'testsession','X-Register-Token':'secret'})
    assert r.status_code == 200
    assert r.json()['status'] == 'pending_approval'
