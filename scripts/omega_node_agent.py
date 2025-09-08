#!/usr/bin/env python3
"""Omega Node Agent
Performs secure self-registration with control backend.
Steps:
 1. Generate RSA keypair (persist locally).
 2. Derive device fingerprint (SHA256 of CPU+MAC+hostname).
 3. Fetch control public key, establish secure session (encrypt random AES key).
 4. Sign challenge (node_id + fingerprint) with node private key.
 5. POST /api/secure/nodes/register with attestation fields.
 6. Periodically send heartbeat.
Environment:
  OMEGA_CONTROL_URL (default http://127.0.0.1:8443)
  OMEGA_NODE_ID (default auto hostname based)
  OMEGA_NODE_TYPE (default compute)
"""
import os, time, base64, json, hashlib, socket, threading
try:
    import psutil  # optional metrics
except Exception:
    psutil = None
import requests
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CONTROL=os.environ.get('OMEGA_CONTROL_URL','http://127.0.0.1:8443')
NODE_ID=os.environ.get('OMEGA_NODE_ID', socket.gethostname().replace('.','-'))
NODE_TYPE=os.environ.get('OMEGA_NODE_TYPE','compute')
STATE_DIR=Path(os.environ.get('OMEGA_NODE_STATE','./node_state'))
STATE_DIR.mkdir(parents=True, exist_ok=True)
KEY_PATH=STATE_DIR/'node_key.pem'

# 1. Keypair
if KEY_PATH.exists():
    priv = serialization.load_pem_private_key(KEY_PATH.read_bytes(), password=None)
else:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    KEY_PATH.write_bytes(priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))

pub_pem = priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

# 2. Fingerprint (simple)
def device_fingerprint():
    host=socket.gethostname()
    try:
        macs=[]
        import uuid
        macs.append(hex(uuid.getnode()))
    except Exception:
        macs=['NA']
    cpu=os.cpu_count()
    raw=f"{host}|{','.join(macs)}|{cpu}".encode()
    return hashlib.sha256(raw).hexdigest()

fingerprint=device_fingerprint()

# 3. Secure session
r=requests.get(f"{CONTROL}/api/secure/public_key", timeout=5)
r.raise_for_status()
control_pub = serialization.load_pem_public_key(r.json()['public_key_pem'].encode())
session_key=os.urandom(32)
enc_key=control_pub.encrypt(session_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
rs=requests.post(f"{CONTROL}/api/secure/session/start", json={'encrypted_key':base64.b64encode(enc_key).decode()})
rs.raise_for_status()
hand=rs.json()
TOKEN=hand['token']; SID=hand['session_id']
headers={'Authorization':f'Bearer {TOKEN}','X-Session-ID':SID,'Content-Type':'application/json'}

def encrypt(payload:dict):
    iv=os.urandom(12)
    aes=AESGCM(session_key)
    pt=json.dumps(payload).encode(); ct=aes.encrypt(iv, pt, None)
    return {'alg':'AES-256-GCM','iv':base64.b64encode(iv).decode(),'ciphertext':base64.b64encode(ct[:-16]).decode(),'tag':base64.b64encode(ct[-16:]).decode()}

# 4. Sign challenge
challenge=(NODE_ID+fingerprint).encode()
signature=priv.sign(challenge, padding.PKCS1v15(), hashes.SHA256())

# 5. Register (encrypted channel expectations: server expects plaintext body but returns encrypted wrapper) => send JSON via secure register endpoint
reg_body={
  'node_id':NODE_ID,'node_type':NODE_TYPE,'hostname':NODE_ID,'ip_address':os.environ.get('OMEGA_NODE_IP','127.0.0.1'),'port':int(os.environ.get('OMEGA_NODE_PORT','8000')),
  'resources':{'cpu_cores':os.cpu_count(),'memory_gb':round((os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'))/1024**3,2) if hasattr(os,'sysconf') else 0},
  'permissions':[], 'description':None,
  'device_fingerprint':fingerprint,
  'public_key_pem':pub_pem,
  'signed_challenge':base64.b64encode(signature).decode(),
  'health_attestation':None,'device_certificate':None,'geoip':None,'behavioral_baseline':{'cpu_cores':os.cpu_count(),'memory_gb':4}
}
reg=requests.post(f"{CONTROL}/api/secure/nodes/register", headers=headers, json=reg_body, timeout=10)
print('Register status', reg.status_code)
print('Register resp (encrypted?)', reg.text[:160])

# 6. Heartbeat thread
stop_flag=False

def heartbeat_loop():
    while not stop_flag:
        try:
            hb={'node_id':NODE_ID,'status':'online'}
            if psutil:
                try:
                    hb['cpu_usage']=float(psutil.cpu_percent(interval=0.2))
                    hb['memory_usage']=float(psutil.virtual_memory().percent)
                except Exception:
                    pass
            requests.post(f"{CONTROL}/api/secure/nodes/heartbeat", headers=headers, json=hb, timeout=5)
        except Exception as e:
            print('Heartbeat error', e)
        time.sleep(20)

threading.Thread(target=heartbeat_loop, daemon=True).start()

print('Node agent running. Press Ctrl+C to exit.')
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    stop_flag=True
    print('Exiting')
