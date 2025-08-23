// Secure API client using AES-GCM session handshake and encrypted responses
export class ApiClient {
  constructor(base) {
    // Determine API base automatically: prefer same-origin backend at port 8443 using current page protocol.
    const pageProto = (typeof location !== 'undefined' ? location.protocol : 'http:');
    const proto = pageProto === 'https:' ? 'https' : 'http';
    const auto = `${proto}://127.0.0.1:8443`;
    const finalBase = base || window.OMEGA_CONFIG?.FALLBACK_API_URL || auto;
    this.base = finalBase.replace(/\/$/, '');
  this.token = null;
  this.sessionId = null;
  this.sessionKey = null; // base64 string (kept in sessionStorage only)
  this.ready = this.initialize();
  this._actionCtr = 0;
  this._polling = false;
  this._storagePrefix = 'omega_session';
  }

  async initialize() {
    // Recover persisted session if available
    try {
      const saved = localStorage.getItem(this._storagePrefix);
      if (saved) {
        const obj = JSON.parse(saved);
        if (obj && obj.sessionId && obj.token) {
          this.sessionId = obj.sessionId;
          this.token = obj.token;
          this._actionCtr = obj.ctr || 0;
          // sessionKey is intentionally stored in sessionStorage only (per-tab memory)
          const key = sessionStorage.getItem(`${this._storagePrefix}_key:${this.sessionId}`);
          if (key) {
            this.sessionKey = key;
            return;
          }
          // no sessionKey available in this tab; fall through to full handshake
        }
      }
    } catch (e) {
      // ignore
    }
    // Check WebCrypto support early
    this.cryptoAvailable = (typeof window !== 'undefined' && window.crypto && window.crypto.subtle);
    if (!this.cryptoAvailable) {
      console.error('[ApiClient] Web Crypto API not available in this environment; secure transport will not function');
    }
    // 1) Authenticate (reuse static admin for now)
    await this._login();
    // 2) Establish secure session
    await this._handshake();
    // 3) Start realtime WS to receive server push notifications (rekey requests, etc.)
    try {
      this._startRealtime();
    } catch (e) {
      // Non-fatal
    }
    // Start polling fallback in case WS is unavailable
    try { this._startPoll(); } catch (e) {}
  }

  _startRealtime() {
    try {
      if (!this.sessionId) return;
      const proto = (location && location.protocol === 'https:') ? 'wss' : 'ws';
      const host = (new URL(this.base)).host;
      const wsUrl = `${proto}://${host}/ws/secure/realtime?session_id=${encodeURIComponent(this.sessionId)}`;
      this._ws = new WebSocket(wsUrl);
      this._ws.addEventListener('open', () => {
        // Ensure polling stops when WS is active
        try { this._stopPoll(); } catch (e) {}
      });
      this._ws.addEventListener('message', async (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data && data.type === 'rekey_request') {
            // Server asks client to rotate its AES key. Perform client-initiated rotate.
            try {
              await this.rotateSessionKey();
              // Optionally notify UI
              console.info('[ApiClient] Performed automatic session rotate on server request');
            } catch (e) {
              console.error('[ApiClient] Automatic rotate failed:', e);
            }
          }
        } catch (e) {
          // ignore
        }
      });
  this._ws.addEventListener('close', () => { this._ws = null; this._startPoll(); });
  this._ws.addEventListener('error', (e) => { /* ignore */ });
    } catch (e) {
      // ignore failures
    }
  }

  _startPoll(interval = 15000) {
    if (this._pollTimer || this._ws) return;
    try {
      this._pollTimer = setInterval(async () => {
        if (!this.sessionId) return;
        try {
          const res = await fetch(`${this.base}/api/secure/session/poll`, { method: 'POST', headers: this._headers({ 'Content-Type':'application/json' }), body: JSON.stringify({ session_id: this.sessionId }) });
          if (!res.ok) return;
          const data = await res.json();
          if (data.status === 'rekey_pending') {
            try {
              window.notify && window.notify('info','Key rotation requested','Server requested key rotation for this session');
              await this.rotateSessionKey();
              window.notify && window.notify('success','Key rotated','Session key rotated successfully');
            } catch (e) {
              window.notify && window.notify('error','Rotation failed',String(e));
            }
          } else if (data.status === 'revoked') {
            window.notify && window.notify('error','Session revoked','This session has been revoked by an administrator');
            // clear local session and force reload
            this.clearSession();
            try { this._stopPoll(); } catch (e) {}
          }
        } catch (e) {
          // ignore transient poll errors
        }
      }, interval);
    } catch (e) {}
  }

  _stopPoll() {
  try { if (this._pollTimer) { clearInterval(this._pollTimer); this._pollTimer = null; } } catch (e) {}
  }

  clearSession() {
  try { localStorage.removeItem(this._storagePrefix); } catch (e){}
  try { if (this.sessionId) sessionStorage.removeItem(`${this._storagePrefix}_key:${this.sessionId}`); } catch (e) {}
  this.sessionId = null; this.sessionKey = null; this.token = null; this._actionCtr = 0;
  try { if (this._ws) this._ws.close(); } catch (e) {}
  }

  async _login() {
    const res = await fetch(`${this.base}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'omega123' })
    });
    if (!res.ok) throw new Error(`Auth failed: ${res.status}`);
    const data = await res.json();
    // Normalize token: server may return encrypted token object or raw string. Keep token as a compact string.
    if (data?.token) {
      const tok = typeof data.token === 'string' ? data.token : btoa(JSON.stringify(data.token));
      this.token = `Bearer ${tok}`;
    } else {
      this.token = 'Bearer bootstrap';
    }
  }

  async _handshake() {
    // 1) fetch server RSA public key
    const pubRes = await fetch(`${this.base}/api/secure/public_key`);
    if (!pubRes.ok) throw new Error('Failed to fetch server public key');
    const pubData = await pubRes.json();
    const pem = pubData.public_key_pem;
    // Import server RSA public key
    const spki = this._pemToArrayBuffer(pem);
    const pubKey = await crypto.subtle.importKey('spki', spki, { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['encrypt']);
    // 2) generate AES-256-GCM key locally
    const rawKey = crypto.getRandomValues(new Uint8Array(32));
    this.sessionKey = this._bytesToB64(rawKey);
    // 3) encrypt AES key with server RSA public key
    const encrypted = await crypto.subtle.encrypt({ name: 'RSA-OAEP' }, pubKey, rawKey);
    const encrypted_b64 = btoa(String.fromCharCode(...new Uint8Array(encrypted)));
    // 4) send encrypted_key to server to register session (server stores key; does NOT return it)
    const res = await fetch(`${this.base}/api/secure/session/start`, {
      method: 'POST',
      headers: { 'Authorization': this.token, 'Content-Type': 'application/json' },
      body: JSON.stringify({ client: 'desktop-app', user_id: 'admin', encrypted_key: encrypted_b64 })
    });
    if (!res.ok) throw new Error(`Handshake failed: ${res.status}`);
    const data = await res.json();
    this.sessionId = data.session_id;
    // If server issued a token, persist it alongside session info
    if (data && data.token) {
      const tok = typeof data.token === 'string' ? data.token : btoa(JSON.stringify(data.token));
      this.token = `Bearer ${tok}`;
    }
    // Persist minimal metadata to localStorage and keep sessionKey in sessionStorage (per-tab)
    try {
      localStorage.setItem(this._storagePrefix, JSON.stringify({ sessionId: this.sessionId, token: this.token, ctr: this._actionCtr }));
      sessionStorage.setItem(`${this._storagePrefix}_key:${this.sessionId}`, this.sessionKey);
    } catch (e) {}
  }

  // AES-GCM decrypt utility matching backend wrap_encrypted
  async _decryptPacket(packet) {
    // If packet has alg field, it's encrypted by wrap_encrypted
    if (!packet || !packet.alg || !packet.ciphertext || !packet.iv || !packet.tag) {
      return packet; // assume plaintext
    }
    const ivBytes = this._b64ToBytes(packet.iv);
    const ctBytes = this._b64ToBytes(packet.ciphertext);
    const tagBytes = this._b64ToBytes(packet.tag);
    const combined = new Uint8Array(ctBytes.length + tagBytes.length);
    combined.set(ctBytes, 0);
    combined.set(tagBytes, ctBytes.length);
    const key = await this._importAesKey(this.sessionKey);
    const plainBuf = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: ivBytes, tagLength: 128 },
      key,
      combined
    );
    const text = new TextDecoder().decode(plainBuf);
    return JSON.parse(text);
  }

  _b64ToBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  async _importAesKey(keyB64) {
    if (this._cryptoKey && this._cryptoKey_b64 === keyB64) return this._cryptoKey;
    const raw = this._b64ToBytes(keyB64);
    this._cryptoKey = await crypto.subtle.importKey('raw', raw, { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);
    this._cryptoKey_b64 = keyB64;
    return this._cryptoKey;
  }

  _headers(extra = {}) {
    return Object.assign({
      'Authorization': this.token,
      'X-Session-ID': this.sessionId,
      'Content-Type': 'application/json'
    }, extra);
  }

  // Client-initiated rekey flow: generate a fresh AES key, encrypt with server RSA pubkey and POST to rotate endpoint
  async rotateSessionKey() {
    // generate fresh AES key
    const rawKey = crypto.getRandomValues(new Uint8Array(32));
    const spkiRes = await fetch(`${this.base}/api/secure/public_key`);
    if (!spkiRes.ok) throw new Error('Failed fetching public key for rotation');
    const spkiData = await spkiRes.json();
    const pem = spkiData.public_key_pem;
    const spki = this._pemToArrayBuffer(pem);
    const pubKey = await crypto.subtle.importKey('spki', spki, { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['encrypt']);
    const encrypted = await crypto.subtle.encrypt({ name: 'RSA-OAEP' }, pubKey, rawKey);
    const encrypted_b64 = btoa(String.fromCharCode(...new Uint8Array(encrypted)));
    const body = { session_id: this.sessionId, encrypted_key: encrypted_b64 };
    const res = await fetch(`${this.base}/api/secure/session/rotate`, { method: 'POST', headers: this._headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body) });
    if (!res.ok) {
      const txt = await res.text().catch(()=>'');
      window.notify && window.notify('error','Rotate failed',`Server error ${res.status}`+(txt?` - ${txt}`:''));
      throw new Error(`Rotate failed: ${res.status} ${txt}`);
    }
    const data = await res.json().catch(()=>null);
    // on success, update locally-held sessionKey and persist
    this.sessionKey = this._bytesToB64(rawKey);
  try { sessionStorage.setItem(`${this._storagePrefix}_key:${this.sessionId}`, this.sessionKey); } catch (e) {}
    window.notify && window.notify('success','Rotate successful','New session key installed');
    return data || true;
  }

  // Admin helper: revoke a session (requires admin privileges on caller)
  async adminRevokeSession(targetSessionId, reason = '') {
    const body = { session_id: targetSessionId, reason };
    const pkt = await this._req('/api/secure/admin/session/revoke', { method: 'POST', body: JSON.stringify(body) });
    return pkt;
  }

  // Admin helper: request server-side rotate which will signal client via WS
  async adminRotateSession(targetSessionId) {
    const body = { session_id: targetSessionId };
    const pkt = await this._req('/api/secure/admin/session/rotate', { method: 'POST', body: JSON.stringify(body) });
    return pkt;
  }

  _pemToArrayBuffer(pem) {
    // remove header/footer and newlines
    const b64 = pem.replace(/-----[^-]+-----/g, '').replace(/\s+/g, '');
    const bin = atob(b64);
    const len = bin.length;
    const buf = new Uint8Array(len);
    for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i);
    return buf.buffer;
  }

  _bytesToB64(bytes) {
    return btoa(String.fromCharCode(...new Uint8Array(bytes)));
  }

  async _req(path, opts = {}) {
    await this.ready; // ensure handshake complete
    const res = await fetch(`${this.base}${path}`, {
      ...opts,
      headers: this._headers(opts.headers || {})
    });
    if (!res.ok) {
      let detail = '';
      try {
        const txt = await res.text();
        if (txt) detail = ` | ${txt.substring(0, 200)}`; // include short body
      } catch {}
      throw new Error(`${path} failed: ${res.status}${detail}`);
    }
    const data = await res.json();
    // Persist CTR and token after successful secure request to reduce race on client crash
    try {
      // save minimal meta (no sessionKey) and keep key in sessionStorage
      if (this.sessionId) {
        localStorage.setItem(this._storagePrefix, JSON.stringify({ sessionId: this.sessionId, token: this.token, ctr: this._actionCtr }));
      }
    } catch (e) {}
    return this._decryptPacket(data);
  }

  // Dashboard and secure data
  async getDashboard() {
    const pkt = await this._req('/api/secure/dashboard');
    return pkt; // already payload-like
  }
  async getNodes() {
    const pkt = await this._req('/api/secure/nodes');
    return pkt;
  }
  async getResources() {
    const pkt = await this._req('/api/secure/resources');
    return pkt;
  }
  async getPerformance() {
    const pkt = await this._req('/api/secure/performance');
    return pkt;
  }
  async getSessions() {
    const pkt = await this._req('/api/secure/sessions');
    return pkt;
  }
  async getProcesses() {
    const pkt = await this._req('/api/secure/processes');
    return pkt;
  }
  async killProcess(pid) {
    const pkt = await this._req('/api/secure/processes/kill', { method: 'POST', body: JSON.stringify({ pid })});
    return pkt;
  }

  // Virtual Desktop APIs (NoVNC over VNC)
  async createVirtualDesktop({ user_id = 'admin', os_image = 'ubuntu-xfce', cpu_cores = 2, memory_gb = 4, gpu_units = 0, packages, resolution, vnc_password, profile } = {}) {
    const body = { user_id, os_image, cpu_cores, memory_gb, gpu_units };
    if (Array.isArray(packages) && packages.length) body.packages = packages;
    if (resolution) body.resolution = resolution;
    if (vnc_password) body.vnc_password = vnc_password;
    if (profile) body.profile = profile;
    const pkt = await this._req('/api/secure/vd/create', {
      method: 'POST',
      body: JSON.stringify(body)
    });
    return pkt;
  }
  async getVirtualDesktopUrl(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/url`);
    return pkt;
  }
  async getVirtualDesktopHealth(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/health`);
    return pkt;
  }
  async getVirtualDesktopPackages(session_id, listCsv) {
    const q = listCsv ? `?q=${encodeURIComponent(listCsv)}` : '';
    const pkt = await this._req(`/api/secure/vd/${session_id}/packages${q}`);
    return pkt;
  }
  async getOsCatalog() {
    const pkt = await this._req('/api/secure/vd/os-list');
    return pkt;
  }
  async getDockerHealth() {
    const pkt = await this._req('/api/secure/vd/docker-health');
    return pkt;
  }
  async getVdProfiles() {
    const pkt = await this._req('/api/secure/vd/profiles');
    return pkt;
  }
  async registerOsImage(body) {
    const pkt = await this._req('/api/secure/vd/os-register', { method: 'POST', body: JSON.stringify(body) });
    return pkt;
  }
  async deleteVirtualDesktop(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}`, { method: 'DELETE' });
    return pkt;
  }
  async pauseVirtualDesktop(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/pause`, { method: 'POST' });
    return pkt;
  }
  async resumeVirtualDesktop(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/resume`, { method: 'POST' });
    return pkt;
  }
  async snapshotVirtualDesktop(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/snapshot`, { method: 'POST' });
    return pkt;
  }
  async listSnapshots(session_id) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/snapshots`);
    return pkt;
  }
  async deleteSnapshot(session_id, tag) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/snapshot/delete`, { method: 'POST', body: JSON.stringify({ tag }) });
    return pkt;
  }
  async restoreSnapshot(session_id, tag) {
    const pkt = await this._req(`/api/secure/vd/${session_id}/snapshot/restore`, { method: 'POST', body: JSON.stringify({ tag }) });
    return pkt;
  }

  // RDP endpoints (Windows)
  async createRdpSession({ user_id = 'admin', host, port = 3389, username, password, domain = '' }) {
    if (!host || !username || !password) throw new Error('host, username, and password are required');
    const pkt = await this._req('/api/secure/rdp/create', { method: 'POST', body: JSON.stringify({ user_id, host, port, username, password, domain }) });
    return pkt;
    }
  async getRdpUrl(session_id) {
    const pkt = await this._req(`/api/secure/rdp/${session_id}/url`);
    return pkt;
  }
  async deleteRdpSession(session_id) {
    const pkt = await this._req(`/api/secure/rdp/${session_id}`, { method: 'DELETE' });
    return pkt;
  }

  // Secure action with nonce/counter
  async secureAction(action, params = {}) {
    const body = {
      action,
      params,
      nonce: (Math.random().toString(36).slice(2) + Date.now().toString(36)),
      ctr: ++this._actionCtr,
      ts: Date.now() / 1000
    };
    const pkt = await this._req('/api/secure/action', { method: 'POST', body: JSON.stringify(body) });
    return pkt;
  }

  async getLogs() {
    const pkt = await this._req('/api/secure/logs');
    return pkt;
  }

  // Security & RBAC
  async whoami() {
    const pkt = await this._req('/api/secure/whoami');
    return pkt;
  }
  async assignRole(username, role) {
    const pkt = await this._req('/api/secure/admin/roles/assign', { method: 'POST', body: JSON.stringify({ username, role }) });
    return pkt;
  }
  async removeRole(username, role) {
    const pkt = await this._req('/api/secure/admin/roles/remove', { method: 'POST', body: JSON.stringify({ username, role }) });
    return pkt;
  }
}
