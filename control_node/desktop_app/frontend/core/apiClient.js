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
    this.sessionKey = null; // base64 string
    this.ready = this.initialize();
  this._actionCtr = 0;
  }

  async initialize() {
    // 1) Authenticate (reuse static admin for now)
    await this._login();
    // 2) Establish secure session
    await this._handshake();
  }

  async _login() {
    const res = await fetch(`${this.base}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'omega123' })
    });
    if (!res.ok) throw new Error(`Auth failed: ${res.status}`);
    const data = await res.json();
    this.token = data?.token ? `Bearer ${JSON.stringify(data.token)}` : 'Bearer bootstrap';
  }

  async _handshake() {
    const res = await fetch(`${this.base}/api/secure/session/start`, {
      method: 'POST',
      headers: { 'Authorization': this.token, 'Content-Type': 'application/json' },
      body: JSON.stringify({ client: 'desktop-app', user_id: 'admin' })
    });
    if (!res.ok) throw new Error(`Handshake failed: ${res.status}`);
    const data = await res.json();
    this.sessionId = data.session_id;
    this.sessionKey = data.session_key; // base64
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
      'X-Session-Key': this.sessionKey,
      'Content-Type': 'application/json'
    }, extra);
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
