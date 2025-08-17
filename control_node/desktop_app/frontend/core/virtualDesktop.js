// Virtual Desktop Manager: track active VD session and URL
export class VirtualDesktopManager {
  constructor(api) {
    this.api = api;
    this.current = null; // { sessionId, url }
    this.listeners = new Set();
  }

  onChange(cb) {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  _emit() {
    for (const cb of this.listeners) {
      try { cb(this.current); } catch {}
    }
  }

  getCurrent() {
    return this.current ? { ...this.current } : null;
  }

  setCurrent(data) {
    this.current = data ? { ...data } : null;
    this._emit();
  }

  async connect(sessionId) {
    const pkt = await this.api.getVirtualDesktopUrl(sessionId);
    const url = pkt.connect_url;
    this.setCurrent({ sessionId, url });
    return url;
  }

  disconnect() {
    this.setCurrent(null);
  }
}
