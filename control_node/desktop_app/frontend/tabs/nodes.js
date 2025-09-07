// Nodes tab with secure backend integration.
// Version stamp: v2025-09-07a (adds Discover busy state + latency jitter formatting fix)
// Establishes AES-GCM session (client generates key, encrypts with server RSA) and decrypts /api/secure responses.
let __omegaSecureSessionPromise=null;
async function ensureSecureSession(){
  if(__omegaSecureSessionPromise) return __omegaSecureSessionPromise;
  __omegaSecureSessionPromise=(async()=>{
    if(window.__omegaSecureSession?.id && window.__omegaSecureSession.aesKey) return window.__omegaSecureSession;
    // Determine API base (try configured, relative, then localhost fallbacks)
    const candidates = [
      (typeof window.OMEGA_API_BASE==='string'? window.OMEGA_API_BASE.trim():''),
      'http://127.0.0.1:8443', // prefer explicit http first to avoid TLS mismatch
      '', // relative (same origin) fallback
      'https://127.0.0.1:8443', // try https only after http
      '//127.0.0.1:8443' // protocol-relative last
    ].filter((v,i,arr)=> v && arr.indexOf(v)===i);
    let lastErr=null, pkJson=null, usedBase=null;
    for(const base of candidates){
      try{
        const url=(base.endsWith('/')? base.slice(0,-1):base)+ '/api/secure/public_key?t=' + Date.now();
        const pkRes = await fetch(url, {cache:'no-store'});
        if(!pkRes.ok){
          lastErr=new Error('status '+pkRes.status+' from '+url);
          console.warn('[SecureSession] public key attempt failed', base, pkRes.status);
          continue;
        }
        pkJson = await pkRes.json();
        usedBase = base;
        break;
      }catch(e){
        lastErr=e;
        console.warn('[SecureSession] public key network error for base', base, e);
      }
    }
    if(!pkJson){
      throw new Error('Public key fetch failed: '+ (lastErr? (lastErr.message||lastErr):'no response'));
    }
    if(usedBase!=='' && usedBase!=null){
      window.__omegaApiBase = usedBase.replace(/\/$/,'');
    } else if(!window.__omegaApiBase) {
      window.__omegaApiBase='';
    }
    const pem = pkJson.public_key_pem || pkJson.public_key || '';
    if(!pem.includes('BEGIN PUBLIC KEY')) throw new Error('Malformed public key PEM');
    const b64 = pem.replace(/-----BEGIN PUBLIC KEY-----/,'').replace(/-----END PUBLIC KEY-----/,'').replace(/\s+/g,'');
    const der = Uint8Array.from(atob(b64), c=>c.charCodeAt(0)).buffer;
    const rsaKey = await crypto.subtle.importKey('spki', der, {name:'RSA-OAEP', hash:'SHA-256'}, false, ['encrypt']);
    const rawKeyBytes=new Uint8Array(32); crypto.getRandomValues(rawKeyBytes);
    const encKeyBuf = await crypto.subtle.encrypt({name:'RSA-OAEP'}, rsaKey, rawKeyBytes);
    const encKeyB64 = btoa(String.fromCharCode(...new Uint8Array(encKeyBuf)));
    // Attempt session start with retries (simple backoff)
    const startUrl = (window.__omegaApiBase||'') + '/api/secure/session/start';
    let startJson=null; let attempt=0; let startErr=null;
    while(attempt<3 && !startJson){
      try{
        const startRes = await fetch(startUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({encrypted_key:encKeyB64,client:'desktop-app',user_id:'admin'})});
        if(!startRes.ok){
          startErr=new Error('session start status '+startRes.status);
          await new Promise(r=>setTimeout(r, 200* (attempt+1)));
        } else {
          startJson=await startRes.json();
        }
      }catch(e){ startErr=e; await new Promise(r=>setTimeout(r, 250*(attempt+1))); }
      attempt++;
    }
    if(!startJson) throw new Error('Session start failed: '+ (startErr? startErr.message:'unknown'));
    const aesKey = await crypto.subtle.importKey('raw', rawKeyBytes,{name:'AES-GCM'},false,['decrypt']);
    window.__omegaSecureSession={id:startJson.session_id,token:startJson.token,aesKey,rawKeyBytes,started:Date.now(),expiresIn:startJson.expires_in, apiBase: window.__omegaApiBase};
    console.info('[SecureSession] established session', {base:window.__omegaApiBase, id:startJson.session_id});
    return window.__omegaSecureSession;
  })();
  return __omegaSecureSessionPromise;
}
function b64Bytes(b64){return Uint8Array.from(atob(b64),c=>c.charCodeAt(0));}
// Build absolute API URL based on discovered backend base (global so all handlers can use)
function apiUrl(path){
  const base = window.__omegaApiBase||'';
  if(!path) return base||'';
  if(path.startsWith('http://')||path.startsWith('https://')||path.startsWith('//')) return path;
  if(!path.startsWith('/')) path='/'+path;
  return (base||'') + path;
}

async function decryptPacket(pkt){
  if(!pkt?.ciphertext) throw new Error('Malformed packet');
  if(!window.__omegaSecureSession?.aesKey) throw new Error('Missing session key');
  const ct=b64Bytes(pkt.ciphertext), tag=b64Bytes(pkt.tag), iv=b64Bytes(pkt.iv);
  const buf=new Uint8Array(ct.length+tag.length); buf.set(ct,0); buf.set(tag,ct.length);
  const plain=await crypto.subtle.decrypt({name:'AES-GCM',iv},window.__omegaSecureSession.aesKey,buf);
  const txt=new TextDecoder().decode(plain); try{return JSON.parse(txt);}catch{throw new Error('Decrypt parse error');}
}

export function renderNodes(root, state){
  state = state || { data: {} };
  state.data.nodes = state.data.nodes || { nodes: [] };
  const nodes = state.data.nodes.nodes || [];
  if(!root) return;
  root.innerHTML = '';
  ensureStyles();

  // Layout (parity structure with network tab: header + subtabs + panels + detail)
  root.innerHTML = `
  <div id="nodes-container" class="nodes-container-v2">
    <div class="nodes-status-header">
      <div class="nodes-overview">
        <div class="nodes-metric"><span>Devices: <strong id="nodes-count">${nodes.length}</strong></span></div>
        <div class="nodes-metric"><span>Online: <strong id="nodes-online">0</strong></span></div>
        <div class="nodes-metric"><span>Avg CPU: <strong id="nodes-avg-cpu">0%</strong></span></div>
        <div class="nodes-metric"><span>Avg MEM: <strong id="nodes-avg-mem">0%</strong></span></div>
        <div class="nodes-metric"><span>Alerts: <strong id="nodes-alerts-total">0</strong></span></div>
      </div>
      <div class="nodes-actions">
  <button id="nodes-refresh" class="nodes-btn">Refresh</button>
  <button id="nodes-discover" class="nodes-btn">Discover</button>
  <button id="nodes-connect" class="nodes-btn">Connect</button>
  <button id="nodes-register" class="nodes-btn">Register</button>
  <button id="nodes-export" class="nodes-btn">Export</button>
      </div>
    </div>
    <div class="nodes-subtabs" id="nodes-subtabs">
      <button class="nodes-subtab active" data-panel="list">List</button>
      <button class="nodes-subtab" data-panel="topology">Topology</button>
      <button class="nodes-subtab" data-panel="resources">Resources</button>
      <button class="nodes-subtab" data-panel="security">Security</button>
      <button class="nodes-subtab" data-panel="settings">Settings</button>
    </div>
    <div class="nodes-panels">
      <div id="nodes-panel-list" class="nodes-panel active">
        <div class="nodes-toolbar">
          <div class="nodes-filters">
            <input id="nodes-search" placeholder="Search" />
            <select id="filter-status"><option value="">Status</option><option value="online">Online</option><option value="offline">Offline</option></select>
          </div>
        </div>
        <table class="nodes-table" id="nodes-table">
          <thead><tr>
            <th>Status</th>
            <th data-sort="node_id" class="sortable">ID</th>
            <th data-sort="node_type" class="sortable">Type</th>
            <th>Class</th>
            <th data-sort="cpu" class="sortable">CPU</th>
            <th data-sort="mem" class="sortable">MEM</th>
            <th data-sort="last" class="sortable">Last Seen</th>
            <th></th>
          </tr></thead>
          <tbody id="nodes-tbody"></tbody>
        </table>
      </div>
      <div id="nodes-panel-topology" class="nodes-panel">
        <div class="nodes-topology-layout">
          <div class="nodes-topology-canvas-container">
            <div class="nodes-topology-controls">
              <div class="nodes-topology-tools">
                <button id="topo-zoom-in" class="btn-small">+</button>
                <button id="topo-zoom-out" class="btn-small">-</button>
                <button id="topo-fit" class="btn-small">Fit</button>
              </div>
              <div class="nodes-topology-meta" id="topo-stats">0 devices • 0 sessions • 0 links</div>
            </div>
            <div id="nodes-topology-canvas" class="nodes-topology-canvas"><div class="topology-empty">No nodes</div></div>
          </div>
          <div class="nodes-topology-sidebar">
            <div class="nodes-topology-panel"><div class="panel-header">Entity</div><div class="panel-body" id="topo-entity-body">Select a node/session</div></div>
            <div class="nodes-topology-panel"><div class="panel-header">Sessions</div><div class="panel-body" id="topo-sessions-body">No sessions</div></div>
          </div>
        </div>
      </div>
      <div id="nodes-panel-resources" class="nodes-panel"><div class="card"><h4>Resources</h4><div id="resources-body">(placeholder)</div></div></div>
      <div id="nodes-panel-security" class="nodes-panel"><div class="card"><h4>Security</h4><div id="security-body">(placeholder)</div></div></div>
      <div id="nodes-panel-settings" class="nodes-panel"><div class="card"><h4>Settings</h4><div id="settings-body">(placeholder)</div></div></div>
    </div>
    <section class="nodes-detail" id="nodes-detail">
      <div class="detail-header">Select a node</div>
      <div id="detail-subtabs" class="detail-subtabs" style="display:none">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="protocol">Protocol</div>
  <div class="tab" data-tab="telemetry">Telemetry</div>
      <div class="tab" data-tab="latency">Latency</div>
  <div class="tab" data-tab="policies">Policies</div>
  <div class="tab" data-tab="diagnostics">Diagnostics</div>
  <div class="tab" data-tab="security">Security</div>
  <div class="tab" data-tab="audit">Audit</div>
  <div class="tab" data-tab="actions">Actions</div>
      </div>
      <div class="detail-body" id="detail-body">No selection</div>
    </section>
  </div>`;

  // ---- State ----
  let selected = null;
  let audit = [];
  let topo = { zoom:1, offsetX:0, offsetY:0, entities:[], links:[], selected:null };
  let sort = { field:'node_id', dir:1 };
  let viewNodes = [];
  const telemetryCache = new Map(); // node_id -> {cpu:[],mem:[],ts:[]}
  const protocolState = new Map(); // node_id -> active protocol
  const policyState = new Map(); // node_id -> {k:v}
  let auditCache = [];

  // Unified secure request helper (handles session + decrypt)
  async function secureRequest(path,{method='GET',body}={}){
    const session = await ensureSecureSession();
    const base = window.__omegaApiBase || '';
    let url = path;
    if(path.startsWith('/api/')) url = base + path; // prefix discovered base for API-relative paths
    const opts = { method, headers:{'X-Session-ID':session.id,'Authorization':'Bearer '+session.token} };
    if(body){ opts.headers['Content-Type']='application/json'; opts.body=JSON.stringify(body); }
    const res = await fetch(url,opts);
    if(!res.ok){ const txt=await res.text().catch(()=>res.status); throw new Error(res.status+': '+txt); }
    const json = await res.json();
    // Auto-detect encrypted packet vs plain JSON (v1 endpoints)
    if(json && typeof json==='object' && json.ciphertext && json.iv && json.tag){
      return decryptPacket(json);
    }
    return json;
  }

  // ---- Subtab switching ----
  root.querySelectorAll('.nodes-subtab').forEach(btn=> btn.addEventListener('click',()=>{
    root.querySelectorAll('.nodes-subtab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.dataset.panel;
    root.querySelectorAll('.nodes-panel').forEach(p=> p.classList.toggle('active', p.id===`nodes-panel-${panel}`));
    if(panel==='topology') renderTopology();
  }));

  // ---- Data Loader ----
  let loading=false;
  async function loadNodes(manual=false){
    try{
      if(loading) return; loading=true;
  const t=root.querySelector('#nodes-tbody'); if(t) t.innerHTML='<tr><td colspan="8" style="padding:18px;text-align:center;color:#9bb5b7">Loading...</td></tr>';
  const plain = await secureRequest('/api/secure/nodes');
      if(!plain.nodes) throw new Error('Payload missing nodes');
      nodes.splice(0,nodes.length,...plain.nodes.map(n=>{
        const metrics = n.metrics || {cpu_usage:0,memory_usage:0};
        return {
          ...n,
          metrics,
          last_seen: metrics.timestamp ? metrics.timestamp*1000 : n.last_seen,
          status:(n.status==='active')?'online':(n.status==='standby'?'offline':(n.status||'offline'))
        };
      }));
      // telemetry accumulation per node (60 samples)
      nodes.forEach(n=>{
        const c = telemetryCache.get(n.node_id) || {cpu:[],mem:[],ts:[]};
        if(typeof n.metrics?.cpu_usage==='number'){ c.cpu.push(n.metrics.cpu_usage); if(c.cpu.length>60) c.cpu.shift(); }
        if(typeof n.metrics?.memory_usage==='number'){ c.mem.push(n.metrics.memory_usage); if(c.mem.length>60) c.mem.shift(); }
        c.ts.push(Date.now()); if(c.ts.length>60) c.ts.shift();
        telemetryCache.set(n.node_id,c);
        if(!protocolState.has(n.node_id)) protocolState.set(n.node_id,'gRPC/QUIC');
      });
  // Rebind selected reference to updated node object
  if(selected){ const updated = nodes.find(x=> x.node_id===selected.node_id); if(updated) selected = updated; }
      updateHeader();
      renderTable();
      if(root.querySelector('#nodes-panel-topology').classList.contains('active')) renderTopology();
      if(selected){ // keep detail panel live
        const activeTab = root.querySelector('.detail-subtabs .tab.active')?.dataset.tab || 'overview';
        renderDetailTab(activeTab);
      }
      if(manual && window.notify) window.notify('info','Nodes','Refreshed');
    }catch(e){
      console.error('[Nodes] load failed',e);
      const t=root.querySelector('#nodes-tbody'); if(t) t.innerHTML='<tr><td colspan="7" style="padding:18px;text-align:center;color:#ff6d6d">Load failed</td></tr>';
      if(window.notify) window.notify('error','Nodes', e.message||'Load failed');
    }finally{loading=false;}
  }

  // ---- Table ----
  function lastSeen(n){ if(!n||!n.last_seen) return '—'; try{ const d=Date.now()-new Date(n.last_seen).getTime(); if(d<60000)return 'just now'; const m=Math.floor(d/60000); if(m<60)return m+'m'; const h=Math.floor(m/60); if(h<48)return h+'h'; return Math.floor(h/24)+'d'; }catch(e){return '—';}}
  function renderTable(){
    const tbody=root.querySelector('#nodes-tbody'); if(!tbody) return; tbody.innerHTML='';
    const q=(root.querySelector('#nodes-search')?.value||'').trim().toLowerCase();
    const statusFilter = root.querySelector('#filter-status')?.value || '';
    viewNodes = nodes.filter(n=>{
      if(statusFilter && (n.status||'')!==statusFilter) return false;
      if(!q) return true;
      const hay = `${n.node_id} ${n.node_type||''} ${(n.tags||[]).join(' ')} ${n.status||''}`.toLowerCase();
      return hay.includes(q);
    });
    // sorting
    const getters = {
      node_id: n=> n.node_id,
      node_type: n=> n.node_type||'',
      cpu: n=> n.metrics?.cpu_usage||0,
      mem: n=> n.metrics?.memory_usage||0,
      last: n=> new Date(n.last_seen||0).getTime()
    };
    if(getters[sort.field]){
      viewNodes.sort((a,b)=>{
        const av=getters[sort.field](a), bv=getters[sort.field](b);
        if(av<bv) return -1*sort.dir; if(av>bv) return 1*sort.dir; return 0;
      });
    }
    if(!viewNodes.length){ tbody.innerHTML='<tr class="empty"><td colspan="8">No matching nodes</td></tr>'; return; }
  viewNodes.forEach(n=>{ const status=n.status||'offline'; const trust =(n.trust_score||0)>=80?'trusted':(n.quarantine?'quarantined':((n.trust_score||0)<50?'untrusted':'neutral')); const klass=n.device_class||'-'; const tr=document.createElement('tr'); tr.innerHTML=`<td><span class="status-dot ${status}"></span></td><td>${n.node_id} <span class="trust-badge ${trust}">${trust}</span></td><td>${n.node_type||'-'}</td><td>${klass}</td><td>${(n.metrics?.cpu_usage||0).toFixed(0)}%</td><td>${(n.metrics?.memory_usage||0).toFixed(0)}%</td><td>${lastSeen(n)}</td><td><button class="btn-small" data-open="${n.node_id}">Open</button></td>`; tbody.appendChild(tr); });
    tbody.querySelectorAll('button[data-open]').forEach(b=> b.onclick=()=> openDetail(nodes.find(x=>x.node_id===b.dataset.open)));
  }
  // wire search & filter
  root.querySelector('#nodes-search')?.addEventListener('input',()=> renderTable());
  root.querySelector('#filter-status')?.addEventListener('change',()=> renderTable());
  // sortable headers
  root.querySelectorAll('th.sortable').forEach(th=> th.addEventListener('click',()=>{
    const f=th.dataset.sort; if(!f) return; if(sort.field===f) sort.dir*=-1; else { sort.field=f; sort.dir=1; }
    root.querySelectorAll('th.sortable').forEach(h=> h.classList.toggle('sorted', h===th));
    renderTable();
  }));
  // export CSV
  root.querySelector('#nodes-export')?.addEventListener('click',()=>{
    if(!viewNodes.length){ return; }
    const header=['id','type','status','cpu','mem','last_seen'];
    const lines=[header.join(',')].concat(viewNodes.map(n=>[
      n.node_id,
      (n.node_type||'').replace(/,/g,';'),
      n.status||'',
      (n.metrics?.cpu_usage||0).toFixed(1),
      (n.metrics?.memory_usage||0).toFixed(1),
      n.last_seen||''
    ].map(v=>`"${String(v)}"`).join(',')));
    const blob=new Blob([lines.join('\n')],{type:'text/csv'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='nodes.csv'; a.click(); URL.revokeObjectURL(url);
  });
  renderTable();

  // ---- Detail ----
  function openDetail(node){ if(!node) return; selected=node; const dh=root.querySelector('.detail-header'); dh.textContent=node.node_id + ' — ' + (node.hostname||node.node_type||'node'); const subt=root.querySelector('#detail-subtabs'); subt.style.display='flex'; renderDetailTab('overview'); subt.querySelectorAll('.tab').forEach(t=> t.onclick=e=>{ subt.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); e.currentTarget.classList.add('active'); renderDetailTab(e.currentTarget.dataset.tab); }); }
  async function loadAudit(){
    try{ const session=await ensureSecureSession(); const res=await fetch('/api/secure/logs',{headers:{'X-Session-ID':session.id,'Authorization':'Bearer '+session.token}}); if(!res.ok) throw new Error('logs '+res.status); const enc=await res.json(); const plain=await decryptPacket(enc); auditCache=plain.events||plain.logs||[]; }
    catch(e){ console.warn('[Nodes] audit fetch failed',e); }
  }
  function sparkline(canvas,data,color){ const ctx=canvas.getContext('2d'); const w=canvas.width=canvas.clientWidth; const h=canvas.height=canvas.clientHeight; ctx.clearRect(0,0,w,h); if(!data.length) return; const max=Math.max(...data,1); ctx.strokeStyle=color; ctx.beginPath(); data.forEach((v,i)=>{ const x=i/(data.length-1||1)*w; const y=h-(v/max)*h; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); }
  async function issueCert(node){ try{ const pem='-----BEGIN PUBLIC KEY-----\nMIIBFAKE==\n-----END PUBLIC KEY-----'; const session=await ensureSecureSession(); const res=await fetch(`/api/secure/nodes/${node.node_id}/issue_cert`,{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({public_key_pem:pem,days:1})}); if(!res.ok) throw new Error('cert '+res.status); const enc=await res.json(); const plain=await decryptPacket(enc); if(window.notify) window.notify('success','Cert','Issued'); return plain.certificate_pem; }catch(e){ if(window.notify) window.notify('error','Cert', e.message||'Failed'); return null; } }
  async function runBenchmark(){ try{ const session=await ensureSecureSession(); const meta=window.__omegaSecureSession; const ctr=(meta.__ctrBench=(meta.__ctrBench||0)+1); const nonce=Math.random().toString(36).slice(2); const res=await fetch('/api/secure/action',{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({action:'run_benchmark',params:{},ctr,nonce,ts:Date.now()/1000})}); if(!res.ok) throw new Error('bench '+res.status); const enc=await res.json(); const plain=await decryptPacket(enc); return plain.result; }catch(e){ console.error('benchmark failed',e); return null; } }
  async function restartNode(node){ try{ const session=await ensureSecureSession(); const meta=window.__omegaSecureSession; const ctr=(meta.__ctrRestart=(meta.__ctrRestart||0)+1); const nonce=Math.random().toString(36).slice(2); const res=await fetch('/api/secure/action',{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({action:'restart_node',params:{node_id:node.node_id},ctr,nonce,ts:Date.now()/1000})}); if(!res.ok) throw new Error('restart '+res.status); const enc=await res.json(); await decryptPacket(enc); if(window.notify) window.notify('info','Restart','Requested'); }catch(e){ if(window.notify) window.notify('error','Restart', e.message||'Failed'); } }
  function renderDetailTab(tab){ const body=root.querySelector('#detail-body'); if(!selected){ body.textContent='No selection'; return;} const n=selected; const trustLabel=(n.trust_score||0)>=80?'Trusted':(n.quarantine?'Quarantined':((n.trust_score||0)<50?'Untrusted':'Neutral')); switch(tab){
  case 'overview': body.innerHTML=`<div class="card"><h4>Overview</h4><div><strong>${n.node_id}</strong> <span class="trust-badge ${(trustLabel||'').toLowerCase()}">${trustLabel}</span></div><div>Type: ${n.node_type||'-'}</div><div>Class: ${n.device_class||'-'}</div><div>Status: ${n.status||'-'}</div><div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}%</div><div>MEM: ${(n.metrics?.memory_usage||0).toFixed(1)}%</div><div>IP: ${n.ip_address||'—'}</div></div>`; break;
  case 'latency': body.innerHTML=`<div class="card"><h4>Ultra Low Latency</h4><div style="display:flex;flex-wrap:wrap;gap:6px"><button class="btn-small" id="lat-start">Negotiate</button><button class="btn-small" id="lat-measure">Measure</button></div><div id="lat-out" style="margin-top:6px;font-size:10px;color:#9bb5b7">Idle</div></div>`; { const out=body.querySelector('#lat-out'); body.querySelector('#lat-start').onclick=async()=>{ out.textContent='Negotiating...'; try{ const r=await secureRequest(`/api/secure/nodes/latency/start`,{method:'POST',body:{node_id:n.node_id}}); out.textContent='Token '+r.token+' proto '+(r.recommended||r.protocols?.join('/')||''); }catch(e){ out.textContent='Failed: '+e.message; } }; body.querySelector('#lat-measure').onclick=async()=>{ out.textContent='Measuring...'; try{ const r=await secureRequest(`/api/secure/nodes/latency/measure?node_id=${encodeURIComponent(n.node_id)}`); out.textContent=`Latency ${r.latency_ms}ms 8${r.jitter_ms} (method ${r.method})`; }catch(e){ out.textContent='Failed: '+e.message; } }; } break;
    case 'protocol': { body.innerHTML=`<div class="card"><h4>Protocol</h4><div id="proto-loading" style="font-size:10px;color:#9bb5b7">Loading...</div><div id="proto-content" style="display:none"><div>Active: <strong id="proto-active"></strong></div><div style="margin-top:6px">Switch <select id="proto-select"></select> <button class="btn-small" id="proto-apply">Apply</button></div><div style="margin-top:6px;font-size:10px;color:#9bb5b7" id="proto-note"></div></div></div>`; (async()=>{ try{ const p=await secureRequest(`/api/secure/nodes/${n.node_id}/protocol`); protocolState.set(n.node_id,p.active); const wrap=body.querySelector('#proto-content'); const sel=body.querySelector('#proto-select'); p.supported.forEach(opt=>{ const o=document.createElement('option'); o.textContent=opt; sel.appendChild(o); }); sel.value=p.active; body.querySelector('#proto-active').textContent=p.active; body.querySelector('#proto-note').textContent='Supported: '+p.supported.join(', '); body.querySelector('#proto-loading').style.display='none'; wrap.style.display='block'; body.querySelector('#proto-apply').onclick=async()=>{ const val=sel.value; try{ const r=await secureRequest(`/api/secure/nodes/${n.node_id}/protocol`,{method:'POST',body:{protocol:val}}); protocolState.set(n.node_id,r.active); body.querySelector('#proto-active').textContent=r.active; if(window.notify) window.notify('success','Protocol','Switched to '+r.active); }catch(e){ if(window.notify) window.notify('error','Protocol', e.message); } }; }catch(e){ body.querySelector('#proto-loading').textContent='Failed: '+e.message; } })(); break; }
    case 'telemetry': { body.innerHTML=`<div class="card"><h4>Telemetry</h4><div id="tel-wrap" style="font-size:10px;color:#9bb5b7">Loading...</div></div>`; (async()=>{ try{ const d=await secureRequest(`/api/secure/nodes/${n.node_id}/telemetry?limit=60`); const c=d.series||{cpu:[],mem:[],ts:[]}; body.querySelector('#tel-wrap').innerHTML=`<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><canvas id="spark-cpu" class="spark" style="height:60px"></canvas><canvas id="spark-mem" class="spark" style="height:60px"></canvas></div><div style="margin-top:6px">Samples: ${c.cpu.length}</div>`; const cv1=body.querySelector('#spark-cpu'); const cv2=body.querySelector('#spark-mem'); if(cv1) sparkline(cv1,c.cpu,'#00e6d1'); if(cv2) sparkline(cv2,c.mem,'#ffb347'); }catch(e){ body.querySelector('#tel-wrap').textContent='Failed: '+e.message;} })(); break; }
  case 'policies': { body.innerHTML=`<div class="card"><h4>Policies</h4><div id="pol-list" style="min-height:40px;font-size:10px;color:#9bb5b7">Loading...</div><div style="margin-top:8px;display:flex;gap:4px"><input id="pol-k" placeholder="key" style="flex:1"/><input id="pol-v" placeholder="value" style="flex:1"/><button class="btn-small" id="pol-add">Add</button></div></div>`; const list=body.querySelector('#pol-list'); function redraw(pol){ list.innerHTML=Object.keys(pol).length?Object.entries(pol).map(([k,v])=>`<div class=\"pol-row\"><code>${k}</code>: <span>${v}</span> <button class=\"btn-small\" data-del=\"${k}\">x</button></div>`).join(''):'<em style=\"font-size:10px;color:#9bb5b7\">No policies</em>'; list.querySelectorAll('button[data-del]').forEach(b=> b.onclick=async()=>{ try{ await secureRequest(`/api/secure/nodes/${n.node_id}/policies/${encodeURIComponent(b.dataset.del)}`,{method:'DELETE'}); const fresh=await secureRequest(`/api/secure/nodes/${n.node_id}/policies`); policyState.set(n.node_id,fresh.policies||{}); redraw(policyState.get(n.node_id)); }catch(e){ if(window.notify) window.notify('error','Policy', e.message); } }); } (async()=>{ try{ const fresh=await secureRequest(`/api/secure/nodes/${n.node_id}/policies`); policyState.set(n.node_id,fresh.policies||{}); redraw(policyState.get(n.node_id)); }catch(e){ list.textContent='Failed: '+e.message; } })(); body.querySelector('#pol-add').onclick=async()=>{ const k=body.querySelector('#pol-k').value.trim(); const v=body.querySelector('#pol-v').value.trim(); if(!k) return; try{ await secureRequest(`/api/secure/nodes/${n.node_id}/policies`,{method:'POST',body:{key:k,value:v}}); const fresh=await secureRequest(`/api/secure/nodes/${n.node_id}/policies`); policyState.set(n.node_id,fresh.policies||{}); redraw(policyState.get(n.node_id)); body.querySelector('#pol-k').value=''; body.querySelector('#pol-v').value=''; }catch(e){ if(window.notify) window.notify('error','Policy', e.message); } }; break; }
    case 'diagnostics': body.innerHTML=`<div class="card"><h4>Diagnostics</h4><div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn-small" id="btn-diag-basic">Run Basic</button><button class="btn-small" id="btn-diag-extended">Extended</button><button class="btn-small" id="btn-bench">Benchmark</button><button class="btn-small" id="btn-restart">Restart</button></div><div id="diag-out" style="margin-top:8px;font-size:10px;color:#9bb5b7">Idle</div></div>`; { const out=body.querySelector('#diag-out'); const runDiag=async(level)=>{ out.textContent='Running diagnostics...'; try{ const d=await secureRequest(`/api/secure/nodes/${n.node_id}/diagnostics`,{method:'POST',body:{level}}); const checks=d.diagnostics?.checks||{}; out.textContent='Checks: '+Object.entries(checks).map(([k,v])=>`${k}=${v}`).join(', '); }catch(e){ out.textContent='Failed: '+e.message; } }; body.querySelector('#btn-diag-basic').onclick=()=>runDiag('basic'); body.querySelector('#btn-diag-extended').onclick=()=>runDiag('extended'); body.querySelector('#btn-bench').onclick=async()=>{ out.textContent='Benchmark...'; const r=await runBenchmark(); out.textContent=r?('Score '+r.score):'Failed'; }; body.querySelector('#btn-restart').onclick=()=> restartNode(n); } break;
  case 'security': body.innerHTML=`<div class="card"><h4>Security</h4><div id="trust-wrap" style="font-size:10px;color:#9bb5b7">Loading...</div><div style="margin-top:6px">Issue Cert <button class="btn-small" id="btn-cert">Issue</button></div><div id="cert-result" style="margin-top:6px;font-size:10px;max-height:140px;overflow:auto"></div><div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap"><button class="btn-small" id="btn-quarantine">Quarantine</button><button class="btn-small" id="btn-unquarantine">Unquarantine</button><button class="btn-small" id="btn-remove">Remove</button><button class="btn-small" id="btn-probe">Probe</button></div></div>`; (async()=>{ try{ const t=await secureRequest(`/api/secure/nodes/${n.node_id}/trust`); const trustBadge=(t.trust_score||0)>=80?'trusted':(t.quarantine?'quarantined':((t.trust_score||0)<50?'untrusted':'neutral')); body.querySelector('#trust-wrap').innerHTML=`Trust: <strong>${t.trust_score}</strong> <span class="trust-badge ${trustBadge}">${trustBadge}</span> &nbsp; Quarantine: ${t.quarantine?'YES':'NO'} &nbsp; Status: ${t.status}`; }catch(e){ body.querySelector('#trust-wrap').textContent='Failed: '+e.message; } })(); body.querySelector('#btn-cert').onclick=async()=>{ body.querySelector('#cert-result').textContent='Issuing...'; const pem=await issueCert(n); body.querySelector('#cert-result').textContent=pem||'Failed'; }; body.querySelector('#btn-quarantine').onclick=()=>nodeQuarantine(n.node_id,true); body.querySelector('#btn-unquarantine').onclick=()=>nodeQuarantine(n.node_id,false); body.querySelector('#btn-remove').onclick=()=>removeNode(n.node_id); body.querySelector('#btn-probe').onclick=()=>probeNode(n.node_id); break;
    case 'audit': body.innerHTML='<div class="card"><h4>Audit</h4><div id="audit-list" style="max-height:200px;overflow:auto;font-size:10px"></div></div>'; (async()=>{ await loadAudit(); const list=body.querySelector('#audit-list'); const filtered=auditCache.filter(e=> (e.node_id===n.node_id) || (e.source||'').includes(n.node_id) || (e.message||'').includes(n.node_id)); list.innerHTML=filtered.length? filtered.map(e=>`<div>${new Date((e.timestamp||e.ts||Date.now())*1000).toLocaleTimeString()} <strong>${e.severity||e.level||''}</strong> ${e.event_type||e.type||''}: ${e.message||e.msg||''}</div>`).join(''):'<em>No events</em>'; })(); break;
    case 'actions': body.innerHTML=`<div class="card"><h4>Actions</h4><div style="display:flex;flex-direction:column;gap:6px"><div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn-small" id="act-refresh">Refresh</button><button class="btn-small" id="act-benchmark">Benchmark</button><button class="btn-small" id="act-restart">Restart</button></div><div style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap"><input id="ota-version" placeholder="version" style="flex:1;min-width:100px"/><button class="btn-small" id="ota-update">OTA Update</button><button class="btn-small" id="ota-rollback">Rollback</button></div></div><div id="act-out" style="font-size:10px;margin-top:8px;color:#9bb5b7"></div></div>`; { const out2=body.querySelector('#act-out'); const runBench=async()=>{ out2.textContent='Benchmark...'; const r=await runBenchmark(); out2.textContent=r?('Score '+r.score):'Failed'; }; body.querySelector('#act-refresh').onclick=()=>loadNodes(true); body.querySelector('#act-benchmark').onclick=runBench; body.querySelector('#act-restart').onclick=()=> restartNode(n); const ota=(action)=>async()=>{ const ver=body.querySelector('#ota-version').value.trim(); out2.textContent='OTA '+action+'...'; try{ const r=await secureRequest(`/api/secure/nodes/${n.node_id}/ota`,{method:'POST',body:{action,version:ver||undefined}}); out2.textContent='OTA '+action+' ok'+(r.version?(' -> '+r.version):''); if(window.notify) window.notify('success','OTA', action+' done'); }catch(e){ out2.textContent='Failed: '+e.message; if(window.notify) window.notify('error','OTA', e.message); } }; body.querySelector('#ota-update').onclick=ota('update'); body.querySelector('#ota-rollback').onclick=ota('rollback'); } break;
    default: body.textContent='N/A'; }
  }

  // ---- Activity (minimal) ----
  function pushAudit(msg){ audit.unshift({ts:Date.now(), msg}); audit=audit.slice(0,200); }
  pushAudit('Nodes UI initialized');

  // ---- Topology (session-aware simplified) ----
  function buildGraph(){ topo.entities=[]; topo.links=[]; const sessionMap=new Map(); nodes.forEach(n=> (n.tags||[]).filter(t=>t.startsWith('session:')).map(t=>t.split(':')[1]).forEach(s=>{ if(!sessionMap.has(s)) sessionMap.set(s,new Set()); sessionMap.get(s).add(n.node_id); })); const sessions=[...sessionMap.keys()].map(id=>({type:'session',id})); const devices=nodes.map(n=>({type:'device',id:n.node_id,ref:n})); topo.entities=[...sessions,...devices]; sessions.forEach(s=> (sessionMap.get(s.id)||[]).forEach(d=> topo.links.push({a:s.id,b:d,type:'membership'}))); const ids=devices.map(d=>d.id); for(let i=0;i<ids.length;i++) for(let j=i+1;j<ids.length;j++){ for(const set of sessionMap.values()){ if(set.has(ids[i])&&set.has(ids[j])){ topo.links.push({a:ids[i],b:ids[j],type:'shared'}); break; } } } // layout
    const left=140,right=620,topY=70,step=50; const ses=topo.entities.filter(e=>e.type==='session'); const dev=topo.entities.filter(e=>e.type==='device'); ses.forEach((s,i)=>{ s.x=left; s.y=topY+i*step; }); dev.forEach((d,i)=>{ d.x=right; d.y=topY+i*step; }); }
  function fitView(){ const c=root.querySelector('#nodes-topology-canvas canvas'); if(!c) return; const ents=topo.entities; if(!ents.length) return; const minX=Math.min(...ents.map(e=>e.x)); const maxX=Math.max(...ents.map(e=>e.x)); const minY=Math.min(...ents.map(e=>e.y)); const maxY=Math.max(...ents.map(e=>e.y)); const w=maxX-minX, h=maxY-minY; const cw=c.width, ch=c.height, m=80; const zx=(cw-m)/w, zy=(ch-m)/h; topo.zoom=Math.min(2,Math.max(.4,Math.min(zx,zy))); topo.offsetX=(cw/2)-(minX+w/2)*topo.zoom; topo.offsetY=(ch/2)-(minY+h/2)*topo.zoom; }
  function drawTopo(){ const wrap=root.querySelector('#nodes-topology-canvas'); if(!wrap) return; let cv=wrap.querySelector('canvas'); if(!cv){ cv=document.createElement('canvas'); wrap.innerHTML=''; wrap.appendChild(cv);} cv.width=wrap.clientWidth||800; cv.height=wrap.clientHeight||460; const ctx=cv.getContext('2d'); ctx.clearRect(0,0,cv.width,cv.height); ctx.save(); ctx.translate(topo.offsetX,topo.offsetY); ctx.scale(topo.zoom,topo.zoom); topo.links.forEach(l=>{ const a=topo.entities.find(e=>e.id===l.a); const b=topo.entities.find(e=>e.id===l.b); if(!a||!b)return; ctx.beginPath(); ctx.strokeStyle=l.type==='shared'? 'rgba(0,230,209,0.18)':'rgba(0,230,209,0.35)'; ctx.lineWidth=l.type==='shared'?1:1.4; ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }); topo.entities.forEach(ent=>{ if(ent.type==='device'){ const n=ent.ref; const status=n.status||'offline'; let fill='#2c3233', stroke='rgba(0,230,209,0.65)'; if(status==='online') fill='#03534c'; else if(status==='offline') stroke='rgba(255,90,90,0.6)'; ctx.beginPath(); ctx.lineWidth=2; ctx.fillStyle=fill; ctx.strokeStyle = (topo.selected && topo.selected.id===ent.id)?'#00e6d1':stroke; ctx.arc(ent.x,ent.y,18,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(n.node_id.slice(0,6), ent.x, ent.y+30); } else { ctx.save(); ctx.translate(ent.x,ent.y); ctx.rotate(Math.PI/4); ctx.beginPath(); ctx.lineWidth=2; ctx.strokeStyle=(topo.selected && topo.selected.id===ent.id)?'#00e6d1':'rgba(0,230,209,0.6)'; ctx.fillStyle='rgba(0,230,209,0.1)'; ctx.rect(-14,-14,28,28); ctx.fill(); ctx.stroke(); ctx.restore(); ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(ent.id.slice(0,6), ent.x, ent.y+30); } }); ctx.restore(); }
  function updateStats(){ const stats=root.querySelector('#topo-stats'); if(!stats) return; stats.textContent=`${topo.entities.filter(e=>e.type==='device').length} devices • ${topo.entities.filter(e=>e.type==='session').length} sessions • ${topo.links.length} links`; }
  function updateEntity(ent){ const body=root.querySelector('#topo-entity-body'); if(!body)return; if(!ent){ body.textContent='Select a node/session'; return;} if(ent.type==='device'){ const n=ent.ref; body.innerHTML=`<div><strong>${n.node_id}</strong><div>Status: ${n.status||'-'}</div><div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}%</div></div>`; } else { const members=topo.links.filter(l=>l.type==='membership'&&l.a===ent.id).length; body.innerHTML=`<div><strong>Session ${ent.id}</strong><div>Devices: ${members}</div></div>`; } }
  function updateSessionList(){ const sb=root.querySelector('#topo-sessions-body'); if(!sb) return; const sessions=topo.entities.filter(e=>e.type==='session'); if(!sessions.length){ sb.textContent='No sessions'; return;} sb.innerHTML=sessions.map(s=>`<div class="session-row" data-sid="${s.id}">${s.id}</div>`).join(''); sb.querySelectorAll('.session-row').forEach(r=> r.onclick=()=>{ topo.selected=topo.entities.find(e=>e.id===r.dataset.sid); updateEntity(topo.selected); drawTopo(); }); }
  function wireCanvas(){ const cv=root.querySelector('#nodes-topology-canvas canvas'); if(!cv) return; let dragging=false,lx=0,ly=0; cv.onmousedown=e=>{dragging=true;lx=e.clientX;ly=e.clientY;}; window.addEventListener('mouseup',()=>dragging=false); window.addEventListener('mousemove',e=>{ if(!dragging)return; topo.offsetX+=e.clientX-lx; topo.offsetY+=e.clientY-ly; lx=e.clientX; ly=e.clientY; drawTopo(); }); cv.onclick=e=>{ const r=cv.getBoundingClientRect(); const x=(e.clientX-r.left - topo.offsetX)/topo.zoom; const y=(e.clientY-r.top - topo.offsetY)/topo.zoom; let hit=null; for(const ent of topo.entities){ if(ent.type==='device'){ const dx=ent.x-x,dy=ent.y-y; if(dx*dx+dy*dy<18*18){ hit=ent; break;} } else { const dx=x-ent.x,dy=y-ent.y; const u=(dx+dy)/Math.SQRT2,v=(dy-dx)/Math.SQRT2; if(Math.abs(u)<14&&Math.abs(v)<14){ hit=ent; break;} } } topo.selected=hit; if(hit && hit.type==='device') openDetail(hit.ref); updateEntity(hit); drawTopo(); }; }
  function wireControls(){ root.querySelector('#topo-zoom-in')?.addEventListener('click',()=>{ topo.zoom=Math.min(2.5, topo.zoom+0.1); drawTopo(); }); root.querySelector('#topo-zoom-out')?.addEventListener('click',()=>{ topo.zoom=Math.max(.3, topo.zoom-0.1); drawTopo(); }); root.querySelector('#topo-fit')?.addEventListener('click',()=>{ fitView(); drawTopo(); }); }
  function renderTopology(){ buildGraph(); drawTopo(); fitView(); drawTopo(); updateStats(); updateSessionList(); wireCanvas(); wireControls(); }

  // ---- Metrics header ----
  function updateHeader(){ const online = nodes.filter(n=>n.status==='online').length; const avgCpu = nodes.length? (nodes.reduce((s,n)=> s+(n.metrics?.cpu_usage||0),0)/nodes.length).toFixed(1):'0.0'; const avgMem = nodes.length? (nodes.reduce((s,n)=> s+(n.metrics?.memory_usage||0),0)/nodes.length).toFixed(1):'0.0'; const cntEl=root.querySelector('#nodes-count'); if(cntEl) cntEl.textContent=nodes.length; root.querySelector('#nodes-online').textContent=online; root.querySelector('#nodes-avg-cpu').textContent=avgCpu+'%'; root.querySelector('#nodes-avg-mem').textContent=avgMem+'%'; }
  updateHeader();

  // ---- Refresh ----
  const discoverBtn = root.querySelector('#nodes-discover');
  discoverBtn?.addEventListener('click', async()=>{
    if(discoverBtn.classList.contains('busy')) return; // already running
    const originalText = discoverBtn.textContent;
    discoverBtn.classList.add('busy');
    discoverBtn.textContent='Discovering...';
    try{
      const session=await ensureSecureSession();
      const meta=window.__omegaSecureSession; const ctr=(meta.__ctrDisc=(meta.__ctrDisc||0)+1); const nonce=Math.random().toString(36).slice(2);
      const attempt = async (primary)=>{
        if(primary){
          const res=await fetch(apiUrl('/api/secure/discover'),{method:'POST',headers:{'X-Session-ID':session.id,'Authorization':'Bearer '+session.token}});
          if(!res.ok) throw new Error('discover '+res.status);
          const enc=await res.json(); return decryptPacket(enc);
        } else {
          const res=await fetch(apiUrl('/api/secure/action'),{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({action:'discover_nodes',params:{},ctr,nonce,ts:Date.now()/1000})});
          if(!res.ok) throw new Error('discover '+res.status);
          const enc=await res.json(); return decryptPacket(enc);
        }
      };
      let plain; try{ plain=await attempt(true);}catch(err){ console.warn('primary discover failed, falling back',err); plain=await attempt(false);} 
      const metaInfo = plain.result? `Found ${plain.result.discovered||0}` + (plain.result.methods_used?` via ${plain.result.methods_used.join('+')}`:'') + (plain.result.self_added?' (self included)':'') : 'Completed';
      if(window.notify) window.notify('success','Discovery', metaInfo);
      pushAudit('Discovery: '+metaInfo);
      loadNodes(true);
    }catch(e){
      const msg = (e && e.message)? e.message : String(e);
      if(window.notify) window.notify('error','Discovery', msg);
      console.error('Discovery error full', e);
      pushAudit('Discovery failed: '+msg);
    } finally {
      discoverBtn.classList.remove('busy');
      discoverBtn.textContent=originalText;
    }
  });
  root.querySelector('#nodes-connect')?.addEventListener('click', async()=>{
    if(!selected){ if(window.notify) window.notify('warn','Connect','Select a node first'); return; }
    try{
      const session=await ensureSecureSession();
      // simple AES-secured request helper inline (reuse secureRequest if supports POST body encryption)
      const meta=window.__omegaSecureSession; const ctr=(meta.__ctrVd=(meta.__ctrVd||0)+1); const nonce=Math.random().toString(36).slice(2);
      // For now vd/start endpoint already encrypted via wrap_encrypted (POST plain JSON allowed through secureRequest path?) reuse fetch for clarity
  const res=await fetch(apiUrl('/api/secure/vd/start'),{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({node_id:selected.node_id,cpu_cores:2,memory_gb:4})});
  async function issueCert(node){ try{ const pem='-----BEGIN PUBLIC KEY-----\nMIIBFAKE==\n-----END PUBLIC KEY-----'; const session=await ensureSecureSession(); const res=await fetch(apiUrl(`/api/secure/nodes/${node.node_id}/issue_cert`),{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({public_key_pem:pem,days:1})}); if(!res.ok) throw new Error('cert '+res.status); const enc=await res.json(); const plain=await decryptPacket(enc); if(window.notify) window.notify('success','Cert','Issued'); return plain.certificate_pem; }catch(e){ if(window.notify) window.notify('error','Cert', e.message||'Failed'); return null; } }
  async function runBenchmark(){ try{ const session=await ensureSecureSession(); const meta=window.__omegaSecureSession; const ctr=(meta.__ctrBench=(meta.__ctrBench||0)+1); const nonce=Math.random().toString(36).slice(2); const res=await fetch(apiUrl('/api/secure/action'),{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({action:'run_benchmark',params:{},ctr,nonce,ts:Date.now()/1000})}); if(!res.ok) throw new Error('bench '+res.status); const enc=await res.json(); const plain=await decryptPacket(enc); return plain.result; }catch(e){ console.error('benchmark failed',e); return null; } }
  async function restartNode(node){ try{ const session=await ensureSecureSession(); const meta=window.__omegaSecureSession; const ctr=(meta.__ctrRestart=(meta.__ctrRestart||0)+1); const nonce=Math.random().toString(36).slice(2); const res=await fetch(apiUrl('/api/secure/action'),{method:'POST',headers:{'Content-Type':'application/json','X-Session-ID':session.id,'Authorization':'Bearer '+session.token},body:JSON.stringify({action:'restart_node',params:{node_id:node.node_id},ctr,nonce,ts:Date.now()/1000})}); if(!res.ok) throw new Error('restart '+res.status); const enc=await res.json(); await decryptPacket(enc); if(window.notify) window.notify('info','Restart','Requested'); }catch(e){ if(window.notify) window.notify('error','Restart', e.message||'Failed'); } }
      if(!res.ok) throw new Error('vd '+res.status);
      const enc=await res.json(); const plain=await decryptPacket(enc);
      if(window.notify) window.notify('success','VD Start', plain.connect_url||plain.session_id);
      // Could open in new window if real viewer available
    }catch(e){ if(window.notify) window.notify('error','VD Start', e.message||'Failed'); }
  });
  root.querySelector('#nodes-register')?.addEventListener('click',()=> openRegistration());

  loadNodes();
  // periodic auto-refresh every 20s (lightweight)
  const autoTimer = setInterval(()=>{ if(document.body.contains(root)) loadNodes(); else clearInterval(autoTimer); },8000);
  // periodic pending refresh (if settings tab open)
  setInterval(()=>{ if(document.body.contains(root)) refreshPending(false); },15000);

  // --- Registration & Approvals (inside Settings panel) ---
  async function refreshPending(manual){
    try{
      const data = await secureRequest('/api/v1/nodes/pending');
      const wrap = document.getElementById('pending-list');
      if(!wrap) return;
      const list = data.pending||[];
      wrap.innerHTML = list.length? list.map(p=>`<div class="pending-row"><code>${p.node_id}</code> <span class="p-status ${p.status}">${p.status}</span> <button class="btn-small" data-approve="${p.node_id}">Approve</button> <button class="btn-small" data-deny="${p.node_id}">Deny</button></div>`).join(''):'<em style="font-size:10px;color:#9bb5b7">No pending nodes</em>';
      wrap.querySelectorAll('button[data-approve]').forEach(b=> b.onclick=()=> approveNode(b.dataset.approve,true));
      wrap.querySelectorAll('button[data-deny]').forEach(b=> b.onclick=()=> denyNode(b.dataset.deny,true));
      if(manual && window.notify) window.notify('info','Nodes','Pending refreshed');
    }catch(e){ console.warn('pending refresh failed',e); }
  }
  async function approveNode(id,notify){ try{ await secureRequest('/api/v1/nodes/approve',{method:'POST',body:{node_id:id}}); if(notify&&window.notify) window.notify('success','Approve',id); refreshPending(false); loadNodes(true);}catch(e){ if(window.notify) window.notify('error','Approve', e.message);} }
  async function denyNode(id,notify){ try{ await secureRequest('/api/v1/nodes/deny',{method:'POST',body:{node_id:id}}); if(notify&&window.notify) window.notify('warn','Deny',id); refreshPending(false); loadNodes(true);}catch(e){ if(window.notify) window.notify('error','Deny', e.message);} }
  function openRegistration(){
    const panel = document.getElementById('settings-body');
    if(!panel) return;
    if(!panel.querySelector('#reg-form')){
      panel.innerHTML = `<div class="card" id="reg-form">
        <h4>Register Node</h4>
        <div class="reg-grid">
          <input id="reg-id" placeholder="node_id" />
          <input id="reg-type" placeholder="type (compute/storage/network)" />
          <input id="reg-hostname" placeholder="hostname" />
          <input id="reg-ip" placeholder="ip_address" />
          <input id="reg-port" placeholder="port" value="8000" />
          <input id="reg-cpu" placeholder="cpu_cores" />
          <input id="reg-mem" placeholder="memory_gb" />
          <textarea id="reg-pubkey" placeholder="device public key PEM (optional advanced)" style="grid-column:1/3;height:70px;background:#141a1c;border:1px solid #2b3a3d;color:#e9f9fa;padding:6px 8px;font-size:10px;border-radius:3px"></textarea>
          <input id="reg-fingerprint" placeholder="device fingerprint hash" style="grid-column:1/3" />
          <input id="reg-signature" placeholder="signed challenge (base64 of signature over node_id+fingerprint)" style="grid-column:1/3" />
          <button class="nodes-btn" id="reg-submit" style="grid-column:1/3">Submit</button>
        </div>
      </div>
      <div class="card"><h4>Pending Approvals</h4><div id="pending-list" style="display:flex;flex-direction:column;gap:4px;font-size:10px"></div><button class="btn-small" id="pending-refresh">Refresh</button></div>`;
      panel.querySelector('#pending-refresh').onclick=()=> refreshPending(true);
      panel.querySelector('#reg-submit').onclick= submitRegistration;
      refreshPending(false);
    }
    // Switch settings subtab if not active
    const btn = root.querySelector('.nodes-subtab[data-panel="settings"]'); if(btn){ btn.click(); }
  }
  async function submitRegistration(){
    const id=document.getElementById('reg-id').value.trim();
    const type=document.getElementById('reg-type').value.trim()||'compute';
    const hn=document.getElementById('reg-hostname').value.trim()||id;
    const ip=document.getElementById('reg-ip').value.trim();
    const port=parseInt(document.getElementById('reg-port').value.trim()||'8000',10);
    if(!id || !ip){ if(window.notify) window.notify('error','Register','id & ip required'); return; }
    const cpu=parseInt(document.getElementById('reg-cpu').value.trim()||'0',10)||0;
    const mem=parseFloat(document.getElementById('reg-mem').value.trim()||'0')||0;
    const pub=document.getElementById('reg-pubkey').value.trim();
    const fp=document.getElementById('reg-fingerprint').value.trim();
    const sig=document.getElementById('reg-signature').value.trim();
    try{
      if(pub && fp && sig){
        // advanced secure registration
        await secureRequest('/api/secure/nodes/register',{method:'POST',body:{
          node_id:id,node_type:type,hostname:hn,ip_address:ip,port,
          resources:{cpu_cores:cpu,memory_gb:mem},
          permissions:[],description:null,device_fingerprint:fp,public_key_pem:pub,signed_challenge:sig,health_attestation:null,device_certificate:null,geoip:null,behavioral_baseline:null
        }});
      } else {
        await secureRequest('/api/v1/nodes/register',{method:'POST',body:{node_id:id,node_type:type,hostname:hn,ip_address:ip,port,resources:{cpu_cores:cpu,memory_gb:mem}}});
      }
      if(window.notify) window.notify('success','Register',id);
      loadNodes(true); refreshPending(false);
    }catch(e){ if(window.notify) window.notify('error','Register', e.message); }
  }

  // --- Node maintenance helpers ---
  async function nodeQuarantine(id, enable){ try{ await secureRequest('/api/secure/nodes/quarantine',{method:'POST',body:{node_id:id,enable}}); if(window.notify) window.notify('info','Quarantine', enable?'Enabled':'Disabled'); loadNodes(true); }catch(e){ if(window.notify) window.notify('error','Quarantine', e.message);} }
  async function removeNode(id){ if(!confirm('Remove node '+id+'?')) return; try{ await secureRequest('/api/secure/nodes/remove',{method:'POST',body:{node_id:id}}); if(window.notify) window.notify('warn','Remove', id); loadNodes(true);}catch(e){ if(window.notify) window.notify('error','Remove', e.message);} }
  async function probeNode(id){ try{ const r=await secureRequest(`/api/secure/nodes/${id}/probe`,{method:'POST'}); if(window.notify) window.notify('info','Probe', Object.entries(r.probe||{}).map(([k,v])=>k+':'+v).join(' ')); }catch(e){ if(window.notify) window.notify('error','Probe', e.message);} }
}

function ensureStyles(){ if(document.getElementById('nodes-v2-styles')) return; const s=document.createElement('style'); s.id='nodes-v2-styles'; s.textContent=`
  /* Resized layout: narrower detail pane & compact buttons */
  /* Height adjustments (more compact) */
  .nodes-container-v2{display:grid;grid-template-columns:1fr 260px;gap:14px;height:auto;padding:10px;box-sizing:border-box;font:400 11px system-ui}
  .nodes-status-header{grid-column:1/3;display:flex;justify-content:space-between;align-items:center;background:#1d2426;border:1px solid #2b3a3d;padding:8px 10px;border-radius:4px}
  .nodes-overview{display:flex;gap:14px}
  .nodes-metric{display:flex;align-items:center;gap:4px;font:400 10px system-ui}
  .nodes-actions{display:flex;flex-wrap:wrap;gap:4px;max-width:260px}
  .nodes-btn{background:#141a1c;border:1px solid #2b3a3d;color:#e9f9fa;padding:4px 8px;border-radius:3px;cursor:pointer;font:500 9px system-ui;line-height:1.1;min-height:26px}
  .nodes-btn:not(:hover){opacity:.92}
  .nodes-btn:hover{border-color:#00e6d1}
  .nodes-subtabs{grid-column:1/3;display:flex;gap:4px;border-bottom:1px solid #2b3a3d}
  .nodes-subtab{background:#1d2426;border:1px solid #2b3a3d;border-bottom:none;color:#b9d7d9;padding:8px 14px;cursor:pointer;font:400 11px system-ui}
  .nodes-subtab{padding:6px 10px;font-size:10px}
  .nodes-subtab.active{background:#111617;color:#00e6d1;border-color:#00e6d1}
  .nodes-panels{grid-column:1/2}
  .nodes-panel{display:none}
  .nodes-panel.active{display:block;animation:fadeIn .18s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
  .nodes-toolbar{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:8px 0 10px}
  .nodes-filters{display:flex;flex-wrap:wrap;gap:6px}
  .nodes-filters input,.nodes-filters select{background:#141a1c;border:1px solid #2b3a3d;color:#e9f9fa;padding:5px 8px;border-radius:3px;height:30px;font:400 11px system-ui}
  .nodes-table{width:100%;border-collapse:collapse;font:400 10px system-ui}
  .nodes-table th{background:#141a1c;color:#00e6d1;text-align:left;padding:6px 8px;font-weight:600;position:sticky;top:0}
  .nodes-table th.sortable{cursor:pointer;user-select:none}
  .nodes-table th.sortable.sorted{color:#fff;text-decoration:underline}
  .nodes-table td{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.05);color:#e9f9fa}
  .nodes-table tr:hover{background:#141a1c}
  .nodes-table .empty td{text-align:center;padding:28px 8px;color:#9bb5b7}
  .status-dot{width:10px;height:10px;display:inline-block;border-radius:50%;background:#777}
  .status-dot.online{background:#00e6d1}
  .status-dot.offline{background:#ff6d6d}
  .nodes-topology-layout{display:grid;grid-template-columns:1fr 260px;gap:10px;height:340px}
  .nodes-topology-canvas{background:#1d2426;border:1px solid #2b3a3d;border-radius:4px;position:relative;height:100%;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#9bb5b7}
  .nodes-topology-controls{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;background:#141a1c;border-bottom:1px solid #2b3a3d}
  .nodes-topology-tools{display:flex;gap:4px}
  .btn-small{background:#1d2426;border:1px solid #2b3a3d;color:#e9f9fa;padding:4px 6px;font-size:10px;cursor:pointer;border-radius:3px}
  .btn-small:hover{border-color:#00e6d1}
  .nodes-topology-sidebar{display:flex;flex-direction:column;gap:12px}
  .nodes-topology-panel{background:#1d2426;border:1px solid #2b3a3d;border-radius:4px;display:flex;flex-direction:column}
  .nodes-topology-panel .panel-header{padding:6px 8px;font:600 11px system-ui;border-bottom:1px solid #2b3a3d;color:#00e6d1}
  .panel-body{padding:6px 8px;overflow:auto;font-size:10px;line-height:1.3}
  .nodes-detail{grid-column:2/3;background:#1d2426;border:1px solid #2b3a3d;border-radius:4px;display:flex;flex-direction:column;min-width:0;max-height:360px;overflow:hidden}
  .detail-header{padding:8px 10px;font:600 11px system-ui;border-bottom:1px solid #2b3a3d;color:#00e6d1}
  .detail-subtabs{display:flex;gap:4px;padding:6px 8px;border-bottom:1px solid #2b3a3d;flex-wrap:wrap}
  .detail-subtabs .tab{background:#141a1c;border:1px solid #2b3a3d;color:#b9d7d9;padding:4px 10px;font-size:10px;cursor:pointer;border-radius:3px}
  .detail-subtabs .tab.active{background:#00e6d1;color:#092121;border-color:#00e6d1}
  .detail-body{padding:8px 10px;font-size:10px;overflow:auto;flex:1;display:flex;flex-direction:column;gap:10px;max-height:300px}
  .card{background:#141a1c;border:1px solid #2b3a3d;border-radius:4px;padding:8px}
  .card h4{margin:0 0 6px;font:600 10px system-ui;color:#00e6d1}
  .trust-badge{display:inline-block;padding:2px 4px;font-size:8px;border-radius:3px;text-transform:uppercase;letter-spacing:.5px;background:#2b3a3d;color:#e9f9fa;margin-left:4px}
  .trust-badge.trusted{background:#00e6d1;color:#062e2c}
  .trust-badge.untrusted{background:#ff6d6d;color:#2b0000}
  .trust-badge.quarantined{background:#ffb347;color:#332100}
  .pol-row{display:flex;align-items:center;gap:6px;font-size:10px;margin:2px 0}
  .spark{width:100%;background:#101415;border:1px solid #2b3a3d;border-radius:3px}
  .reg-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px}
  .reg-grid input{background:#141a1c;border:1px solid #2b3a3d;color:#e9f9fa;padding:6px 8px;font-size:10px;border-radius:3px}
  .pending-row{display:flex;align-items:center;gap:6px;background:#141a1c;padding:4px 6px;border:1px solid #2b3a3d;border-radius:3px}
  .p-status{font-size:9px;text-transform:uppercase;letter-spacing:.5px;background:#2b3a3d;padding:2px 4px;border-radius:2px}
`; document.head.appendChild(s); }

// End minimal nodes.js

