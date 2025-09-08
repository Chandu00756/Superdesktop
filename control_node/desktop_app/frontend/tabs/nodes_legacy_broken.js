// Backup of previous corrupted nodes.js for reference.
// Clean rewritten Nodes tab (parity with network.js layout)
export function renderNodes(root, state){
  state = state||{data:{}}; state.data.nodes = state.data.nodes||{nodes:[]};
  let nodes = state.data.nodes.nodes||[];
  if(!root) return; root.innerHTML='';
  ensureStyles();
  root.innerHTML = `
  <div id="nodes-container" class="nodes-container-v2">
    <div class="nodes-status-header">
      <div class="nodes-overview">
        <div class="nodes-metric"><i class="fas fa-server"></i><span>Devices: <strong id="nodes-count">${nodes.length}</strong></span></div>
        <div class="nodes-metric"><i class="fas fa-signal"></i><span>Online: <strong id="nodes-online">0</strong></span></div>
        <div class="nodes-metric"><i class="fas fa-microchip"></i><span>Avg CPU: <strong id="nodes-avg-cpu">0%</strong></span></div>
        <div class="nodes-metric"><i class="fas fa-memory"></i><span>Avg MEM: <strong id="nodes-avg-mem">0%</strong></span></div>
        <div class="nodes-metric"><i class="fas fa-exclamation-triangle"></i><span>Alerts: <strong id="nodes-alerts-total">0</strong></span></div>
      </div>
      <div class="nodes-actions">
        <button id="nodes-refresh" class="nodes-btn"><i class="fas fa-sync"></i>Refresh</button>
        <button id="nodes-discover" class="nodes-btn"><i class="fas fa-satellite-dish"></i>Discover</button>
        <button id="nodes-export" class="nodes-btn"><i class="fas fa-download"></i>Export</button>
      </div>
    </div>
    <div class="nodes-subtabs" id="nodes-subtabs">
      <button class="nodes-subtab active" data-panel="list"><i class="fas fa-list"></i><span>List</span></button>
      <button class="nodes-subtab" data-panel="topology"><i class="fas fa-project-diagram"></i><span>Topology</span></button>
      <button class="nodes-subtab" data-panel="resources"><i class="fas fa-cubes"></i><span>Resources</span></button>
      <button class="nodes-subtab" data-panel="security"><i class="fas fa-shield-alt"></i><span>Security</span></button>
      <button class="nodes-subtab" data-panel="plugins"><i class="fas fa-puzzle-piece"></i><span>Plugins</span></button>
      <button class="nodes-subtab" data-panel="settings"><i class="fas fa-sliders-h"></i><span>Settings</span></button>
      <button class="nodes-subtab" data-panel="policies"><i class="fas fa-scroll"></i><span>Policies</span></button>
      <button class="nodes-subtab" data-panel="diagnostics"><i class="fas fa-stethoscope"></i><span>Diagnostics</span></button>
    </div>
    <div class="nodes-panels">
      <div id="nodes-panel-list" class="nodes-panel active">
        <div class="nodes-toolbar">
          <div class="nodes-filters">
            <input id="nodes-search" placeholder="Search..."/>
            <select id="filter-role"><option value="">Role</option></select>
            <select id="filter-status"><option value="">Status</option><option value="online">Online</option><option value="offline">Offline</option><option value="quarantined">Quarantined</option></select>
            <input id="filter-tag" placeholder="Tag"/>
            <select id="filter-zone"><option value="">Zone</option></select>
            <input id="filter-os" placeholder="OS" style="width:90px"/>
            <label class="cpu-filter">CPU ≥ <input id="filter-cpu" type="number" min="0" max="100" style="width:50px"/></label>
          </div>
          <div class="nodes-batch">
            <button id="batch-approve" class="nodes-mini-btn">Approve</button>
            <button id="batch-quarantine" class="nodes-mini-btn">Quarantine</button>
            <button id="batch-remove" class="nodes-mini-btn danger">Remove</button>
            <button id="batch-assign-tags" class="nodes-mini-btn">Tags</button>
          </div>
        </div>
        <div class="nodes-list-grid">
          <div class="nodes-list-main">
            <table class="nodes-table" id="nodes-table">
              <thead><tr><th><input id="select-all-nodes" type="checkbox"/></th><th>Status</th><th>ID</th><th>Type</th><th>Tags</th><th>Zone</th><th>Res</th><th>Agent</th><th>Alerts</th><th>Last Seen</th><th>Actions</th></tr></thead>
              <tbody id="nodes-tbody"></tbody>
            </table>
          </div>
          <div class="nodes-list-side">
            <div class="nodes-side-panel"><div class="panel-header">Alerts</div><div id="alerts-feed" class="panel-body panel-scroll"></div></div>
            <div class="nodes-side-panel"><div class="panel-header">Activity</div>
              <div class="activity-toolbar"><div id="activity-filters" class="activity-filters">
                <button class="act-filter active" data-filter="all">All</button>
                <button class="act-filter" data-filter="events">Events</button>
                <button class="act-filter" data-filter="alerts">Alerts</button>
                <button class="act-filter" data-filter="security">Security</button>
                <button class="act-filter" data-filter="errors">Errors</button>
              </div><div class="activity-meta"><span id="activity-total">0</span></div></div>
              <div id="audit-log" class="panel-body panel-scroll"></div>
            </div>
          </div>
        </div>
      </div>
      <div id="nodes-panel-topology" class="nodes-panel">
        <div class="nodes-topology-layout">
          <div class="nodes-topology-canvas-container">
            <div class="nodes-topology-controls">
              <div class="nodes-topology-tools">
                <button id="nodes-topo-zoom-in" class="btn-small" data-variant="ghost">+</button>
                <button id="nodes-topo-zoom-out" class="btn-small" data-variant="ghost">−</button>
                <button id="nodes-topo-reset" class="btn-small" data-variant="ghost">Reset</button>
                <button id="nodes-topo-autolayout" class="btn-small" data-variant="ghost">Layout</button>
                <button id="nodes-topo-fit" class="btn-small" data-variant="ghost">Fit</button>
                <div class="zoom-box"><input id="nodes-topo-zoom" type="range" min="40" max="200" value="100"/></div>
              </div>
              <div class="nodes-topology-meta"><span id="nodes-topo-stats" class="topology-stats">0 devices • 0 sessions • 0 links</span></div>
            </div>
            <div id="nodes-topology-canvas" class="nodes-topology-canvas"><div class="topology-empty">No nodes</div></div>
          </div>
          <div class="nodes-topology-sidebar">
            <div class="nodes-topology-panel"><div class="panel-header">Entity</div><div class="panel-body" id="nodes-topology-entity-body">Select a node or session</div></div>
            <div class="nodes-topology-panel"><div class="panel-header">Legend</div><div class="panel-body legend-body">
              <div class="legend-item"><span class="legend-color device online"></span><span>Device (Online)</span></div>
              <div class="legend-item"><span class="legend-color device offline"></span><span>Device (Offline)</span></div>
              <div class="legend-item"><span class="legend-color device quarantine"></span><span>Device (Quarantined)</span></div>
              <div class="legend-item"><span class="legend-color session"></span><span>Session</span></div>
            </div></div>
            <div class="nodes-topology-panel"><div class="panel-header">Sessions</div><div class="panel-body" id="nodes-topology-sessions-body">No sessions</div></div>
          </div>
        </div>
      </div>
      <div id="nodes-panel-resources" class="nodes-panel"><div class="card"><h4>Resources</h4><div id="resources-body">Loading...</div></div></div>
      <div id="nodes-panel-security" class="nodes-panel"><div class="card"><h4>Security</h4><div id="security-body">Loading...</div></div></div>
      <div id="nodes-panel-plugins" class="nodes-panel"><div class="card"><h4>Plugins</h4><div id="plugin-market">Loading...</div></div></div>
      <div id="nodes-panel-settings" class="nodes-panel"><div class="card"><h4>Settings</h4><div id="settings-body">Loading...</div></div></div>
      <div id="nodes-panel-policies" class="nodes-panel"><div class="card"><h4>Policies</h4><div>Policy editor placeholder</div></div></div>
      <div id="nodes-panel-diagnostics" class="nodes-panel"><div class="card"><h4>Diagnostics</h4><div id="diag-body">Ready</div></div></div>
    </div>
    <section class="nodes-detail" id="nodes-detail">
      <div class="detail-header">Select a node <span id="detail-protocol" class="protocol-badge" style="display:none;margin-left:8px"></span></div>
      <div id="detail-subtabs" class="detail-subtabs" style="display:none">
        <div class="tab active" data-tab="overview">Overview</div>
        <div class="tab" data-tab="metrics">Metrics</div>
        <div class="tab" data-tab="processes">Processes</div>
        <div class="tab" data-tab="logs">Logs</div>
        <div class="tab" data-tab="security">Security</div>
        <div class="tab" data-tab="maintenance">Maint</div>
        <div class="tab" data-tab="policies">Policies</div>
        <div class="tab" data-tab="diagnostics">Diag</div>
      </div>
      <div class="detail-body" id="detail-body">No selection</div>
    </section>
  </div>`;

  // --- State ---
  let selected=null; let audit=[]; let activeActivityFilter='all';

  // Subtabs
  root.querySelectorAll('.nodes-subtab').forEach(b=> b.addEventListener('click',()=>{ root.querySelectorAll('.nodes-subtab').forEach(x=>x.classList.remove('active')); b.classList.add('active'); const p=b.dataset.panel; root.querySelectorAll('.nodes-panel').forEach(pn=> pn.classList.toggle('active', pn.id==='nodes-panel-'+p)); if(p==='topology') renderTopology(); if(p==='resources') renderResources(); if(p==='security') renderSecurity(); if(p==='plugins') renderPlugins(); if(p==='settings') renderSettings(); }));

  // Table
  function lastSeenText(n){ if(!n||!n.last_seen) return '—'; try{ const diff=Date.now()-new Date(n.last_seen).getTime(); if(diff<60000) return 'just now'; const m=Math.floor(diff/60000); if(m<60) return m+'m'; const h=Math.floor(m/60); if(h<48) return h+'h'; return Math.floor(h/24)+'d'; }catch(e){return '—';} }
  function renderTable(){ const tbody=root.querySelector('#nodes-tbody'); if(!tbody) return; tbody.innerHTML=''; let rendered=0; nodes.forEach(n=>{ const tr=document.createElement('tr'); tr.dataset.node=n.node_id; const status=n.status||'offline'; tr.innerHTML=`<td><input class="row-select" data-id="${n.node_id}" type="checkbox"/></td><td><span class="status-dot ${status}"></span></td><td>${n.node_id}</td><td>${n.node_type||'-'}</td><td>${(n.tags||[]).slice(0,3).join(',')}</td><td>${n.zone||'-'}</td><td>${(n.metrics?.cpu_usage||0).toFixed(0)}%</td><td>${n.agent_version||'?'}</td><td>${(n.alerts||0)}</td><td>${lastSeenText(n)}</td><td><button class="btn-small open-detail" data-id="${n.node_id}">Open</button></td>`; tbody.appendChild(tr); rendered++; }); if(!rendered) tbody.innerHTML='<tr class="empty"><td colspan="11">No nodes</td></tr>'; tbody.querySelectorAll('.open-detail').forEach(b=> b.onclick=()=> openDetail(nodes.find(x=>x.node_id===b.dataset.id))); }
  renderTable();

  function openDetail(node){ if(!node) return; selected=node; root.querySelector('.detail-header').textContent = node.node_id+' — '+(node.hostname||node.node_type||'node'); const subtabs=root.querySelector('#detail-subtabs'); subtabs.style.display='flex'; renderDetailTab('overview'); subtabs.querySelectorAll('.tab').forEach(t=> t.onclick=(e)=>{ subtabs.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); e.currentTarget.classList.add('active'); renderDetailTab(e.currentTarget.dataset.tab); }); }
  function renderDetailTab(tab){ const body=root.querySelector('#detail-body'); const n=selected; if(!n){ body.textContent='No selection'; return;} switch(tab){ case 'overview': body.innerHTML=`<div class="card"><h4>System</h4><div>ID: ${n.node_id}</div><div>Host: ${n.hostname||'-'}</div><div>IP: ${n.ip_address||'-'}</div><div>Role: ${n.node_type||'-'}</div><div>Status: ${n.status||'-'}</div></div>`; break; case 'metrics': body.innerHTML=`<div class="card"><h4>Metrics</h4><div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}%</div><div>MEM: ${(n.metrics?.memory_usage||0).toFixed(1)}%</div></div>`; break; case 'logs': body.innerHTML=`<div class="card"><h4>Logs</h4><div id="log-output" style="max-height:260px;overflow:auto"></div></div>`; break; case 'security': body.innerHTML=`<div class="card"><h4>Security</h4><div>Protocol: ${n.protocol||'?'}</div><div>mTLS: ${n.mtls?'yes':'no'}</div></div>`; break; default: body.innerHTML=`<div class="card"><h4>${tab}</h4><div>Coming soon</div></div>`; }}

  // Activity
  root.querySelectorAll('.act-filter').forEach(f=> f.addEventListener('click',()=>{ root.querySelectorAll('.act-filter').forEach(x=>x.classList.remove('active')); f.classList.add('active'); activeActivityFilter=f.dataset.filter; renderAudit(); }));
  function pushAudit(msg,cat='events'){ audit.unshift({ts:Date.now(),msg,cat}); audit=audit.slice(0,400); renderAudit(); }
  function renderAudit(){ const feed=root.querySelector('#audit-log'); if(!feed) return; const filtered=audit.filter(a=> activeActivityFilter==='all'||a.cat===activeActivityFilter); feed.innerHTML=filtered.slice(0,120).map(a=>`<div class="audit-row ${a.cat}"><span class="ts">${new Date(a.ts).toLocaleTimeString()}</span> ${a.msg}</div>`).join(''); root.querySelector('#activity-total').textContent=audit.length; }

  // Topology
  const topo={zoom:1,offsetX:0,offsetY:0,entities:[],links:[],selected:null};
  function buildGraph(){ topo.entities=[]; topo.links=[]; const sessionMap=new Map(); nodes.forEach(n=> (n.tags||[]).filter(t=>t.startsWith('session:')).map(t=>t.split(':')[1]).forEach(s=>{ if(!sessionMap.has(s)) sessionMap.set(s,new Set()); sessionMap.get(s).add(n.node_id); })); const sessions=[...sessionMap.keys()].map(id=>({type:'session',id})); const devices=nodes.map(n=>({type:'device',id:n.node_id,ref:n})); topo.entities=[...sessions,...devices]; sessions.forEach(s=> (sessionMap.get(s.id)||[]).forEach(d=> topo.links.push({a:s.id,b:d,type:'membership'}))); const ids=devices.map(d=>d.id); for(let i=0;i<ids.length;i++) for(let j=i+1;j<ids.length;j++){ let shared=false; for(const set of sessionMap.values()){ if(set.has(ids[i])&&set.has(ids[j])) {shared=true; break;} } if(shared) topo.links.push({a:ids[i],b:ids[j],type:'shared'}); } autolayout(); }
  function autolayout(){ const sessions=topo.entities.filter(e=>e.type==='session'); const devices=topo.entities.filter(e=>e.type==='device'); const left=160,right=640,v=50,top=60; sessions.forEach((s,i)=>{ s.x=left; s.y=top+i*v; }); devices.forEach((d,i)=>{ d.x=right; d.y=top+i*v; }); }
  function fitView(){ const c=root.querySelector('#nodes-topology-canvas canvas'); if(!c) return; const ents=topo.entities; if(!ents.length) return; const minX=Math.min(...ents.map(e=>e.x)),maxX=Math.max(...ents.map(e=>e.x)),minY=Math.min(...ents.map(e=>e.y)),maxY=Math.max(...ents.map(e=>e.y)); const w=maxX-minX,h=maxY-minY,cw=c.width,ch=c.height,m=80; const zx=(cw-m)/w, zy=(ch-m)/h; topo.zoom=Math.min(2,Math.max(.4,Math.min(zx,zy))); topo.offsetX=(cw/2)-(minX+w/2)*topo.zoom; topo.offsetY=(ch/2)-(minY+h/2)*topo.zoom; root.querySelector('#nodes-topo-zoom').value=Math.round(topo.zoom*100); }
  function drawTopo(){ const wrap=root.querySelector('#nodes-topology-canvas'); if(!wrap) return; let cv=wrap.querySelector('canvas'); if(!cv){ cv=document.createElement('canvas'); wrap.innerHTML=''; wrap.appendChild(cv);} cv.width=wrap.clientWidth||800; cv.height=wrap.clientHeight||480; const ctx=cv.getContext('2d'); ctx.clearRect(0,0,cv.width,cv.height); ctx.save(); ctx.translate(topo.offsetX,topo.offsetY); ctx.scale(topo.zoom,topo.zoom); topo.links.forEach(l=>{ const a=topo.entities.find(e=>e.id===l.a); const b=topo.entities.find(e=>e.id===l.b); if(!a||!b) return; ctx.beginPath(); ctx.strokeStyle=l.type==='shared'?'rgba(0,230,209,0.18)':'rgba(0,230,209,0.35)'; ctx.lineWidth=l.type==='shared'?1:1.4; ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }); topo.entities.forEach(ent=>{ if(ent.type==='device'){ const n=ent.ref; const status=n.status||'offline'; let fill='#2c3233',stroke='rgba(0,230,209,0.65)'; if(status==='online') fill='#03534c'; else if(status==='quarantined'){ fill='#5a4720'; stroke='rgba(255,180,60,0.6)'; } else if(status==='offline') stroke='rgba(255,90,90,0.6)'; if(topo.selected && topo.selected.id===ent.id) stroke='#00e6d1'; ctx.beginPath(); ctx.lineWidth=2; ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.arc(ent.x,ent.y,18,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(n.node_id.slice(0,6),ent.x,ent.y+30); } else { ctx.save(); ctx.translate(ent.x,ent.y); ctx.rotate(Math.PI/4); ctx.beginPath(); ctx.lineWidth=2; ctx.strokeStyle=(topo.selected&&topo.selected.id===ent.id)?'#00e6d1':'rgba(0,230,209,0.6)'; ctx.fillStyle='rgba(0,230,209,0.1)'; ctx.rect(-14,-14,28,28); ctx.fill(); ctx.stroke(); ctx.restore(); ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(ent.id.slice(0,6),ent.x,ent.y+30); } }); ctx.restore(); }
  function updateInfo(ent){ const body=root.querySelector('#nodes-topology-entity-body'); if(!body) return; if(!ent){ body.textContent='Select a node or session'; return;} if(ent.type==='device'){ const n=ent.ref; body.innerHTML=`<div style="display:flex;flex-direction:column;gap:6px"><strong>${n.node_id}</strong><div>Status: ${n.status||'-'}</div><div>Role: ${n.node_type||'-'}</div><div>Zone: ${n.zone||'-'}</div><div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}%</div></div>`; } else { const members=topo.links.filter(l=>l.type==='membership'&&l.a===ent.id).length; body.innerHTML=`<div><strong>Session ${ent.id}</strong><div>Devices: ${members}</div></div>`; } }
  function updateSessionList(){ const sb=root.querySelector('#nodes-topology-sessions-body'); if(!sb) return; const sessions=topo.entities.filter(e=>e.type==='session'); if(!sessions.length){ sb.textContent='No sessions'; return;} sb.innerHTML=sessions.map(s=>`<div class="session-row" data-sid="${s.id}">${s.id}</div>`).join(''); sb.querySelectorAll('.session-row').forEach(r=> r.onclick=()=>{ topo.selected=topo.entities.find(e=>e.id===r.dataset.sid); updateInfo(topo.selected); drawTopo();}); }
  function updateStats(){ const st=root.querySelector('#nodes-topo-stats'); if(st) st.textContent=`${topo.entities.filter(e=>e.type==='device').length} devices • ${topo.entities.filter(e=>e.type==='session').length} sessions • ${topo.links.length} links`; }
  function wireCanvas(){ const cv=root.querySelector('#nodes-topology-canvas canvas'); if(!cv) return; let dragging=false,lastX=0,lastY=0; cv.onmousedown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;}; window.addEventListener('mouseup',()=>dragging=false); window.addEventListener('mousemove',e=>{ if(!dragging) return; const dx=e.clientX-lastX,dy=e.clientY-lastY; lastX=e.clientX;lastY=e.clientY; topo.offsetX+=dx; topo.offsetY+=dy; drawTopo();}); cv.onwheel=e=>{ e.preventDefault(); topo.zoom=Math.min(2.5,Math.max(.3,topo.zoom+(e.deltaY>0?-0.1:0.1))); root.querySelector('#nodes-topo-zoom').value=Math.round(topo.zoom*100); drawTopo();}; cv.onclick=e=>{ const r=cv.getBoundingClientRect(); const x=(e.clientX-r.left - topo.offsetX)/topo.zoom; const y=(e.clientY-r.top - topo.offsetY)/topo.zoom; let hit=null; for(const ent of topo.entities){ if(ent.type==='device'){ const dx=ent.x-x,dy=ent.y-y; if(dx*dx+dy*dy<18*18){ hit=ent; break;} } else { const dx=x-ent.x,dy=y-ent.y; const u=(dx+dy)/Math.SQRT2,v=(dy-dx)/Math.SQRT2; if(Math.abs(u)<14&&Math.abs(v)<14){ hit=ent; break;} } } topo.selected=hit; if(hit&&hit.type==='device') openDetail(hit.ref); updateInfo(hit); drawTopo(); }; }
  function wireControls(){ root.querySelector('#nodes-topo-zoom-in')?.addEventListener('click',()=>{ topo.zoom=Math.min(2.5,topo.zoom+0.1); root.querySelector('#nodes-topo-zoom').value=Math.round(topo.zoom*100); drawTopo();}); root.querySelector('#nodes-topo-zoom-out')?.addEventListener('click',()=>{ topo.zoom=Math.max(.3,topo.zoom-0.1); root.querySelector('#nodes-topo-zoom').value=Math.round(topo.zoom*100); drawTopo();}); root.querySelector('#nodes-topo-reset')?.addEventListener('click',()=>{ topo.zoom=1; topo.offsetX=0; topo.offsetY=0; root.querySelector('#nodes-topo-zoom').value=100; drawTopo();}); root.querySelector('#nodes-topo-autolayout')?.addEventListener('click',()=>{ autolayout(); fitView(); drawTopo();}); root.querySelector('#nodes-topo-fit')?.addEventListener('click',()=>{ fitView(); drawTopo();}); root.querySelector('#nodes-topo-zoom')?.addEventListener('input',e=>{ topo.zoom=Number(e.target.value)/100; drawTopo();}); }
  function renderTopology(){ buildGraph(); drawTopo(); fitView(); drawTopo(); updateSessionList(); updateStats(); wireCanvas(); wireControls(); }

  // Other panels placeholders
  function renderResources(){ const b=root.querySelector('#resources-body'); if(b) b.textContent='Aggregate resource view (todo)'; }
  function renderSecurity(){ const b=root.querySelector('#security-body'); if(b) b.textContent='Security overview (certs, RBAC, mTLS)'; }
  function renderPlugins(){ const b=root.querySelector('#plugin-market'); if(b) b.textContent='Plugins list (todo)'; }
  function renderSettings(){ const b=root.querySelector('#settings-body'); if(b) b.innerHTML=`<div>Cluster Name: <input id="set-cluster-name" value="${state.data.cluster_name||''}"/></div><div style="margin-top:8px"><button id="save-settings" class="btn-small">Save</button></div>`; }

  // Metrics header updater
  function updateHeaderMetrics(){ const online=nodes.filter(n=>n.status==='online').length; const avgCpu=nodes.length? (nodes.reduce((s,n)=> s+(n.metrics?.cpu_usage||0),0)/nodes.length).toFixed(1):'0.0'; const avgMem=nodes.length? (nodes.reduce((s,n)=> s+(n.metrics?.memory_usage||0),0)/nodes.length).toFixed(1):'0.0'; root.querySelector('#nodes-online').textContent=online; root.querySelector('#nodes-avg-cpu').textContent=avgCpu+'%'; root.querySelector('#nodes-avg-mem').textContent=avgMem+'%'; }
  updateHeaderMetrics();
  pushAudit('Nodes interface initialized'); if(nodes.length) pushAudit(nodes.length+' nodes loaded');
  root.querySelector('#nodes-refresh')?.addEventListener('click',()=>{ pushAudit('Manual refresh'); renderTable(); if(root.querySelector('#nodes-panel-topology').classList.contains('active')) renderTopology(); updateHeaderMetrics(); });
  root.querySelector('#nodes-discover')?.addEventListener('click',()=> pushAudit('Discovery triggered','events'));
  root.querySelector('#nodes-export')?.addEventListener('click',()=> pushAudit('Export started','events'));
}

function ensureStyles(){ if(document.getElementById('nodes-v2-styles')) return; const s=document.createElement('style'); s.id='nodes-v2-styles'; s.textContent=`
  .nodes-container-v2{display:grid;grid-template-columns:1fr 360px;gap:16px;height:100%;padding:12px;box-sizing:border-box;font:400 11px var(--font-mono,system-ui)}
  .nodes-status-header{grid-column:1/3;display:flex;justify-content:space-between;align-items:center;background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);padding:10px 14px;border-radius:4px}
  .nodes-overview{display:flex;gap:22px}
  .nodes-metric{display:flex;align-items:center;gap:6px;font:400 11px var(--font-mono)}
  .nodes-actions{display:flex;gap:8px}
  .nodes-btn{background:var(--omega-dark-4,#141a1c);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-white,#e9f9fa);padding:6px 12px;border-radius:3px;cursor:pointer;font:400 10px var(--font-mono);transition:.15s}
  .nodes-btn:hover{border-color:var(--omega-cyan,#00e6d1)}
  .nodes-subtabs{grid-column:1/3;display:flex;gap:4px;border-bottom:1px solid var(--omega-gray-1,#2b3a3d)}
  .nodes-subtab{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);border-bottom:none;color:var(--omega-light-1,#b9d7d9);padding:8px 14px;cursor:pointer;display:flex;gap:6px;align-items:center;font:400 11px var(--font-mono);transition:.15s}
  .nodes-subtab.active{background:var(--omega-dark-1,#111617);color:var(--omega-cyan,#00e6d1);border-color:var(--omega-cyan,#00e6d1)}
  .nodes-panels{position:relative;grid-column:1/2}
  .nodes-panel{display:none;animation:fadeIn .18s ease}
  .nodes-panel.active{display:block}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
  .nodes-toolbar{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:8px 0 10px}
  .nodes-filters{display:flex;flex-wrap:wrap;gap:6px}
  .nodes-filters input,.nodes-filters select{background:var(--omega-dark-4,#141a1c);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-white,#e9f9fa);padding:5px 8px;border-radius:3px;height:30px;font:400 11px var(--font-mono)}
  .nodes-batch{display:flex;gap:6px}
  .nodes-mini-btn{background:var(--omega-dark-4,#141a1c);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-white,#e9f9fa);padding:4px 8px;border-radius:3px;cursor:pointer;font:400 10px var(--font-mono);transition:.15s}
  .nodes-mini-btn:hover{border-color:var(--omega-cyan,#00e6d1)}
  .nodes-mini-btn.danger{border-color:#a44242;color:#ffa8a8}
  .nodes-list-grid{display:grid;grid-template-columns:1fr 320px;gap:12px}
  .nodes-table{width:100%;border-collapse:collapse;font:400 10px var(--font-mono)}
  .nodes-table th{background:var(--omega-dark-4,#141a1c);color:var(--omega-cyan,#00e6d1);text-align:left;padding:6px 8px;font-weight:600;position:sticky;top:0}
  .nodes-table td{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.05);color:var(--omega-white,#e9f9fa)}
  .nodes-table tr:hover{background:var(--omega-dark-4,#141a1c)}
  .nodes-table .empty td{text-align:center;padding:28px 8px;color:var(--omega-light-1,#9bb5b7)}
  .status-dot{width:10px;height:10px;display:inline-block;border-radius:50%;background:#777}
  .status-dot.online{background:#00e6d1}
  .status-dot.offline{background:#ff6d6d}
  .status-dot.quarantined{background:#f5c041}
  .nodes-list-side{display:flex;flex-direction:column;gap:12px}
  .nodes-side-panel{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);border-radius:4px;display:flex;flex-direction:column;max-height:240px}
  .nodes-side-panel .panel-header{font:600 11px var(--font-mono);padding:8px 10px;border-bottom:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-cyan,#00e6d1)}
  .panel-body{padding:6px 8px;overflow:auto;font-size:10px;line-height:1.3}
  .activity-toolbar{display:flex;justify-content:space-between;align-items:center;padding:4px 6px;border-bottom:1px solid var(--omega-gray-1,#2b3a3d);background:var(--omega-dark-4,#141a1c)}
  .activity-filters{display:flex;gap:4px}
  .activity-filters .act-filter{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-light-1,#b9d7d9);padding:3px 8px;font-size:10px;cursor:pointer;border-radius:3px}
  .activity-filters .act-filter.active{background:var(--omega-cyan,#00e6d1);color:#062b29}
  .audit-row{padding:2px 0;border-bottom:1px dashed rgba(255,255,255,0.04)}
  .nodes-topology-layout{display:grid;grid-template-columns:1fr 320px;gap:12px;height:460px}
  .nodes-topology-canvas{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);border-radius:4px;position:relative;height:100%;overflow:hidden;display:flex;align-items:center;justify-content:center;color:var(--omega-light-1,#9bb5b7)}
  .nodes-topology-controls{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;background:var(--omega-dark-4,#141a1c);border-bottom:1px solid var(--omega-gray-1,#2b3a3d)}
  .nodes-topology-tools{display:flex;gap:4px}
  .nodes-topology-tools .btn-small{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-white,#e9f9fa);padding:4px 6px;font-size:10px;cursor:pointer;border-radius:3px}
  .nodes-topology-tools .btn-small:hover{border-color:var(--omega-cyan,#00e6d1)}
  .nodes-topology-sidebar{display:flex;flex-direction:column;gap:12px}
  .nodes-topology-panel{background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);border-radius:4px;display:flex;flex-direction:column}
  .nodes-topology-panel .panel-header{padding:6px 8px;font:600 11px var(--font-mono);border-bottom:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-cyan,#00e6d1)}
  .legend-item{display:flex;align-items:center;gap:6px;font-size:10px}
  .legend-color{width:12px;height:12px;border-radius:3px;background:#00e6d1}
  .legend-color.device.offline{background:#ff6d6d}
  .legend-color.device.quarantine{background:#f5c041}
  .legend-color.session{background:#00e6d1;opacity:.5}
  .nodes-detail{grid-column:2/3;background:var(--omega-dark-3,#1d2426);border:1px solid var(--omega-gray-1,#2b3a3d);border-radius:4px;display:flex;flex-direction:column}
  .detail-header{padding:10px 12px;font:600 12px var(--font-mono);border-bottom:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-cyan,#00e6d1)}
  .detail-subtabs{display:flex;gap:4px;padding:6px 8px;border-bottom:1px solid var(--omega-gray-1,#2b3a3d);flex-wrap:wrap}
  .detail-subtabs .tab{background:var(--omega-dark-4,#141a1c);border:1px solid var(--omega-gray-1,#2b3a3d);color:var(--omega-light-1,#b9d7d9);padding:4px 10px;font-size:10px;cursor:pointer;border-radius:3px}
  .detail-subtabs .tab.active{background:var(--omega-cyan,#00e6d1);color:#092121;border-color:var(--omega-cyan,#00e6d1)}
  .detail-body{padding:10px 12px;font-size:11px;overflow:auto;flex:1;display:flex;flex-direction:column;gap:12px}
  .card{background:var(--omega-dark-4,#141a1c);border:1px solid var(--omega-gray-1,#2b3a3d);border-radius:4px;padding:10px}
  .card h4{margin:0 0 8px;font:600 11px var(--font-mono);color:var(--omega-cyan,#00e6d1)}
`; document.head.appendChild(s); }
  // RBAC check helper
  // permission cache + RBAC helper
  const permissionCache = new Set();
  let permissionsLoaded = false;
  async function loadPermissions() {
    // best-effort: try window.omegaAPI, then /api/v1/me/permissions
    try {
      let res = null;
      const client = getApiClient();
      if (client && typeof client._req === 'function') {
        try { res = await client._req('/api/secure/me/permissions'); } catch(e) { res = null; }
      }
      if (!res && window.omegaAPI && typeof window.omegaAPI._req === 'function') {
        try { res = await window.omegaAPI._req('/api/v1/me/permissions'); } catch(e) { res = null; }
      }
      if (!res) {
        try { res = await apiRequest('/api/v1/me/permissions'); } catch(e){ res = null; }
      }
      // accept shape { permissions: [...] } or array directly
      const perms = Array.isArray(res) ? res : (res && Array.isArray(res.permissions) ? res.permissions : null);
      if (perms) { perms.forEach(p=>permissionCache.add(p)); permissionsLoaded = true; }
    } catch(e) { /* ignore */ }
    permissionsLoaded = true;
  }

  function hasPermission(action) {
    // check cache first
    if (permissionCache.has(action)) return true;
    // if not loaded yet, allow window.omegaAPI check or admin token fallback
    if (!permissionsLoaded) {
      if (window.omegaAPI && typeof window.omegaAPI.hasPermission === 'function') return window.omegaAPI.hasPermission(action);
      return Boolean(window.OMEGA_ADMIN_TOKEN);
    }
    // loaded and not present -> deny
    if (window.omegaAPI && typeof window.omegaAPI.hasPermission === 'function') return window.omegaAPI.hasPermission(action);
    return false;
  }

  // Helper to access secure ApiClient if available
  function getApiClient() {
    try {
      if (window.ApiClient && window.ApiClient.prototype) {
        // older integration where ApiClient is global class
        if (!window._omega_api_client_instance) window._omega_api_client_instance = new window.ApiClient();
        return window._omega_api_client_instance;
      }
      if (window.omegaAPI && window.omegaAPI.client) return window.omegaAPI.client;
      if (window.apiClient) return window.apiClient;
    } catch (e) {}
    return null;
  }

  // Centralized request helper: prefer secure ApiClient._req, then window.omegaAPI._req, then fetch
  async function apiRequest(path, opts = {}) {
    const client = getApiClient();
    const body = opts.body;
    const method = opts.method || (body ? 'POST' : 'GET');
    // normalize headers
    const headers = Object.assign({'Content-Type': body ? 'application/json' : 'text/plain'}, opts.headers || {});
    try {
      if (client && typeof client._req === 'function') {
        return await client._req(path, { method, body: typeof body === 'string' ? body : (body ? JSON.stringify(body) : undefined), headers });
      }
    } catch (e) {
      // fallthrough to legacy
    }
    try {
      if (window.omegaAPI && typeof window.omegaAPI._req === 'function') {
        // translate secure path to legacy when needed
        const legacyPath = path.replace(/^\/api\/secure/, '/api/v1');
        return await window.omegaAPI._req(legacyPath, { method, body: typeof body === 'string' ? body : (body ? JSON.stringify(body) : undefined), headers });
      }
    } catch (e) {}

    // final fallback to fetch (translate secure->v1)
    const fetchPath = path.replace(/^\/api\/secure/, '/api/v1');
    const res = await fetch(fetchPath, { method, body: typeof body === 'string' ? body : (body ? JSON.stringify(body) : undefined), headers });
    if (!res.ok) throw new Error('Request failed: ' + res.status);
    try { return await res.json(); } catch(e){ return null; }
  }

  // Batch action handler
  async function batchAction(action) {
    const selected = Array.from(root.querySelectorAll('.row-select:checked')).map(cb=>cb.dataset.id);
    if (!selected.length) { notify('info','Batch','No nodes selected'); return; }
    if (!hasPermission(action)) { notify('error','Permission','Action not permitted'); return; }
    try {
  await Promise.all(selected.map(id => apiRequest(`/api/secure/nodes/${action}`, { method: 'POST', body: { node_id: id } })));
      notify('success','Batch',`${action} applied to ${selected.length} nodes`);
    } catch(e){ notify('error','Batch',e.message||e); }
  }

  function renderTable(filterText) {
    tbody.innerHTML = '';
    const f = (filterText || '').toLowerCase();
  let rendered = 0;
  const cpuThreshold = Number(root.querySelector('#filter-cpu')?.value || 0);
    nodes.forEach(n => {
      if (f) {
        const hay = `${n.node_id} ${n.node_type} ${(n.tags||[]).join(' ')} ${n.ip_address || ''}`.toLowerCase();
        if (!hay.includes(f)) return;
      }
      const statusBadge = n.status === 'online' ? '<span style="color:#5de68a">●</span>' : (n.status==='quarantined'?'<span style="color:#f5c041">●</span>':'<span style="color:#ff6b6b">●</span>');
      const agent = n.agent_version || '-';
      const alerts = (n.alerts||[]).length || 0;
      const tr = document.createElement('tr');
      tr.dataset.node = n.node_id;
      tr.innerHTML = `
        <td><input type="checkbox" class="row-select" data-id="${n.node_id}"/></td>
        <td>${statusBadge}</td>
        <td class="mono">${n.node_id}</td>
        <td>${n.node_type || '-'}</td>
        <td><span class="protocol-badge">${n.protocol || 'unknown'}</span></td>
        <td>${(n.tags||[]).slice(0,3).map(t=>`<span class="tag">${t}</span>`).join(' ')}</td>
        <td>${n.zone||'-'}</td>
  <td>${Math.round((n.metrics?.cpu_usage||0))}% CPU • ${Math.round((n.metrics?.memory_usage||0))}% MEM<div style="margin-top:6px"><canvas class="spark-canvas" data-id="${n.node_id}" width="80" height="20"></canvas></div></td>
        <td>${agent}</td>
        <td>${alerts}</td>
        <td>${lastSeenText(n)}</td>
  <td><button class="btn-small view-btn" data-id="${n.node_id}">View</button></td>
      `;
  tbody.appendChild(tr);
  rendered++;
    });

    tbody.querySelectorAll('.view-btn').forEach(b => b.onclick = (e) => {
      const id = e.currentTarget.dataset.id;
      const node = nodes.find(x=>x.node_id===id);
      openDetail(node);
    });

    // render small sparklines: prefer Chart.js if available, else fallback drawSparkline
    tbody.querySelectorAll('.spark-canvas').forEach(c=>{
      try {
        const id = c.dataset.id; const n = nodes.find(x=>x.node_id===id);
        const values = (n?.metrics?.history?.cpu||[]).slice(-20);
        if (window.Chart && typeof window.Chart === 'function') {
          try {
            // create tiny line chart
            new Chart(c.getContext('2d'), {
              type: 'line',
              data: { labels: values.map((_,i)=>i), datasets: [{ data: values, borderColor: 'rgba(5,179,200,0.95)', borderWidth: 1, tension: 0.3, pointRadius: 0 }] },
              options: { responsive:false, animation:false, plugins:{ legend:{ display:false } }, scales:{ x:{ display:false }, y:{ display:false } }, elements:{ line:{ borderWidth:1 } } }
            });
            return;
          } catch(e){}
        }
        drawSparkline(c, values);
      } catch(e){}
    });
    // empty state
    if (!rendered) {
      const tr = document.createElement('tr'); tr.className='empty'; tr.innerHTML = `<td colspan="11">No nodes found. Try <button id="quick-discover" class="btn-small">Discover</button> or <button id="quick-register" class="btn-small">Register</button> or check <button id="quick-pending" class="btn-small">Pending</button>.</td>`;
      tbody.appendChild(tr);
      // wire quick actions
      setTimeout(()=>{
        root.querySelector('#quick-discover')?.addEventListener('click', ()=> root.querySelector('#discover-nodes')?.click());
        root.querySelector('#quick-register')?.addEventListener('click', ()=> root.querySelector('#open-register')?.click());
        root.querySelector('#quick-pending')?.addEventListener('click', ()=> root.querySelector('#show-pending')?.click());
      },10);
    }
    // update sidebar count
    const c = root.querySelector('.ns-count'); if (c) c.textContent = `(${nodes.length})`;
  renderTopologyIfActive();
  }

  let selected = null;
  function openDetail(node) {
    selected = node;
    root.querySelector('.detail-header').textContent = node.node_id + ' — ' + (node.hostname || node.node_type || 'node');
  detailSubtabs.style.display = 'flex';
  // show protocol badge
  const pb = root.querySelector('#detail-protocol'); if (pb) { pb.style.display='inline-block'; pb.textContent = (node.protocol || 'unknown') + (node.mtls? ' • mTLS':''); }
    renderDetailTab('overview');
    root.querySelectorAll('#detail-subtabs .tab').forEach(t => t.onclick = (ev)=>{
      root.querySelectorAll('#detail-subtabs .tab').forEach(x=>x.classList.remove('active'));
      ev.currentTarget.classList.add('active');
      renderDetailTab(ev.currentTarget.dataset.tab);
    });
  }

  /* ================= Topology Visualization ================= */
  // Session-aware topology (devices + session hubs) with shared-session links
  const nodesTopology = {
    initialized:false,
    zoom:1,
    offsetX:0,
    offsetY:0,
    seed:Date.now(),
    entities:[], // {type:'device'|'session', id, ref?, x,y}
    links:[],    // {a,b,type:'membership'|'shared'}
    selected:null
  };

  function buildSessionGraph(){
    nodesTopology.entities = [];
    nodesTopology.links = [];
    if (!nodes.length) return;
    // collect session ids from node tags (tag format: session:<id>)
    const deviceEntities = nodes.map(n=>({ type:'device', id:n.node_id, ref:n }));
    const sessionMap = new Map(); // id -> Set(deviceIds)
    deviceEntities.forEach(d=>{
      const tags = Array.isArray(d.ref.tags)? d.ref.tags: [];
      tags.filter(t=> t && typeof t==='string' && t.startsWith('session:'))
        .map(t=> t.split(':')[1])
        .filter(Boolean)
        .forEach(sid=>{ if(!sessionMap.has(sid)) sessionMap.set(sid, new Set()); sessionMap.get(sid).add(d.id); });
    });
    const sessionEntities = Array.from(sessionMap.keys()).map(sid=>({ type:'session', id:sid }));
    nodesTopology.entities = [...sessionEntities, ...deviceEntities];
    // membership links session->device
    sessionEntities.forEach(se=>{ const members = sessionMap.get(se.id) || new Set(); members.forEach(devId=>{
      nodesTopology.links.push({ a: se.id, b: devId, type:'membership' });
    }); });
    // shared-session links between devices (if share >=1 session)
    const devIndex = new Map(deviceEntities.map(d=>[d.id,d]));
    const deviceIds = deviceEntities.map(d=>d.id);
    for (let i=0;i<deviceIds.length;i++){
      for (let j=i+1;j<deviceIds.length;j++){
        const a=deviceIds[i], b=deviceIds[j];
        let shared = false;
        for (const [sid,set] of sessionMap.entries()){
          if (set.has(a) && set.has(b)) { shared=true; break; }
        }
        if (shared) nodesTopology.links.push({ a, b, type:'shared' });
      }
    }
    autolayout();
  }

  function autolayout(){
    const width =  (root.querySelector('#nodes-topology-canvas')?.clientWidth)||800;
    const height = (root.querySelector('#nodes-topology-canvas')?.clientHeight)||480;
    const sessions = nodesTopology.entities.filter(e=>e.type==='session');
    const devices  = nodesTopology.entities.filter(e=>e.type==='device');
    // columns: sessions left, devices right
    const leftX  = 180; const rightX = width-180;
    const vPad = 48; const topMargin=60;
    sessions.forEach((s,i)=>{ s.x = leftX + (Math.sin(i+nodesTopology.seed)*8); s.y = topMargin + i*vPad; });
    devices.forEach((d,i)=>{ d.x = rightX + (Math.sin(i*1.7+nodesTopology.seed)*10); d.y = topMargin + i*vPad; });
    // if exceeds height compress spacing
    const totalHeight = topMargin + Math.max(sessions.length, devices.length)*vPad;
    if (totalHeight > height) {
      const scale = (height-topMargin)/ (Math.max(sessions.length, devices.length)*vPad+1);
      [...sessions, ...devices].forEach(e=>{ e.y = topMargin + (e.y-topMargin)*scale; });
    }
  }

  function fitView(){
    const canvas = root.querySelector('#nodes-topology-canvas canvas');
    if(!canvas) return; const ctx = canvas.getContext('2d');
    const ents = nodesTopology.entities; if(!ents.length) return;
    const minX = Math.min(...ents.map(e=>e.x));
    const maxX = Math.max(...ents.map(e=>e.x));
    const minY = Math.min(...ents.map(e=>e.y));
    const maxY = Math.max(...ents.map(e=>e.y));
    const w = maxX-minX; const h = maxY-minY;
    const cw = canvas.width, ch = canvas.height;
    const margin = 80;
    const zx = (cw - margin)/w; const zy = (ch - margin)/h;
    nodesTopology.zoom = Math.min(2, Math.max(0.4, Math.min(zx, zy)));
    nodesTopology.offsetX = (cw/2) - (minX + w/2)*nodesTopology.zoom;
    nodesTopology.offsetY = (ch/2) - (minY + h/2)*nodesTopology.zoom;
    const slider = root.querySelector('#nodes-topo-zoom'); if (slider) slider.value = Math.round(nodesTopology.zoom*100);
  }

  function drawSessionTopology(){
    const container = root.querySelector('#nodes-topology-canvas');
    if (!container) return;
    let canvas = container.querySelector('canvas');
    if (!canvas) { canvas = document.createElement('canvas'); container.innerHTML=''; container.appendChild(canvas); }
    canvas.width = container.clientWidth || 800;
    canvas.height = container.clientHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.save();
    ctx.translate(nodesTopology.offsetX, nodesTopology.offsetY);
    ctx.scale(nodesTopology.zoom, nodesTopology.zoom);
    // links
    nodesTopology.links.forEach(l=>{
      const a = nodesTopology.entities.find(e=>e.id===l.a);
      const b = nodesTopology.entities.find(e=>e.id===l.b);
      if(!a||!b) return;
      ctx.beginPath();
      if (l.type==='shared') ctx.strokeStyle='rgba(0,230,209,0.18)'; else ctx.strokeStyle='rgba(0,230,209,0.32)';
      ctx.lineWidth = (l.type==='shared'?1:1.6);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });
    // entities
    nodesTopology.entities.forEach(ent=>{
      if (ent.type==='device') {
        const n = ent.ref; const status = n.status||'offline';
        let fill='#303335'; let stroke='rgba(0,230,209,0.65)';
        if(status==='online') fill='#03534c'; else if(status==='quarantined') { fill='#5a4720'; stroke='rgba(255,180,60,0.65)'; } else if(status==='offline') stroke='rgba(255,90,90,0.65)';
        ctx.beginPath(); ctx.lineWidth=2; ctx.fillStyle=fill; ctx.strokeStyle = (nodesTopology.selected && nodesTopology.selected.id===ent.id)?'#00e6d1':stroke; ctx.arc(ent.x, ent.y, 19, 0, Math.PI*2); ctx.fill(); ctx.stroke();
        ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(n.node_id.slice(0,6), ent.x, ent.y+30);
      } else { // session hub (diamond)
        ctx.save(); ctx.translate(ent.x, ent.y); ctx.rotate(Math.PI/4); ctx.beginPath();
        const size = 18; ctx.strokeStyle = (nodesTopology.selected && nodesTopology.selected.id===ent.id)?'#00e6d1':'rgba(0,230,209,0.6)';
        ctx.fillStyle='rgba(0,230,209,0.1)'; ctx.lineWidth=2; ctx.rect(-size/2,-size/2,size,size); ctx.fill(); ctx.stroke(); ctx.restore();
        ctx.fillStyle='#00e6d1'; ctx.font='10px system-ui'; ctx.textAlign='center'; ctx.fillText(ent.id.slice(0,8), ent.x, ent.y+30);
      }
    });
    ctx.restore();
  }

  function updateInfoPanel(entity){
    const body = root.querySelector('#nodes-topology-entity-body'); if(!body) return;
    if(!entity){ body.textContent='Select a node or session'; return; }
    if (entity.type==='device') {
      const n = entity.ref;
      body.innerHTML = `<div style="display:flex;flex-direction:column;gap:6px">\n        <div><strong>${n.node_id}</strong></div>\n        <div>Status: <span style="color:${n.status==='online'?'#00e6d1':(n.status==='quarantined'?'#f5c041':'#ff7878')}">${n.status||'unknown'}</span></div>\n        <div>Role: ${n.node_type||'-'}</div>\n        <div>Zone: ${n.zone||'-'}</div>\n        <div>Protocol: ${n.protocol||'-'} ${n.mtls?'• mTLS':''}</div>\n        <div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}% • MEM: ${(n.metrics?.memory_usage||0).toFixed(1)}%</div>\n        <div style="margin-top:4px"><button id="info-open-detail" class="btn-small" data-variant="ghost">Open Detail</button></div>\n      </div>`;
      root.querySelector('#info-open-detail')?.addEventListener('click', ()=> openDetail(n));
    } else {
      // session summary
      const members = nodesTopology.links.filter(l=> l.type==='membership' && l.a===entity.id).map(l=> l.b);
      body.innerHTML = `<div style="display:flex;flex-direction:column;gap:6px">\n        <div><strong>Session: ${entity.id}</strong></div>\n        <div>Devices: ${members.length}</div>\n        <div style="font-size:11px;opacity:.8">Shared edges depict multi-session overlap between devices.</div>\n      </div>`;
    }
  }

  function updateSessionList(){
    const sb = root.querySelector('#nodes-topology-sessions-body'); if(!sb) return;
    const sessions = nodesTopology.entities.filter(e=>e.type==='session');
    if(!sessions.length){ sb.textContent='No sessions'; return; }
    sb.innerHTML = sessions.map(s=>{
      const memberCount = nodesTopology.links.filter(l=>l.type==='membership' && l.a===s.id).length;
      return `<div class="session-row" data-sid="${s.id}"><span>${s.id}</span><span style="opacity:.7;font-size:11px">${memberCount} dev</span></div>`;
    }).join('');
    sb.querySelectorAll('.session-row').forEach(r=> r.addEventListener('click',()=>{
      const sid = r.dataset.sid; const ent = nodesTopology.entities.find(e=>e.id===sid); nodesTopology.selected = ent; updateInfoPanel(ent); drawSessionTopology(); }));
  }

  function wireCanvasInteractions(){
    const canvas = root.querySelector('#nodes-topology-canvas canvas'); if(!canvas) return;
    let dragging=false; let lastX=0,lastY=0;
    canvas.onmousedown = (e)=>{ dragging=true; lastX=e.clientX; lastY=e.clientY; };
    window.addEventListener('mouseup', ()=> dragging=false);
    window.addEventListener('mousemove', (e)=>{ if(!dragging) return; const dx=e.clientX-lastX; const dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY; nodesTopology.offsetX+=dx; nodesTopology.offsetY+=dy; drawSessionTopology(); });
    canvas.addEventListener('wheel', (e)=>{ e.preventDefault(); const delta = e.deltaY>0?-0.1:0.1; nodesTopology.zoom = Math.min(2.5, Math.max(0.3, nodesTopology.zoom+delta)); const slider=root.querySelector('#nodes-topo-zoom'); if(slider) slider.value = Math.round(nodesTopology.zoom*100); drawSessionTopology(); }, {passive:false});
    canvas.addEventListener('click', (e)=>{
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX-rect.left - nodesTopology.offsetX)/nodesTopology.zoom;
      const y = (e.clientY-rect.top - nodesTopology.offsetY)/nodesTopology.zoom;
      let hit = null;
      for(const ent of nodesTopology.entities){
        if(ent.type==='device') { const dx=ent.x-x, dy=ent.y-y; if (dx*dx+dy*dy < 20*20) { hit=ent; break; } }
        else { // diamond detection via rotation
          const dx = x - ent.x; const dy = y - ent.y; const u = (dx+dy)/Math.SQRT2; const v = (dy-dx)/Math.SQRT2; if (Math.abs(u) < 14 && Math.abs(v) < 14) { hit=ent; break; }
        }
      }
      nodesTopology.selected = hit; updateInfoPanel(hit); if(hit && hit.type==='device') openDetail(hit.ref); drawSessionTopology();
    });
  }

  function updateStats(){
    const stats = root.querySelector('#nodes-topo-stats'); if(!stats) return;
    const deviceCount = nodesTopology.entities.filter(e=>e.type==='device').length;
    const sessionCount = nodesTopology.entities.filter(e=>e.type==='session').length;
    const linkCount = nodesTopology.links.length;
    stats.textContent = `${deviceCount} devices • ${sessionCount} sessions • ${linkCount} links`;
  }

  function ensureControlWiring(){
    if(nodesTopology.controlsWired) return;
    nodesTopology.controlsWired = true;
    root.querySelector('#nodes-topo-zoom-in')?.addEventListener('click', ()=>{ nodesTopology.zoom=Math.min(2.5,nodesTopology.zoom+0.1); const s=root.querySelector('#nodes-topo-zoom'); if(s) s.value=Math.round(nodesTopology.zoom*100); drawSessionTopology(); });
    root.querySelector('#nodes-topo-zoom-out')?.addEventListener('click', ()=>{ nodesTopology.zoom=Math.max(0.3,nodesTopology.zoom-0.1); const s=root.querySelector('#nodes-topo-zoom'); if(s) s.value=Math.round(nodesTopology.zoom*100); drawSessionTopology(); });
    root.querySelector('#nodes-topo-reset')?.addEventListener('click', ()=>{ nodesTopology.zoom=1; nodesTopology.offsetX=0; nodesTopology.offsetY=0; const s=root.querySelector('#nodes-topo-zoom'); if(s) s.value=100; drawSessionTopology(); });
    root.querySelector('#nodes-topo-autolayout')?.addEventListener('click', ()=>{ nodesTopology.seed=Date.now(); autolayout(); fitView(); drawSessionTopology(); });
    root.querySelector('#nodes-topo-fit')?.addEventListener('click', ()=>{ fitView(); drawSessionTopology(); });
    root.querySelector('#nodes-topo-zoom')?.addEventListener('input', (e)=>{ nodesTopology.zoom = Number(e.target.value)/100; drawSessionTopology(); });
    window.addEventListener('resize', ()=>{ drawSessionTopology(); });
  }

  function renderTopologyIfActive(){
    const panel = root.querySelector('#mode-topology');
    if(!panel || panel.style.display==='none') return;
    // build graph each time (cheap for small counts)
    buildSessionGraph();
    const container = root.querySelector('#nodes-topology-canvas');
    if(container && !nodes.length){ container.innerHTML='<div class="topology-empty">No nodes discovered</div>'; }
    else {
      drawSessionTopology();
      fitView();
      drawSessionTopology();
      wireCanvasInteractions();
    }
    updateSessionList();
    updateStats();
    ensureControlWiring();
  }


  // context menu for node rows
  const ctx = root.querySelector('#ctx-menu');
  root.addEventListener('contextmenu', (ev)=>{
    const row = ev.target.closest('tr[data-node]');
    if (!row) return; ev.preventDefault(); const nid = row.dataset.node; ctx.style.left = ev.pageX + 'px'; ctx.style.top = ev.pageY + 'px'; ctx.style.display = 'block'; ctx.innerHTML = `<div style="padding:6px"><div class="ctx-item" data-action="copy">Copy ID</div><div class="ctx-item" data-action="properties">Properties</div><div class="ctx-item" data-action="remove">Remove</div></div>`; ctx.querySelectorAll('.ctx-item').forEach(it=> it.onclick = async (e)=>{ const a = e.currentTarget.dataset.action; ctx.style.display='none'; if (a==='copy') { navigator.clipboard && navigator.clipboard.writeText(nid); notify('info','Clipboard','Copied '+nid); } if (a==='properties') { const node = nodes.find(x=>x.node_id===nid); openDetail(node); } if (a==='remove') { confirmTitle.textContent='Remove Node'; confirmBody.textContent='Remove '+nid+'?'; confirmModal.style.display='flex'; confirmSubmit.onclick = async ()=>{ try{ await apiRequest(`/api/v1/nodes/remove`, { method:'POST', body:{ node_id: nid } }); notify('success','Remove','Removed '+nid); confirmModal.style.display='none'; }catch(e){ notify('error','Remove', e.message||e); } }; } });
  });
  document.addEventListener('click', ()=>{ if (ctx) ctx.style.display='none'; });

  // status bar clock
  const clockEl = root.querySelector('#status-clock');
  if (clockEl) { setInterval(()=> { clockEl.textContent = new Date().toLocaleString(); }, 1000); }

  // Protocol modal wiring: open when protocol-badge clicked
  root.addEventListener('click', (ev)=>{
    const pb = ev.target.closest('.protocol-badge');
    if (!pb) return;
    // find node id from row or detail
    const row = ev.target.closest('tr[data-node]');
    const nid = row ? row.dataset.node : (selected ? selected.node_id : null);
    if (!nid) return;
    const pm = root.querySelector('#protocol-modal');
    pm.style.display = 'flex';
    root.querySelector('#protocol-node-id').textContent = nid;
  });

  root.querySelector('#protocol-close')?.addEventListener('click', ()=> root.querySelector('#protocol-modal').style.display = 'none');
  root.querySelector('#protocol-apply')?.addEventListener('click', async ()=>{
    const nid = root.querySelector('#protocol-node-id').textContent;
    const proto = Array.from(root.querySelectorAll('input[name="proto"]')).find(r=>r.checked)?.value;
    if (!proto || !nid) { notify('error','Protocol','Select a protocol'); return; }
    try { await apiRequest(`/api/v1/nodes/${nid}/protocol`, { method:'POST', body:{ protocol: proto } }); notify('success','Protocol','Protocol applied'); root.querySelector('#protocol-modal').style.display='none'; } catch(e){ notify('error','Protocol', e.message||e); }
  });

  // PEM modal wiring: open when cert-view clicked in security tab
  async function openPemModalForNode(nodeId) {
    try {
      const resp = await apiRequest(`/api/v1/nodes/${nodeId}/certificate/pem`);
      const pem = resp && (resp.pem || resp);
      root.querySelector('#pem-block').textContent = pem || 'No PEM available';
      root.querySelector('#pem-info').textContent = `Node: ${nodeId}`;
      root.querySelector('#pem-modal').style.display = 'flex';
      // expiry timer if cert metadata available
      try { const meta = await apiRequest(`/api/v1/nodes/${nodeId}/certificate`); if (meta && meta.expiry) { startPemExpiryTimer(new Date(meta.expiry)); } } catch(e){}
    } catch(e){ notify('error','Certificate','Unable to fetch PEM'); }
  }
  root.querySelector('#pem-close')?.addEventListener('click', ()=> { root.querySelector('#pem-modal').style.display='none'; stopPemExpiryTimer(); });
  root.querySelector('#pem-download')?.addEventListener('click', ()=>{
    const txt = root.querySelector('#pem-block').textContent || '';
    const blob = new Blob([txt], { type: 'text/plain' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'cert.pem'; a.click(); URL.revokeObjectURL(url);
  });

  let pemTimer = null;
  function startPemExpiryTimer(dt) {
    stopPemExpiryTimer();
    function update(){ const now = new Date(); const diff = dt - now; if (diff<=0) { root.querySelector('#pem-expiry').textContent = 'Expired'; stopPemExpiryTimer(); return; } const days = Math.floor(diff/86400000); const hrs = Math.floor((diff%86400000)/3600000); root.querySelector('#pem-expiry').textContent = `Expires in ${days}d ${hrs}h`; }
    update(); pemTimer = setInterval(update, 60000);
  }
  function stopPemExpiryTimer(){ if (pemTimer) { clearInterval(pemTimer); pemTimer = null; } }

  // When security tab renders, wire cert-view click to open pem modal
  const origRenderDetailTab = renderDetailTab;
  renderDetailTab = function(tab){ origRenderDetailTab(tab); if (tab==='security' && selected) { const cv = root.querySelector('#cert-view'); if (cv) cv.style.cursor='pointer'; cv && (cv.onclick = ()=> openPemModalForNode(selected.node_id)); } };

  // Drag-resize between sidebar and main area
  const dragHandle = root.querySelector('#drag-handle');
  if (dragHandle) {
    let dragging = false; let startX = 0; let startLeft = 260;
    dragHandle.addEventListener('mousedown', (ev)=>{ dragging=true; startX = ev.clientX; const style = window.getComputedStyle(container); const cols = style.gridTemplateColumns; const left = parseInt(cols.split(' ')[0]) || 260; startLeft = left; document.body.style.cursor='col-resize'; ev.preventDefault(); });
    window.addEventListener('mousemove', (ev)=>{ if (!dragging) return; const dx = ev.clientX - startX; const newLeft = Math.max(180, Math.min(480, startLeft + dx)); container.style.gridTemplateColumns = `${newLeft}px 1fr 340px`; dragHandle.style.left = newLeft + 'px'; });
    window.addEventListener('mouseup', ()=>{ if (dragging) { dragging=false; document.body.style.cursor='default'; } });
  }

  // Simple drag-reorder for resource cards in resources mode
  function enableResourceDrag() {
    const area = root.querySelector('#mode-resources'); if (!area) return;
    const cards = Array.from(area.querySelectorAll('.card'));
    cards.forEach(c=>{ c.draggable = true; c.classList.add('draggable'); c.addEventListener('dragstart', (e)=>{ e.dataTransfer.setData('text/node-card', c.id||''); c.classList.add('dragging'); }); c.addEventListener('dragend', ()=>{ c.classList.remove('dragging'); }); });
    area.addEventListener('dragover', (e)=>{ e.preventDefault(); const after = getDragAfterElement(area, e.clientY); const dragging = area.querySelector('.dragging'); if (!dragging) return; if (after == null) area.appendChild(dragging); else area.insertBefore(dragging, after); });
    function getDragAfterElement(container, y) { const draggableElements = [...container.querySelectorAll('.card:not(.dragging)')]; return draggableElements.reduce((closest, child) => { const box = child.getBoundingClientRect(); const offset = y - box.top - box.height/2; if (offset < 0 && offset > closest.offset) { return { offset: offset, element: child }; } return closest; }, { offset: Number.NEGATIVE_INFINITY }).element;
    }
  }
  // enable when resources mode opened
  const origModeClick = () => {}; // placeholder
  // call enableResourceDrag when resources rendered
  const origRenderResources = renderResources;
  renderResources = async function(){ await origRenderResources(); enableResourceDrag(); };

  function renderDetailTab(tab) {
    if (!selected) return;
    const n = selected;
    switch(tab) {
      case 'overview':
        detailBody.innerHTML = `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="card"><h4>System</h4><div>ID: ${n.node_id}</div><div>Host: ${n.hostname||'-'}</div><div>IP: ${n.ip_address||'-'}</div><div>Role: ${n.node_type||'-'}</div></div>
            <div class="card"><h4>Status</h4><div>Last Seen: ${lastSeenText(n)}</div><div>Health: ${calculateHealthScore(n.metrics)}/100</div><div>Agent: ${n.agent_version||'unknown'}</div>
              <div class="quick-actions">
                <button class="btn-small" id="action-pause">Pause</button>
                <button class="btn-small" id="action-drain">Drain</button>
                <button class="btn-small" id="action-reboot">Reboot</button>
              </div>
            </div>
            <div class="card" style="grid-column:1 / -1"><h4>Resources</h4>
              <div>CPU: ${(n.metrics?.cpu_usage||0).toFixed(1)}% • MEM: ${(n.metrics?.memory_usage||0).toFixed(1)}%</div>
              <div style="margin-top:8px"><canvas id="overview-chart" style="width:100%;height:120px"></canvas></div>
            </div>
            <div class="card" style="grid-column:1 / -1"><h4>Tags</h4>
              <div id="tag-list">${(n.tags||[]).map(t=>`<span class="tag">${t}<span class="remove" data-tag="${t}">×</span></span>`).join(' ')}</div>
              <div style="margin-top:8px"><input id="new-tag" placeholder="Add tag"/> <button id="add-tag" class="btn-small">Add</button></div>
            </div>
          </div>
        `;
        // tag management wiring
        setTimeout(()=>{
          const add = root.querySelector('#add-tag');
          const newTag = root.querySelector('#new-tag');
          add && (add.onclick = ()=>{
            const v = (newTag.value||'').trim(); if (!v) return; selected.tags = selected.tags||[]; selected.tags.push(v); renderDetailTab('overview'); renderTable(root.querySelector('#nodes-search').value||'');
          });
          root.querySelectorAll('.tag .remove').forEach(r=> r.onclick = (ev)=>{ const t = ev.currentTarget.dataset.tag; selected.tags = (selected.tags||[]).filter(x=>x!==t); renderDetailTab('overview'); renderTable(root.querySelector('#nodes-search').value||''); });
        },50);
        setTimeout(()=>{
          const c = root.querySelector('#overview-chart'); if (c && c.getContext) { const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='var(--omega-dark-4)'; ctx.fillRect(0,0,c.width,c.height); ctx.fillStyle='var(--omega-cyan)'; ctx.fillText('Overview chart',10,20); }
        },50);
        break;
      case 'performance':
  detailBody.innerHTML = `<div class="card"><h4>Performance</h4><div id="perf-charts">(charts loading...)</div></div>`;
  // draw charts for selected node
  setTimeout(()=> drawDetailCharts(n), 20);
        break;
      case 'processes':
        detailBody.innerHTML = `<div class="card"><h4>Processes</h4><div id="proc-list">Loading processes...</div></div>`;
        setTimeout(async ()=>{
          // try fetch process list
          try { const procs = await apiRequest(`/api/v1/nodes/${n.node_id}/processes`); const list = (procs||[]).map(p=>`<div style="display:flex;justify-content:space-between;padding:6px"><div>${p.pid} — ${p.name}</div><div>${p.cpu}% • ${p.mem}%</div></div>`).join(''); root.querySelector('#proc-list').innerHTML = list || '<div>No processes</div>'; } catch(e){ root.querySelector('#proc-list').textContent = 'Error loading processes'; }
        },20);
        break;
      case 'logs':
        detailBody.innerHTML = `<div class="card"><h4>Logs</h4><div style="margin-bottom:8px"><button id="logs-refresh" class="btn-small">Refresh</button> <select id="logs-level"><option value="all">All</option><option value="error">Error</option><option value="warn">Warn</option></select></div><div id="log-output" style="max-height:300px;overflow:auto;background:#0b0b0b;padding:8px;border-radius:6px"></div></div>`;
        setTimeout(async ()=>{ const out = root.querySelector('#log-output'); const refresh = root.querySelector('#logs-refresh'); async function loadLogs(){ try{ const j = await apiRequest(`/api/v1/nodes/${n.node_id}/logs?level=${root.querySelector('#logs-level').value}`); out.innerHTML = (j||[]).map(l=>`<div style="padding:6px;border-bottom:1px solid rgba(60,60,60,0.04)"><div style="font-size:11px;color:var(--omega-light-2)">${new Date(l.ts).toLocaleString()}</div><div>${l.msg}</div></div>`).join(''); }catch(e){ out.textContent='Error loading logs'; } }
          refresh.onclick = loadLogs; loadLogs(); },20);
        break;
      case 'security':
        detailBody.innerHTML = `<div class="card"><h4>Security</h4><div id="cert-view">Loading certificate info...</div><div style="margin-top:8px"><button id="revoke-cert" class="btn-small btn-danger">Revoke Certificate</button> <button id="re-register" class="btn-small">Re-register</button></div></div>`;
        setTimeout(async ()=>{ try{ const cert = await apiRequest(`/api/v1/nodes/${n.node_id}/certificate`); root.querySelector('#cert-view').innerHTML = `<div>Subject: ${cert?.subject||'--'}</div><div>Issuer: ${cert?.issuer||'--'}</div><div>Expiry: ${cert?.expiry||'--'}</div>`; root.querySelector('#revoke-cert').onclick = async ()=>{ await apiRequest(`/api/v1/nodes/${n.node_id}/certificate/revoke`, { method:'POST' }); notify('success','Security','Certificate revoked'); }; root.querySelector('#re-register').onclick = ()=> root.querySelector('#open-register').click(); }catch(e){ root.querySelector('#cert-view').textContent='No certificate info'; } },20);
        break;
      case 'maintenance':
        detailBody.innerHTML = `<div class="card"><h4>Maintenance</h4><div><button id="enter-maint" class="btn-small">Enter Maintenance</button> <button id="exit-maint" class="btn-small">Exit Maintenance</button> <button id="quarantine-node" class="btn-small btn-danger">Quarantine</button></div><div id="maint-output" style="margin-top:8px"></div></div>`;
        setTimeout(()=>{ root.querySelector('#enter-maint').onclick = async ()=>{ try{ await apiRequest(`/api/v1/nodes/${n.node_id}/maintenance`, { method:'POST', body:{ action:'enter' } }); notify('success','Maintenance','Node set to maintenance'); }catch(e){ notify('error','Maintenance', e.message||e); } }; root.querySelector('#exit-maint').onclick = async ()=>{ try{ await apiRequest(`/api/v1/nodes/${n.node_id}/maintenance`, { method:'POST', body:{ action:'exit' } }); notify('success','Maintenance','Node maintenance exited'); }catch(e){ notify('error','Maintenance', e.message||e); } }; root.querySelector('#quarantine-node').onclick = async ()=>{ try{ await apiRequest(`/api/v1/nodes/${n.node_id}/quarantine`, { method:'POST' }); notify('success','Quarantine','Node quarantined'); }catch(e){ notify('error','Quarantine', e.message||e); } }; },20);
        break;
      case 'topology':
  detailBody.innerHTML = `<div class="card"><h4>Topology</h4><div id="topo-canvas" style="height:300px;background:var(--omega-dark-4);display:flex;align-items:center;justify-content:center;color:var(--omega-light-1)"></div></div>`;
  // render simple detail topology (peers of this node)
  setTimeout(()=> renderDetailTopology(n), 20);
        break;
      case 'policies':
        detailBody.innerHTML = `<div class="card"><h4>Policies</h4><div id="policy-editor">(policy editor placeholder)</div></div>`;
        break;
      case 'diagnostics':
        detailBody.innerHTML = `<div class="card"><h4>Diagnostics</h4><div><button id="run-diag" class="btn-small">Run Diagnostics</button> <button id="open-shell" class="btn-small">Open Shell</button></div><div id="diag-output" style="margin-top:8px"></div></div>`;
        setTimeout(()=>{
          const run = root.querySelector('#run-diag');
          run && (run.onclick = async ()=>{ root.querySelector('#diag-output').textContent = 'Running...'; try { await (window.omegaAPI?.secureAction?.('run_diagnostics',{node_id:selected.node_id})|| Promise.resolve()); root.querySelector('#diag-output').textContent='Diagnostics finished'; } catch(e){ root.querySelector('#diag-output').textContent = 'Error: '+e.message; } });
        },10);
        break;
    }
  }

  function renderDetailTopology(node) {
    const container = root.querySelector('#topo-canvas'); if (!container) return;
    container.innerHTML = '';
    const svgNS = 'http://www.w3.org/2000/svg';
    const w = container.clientWidth || 600; const h = 300;
    const svg = document.createElementNS(svgNS,'svg'); svg.setAttribute('width','100%'); svg.setAttribute('height',h); svg.style.background='transparent';
    const cx = Math.round(w/2); const cy = Math.round(h/2);
    // central node
    const g = document.createElementNS(svgNS,'g'); g.setAttribute('transform',`translate(${cx},${cy})`);
    const circ = document.createElementNS(svgNS,'circle'); circ.setAttribute('r','28'); circ.setAttribute('fill', node.status==='online'?'#2ecc71':'#e74c3c'); circ.setAttribute('stroke','#222'); circ.setAttribute('stroke-width','2');
    const txt = document.createElementNS(svgNS,'text'); txt.setAttribute('y','6'); txt.setAttribute('text-anchor','middle'); txt.setAttribute('font-size','12'); txt.setAttribute('fill','#fff'); txt.textContent = (node.hostname||node.node_id).slice(0,14);
    g.appendChild(circ); g.appendChild(txt); svg.appendChild(g);
    const peers = (node.peers||[]).slice(0,8);
    peers.forEach((p,i)=>{
      const ang = (i / peers.length) * Math.PI*2;
      const px = Math.round(cx + Math.cos(ang)*100); const py = Math.round(cy + Math.sin(ang)*100);
      const pg = document.createElementNS(svgNS,'g'); pg.setAttribute('transform',`translate(${px},${py})`);
      const pc = document.createElementNS(svgNS,'circle'); pc.setAttribute('r','14'); pc.setAttribute('fill','#555'); pc.setAttribute('stroke','#222'); pc.setAttribute('stroke-width','1');
      const pt = document.createElementNS(svgNS,'text'); pt.setAttribute('y','4'); pt.setAttribute('text-anchor','middle'); pt.setAttribute('font-size','10'); pt.setAttribute('fill','#fff'); pt.textContent = (p.node_id||'').slice(0,10);
      pg.appendChild(pc); pg.appendChild(pt); svg.appendChild(pg);
      const ln = document.createElementNS(svgNS,'line'); ln.setAttribute('x1',cx); ln.setAttribute('y1',cy); ln.setAttribute('x2',px); ln.setAttribute('y2',py); ln.setAttribute('stroke','#666'); ln.setAttribute('stroke-width','1'); svg.insertBefore(ln, svg.firstChild);
    });
    container.appendChild(svg);
  }

  root.querySelector('#open-register').onclick = ()=>{
    modalTitle.textContent = 'Register Node';
    modalBody.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <input id="m_node_id" placeholder="Node ID" />
        <input id="m_hostname" placeholder="Hostname" />
        <input id="m_ip" placeholder="IP Address" />
        <select id="m_role"><option value="compute">compute</option><option value="storage">storage</option></select>
      </div>
      <label style="display:block;margin-top:8px"><input type="checkbox" id="m_ssh"/> Advanced (SSH)</label>
      <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <input id="m_cert" placeholder="Certificate PEM or path (optional)" />
        <input id="m_tpm" placeholder="TPM ID / Attestation (optional)" />
      </div>
      <div style="margin-top:8px"><label>Endpoint URL: <input id="m_endpoint" placeholder="https://node:8443" style="width:60%"/></label></div>
    `;
    modal.style.display = 'flex';
    modalSubmit.onclick = async ()=>{
      const node_id = modalBody.querySelector('#m_node_id')?.value;
      const hostname = modalBody.querySelector('#m_hostname')?.value;
      const ip = modalBody.querySelector('#m_ip')?.value;
      const client = getApiClient();
      try {
  const payload = { node_id, node_type: modalBody.querySelector('#m_role')?.value||'compute', hostname, ip, cert: modalBody.querySelector('#m_cert')?.value||null, tpm: modalBody.querySelector('#m_tpm')?.value||null, endpoint: modalBody.querySelector('#m_endpoint')?.value||null };
  await apiRequest('/api/secure/nodes/register', { method: 'POST', body: payload });
        modal.style.display='none';
        notify('success','Register','Node registered');
      } catch(e){ notify('error','Register', e.message||e); }
    };
  };
  root.querySelector('#modal-close').onclick = ()=> modal.style.display='none';

  root.querySelector('#show-pending').onclick = async ()=>{
    modalTitle.textContent = 'Pending Registrations';
    modalBody.innerHTML = '<div id="pending-list">Loading...</div>';
    modal.style.display = 'flex';
    try{
      const client = getApiClient();
      let pkt = null;
      if (client && typeof client._req === 'function') {
        try { pkt = await client._req('/api/secure/nodes/pending'); } catch(e){ pkt = null; }
      }
      if (!pkt) {
        try { pkt = await apiRequest('/api/v1/nodes/pending'); } catch(e){ pkt = { pending: [] }; }
      }
      const items = pkt.pending || [];
      const list = items.map(i=>`<div style="display:flex;justify-content:space-between;padding:6px;border-bottom:1px solid var(--omega-gray-2)"><span>${i.node_id||i}</span><span><button class="approve" data-id="${i.node_id||i}">Approve</button> <button class="deny" data-id="${i.node_id||i}">Deny</button></span></div>`).join('')||'<div>No pending</div>';
      modalBody.querySelector('#pending-list').innerHTML = list;
  modalBody.querySelectorAll('.approve').forEach(b=>b.onclick=async (e)=>{ const id=e.currentTarget.dataset.id; confirmTitle.textContent='Approve Node'; confirmBody.textContent = 'Approve '+id+'?'; confirmModal.style.display='flex'; confirmSubmit.onclick = async ()=>{ try { await apiRequest('/api/secure/nodes/approve', { method:'POST', body: { node_id: id } }); confirmModal.style.display='none'; modal.style.display='none'; notify('success','Approve',id+' approved'); } catch(e){ notify('error','Approve', e.message||e); } }; });
  modalBody.querySelectorAll('.deny').forEach(b=>b.onclick=async (e)=>{ const id=e.currentTarget.dataset.id; confirmTitle.textContent='Deny Node'; confirmBody.textContent = 'Deny '+id+'?'; confirmModal.style.display='flex'; confirmSubmit.onclick = async ()=>{ try { await apiRequest('/api/secure/nodes/deny', { method:'POST', body: { node_id: id } }); confirmModal.style.display='none'; modal.style.display='none'; notify('success','Deny',id+' denied'); } catch(e){ notify('error','Deny', e.message||e); } }; });
  confirmClose.onclick = ()=> confirmModal.style.display='none'; confirmCancel.onclick = ()=> confirmModal.style.display='none';
    }catch(e){ modalBody.querySelector('#pending-list').textContent = 'Error fetching'; }
  };

  root.querySelector('#discover-nodes').onclick = async ()=>{
    try {
      const client = getApiClient();
      if (client && typeof client.secureAction === 'function') {
        await client.secureAction('discover_nodes', {});
      } else if (window.omegaAPI && typeof window.omegaAPI.secureAction === 'function') {
        await window.omegaAPI.secureAction('discover_nodes', {});
      }
      notify('success','Discovery','Discovery triggered');
      let disc = null;
      if (client && typeof client._req === 'function') {
        try { disc = await client._req('/api/secure/nodes/discovered'); } catch(e){ disc = null; }
      }
  if (!disc) { try { disc = await apiRequest('/api/v1/nodes/discovered'); } catch(e){ disc = { discovered: [] }; } }
      const list = (disc.discovered||[]);
      const panel = root.querySelector('#discovery-panel');
      const dl = root.querySelector('#discovery-list');
      if (!list.length) { dl.innerHTML = '<div style="padding:8px">No discovered devices</div>'; panel.style.display='block'; return; }
      dl.innerHTML = list.map(d=>`<div style="padding:6px;border-bottom:1px solid var(--omega-gray-2);display:flex;justify-content:space-between"><div>${d.node_id||d.ip||d.hostname}</div><div><button class="reg" data-id="${d.node_id||d.ip}">Register</button></div></div>`).join('');
      panel.style.display='block';
      dl.querySelectorAll('.reg').forEach(b=>b.onclick=()=>{ root.querySelector('#open-register').click(); modalBody.querySelector('#m_node_id') && (modalBody.querySelector('#m_node_id').value = b.dataset.id); });
    } catch(e){ notify('error','Discovery',e.message||e); }
  };

  selectAll && (selectAll.onchange = (e)=>{ root.querySelectorAll('.row-select').forEach(cb=>cb.checked = e.target.checked); });

  // wire batch action buttons
  root.querySelector('#batch-approve')?.addEventListener('click', ()=>batchAction('approve'));
  root.querySelector('#batch-quarantine')?.addEventListener('click', ()=>batchAction('quarantine'));
  root.querySelector('#batch-remove')?.addEventListener('click', ()=>batchAction('remove'));

  // quick action wiring (in detail view)
  root.addEventListener('click', async (ev)=>{
    if (ev.target && ev.target.id && ev.target.id.startsWith('action-')) {
      const act = ev.target.id.replace('action-','');
      if (!selected) return notify('error','Action','No node selected');
      if (!hasPermission(act)) return notify('error','Permission','Not allowed');
      try {
        const client = getApiClient();
        if (client && typeof client.secureAction === 'function') {
          await client.secureAction(act, { node_id: selected.node_id });
        } else if (window.omegaAPI && typeof window.omegaAPI.secureAction === 'function') {
          await window.omegaAPI.secureAction(act, { node_id: selected.node_id });
        } else {
          await apiRequest(`/api/v1/nodes/${act}`, { method: 'POST', body: { node_id: selected.node_id } });
        }
        notify('success','Action', act + ' executed');
      } catch(e){ notify('error','Action', e.message||e); }
    }
  });

  // build sidebar and init filters
  buildSidebarTree(nodes);
  initFiltersAndSorting();

  // mode tab wiring
  root.querySelectorAll('.mode-tab').forEach(tb=> tb.onclick = (e)=>{
    root.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active'));
    e.currentTarget.classList.add('active');
    const mode = e.currentTarget.dataset.mode;
    ['list','topology','dashboard','resources','sessions','network','performance','security','plugins','ai','settings','policies','diagnostics'].forEach(m=>{
      const el = root.querySelector('#mode-'+m);
      if (el) el.style.display = (m===mode)?'block':'none';
    });
    // lazy renderers
    if (mode === 'dashboard') renderDashboard();
    if (mode === 'resources') renderResources();
    if (mode === 'sessions') renderSessions();
    if (mode === 'network') renderNetwork();
    if (mode === 'performance') renderPerformance();
    if (mode === 'security') renderSecurity();
    if (mode === 'plugins') renderPlugins();
    if (mode === 'ai') renderAI();
    if (mode === 'settings') renderSettings();
  });
  // Removed nodes-active toggle logic (was hiding global side widgets unintentionally)
  // ensure switching to topology triggers render
  root.querySelector('[data-mode="topology"]')?.addEventListener('click', ()=> setTimeout(()=> renderTopologyIfActive(),40));
  // ensure list refresh
  root.querySelector('[data-mode="list"]')?.addEventListener('click', ()=> setTimeout(()=> renderTable(root.querySelector('#nodes-search')?.value||''),50));

  // audit-log click opens modal and loads page 1
  const auditLogEl = root.querySelector('#audit-log');
  if (auditLogEl) {
    auditLogEl.style.cursor = 'pointer';
    auditLogEl.addEventListener('click', ()=>{ root.querySelector('#audit-modal').style.display='flex'; loadAuditPage(1); });
  }

  // RBAC enforcement for detail quick actions and batch actions
  function applyRBAC() {
    // permissions mapping (assumed permission names)
    const permMap = {
      'pause':'nodes.manage', 'drain':'nodes.manage', 'reboot':'nodes.manage',
      'approve':'nodes.approve', 'quarantine':'nodes.quarantine', 'remove':'nodes.remove', 'assign-tags':'nodes.tags',
      'discover':'nodes.discover', 'register':'nodes.register', 'policy':'nodes.policy', 'plugin':'nodes.plugin', 'controls':'nodes.controls', 'audit':'audit.view'
    };
    // helper to check (wait for load to finish)
    function allowed(key) { if (!permissionsLoaded) return false; const p = permMap[key] || key; return hasPermission(p); }

    // quick action buttons
    ['pause','drain','reboot'].forEach(a=>{
      const btn = root.querySelector('#action-'+a);
      if (btn) { btn.disabled = !allowed(a); if (btn.disabled) btn.title = 'Insufficient permissions'; else btn.title = ''; }
    });

    // batch actions
    const batchMap = { 'batch-approve':'approve', 'batch-quarantine':'quarantine', 'batch-remove':'remove', 'batch-assign-tags':'assign-tags' };
    Object.keys(batchMap).forEach(id=>{
      const b = root.querySelector('#'+id);
      if (b) { b.disabled = !allowed(batchMap[id]); if (b.disabled) b.title = 'Insufficient permissions'; else b.title=''; }
    });

    // top-level actions
    const discoverBtn = root.querySelector('#discover-nodes'); if (discoverBtn) { discoverBtn.disabled = !allowed('discover'); discoverBtn.title = discoverBtn.disabled ? 'Insufficient permissions':''; }
    const pendingBtn = root.querySelector('#show-pending'); if (pendingBtn) { pendingBtn.disabled = !allowed('approve'); pendingBtn.title = pendingBtn.disabled ? 'Insufficient permissions':''; }
    const registerBtn = root.querySelector('#open-register'); if (registerBtn) { registerBtn.disabled = !allowed('register'); registerBtn.title = registerBtn.disabled ? 'Insufficient permissions':''; }

    // policy save
    const policySave = root.querySelector('#policy-save'); if (policySave) { policySave.disabled = !allowed('policy'); policySave.title = policySave.disabled ? 'Insufficient permissions':''; }

    // resource controls save (if present)
    const ctlSave = root.querySelector('#ctl-save'); if (ctlSave) { ctlSave.disabled = !allowed('controls'); ctlSave.title = ctlSave.disabled ? 'Insufficient permissions':''; }

    // plugin toggles (if present in DOM) - disable by default when not allowed
    root.querySelectorAll('.plugin-toggle').forEach(b=>{ b.disabled = !allowed('plugin'); if (b.disabled) b.title = 'Insufficient permissions'; else b.title=''; });

    // audit access
    const auditEl = root.querySelector('#audit-log'); if (auditEl) { auditEl.style.opacity = allowed('audit') ? '1' : '0.6'; auditEl.title = allowed('audit') ? '' : 'Insufficient permissions to view full audit'; }
  }
  applyRBAC();

  // start loading permissions and re-apply RBAC once ready
  (async ()=>{ await loadPermissions(); applyRBAC(); })();

  // --- Live feed: WebSocket or polling fallback ---
  let ws = null; let pollingId = null;
  function startLiveFeed() {
    const client = getApiClient();
    if (client && typeof client.connect === 'function') {
      // ApiClient provided websocket wrapper (secure)
      try { ws = client.connect('/ws/secure/realtime'); ws.onmessage = (m)=>{ try{ const p=JSON.parse(m.data); if (p.node_id) { const idx = nodes.findIndex(x=>x.node_id===p.node_id); if (idx>=0) { nodes[idx] = Object.assign({}, nodes[idx], p); renderTable(root.querySelector('#nodes-search').value||''); } } }catch(e){} }; }
      catch(e){ ws=null; }
    } else if (window.omegaAPI && typeof window.omegaAPI.connect === 'function') {
      // app-provided websocket wrapper (legacy)
      try { ws = window.omegaAPI.connect('/api/v1/nodes/stream'); ws.onmessage = (m)=>{ try{ const p=JSON.parse(m.data); if (p.node_id) { const idx = nodes.findIndex(x=>x.node_id===p.node_id); if (idx>=0) { nodes[idx] = Object.assign({}, nodes[idx], p); renderTable(root.querySelector('#nodes-search').value||''); } } }catch(e){} }; }
      catch(e){ ws=null; }
    }
  if (!ws && 'WebSocket' in window) {
      try {
        ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/api/v1/nodes/stream');
        ws.onmessage = (m)=>{ try{ const p=JSON.parse(m.data); if (p.node_id) { const idx = nodes.findIndex(x=>x.node_id===p.node_id); if (idx>=0) { nodes[idx] = Object.assign({}, nodes[idx], p); renderTable(root.querySelector('#nodes-search').value||''); } } }catch(e){} };
      } catch(e){ ws=null; }
    }
    if (!ws) {
      // polling fallback (explicit checks to avoid complex inline expressions)
      pollingId = setInterval(async () => {
        try {
            let latest;
            const client = getApiClient();
            try { latest = await apiRequest('/api/v1/nodes/list'); } catch(e){ latest = { nodes: [] }; }
          if (latest && Array.isArray(latest.nodes)) {
            latest.nodes.forEach(nu => {
              const i = nodes.findIndex(x => x.node_id === nu.node_id);
              if (i >= 0) nodes[i] = Object.assign({}, nodes[i], nu);
            });
            renderTable(root.querySelector('#nodes-search').value || '');
          }
        } catch (e) {
          // ignore transient errors
        }
      }, 8000);
    }
  }
  startLiveFeed();

  // Alerts and audit in-memory stores
  const alerts = [];
  const audit = [];
  let activeActivityFilter = 'all';
  // try to restore recent audit from localStorage
  try { const saved = localStorage.getItem('nodes_audit_v1'); if (saved) { const parsed = JSON.parse(saved); if (Array.isArray(parsed)) { audit.push(...parsed.slice(0,50)); } } } catch(e){}
  function pushAlert(msg) { alerts.unshift({ts:Date.now(), msg}); if (alerts.length>200) alerts.pop(); renderAlerts(); }
  function classifyAudit(msg){
    const m = msg.toLowerCase();
    if (m.includes('error')||m.includes('failed')) return 'errors';
    if (m.includes('quarantine')||m.includes('cert')||m.includes('rbac')||m.includes('policy')) return 'security';
    if (m.includes('alert')) return 'alerts';
    if (m.includes('node')||m.includes('ws ')||m.includes('benchmark')||m.includes('limits')) return 'events';
    return 'events';
  }
  function pushAudit(msg) {
    const entry = { ts: Date.now(), msg, cat: classifyAudit(msg) };
    audit.unshift(entry); if (audit.length>400) audit.pop();
    renderAudit();
    try { localStorage.setItem('nodes_audit_v1', JSON.stringify(audit.slice(0,150))); } catch(e){}
    try { if (typeof saveAuditToServer === 'function') saveAuditToServer(entry); } catch(e){}
  }
  function renderAlerts(){ const feed=root.querySelector('#alerts-feed'); if (!feed) return; feed.innerHTML = alerts.slice(0,30).map(a=>`<div><span class="ts">${new Date(a.ts).toLocaleTimeString()}</span><span>${a.msg}</span></div>`).join(''); }
  function renderAudit(){ const feed=root.querySelector('#audit-log'); if (!feed) return; const filtered = audit.filter(a=> activeActivityFilter==='all' || a.cat===activeActivityFilter || (activeActivityFilter==='alerts' && a.cat==='alerts')); const slice = filtered.slice(0,120); feed.innerHTML = slice.map(a=>`<div class="${a.cat.slice(0,-1)}-entry"><span class="ts">${new Date(a.ts).toLocaleTimeString()}</span><span>${a.msg}</span></div>`).join(''); const totalEl = root.querySelector('#activity-total'); if (totalEl) totalEl.textContent = filtered.length; }
  // hook up filter buttons (after DOM flush later)
  setTimeout(()=>{ root.querySelectorAll('.act-filter').forEach(b=> b.addEventListener('click', e=>{ root.querySelectorAll('.act-filter').forEach(x=>x.classList.remove('active')); e.currentTarget.classList.add('active'); activeActivityFilter = e.currentTarget.dataset.filter; renderAudit(); })); }, 50);

  // WebSocket reconnection/backoff
  let wsConn = null; let wsAttempts = 0; let wsTimer = null;
  function handleWsMessage(data){ try{ const p = typeof data === 'string' ? JSON.parse(data) : data; if (p && p.type==='node_update' && p.node_id) { const idx = nodes.findIndex(x=>x.node_id===p.node_id); if (idx>=0) { nodes[idx]=Object.assign({}, nodes[idx], p.payload||p); renderTable(root.querySelector('#nodes-search').value||''); pushAudit('Node '+p.node_id+' updated'); } } if (p && p.type==='alert') { pushAlert(p.msg||JSON.stringify(p)); } }catch(e){} }
  function connectWs(){
    if (wsConn) try{ wsConn.close(); }catch(e){}
    const url = (location.protocol==='https:'?'wss://':'ws://') + location.host + '/api/v1/nodes/stream';
    try{
      wsConn = new WebSocket(url);
      wsConn.onopen = ()=>{ wsAttempts=0; pushAudit('WS connected'); };
      wsConn.onmessage = (ev)=> handleWsMessage(ev.data);
      wsConn.onclose = ()=>{ pushAudit('WS closed'); scheduleReconnect(); };
      wsConn.onerror = (e)=>{ pushAudit('WS error'); wsConn.close(); };
    }catch(e){ scheduleReconnect(); }
  }
  function scheduleReconnect(){ wsAttempts++; const delay = Math.min(30000, 1000 * Math.pow(1.5, wsAttempts)); if (wsTimer) clearTimeout(wsTimer); wsTimer = setTimeout(()=> connectWs(), delay); }
  connectWs();

  // --- New mode renderers under Nodes tab ---
  async function renderDashboard() {
    const container = root.querySelector('#mode-dashboard');
    if (!container) return;
    container.innerHTML = `<div style="display:grid;grid-template-columns:1fr 320px;gap:12px"><div id="dash-left"><div class=\"card\"><h4>Cluster Overview</h4><div id=\"cluster-topo\" style=\"height:260px;background:var(--omega-dark-4);border-radius:6px\">Topology</div><div style=\"display:flex;gap:8px;margin-top:8px\"><div class=\"card\"><div>Active</div><div id=\"dash-active\">0</div></div><div class=\"card\"><div>Standby</div><div id=\"dash-standby\">0</div></div><div class=\"card\"><div>Sessions</div><div id=\"dash-sessions\">0</div></div></div></div></div><div id="dash-right"><div class=\"card\"><h4>Quick Actions</h4><div style=\"display:flex;flex-direction:column;gap:8px\"><button id=\"qa-discover\" class=\"btn-small\">Discover Nodes</button><button id=\"qa-new-session\" class=\"btn-small\">New Session</button><button id=\"qa-run-bench\" class=\"btn-small\">Run Benchmark</button><button id=\"qa-health\" class=\"btn-small\">Health Check</button></div></div><div class=\"card\" style=\"margin-top:12px\"><h4>Alerts</h4><div id=\"dash-alerts\">(recent alerts)</div></div></div></div>`;
    // populate counters
    try { const s = await apiRequest('/api/v1/cluster/summary'); document.getElementById('dash-active').textContent = s.active || 0; document.getElementById('dash-standby').textContent = s.standby || 0; document.getElementById('dash-sessions').textContent = s.sessions || 0; } catch(e){}
    root.querySelector('#qa-discover')?.addEventListener('click', ()=> root.querySelector('#discover-nodes')?.click());
    root.querySelector('#qa-run-bench')?.addEventListener('click', ()=> notify('info','Benchmark','Use Performance tab to run benchmarks'));
  }

  async function renderResources() {
    root.querySelector('#res-cpu-body').textContent = 'Loading CPU aggregate...';
    root.querySelector('#res-gpu-body').textContent = 'Loading GPU aggregate...';
    root.querySelector('#res-mem-body').textContent = 'Loading memory...';
    root.querySelector('#res-storage-body').textContent = 'Loading storage...';
    try { const r = await apiRequest('/api/v1/resources/summary'); root.querySelector('#res-cpu-body').textContent = `${r.cpu_usage||0}% overall`; root.querySelector('#res-gpu-body').textContent = `${r.gpu_count||0} GPUs`; root.querySelector('#res-mem-body').textContent = `${r.mem_usage||0}% used`; root.querySelector('#res-storage-body').textContent = `${r.storage_iops||0} IOPS`; } catch(e){ /* ignore */ }
  }

  async function renderSessions(){ root.querySelector('#sessions-list').textContent='Loading...'; try{ const s = await apiRequest('/api/v1/sessions/list'); root.querySelector('#sessions-list').innerHTML = (s||[]).map(ss=>`<div style="padding:8px;border-bottom:1px solid rgba(60,60,60,0.04)"><div style="display:flex;justify-content:space-between"><div><strong>${ss.name}</strong> <small>${ss.status}</small></div><div><button class=\"btn-small\" data-id=\"${ss.id}\">Open</button></div></div></div>`).join(''); }catch(e){ root.querySelector('#sessions-list').textContent='No sessions'; } }

  async function renderNetwork(){ root.querySelector('#network-controls').textContent='Loading network interfaces...'; try{ const n = await apiRequest('/api/v1/network/interfaces'); root.querySelector('#network-controls').innerHTML = (n||[]).map(i=>`<div style="padding:6px;border-bottom:1px solid rgba(60,60,60,0.04)"><div>${i.name} — ${i.link_speed||'-'} — RX:${i.rx||0} TX:${i.tx||0}</div></div>`).join(''); }catch(e){ root.querySelector('#network-controls').textContent='No interfaces'; } }

  async function renderPerformance(){ root.querySelector('#bench-summary').textContent='Fetching latest results...'; try{ const b = await apiRequest('/api/v1/benchmarks/latest'); root.querySelector('#bench-summary').textContent = b ? `Score: ${b.score||0}` : 'No results'; root.querySelector('#run-full-bench')?.addEventListener('click', async ()=>{ notify('info','Benchmark','Running full benchmark'); try{ await apiRequest('/api/v1/benchmarks/run', { method:'POST', body:{ mode:'full' } }); notify('success','Benchmark','Started full benchmark'); }catch(e){ notify('error','Benchmark', e.message||e); } }); root.querySelector('#run-quick-bench')?.addEventListener('click', async ()=>{ try{ await apiRequest('/api/v1/benchmarks/run', { method:'POST', body:{ mode:'quick' } }); notify('success','Benchmark','Started quick benchmark'); }catch(e){ notify('error','Benchmark', e.message||e); } }); }catch(e){ /* ignore */ } }

  async function renderSecurity(){ root.querySelector('#security-body').textContent='Loading security overview...'; try{ const s = await apiRequest('/api/v1/security/summary'); root.querySelector('#security-body').innerHTML = `<div>Certificates: ${s.cert_count||0}</div><div>RBAC roles: ${s.roles||0}</div>`; }catch(e){} }

  async function renderPlugins(){ root.querySelector('#plugin-market').textContent='Loading plugins...'; try{ const p = await apiRequest('/api/v1/plugins/list'); root.querySelector('#plugin-market').innerHTML = (p||[]).map(pl=>`<div style="padding:8px;border-bottom:1px solid rgba(60,60,60,0.04)"><div style="display:flex;justify-content:space-between"><div>${pl.name} <small>v${pl.version}</small></div><div><button class=\"btn-small\">Enable</button></div></div></div>`).join(''); }catch(e){ root.querySelector('#plugin-market').textContent='No plugins'; } }

  async function renderAI(){ root.querySelector('#ai-body').textContent='Loading AI engine status...'; try{ const a = await apiRequest('/api/v1/ai/status'); root.querySelector('#ai-body').innerHTML = `<div>Model: ${a.model||'n/a'}</div><div>Status: ${a.state||'idle'}</div>`; }catch(e){} }

  async function renderSettings(){ root.querySelector('#settings-body').innerHTML = `<div>Cluster name: <input id=\"set-cluster-name\" value=\"${(await (async()=>{try{return (await apiRequest('/api/v1/settings')).cluster_name}catch(e){return ''}})())||''}\"/></div><div style=\"margin-top:8px\"><button id=\"save-settings\" class=\"btn-small\">Save</button></div>`; root.querySelector('#save-settings')?.addEventListener('click', async ()=>{ try{ const v = root.querySelector('#set-cluster-name').value; await apiRequest('/api/v1/settings', { method:'POST', body:{ cluster_name: v } }); notify('success','Settings','Saved'); }catch(e){ notify('error','Settings', e.message||e); } }); }

  

  // Policy modal wiring
  root.querySelector('#policy-close')?.addEventListener('click', ()=> root.querySelector('#policy-modal').style.display='none');
  root.querySelector('#policy-save')?.addEventListener('click', async () => {
    const txt = root.querySelector('#policy-text').value;
    // basic validation: ensure valid JSON
    try {
      JSON.parse(txt);
    } catch (e) {
      notify('error','Policy','Invalid JSON: '+(e.message||e));
      return;
    }
      try {
        await apiRequest('/api/v1/nodes/policy', { method: 'POST', body: txt });
        pushAudit('Policy saved');
        root.querySelector('#policy-modal').style.display = 'none';
        notify('success', 'Policy', 'Saved');
      } catch (e) {
        notify('error', 'Policy', e.message || e);
      }
  });

  // persist audit to server (best-effort)
  async function saveAuditToServer(entry) {
    try {
      await apiRequest('/api/v1/audit', { method: 'POST', body: entry });
    } catch (e) { /* ignore persistence errors */ }
  }

  // Batch assign tags flow
  root.querySelector('#batch-assign-tags')?.addEventListener('click', ()=>{
    const selected = Array.from(root.querySelectorAll('.row-select:checked')).map(cb=>cb.dataset.id);
    if (!selected.length) { notify('info','Tags','No nodes selected'); return; }
    // reuse confirm modal as a small form
    confirmTitle.textContent = 'Assign Tags';
    confirmBody.innerHTML = `<div style="display:flex;flex-direction:column;gap:8px"><input id="batch-tags-input" placeholder="comma separated tags"/></div>`;
    confirmModal.style.display = 'flex';
    confirmSubmit.onclick = async ()=>{
      const val = confirmModal.querySelector('#batch-tags-input')?.value || '';
      const tags = val.split(',').map(s=>s.trim()).filter(Boolean);
      if (!tags.length) { notify('info','Tags','No tags entered'); return; }
      confirmModal.style.display='none';
      try {
  await Promise.all(selected.map(id => apiRequest(`/api/v1/nodes/${id}/tags`, { method: 'POST', body: { tags } })));
        pushAudit('Tags '+tags.join(',')+' assigned to '+selected.length+' nodes');
        notify('success','Tags','Assigned to '+selected.length+' nodes');
      } catch(e){ notify('error','Tags', e.message||e); }
    };
  });

  // plugin manager placeholder in detail tab when a node selected
  function renderPluginManager(node) {
    const pm = document.createElement('div'); pm.innerHTML = `<h4>Plugins</h4><div id="plugin-list">${(node.plugins||[]).map(p=>`<div style="display:flex;justify-content:space-between;padding:6px"><div>${p.name} <span style="font-size:11px;color:var(--omega-light-2)">v${p.version}</span></div><div><button class='btn-small plugin-toggle' data-name='${p.name}'>${p.enabled? 'Disable':'Enable'}</button></div></div>`).join('') || '<div>No plugins</div>'}</div>`;
    const detail = root.querySelector('#detail-body'); detail.appendChild(pm);
  detail.querySelectorAll('.plugin-toggle').forEach(b=> b.onclick = async (ev)=>{ const name=ev.currentTarget.dataset.name; try { await apiRequest(`/api/v1/nodes/${node.node_id}/plugins/toggle`, { method: 'POST', body: { name } }); pushAudit((node.node_id||'')+' plugin toggled '+name); notify('success','Plugin','Toggled '+name); } catch(e){ notify('error','Plugin', e.message||e); } });
  }

  // plugin toggle helper for tests
  async function pluginToggleAPI(nodeId, name) {
    try { await apiRequest(`/api/v1/nodes/${nodeId}/plugins/toggle`, { method:'POST', body: { name } }); return true; } catch(e){ return false; }
  }

  // hook into openDetail to render plugin manager
  const origOpen = openDetail;
  openDetail = function(node) {
    origOpen(node);
    if (node) renderPluginManager(node);
    renderTopologyIfActive();
  };

  // draw detailed charts in performance tab
  function drawDetailCharts(node) {
    const perf = root.querySelector('#perf-charts'); if (!perf) return;
    perf.innerHTML = `<div style="display:flex;gap:12px"><canvas id="cpu-chart" width="460" height="120"></canvas><canvas id="mem-chart" width="240" height="120"></canvas></div>`;
    const cpuCanvas = perf.querySelector('#cpu-chart'); const memCanvas = perf.querySelector('#mem-chart');
    const cpuHistory = (node.metrics?.history?.cpu||[]).slice(-120); const memHistory = (node.metrics?.history?.mem||[]).slice(-120);
    // use Chart.js if available
    if (window.Chart && typeof window.Chart === 'function') {
      try {
        new Chart(cpuCanvas.getContext('2d'), { type:'line', data:{ labels:cpuHistory.map((_,i)=>i), datasets:[{ label:'CPU', data:cpuHistory, borderColor:'#05b3c8', fill:false }] }, options:{ responsive:false, animation:false, scales:{ x:{ display:false } } } });
        new Chart(memCanvas.getContext('2d'), { type:'line', data:{ labels:memHistory.map((_,i)=>i), datasets:[{ label:'Mem', data:memHistory, borderColor:'#f5c041', fill:false }] }, options:{ responsive:false, animation:false, scales:{ x:{ display:false } } } });
        return;
      } catch(e){}
    }
    // fallback simple canvas draw
    function drawLine(canvas, values, color){ const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); if (!values.length) { ctx.fillStyle='#222'; ctx.fillRect(0,0,canvas.width,canvas.height); return; } const w=canvas.width,h=canvas.height; const max=Math.max(...values), min=Math.min(...values); ctx.beginPath(); ctx.strokeStyle=color; values.forEach((v,i)=>{ const x = (i/(values.length-1))*(w-10)+5; const y = h-5 - ((v-min)/(max-min||1))*(h-10); if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); }
    drawLine(cpuCanvas, cpuHistory, '#05b3c8'); drawLine(memCanvas, memHistory, '#f5c041');
  }

  // resource control sliders (in overview)
  function wireResourceControls(node) {
    const controlsArea = root.querySelector('#resource-controls'); if (!controlsArea) return;
    controlsArea.innerHTML = `<div style="display:flex;gap:8px;align-items:center"><label>CPU limit</label><input id="ctl-cpu" type="range" min="0" max="100" value="${node.limits?.cpu||100}"/> <span id="ctl-cpu-val">${node.limits?.cpu||100}%</span></div><div style="display:flex;gap:8px;align-items:center;margin-top:8px"><label>Mem limit</label><input id="ctl-mem" type="range" min="0" max="100" value="${node.limits?.mem||100}"/> <span id="ctl-mem-val">${node.limits?.mem||100}%</span></div><div style="margin-top:8px"><button id="ctl-save" class="btn-small">Save Limits</button></div>`;
    const cpu = controlsArea.querySelector('#ctl-cpu'); const cpv = controlsArea.querySelector('#ctl-cpu-val'); const mem = controlsArea.querySelector('#ctl-mem'); const mpv = controlsArea.querySelector('#ctl-mem-val');
    cpu.oninput = ()=> cpv.textContent = cpu.value + '%'; mem.oninput = ()=> mpv.textContent = mem.value + '%';
    controlsArea.querySelector('#ctl-save').onclick = async ()=>{
      const payload = { cpu: Number(cpu.value), mem: Number(mem.value) };
      try { await apiRequest(`/api/v1/nodes/${node.node_id}/controls`, { method:'POST', body: payload }); pushAudit(node.node_id+' limits updated'); notify('success','Limits','Saved'); } catch(e){ notify('error','Limits', e.message||e); }
    };
  }

  // dynamic lib loader: Chart.js and D3 (CDN) — optional
  function loadLib(url){ return new Promise((res,rej)=>{ const s=document.createElement('script'); s.src=url; s.onload=res; s.onerror=rej; document.head.appendChild(s); }); }
  (async ()=>{
    try { if (!window.Chart) await loadLib('https://cdn.jsdelivr.net/npm/chart.js'); } catch(e){}
    try { if (!window.d3) await loadLib('https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.2/d3.min.js'); } catch(e){}
  })();


  // Audit browser wiring (server-backed pagination if available, else local fallback)
  root.querySelector('#audit-close')?.addEventListener('click', ()=> root.querySelector('#audit-modal').style.display='none');
  // helper to fetch audit from server (best-effort)
  async function fetchAuditPage(page=1, limit=50, q=''){
    try {
      if (window.omegaAPI && typeof window.omegaAPI._req === 'function') {
        const res = await window.omegaAPI._req(`/api/v1/audit?page=${page}&limit=${limit}&q=${encodeURIComponent(q)}`);
        return res && Array.isArray(res.items) ? { items: res.items, total: res.total||res.items.length } : null;
      }
  try { const j = await apiRequest(`/api/v1/audit?page=${page}&limit=${limit}&q=${encodeURIComponent(q)}`); return { items: j.items||[], total: j.total|| (j.items? j.items.length:0) }; } catch(e){ return null; }
    } catch(e){ return null; }
  }

  function renderAuditResults(items, page=1, total=0){
    const out = (items||[]).map(r=>`<div style="padding:6px;border-bottom:1px solid rgba(60,60,60,0.08)"><div style="font-size:11px;color:var(--omega-light-2)">${new Date(r.ts).toLocaleString()}</div><div>${r.msg}</div></div>`).join('');
    root.querySelector('#audit-results').innerHTML = out || '<div style="padding:8px;color:var(--omega-light-2)">No audit entries</div>';
    // pagination footer
    const footerId = 'audit-pager';
    let footer = root.querySelector('#'+footerId);
    if (!footer) { footer = document.createElement('div'); footer.id = footerId; footer.style.padding='8px'; footer.style.display='flex'; footer.style.justifyContent='space-between'; root.querySelector('#audit-results').appendChild(footer); }
    footer.innerHTML = `<div style="font-size:12px;color:var(--omega-light-2)">Page ${page} — ${total} entries</div><div><button id="audit-prev" class="btn-small" ${page<=1? 'disabled':''}>Prev</button> <button id="audit-next" class="btn-small" ${ (page*50)>=total ? 'disabled':''}>Next</button></div>`;
    root.querySelector('#audit-prev')?.addEventListener('click', ()=> { loadAuditPage(Math.max(1, currentAuditPage-1)); });
    root.querySelector('#audit-next')?.addEventListener('click', ()=> { loadAuditPage(currentAuditPage+1); });
  }

  let currentAuditPage = 1;
  async function loadAuditPage(page=1){
    const q = root.querySelector('#audit-filter')?.value || '';
    // try server
    const srv = await fetchAuditPage(page, 50, q);
    if (srv && Array.isArray(srv.items)) { currentAuditPage = page; renderAuditResults(srv.items, page, srv.total||0); return; }
    // fallback: client-side filter/paginate
    const list = (audit||[]).filter(a=> a.msg.toLowerCase().includes((q||'').toLowerCase()));
    const total = list.length; const start = (page-1)*50; const pageItems = list.slice(start, start+50); currentAuditPage = page; renderAuditResults(pageItems, page, total);
  }

  root.querySelector('#audit-refresh')?.addEventListener('click', ()=> loadAuditPage(1));
  // initial load when modal opens
  const origAuditOpener = document.querySelector('[data-open-audit]');
  root.querySelector('#audit-export')?.addEventListener('click', ()=>{ const blob = new Blob([JSON.stringify(audit)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='audit.json'; a.click(); URL.revokeObjectURL(url); });

  function notify(level, title, msg){ window.notify && window.notify(level, title, msg); }

  function calculateHealthScore(metrics) {
    if (!metrics) return 85;
    const cpu = metrics.cpu_usage || 0;
    const mem = metrics.memory_usage || 0;
    return Math.max(0, Math.round(100 - (cpu + mem) / 2));
  }

  renderTable('');

  // Removed temporary inline override stylesheet & debug badge (no longer needed)
