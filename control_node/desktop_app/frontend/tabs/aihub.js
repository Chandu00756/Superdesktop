export async function renderAIHub(root,state){
  root.innerHTML = `<div class='grid'>
    <div class='card'><h3>Recommendations</h3><div id='ai-recs' class='mono'>Loading...</div></div>
    <div class='card'><h3>Models</h3><div id='ai-models' class='mono'>Loading...</div></div>
    <div class='card'><h3>Controls</h3><div class='mono'><button id='btn-ai-refresh'>REFRESH</button></div></div>
  </div>`;
  async function load(){ try{ const data = await window.omegaAPI.aihub(); state.data.aihub=data; paint(data); }catch(e){ document.getElementById('ai-recs').textContent='Error'; } }
  function paint(data){ const recDiv=document.getElementById('ai-recs'); if(!recDiv) return; recDiv.innerHTML=(data.recommendations||[]).map(r=>`<div>[${r.impact}] ${r.title} (${(r.confidence*100).toFixed(1)}%)</div>`).join('')||'None'; const mDiv=document.getElementById('ai-models'); mDiv.innerHTML=(data.models||[]).map(m=>`<div>${m.name} v${m.version} - ${m.status}</div>`).join(''); }
  document.getElementById('btn-ai-refresh').onclick=()=>load();
  if(!state.data.aihub) load(); else paint(state.data.aihub);
}
