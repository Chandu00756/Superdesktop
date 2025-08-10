// Simple notification center
export function notify(type, title, message, timeout=4000){
  let c = document.getElementById('notify-container');
  if(!c){ c=document.createElement('div'); c.id='notify-container'; c.style.cssText='position:fixed;top:16px;right:16px;display:flex;flex-direction:column;gap:8px;z-index:99999;font:12px Consolas,monospace;'; document.body.appendChild(c);}  
  const el=document.createElement('div');
  el.setAttribute('role','alert');
  el.style.cssText='min-width:240px;max-width:380px;padding:10px 12px;border:1px solid #333;background:#111;color:#fff;border-left:4px solid '+color(type)+';box-shadow:0 2px 6px rgba(0,0,0,.5);opacity:0;transform:translateX(16px);transition:all .25s ease;';
  el.innerHTML=`<div style='font-weight:600;font-size:12px;margin-bottom:4px;'>${escapeHtml(title)}</div><div style='font-size:11px;line-height:1.4;white-space:pre-wrap;'>${escapeHtml(message)}</div>`;
  c.appendChild(el); requestAnimationFrame(()=>{ el.style.opacity='1'; el.style.transform='translateX(0)'; });
  setTimeout(()=>close(), timeout);
  el.addEventListener('click', close);
  function close(){ el.style.opacity='0'; el.style.transform='translateX(16px)'; setTimeout(()=>{ el.remove(); if(c.childElementCount===0) c.remove(); },250); }
}
function color(t){ switch(t){ case 'success': return '#00ff7f'; case 'error':return '#ff4444'; case 'warn':return '#ffaa00'; case 'info': default: return '#00f5ff'; } }
function escapeHtml(s){ return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }
