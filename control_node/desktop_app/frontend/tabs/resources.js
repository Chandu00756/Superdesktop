export function renderResources(root,state){
  const r = state.data.resources || {}; const cpu=r.cpu||{}; const gpu=r.gpu||{}; const mem=r.memory||{}; const storage=r.storage||{};
  root.innerHTML = `<div class='grid'>
    ${section('CPU', cpuSection(cpu))}
    ${section('GPU', gpuSection(gpu))}
    ${section('Memory', memSection(mem))}
    ${section('Storage', storageSection(storage))}
  </div>`;
  function section(title,body){ return `<div class='card'><h3>${title}</h3>${body}</div>`; }
  function cpuSection(d){ return `<div class='mono'>Total cores: ${d.total_cores||0}<br>Active cores: ${d.active_cores||0}<br>Usage: ${d.usage_percentage||0}%<br>${(d.nodes||[]).map(n=>`${n.node_id}:${n.usage||0}%`).join('<br>')}</div>`; }
  function gpuSection(d){ return `<div class='mono'>Total units: ${d.total_units||0}<br>Active: ${d.active_units||0}<br>VRAM ${d.used_vram||0}/${d.total_vram||0} GB</div>`; }
  function memSection(d){ return `<div class='mono'>Total: ${d.total_ram||0} GB<br>Allocated: ${d.allocated_ram||0} GB<br>Cached: ${d.cached_ram||0} GB<br>Swap: ${d.swap_usage||0}%</div>`; }
  function storageSection(d){ return `<div class='mono'>NVMe: ${d.nvme_pool?d.nvme_pool.used:0}/${d.nvme_pool?d.nvme_pool.capacity:0} TB<br>SATA: ${d.sata_pool?d.sata_pool.used:0}/${d.sata_pool?d.sata_pool.capacity:0} TB</div>`; }
}
