export const AppState = {
  activeTab: 'dashboard',
  decrypted:false,
  data: {}
};

export function mergeRealtime(state, msg){
  if(msg.type==='rt_delta'){
    if(msg.nodes_full) state.data.nodes={nodes: msg.nodes_full};
    if(msg.nodes_changed && state.data.nodes){
      const map=new Map(state.data.nodes.nodes.map(n=>[n.node_id,n]));
      msg.nodes_changed.forEach(n=>map.set(n.node_id,n));
      state.data.nodes.nodes=[...map.values()];
    }
    if(msg.sessions_full) state.data.sessions={sessions: msg.sessions_full};
    if(msg.sessions_changed && state.data.sessions){
      const smap=new Map(state.data.sessions.sessions.map(s=>[s.session_id,s]));
      msg.sessions_changed.forEach(s=>smap.set(s.session_id,s));
      state.data.sessions.sessions=[...smap.values()];
    }
  }
}
