import asyncio, pathlib, importlib.util, json
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

# Dynamic load orchestrator module
_orch_path = pathlib.Path(__file__).resolve().parent.parent / 'omega-orchestrator' / 'main.py'
spec = importlib.util.spec_from_file_location('omega_orchestrator_autoevents', _orch_path)
mod = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore
app = mod.app
orch = mod.orch

def test_autoscaling_event_persistence_basic():
    async def run():
        if orch.cluster_state != 'active':
            await orch.initialize()
        # Seed a node so utilization calc >0
        if not orch.nodes:
            from datetime import datetime, timezone
            n = mod.NodeSpec(
                node_id='seed-node',
                node_type='cpu_node',
                resources={'cpu':4,'memory':16,'gpu':0},
                status='active',
                last_heartbeat=datetime.now(timezone.utc),
                labels={'autoscaled':'true'},
                annotations={},
                network_config={}
            )
            orch.nodes[n.node_id]=n
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            # Trigger manual scale out then scale in
            r1 = await client.post('/autoscaling/scale_out', params={'count':1,'reason':'test'})
            assert r1.status_code == 200, r1.text
            r2 = await client.post('/autoscaling/scale_in', params={'count':1,'reason':'test'})
            assert r2.status_code == 200, r2.text
            # Fetch events
            r3 = await client.get('/autoscaling/events')
            assert r3.status_code == 200
            events = r3.json().get('events', [])
            assert any(e['action']=='scale_out' for e in events), 'scale_out event missing'
            assert any(e['action']=='scale_in' for e in events), 'scale_in event missing'
    asyncio.run(run())
