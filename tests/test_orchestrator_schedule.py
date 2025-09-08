import asyncio
from httpx import AsyncClient
from httpx import ASGITransport
from fastapi import FastAPI

# Import orchestrator app
from datetime import datetime, timezone
import importlib.util, pathlib, types

# Dynamically load orchestrator module (directory name has hyphen)
_orch_path = pathlib.Path(__file__).resolve().parent.parent / 'omega-orchestrator' / 'main.py'
spec = importlib.util.spec_from_file_location('omega_orchestrator_main', _orch_path)
orch_mod = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(orch_mod)  # type: ignore
orch_app = orch_mod.app
orch = orch_mod.orch
NodeSpec = orch_mod.NodeSpec

def test_orchestrator_health_and_schedule():
    async def run():
        if orch.cluster_state != 'active':
            await orch.initialize()
        # Register a mock node if not present
        if 'node-test-1' not in orch.nodes:
            node = NodeSpec(
                node_id='node-test-1',
                node_type='cpu_node',
                resources={'cpu': 16, 'memory': 64, 'gpu': 0},
                status='active',
                last_heartbeat=datetime.now(timezone.utc),
                labels={'region':'local'},
                annotations={},
                network_config={}
            )
            orch.nodes[node.node_id] = node
        transport = ASGITransport(app=orch_app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            r = await client.get('/health')
            assert r.status_code == 200
            body = r.json()
            assert 'status' in body
            req = {
                'session_id': 'sess-123',
                'resource_requirements': {'cpu': 2, 'memory': 4},
                'constraints': [],
                'preferences': []
            }
            r2 = await client.post('/schedule', json=req)
            assert r2.status_code == 200, r2.text
            dec = r2.json()
            assert dec['session_id'] == 'sess-123'
            assert isinstance(dec['selected_nodes'], list)
            assert dec['selected_nodes'], 'No node selected'
    asyncio.run(run())
