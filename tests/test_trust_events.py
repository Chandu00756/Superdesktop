import asyncio
import importlib.util
import pathlib
import time

import pytest


def _load_orchestrator_module():
    orch_path = pathlib.Path(__file__).parent.parent / 'omega-orchestrator' / 'main.py'
    spec = importlib.util.spec_from_file_location('omega_orchestrator_events', orch_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_handle_trust_event_unit():
    mod = _load_orchestrator_module()
    # Create a node and attach to orchestrator
    node_id = 'evt-node-1'
    spec = mod.NodeSpec(
        node_id=node_id,
        node_type='cpu_node',
        resources={'cpu': 4, 'memory': 8},
        status='active',
        last_heartbeat=mod.datetime.now(mod.timezone.utc),
        labels={}, annotations={}, network_config={}, trust_score=1.0
    )
    mod.orch.nodes[node_id] = spec

    # Apply negative event and verify trust decreased
    new_trust = mod.orch._handle_trust_event('node.failed', {'node_id': node_id})
    assert new_trust is not None
    assert 0.79 <= float(new_trust) <= 0.81  # -0.2 from 1.0 with clamping

    # Apply recovery event and verify trust increases
    new_trust = mod.orch._handle_trust_event('node.recovered', {'node_id': node_id})
    assert new_trust is not None
    # 0.8 + 0.05 => ~0.85
    assert 0.84 <= float(new_trust) <= 0.86


@pytest.mark.asyncio
async def test_trust_consumer_integration_memory_bus():
    mod = _load_orchestrator_module()
    # Build a fresh orchestrator for isolation
    orch = mod.OmegaOrchestrator()
    # Register one node in-memory
    node_id = 'evt-node-2'
    orch.nodes[node_id] = mod.NodeSpec(
        node_id=node_id,
        node_type='cpu_node',
        resources={'cpu': 4, 'memory': 8},
        status='active',
        last_heartbeat=mod.datetime.now(mod.timezone.utc),
        labels={}, annotations={}, network_config={}, trust_score=0.9
    )

    # Start the consumer task
    consumer_task = asyncio.create_task(orch._trust_anomaly_consumer())

    # Publish a couple of events via the in-memory event bus
    from common.event_bus import get_event_bus
    bus = await get_event_bus()
    await bus.publish('cluster', 'node.unhealthy', {'node_id': node_id})
    await bus.publish('cluster', 'schedule.decision', {'nodes': [node_id], 'score': 1.23})

    # Allow the consumer to process
    await asyncio.sleep(0.1)

    # Verify trust adjusted: -0.05 + 0.01 => net -0.04 => 0.86
    trust = orch.nodes[node_id].trust_score
    assert 0.85 <= float(trust) <= 0.87

    # Cleanup: cancel consumer
    consumer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer_task
