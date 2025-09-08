import pytest
from datetime import datetime, timezone
import importlib.util, types, pathlib

# Dynamically load omega-orchestrator/main.py as a module since directory has hyphen
_orch_path = pathlib.Path(__file__).parent.parent / 'omega-orchestrator' / 'main.py'
_spec = importlib.util.spec_from_file_location('omega_orchestrator_dyn', _orch_path)
_orch_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_orch_mod)  # type: ignore

OmegaOrchestrator = _orch_mod.OmegaOrchestrator
PlacementRequest = _orch_mod.PlacementRequest
NodeSpec = _orch_mod.NodeSpec

@pytest.mark.asyncio
async def test_predictive_fairness_penalty_balances_nodes():
    orch = OmegaOrchestrator()
    now = datetime.now(timezone.utc)
    orch.nodes['n1'] = NodeSpec(node_id='n1', node_type='cpu_node', resources={'cpu':8,'memory':32768}, status='active', last_heartbeat=now, labels={}, annotations={}, network_config={})
    orch.nodes['n2'] = NodeSpec(node_id='n2', node_type='cpu_node', resources={'cpu':8,'memory':32768}, status='active', last_heartbeat=now, labels={}, annotations={}, network_config={})

    orch._node_schedule_counts['n1'] = 20
    orch._node_schedule_counts['n2'] = 5

    req = PlacementRequest(session_id='s1', resource_requirements={'cpu':2,'memory':1024}, constraints=[], preferences=[])
    scores = await orch._calculate_placement_scores(req, candidates=orch.nodes)
    assert scores['n2'] >= scores['n1'], f"Expected fairness penalty to reduce n1 score. Scores: {scores}"

@pytest.mark.asyncio
async def test_predictive_penalty_zero_when_low_history():
    orch = OmegaOrchestrator()
    now = datetime.now(timezone.utc)
    orch.nodes['n1'] = NodeSpec(node_id='n1', node_type='cpu_node', resources={'cpu':4,'memory':8192}, status='active', last_heartbeat=now, labels={}, annotations={}, network_config={})
    orch._node_schedule_counts['n1'] = 2  # total < 10 so penalty should be zero
    req = PlacementRequest(session_id='s2', resource_requirements={'cpu':1,'memory':512}, constraints=[], preferences=[])
    scores = await orch._calculate_placement_scores(req, candidates=orch.nodes)
    assert 'n1' in scores and scores['n1'] > 0
