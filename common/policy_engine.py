"""Policy Engine Scaffold

This module provides a structured representation for policies with
YAML/JSON loading, basic validation, and evaluation stubs. It is a
progression path from the ad-hoc evaluate_policies logic currently in
`backend.api_server`.

Policy Document (YAML/JSON) schema (v1):

id: <string>
version: 1
meta:
  description: <string>
  labels:
    env: prod
spec:
  targets:
    - kind: user|role|group|node
      selector:
        id: <id or pattern>
  statements:
    - effect: allow|deny
      actions: ["nodes:view", "sessions:start"]
      conditions:
        any:
          - attr: time.hour
            op: in
            value: [8,9,10]
        all: []
      reason: <string>

Conditions are NOT fully implemented yet; the structure is preserved for
future advanced evaluation (attribute, operation, value trees).
"""
from __future__ import annotations
import os, json, yaml, time, fnmatch, threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class PolicyStatement:
    effect: str  # allow|deny
    actions: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

@dataclass
class PolicySpec:
    targets: List[Dict[str, Any]]
    statements: List[PolicyStatement]

@dataclass
class PolicyDoc:
    id: str
    version: int
    meta: Dict[str, Any]
    spec: PolicySpec
    loaded_at: float = field(default_factory=time.time)

class PolicyLoader:
    def __init__(self, directory: str = "policies"):
        self.directory = directory
        self._lock = threading.RLock()
        self._policies: Dict[str, PolicyDoc] = {}
    def load_all(self) -> Dict[str, PolicyDoc]:
        with self._lock:
            self._policies.clear()
            if not os.path.isdir(self.directory):
                return self._policies
            for name in os.listdir(self.directory):
                if not (name.endswith('.yml') or name.endswith('.yaml') or name.endswith('.json')):
                    continue
                path = os.path.join(self.directory, name)
                try:
                    with open(path,'r') as f:
                        if name.endswith('.json'):
                            raw = json.load(f)
                        else:
                            raw = yaml.safe_load(f)
                    pol = self._parse(raw)
                    self._policies[pol.id] = pol
                except Exception:
                    continue
            return dict(self._policies)
    def snapshot(self) -> Dict[str, PolicyDoc]:
        with self._lock:
            return dict(self._policies)
    def _parse(self, raw: Dict[str, Any]) -> PolicyDoc:
        pid = raw.get('id') or raw.get('name') or 'unknown'
        version = int(raw.get('version') or 1)
        meta = raw.get('meta') or {}
        spec_raw = raw.get('spec') or {}
        stmts = []
        for s in spec_raw.get('statements', []):
            stmts.append(PolicyStatement(
                effect=(s.get('effect') or 'deny').lower(),
                actions=[a for a in s.get('actions', []) if isinstance(a,str)],
                conditions=s.get('conditions') or {},
                reason=s.get('reason')
            ))
        spec = PolicySpec(targets=spec_raw.get('targets') or [], statements=stmts)
        return PolicyDoc(id=pid, version=version, meta=meta, spec=spec)

class PolicyEvaluator:
    def __init__(self, loader: PolicyLoader):
        self.loader = loader
    def evaluate(self, subject: str, roles: List[str], base_permissions: List[str]) -> Tuple[set,set]:
        pols = self.loader.snapshot()
        allow = set()
        deny = set()
        eff = set(base_permissions)
        for pol in pols.values():
            for tgt in pol.spec.targets:
                kind = tgt.get('kind')
                sel = tgt.get('selector') or {}
                sid = sel.get('id') or '*'
                matches = False
                if kind == 'user' and fnmatch.fnmatch(subject, sid):
                    matches = True
                elif kind == 'role':
                    for r in roles:
                        if fnmatch.fnmatch(r, sid):
                            matches = True; break
                if not matches:
                    continue
                for stmt in pol.spec.statements:
                    if stmt.effect == 'allow':
                        allow.update(stmt.actions)
                    elif stmt.effect == 'deny':
                        deny.update(stmt.actions)
        eff |= allow
        eff -= deny
        return eff, deny

__all__ = ['PolicyLoader','PolicyEvaluator','PolicyDoc','PolicySpec','PolicyStatement']
