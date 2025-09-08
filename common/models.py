"""
Omega Super Desktop Console - Common Models and Utilities
Initial prototype shared models and utility functions.
"""

from typing import Dict, Any, List

class NodeInfo:
    def __init__(self, node_id: str, node_type: str, status: str, resources: Dict[str, Any]):
        self.node_id = node_id
        self.node_type = node_type
        self.status = status
        self.resources = resources

class ResourceRequest:
    def __init__(self, node_id: str, resource_type: str, amount: Any):
        self.node_id = node_id
        self.resource_type = resource_type
        self.amount = amount

class TaskStatus:
    def __init__(self, task_id: str, status: str, result: Any):
        self.task_id = task_id
        self.status = status
        self.result = result

# Utility: Generate unique IDs
import uuid
try:
    import ulid
except Exception:  # ulid may not be installed in minimal env; fallback gracefully
    ulid = None  # type: ignore

def generate_id(prefix: str) -> str:
    """Generate a lexicographically sortable, globally unique identifier.

    Preference order:
    1. ULID (time sortable) if library available
    2. UUID4 hex
    Result is prefixed with provided prefix and a hyphen for clarity.
    """
    if ulid is not None:  # type: ignore
        try:
            return f"{prefix}-{ulid.new().str.lower()}"  # type: ignore[attr-defined]
        except Exception:
            pass
    return f"{prefix}-{uuid.uuid4().hex}"

# Utility: Logging setup
import logging

def setup_logging(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
