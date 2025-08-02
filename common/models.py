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

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"

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
