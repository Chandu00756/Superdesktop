"""
UFO-Task: Task Graph Abstractions
Provides the basic primitives for defining Directed Acyclic Graphs (DAGs) of tasks
that operate over the Partitioned Global Address Space (PGAS).
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass, field
import uuid
from enum import Enum

from memory_fabric.pgas_manager import UfoGlobalPointer


class TaskCapability(Enum):
    CPU_GENERAL = "cpu_general"
    CPU_VECTOR = "cpu_vector"
    GPU_CUDA = "gpu_cuda"
    GPU_TENSOR = "gpu_tensor"
    NPU_EDGE = "npu_edge"


@dataclass
class UfoTaskDescriptor:
    """
    Metadata describing a specific UFO computation task that can be scheduled
    across the fabric.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_task"
    
    # Requirements
    required_capabilities: List[TaskCapability] = field(default_factory=list)
    latency_sensitive: bool = False
    
    # Data Dependencies (PGAS Pointers)
    ufo_in: List[UfoGlobalPointer] = field(default_factory=list)
    ufo_out: List[UfoGlobalPointer] = field(default_factory=list)
    ufo_inout: List[UfoGlobalPointer] = field(default_factory=list)
    
    # Execution Payload (e.g. serialized python func, docker image, or path to kernel)
    exec_payload: bytes = b""
    exec_state: bytes = b"" # For suspend/resume migration
    
    # Graph Dependencies
    depends_on: Set[str] = field(default_factory=set)
    
    def add_dependency(self, parent_task_id: str):
        self.depends_on.add(parent_task_id)


class UfoTaskGraph:
    """Represents a DAG of UfoTasks to be submitted to the scheduler."""
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, UfoTaskDescriptor] = {}
        
    def add_task(self, task: UfoTaskDescriptor):
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists in graph.")
        self.tasks[task.task_id] = task
        
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[UfoTaskDescriptor]:
        """Returns tasks whose dependencies have been met."""
        ready = []
        for task in self.tasks.values():
            if task.task_id not in completed_tasks:
                if task.depends_on.issubset(completed_tasks):
                    ready.append(task)
        return ready

    def is_complete(self, completed_tasks: Set[str]) -> bool:
        return len(completed_tasks) == len(self.tasks)
