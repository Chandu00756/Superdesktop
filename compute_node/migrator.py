"""
UFO-Task: Process/State Migration Engine
Facilitates the serialization and physical movement of a running UfoTask
from one Compute Node to another, handling PGAS pointers intrinsically.
"""

from typing import Any, Tuple
import dill # Used for deeply serializing python closures/generators
import logging

from memory_fabric.pgas_manager import UfoGlobalPointer, PGASManager
from .task_graph import UfoTaskDescriptor

logger = logging.getLogger(__name__)


class UfoMigrator:
    """
    Suspends a task, serializes it, and resumes.
    This prototype relies on `dill` to capture generator/closure state.
    """
    def __init__(self, pgas_manager: PGASManager, local_node_id: int):
        self.pgas_manager = pgas_manager
        self.local_node_id = local_node_id

    def suspend_task((self, task: UfoTaskDescriptor, live_object: Any)) -> bytes:
        """
        Takes a running execution state (e.g. a Python generator) and
        serializes it into a byte payload for transmission.
        """
        logger.info(f"Suspending task {task.name} ({task.task_id})")
        # In a real OS, this dumps the process memory space.
        # Here we dump the python object state.
        state = dill.dumps(live_object)
        task.exec_state = state
        return state

    def resume_task(self, task: UfoTaskDescriptor) -> Any:
        """
        Unpacks the suspended state on a new node and returns the runnable object.
        """
        logger.info(f"Resuming task {task.name} ({task.task_id}) on Node {self.local_node_id}")
        if not task.exec_state:
            raise ValueError("Task has no suspended state to resume.")
            
        live_object = dill.loads(task.exec_state)
        # Note: All PGAS pointers inside `live_object` remain valid because
        # the UFO-PGAS namespace is globally accessible from any node!
        return live_object
