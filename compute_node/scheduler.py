"""
UFO-Task: Capability-aware placement
Schedules UfoTaskDescriptors to optimal compute nodes balancing
fabric latency, available capabilities (Cuda vs CPU), and data location.
"""

from typing import Dict, List, Set
import logging
from .task_graph import UfoTaskGraph, UfoTaskDescriptor, TaskCapability

logger = logging.getLogger(__name__)

class NodeInfo:
    def __init__(self, node_id: int, capabilities: Set[TaskCapability]):
        self.node_id = node_id
        self.capabilities = capabilities
        self.running_tasks = 0

class UfoScheduler:
    def __init__(self):
        self.compute_nodes: Dict[int, NodeInfo] = {}

    def register_node(self, node_id: int, capabilities: List[TaskCapability]):
        self.compute_nodes[node_id] = NodeInfo(node_id, set(capabilities))
        logger.info(f"Registered compute node {node_id} with {capabilities}")
        
    def _score_placement(self, task: UfoTaskDescriptor, node: NodeInfo) -> float:
        """
        Scores a compute node based on task requirements.
        Higher is better.
        """
        # 1. Capability Check - Soft constraint (Needs to match at least ONE required capability)
        intersection = set(task.required_capabilities).intersection(node.capabilities)
        if not intersection:
            return -1.0
            
        score = 100.0
        
        # 2. Data Locality - Bonus for being near memory
        local_data_count = 0
        all_data_ptrs = task.ufo_in + task.ufo_out + task.ufo_inout
        for ptr in all_data_ptrs:
            if ptr.node_id == node.node_id:
                local_data_count += 1
                
        if len(all_data_ptrs) > 0:
            locality_ratio = local_data_count / len(all_data_ptrs)
            score += locality_ratio * 50.0 # Up to 50 pts for perfect locality
            
        # 3. Load Balancing
        score -= (node.running_tasks * 10)
        
        return score

    def schedule(self, graph: UfoTaskGraph, completed: Set[str]) -> Dict[str, int]:
        """
        Returns a mapping of {task_id : node_id} reflecting where ready tasks
        should be executed.
        """
        ready_tasks = graph.get_ready_tasks(completed)
        placement = {}
        
        for task in ready_tasks:
            best_node = None
            best_score = -1.0
            
            for node in self.compute_nodes.values():
                score = self._score_placement(task, node)
                if score > best_score:
                    best_score = score
                    best_node = node
                    
            if best_node is not None:
                placement[task.task_id] = best_node.node_id
                best_node.running_tasks += 1
                logger.debug(f"Scheduled task {task.task_id} on Node {best_node.node_id}")
            else:
                logger.warning(f"Could not find a valid compute node for task {task.task_id}")
                
        return placement
