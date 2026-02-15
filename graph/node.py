"""
Node class for dependency graph.

A Node represents a unit of computation with dependencies.
"""

from typing import Any, Callable, List, Optional, Dict
from enum import Enum
import time
import hashlib
import pickle


class NodeStatus(Enum):
    """Status of node execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Node:
    """
    Computation node in a dependency graph.
    
    A node represents a function that depends on other nodes' outputs.
    
    Args:
        node_id: Unique identifier for this node
        fn: Function to execute
        depends_on: List of node IDs this depends on
        description: Optional description
        
    Example:
        >>> preprocess = Node("preprocess", fn=load_data)
        >>> train = Node("train", fn=train_model, depends_on=["preprocess"])
    """
    
    def __init__(
        self,
        node_id: str,
        fn: Callable,
        depends_on: Optional[List[str]] = None,
        description: Optional[str] = None,
        cache_result: bool = True
    ):
        self.node_id = node_id
        self.fn = fn
        self.depends_on = depends_on or []
        self.description = description or f"Node {node_id}"
        self.cache_result = cache_result
        
        # Execution state
        self.status = NodeStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute this node's function.
        
        Args:
            inputs: Dictionary mapping dependency node_ids to their results
            
        Returns:
            Result of executing this node's function
            
        Raises:
            Exception: If execution fails
        """
        self.status = NodeStatus.RUNNING
        self.start_time = time.time()
        
        try:
            # Prepare arguments from dependencies
            args = [inputs.get(dep_id) for dep_id in self.depends_on]
            
            # Execute function
            self.result = self.fn(*args)
            
            self.status = NodeStatus.COMPLETED
            self.end_time = time.time()
            
            return self.result
            
        except Exception as e:
            self.status = NodeStatus.FAILED
            self.error = e
            self.end_time = time.time()
            raise
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def reset(self):
        """Reset node to pending state"""
        self.status = NodeStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def get_hash(self) -> str:
        """
        Get hash of node function and dependencies.
        
        Used for caching - same hash means same computation.
        """
        try:
            # Try to pickle function for hashing
            fn_bytes = pickle.dumps(self.fn)
        except:
            # If function can't be pickled, use code
            fn_bytes = self.fn.__code__.co_code
        
        deps_bytes = pickle.dumps(sorted(self.depends_on))
        
        hash_obj = hashlib.sha256(fn_bytes + deps_bytes)
        return hash_obj.hexdigest()[:16]
    
    def __repr__(self) -> str:
        """String representation"""
        duration = self.get_duration()
        duration_str = f", {duration:.2f}s" if duration else ""
        return f"Node(id='{self.node_id}', status={self.status.value}{duration_str})"
    
    def __hash__(self):
        """Hash based on node_id"""
        return hash(self.node_id)
    
    def __eq__(self, other):
        """Equality based on node_id"""
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id


class ConditionalNode(Node):
    """
    Node that executes conditionally based on a predicate.
    
    Args:
        node_id: Node identifier
        fn: Function to execute
        condition: Function that returns True/False
        depends_on: Dependencies
    """
    
    def __init__(
        self,
        node_id: str,
        fn: Callable,
        condition: Callable[..., bool],
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(node_id, fn, depends_on, **kwargs)
        self.condition = condition
    
    def should_execute(self, inputs: Dict[str, Any]) -> bool:
        """
        Check if node should execute.
        
        Args:
            inputs: Dependency results
            
        Returns:
            True if node should execute
        """
        try:
            args = [inputs.get(dep_id) for dep_id in self.depends_on]
            return self.condition(*args)
        except Exception:
            return False
    
    def execute(self, inputs: Dict[str, Any]) -> Any:
        """Execute only if condition is met"""
        if not self.should_execute(inputs):
            self.status = NodeStatus.SKIPPED
            return None
        
        return super().execute(inputs)