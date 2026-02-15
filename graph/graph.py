"""
Graph class for managing dependency graphs.

A Graph manages nodes and their dependencies, executing them in
topologically sorted order.
"""

from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from graph.node import Node, NodeStatus
from core.exceptions import FramewormError


class GraphError(FramewormError):
    """Raised when graph operations fail"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.set_doc_link("https://frameworm.readthedocs.io/graph/errors")


class CycleDetectedError(GraphError):
    """Raised when a cycle is detected in the graph"""
    
    def __init__(self, cycle_path: List[str], **kwargs):
        message = "Cycle detected in dependency graph"
        super().__init__(message, cycle_path=cycle_path, **kwargs)
        
        cycle_str = " → ".join(cycle_path)
        self.add_cause(f"Circular dependency: {cycle_str}")
        self.add_suggestion("Remove one of the dependencies to break the cycle")
        self.add_suggestion("Review your pipeline structure")
    
    def get_details(self) -> Dict[str, Any]:
        return {'Cycle': " → ".join(self.cycle_path)}


class Graph:
    """
    Dependency graph for managing computational workflows.
    
    Manages nodes and their dependencies, executing them in correct order.
    
    Example:
        >>> graph = Graph()
        >>> graph.add_node(Node("a", fn=lambda: 1))
        >>> graph.add_node(Node("b", fn=lambda x: x + 1, depends_on=["a"]))
        >>> results = graph.execute()
        >>> print(results["b"])  # 2
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._execution_order: Optional[List[str]] = None
        self._cache: Dict[str, Any] = {}
    
    def add_node(self, node: Node):
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            
        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in graph")
        
        self.nodes[node.node_id] = node
        self._execution_order = None  # Invalidate cached order
    
    def add_nodes(self, nodes: List[Node]):
        """Add multiple nodes"""
        for node in nodes:
            self.add_node(node)
    
    def get_node(self, node_id: str) -> Node:
        """
        Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node object
            
        Raises:
            KeyError: If node not found
        """
        if node_id not in self.nodes:
            available = list(self.nodes.keys())
            raise KeyError(
                f"Node '{node_id}' not found. Available: {available}"
            )
        return self.nodes[node_id]
    
    def remove_node(self, node_id: str):
        """
        Remove a node from the graph.
        
        Args:
            node_id: Node to remove
            
        Raises:
            ValueError: If other nodes depend on this node
        """
        # Check if any other nodes depend on this
        dependents = [
            n_id for n_id, node in self.nodes.items()
            if node_id in node.depends_on
        ]
        
        if dependents:
            raise ValueError(
                f"Cannot remove node '{node_id}': other nodes depend on it: {dependents}"
            )
        
        del self.nodes[node_id]
        self._execution_order = None
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get direct dependencies of a node"""
        return self.nodes[node_id].depends_on.copy()
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get nodes that depend on this node"""
        return [
            n_id for n_id, node in self.nodes.items()
            if node_id in node.depends_on
        ]


    def has_cycle(self) -> bool:
        """
        Check if graph has a cycle.
        
        Returns:
            True if cycle exists
        """
        # Colors for DFS: 0=white (unvisited), 1=gray (in progress), 2=black (done)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            """DFS to detect back edge (cycle)"""
            color[node_id] = GRAY
            
            for dep_id in self.nodes[node_id].depends_on:
                if color[dep_id] == GRAY:  # Back edge = cycle
                    return True
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            
            color[node_id] = BLACK
            return False
        
        # Check from each unvisited node
        return any(dfs(node_id) for node_id in self.nodes if color[node_id] == WHITE)
    
    def find_cycle(self) -> Optional[List[str]]:
        """
        Find a cycle in the graph if one exists.
        
        Returns:
            List of node IDs forming a cycle, or None if no cycle
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in self.nodes}
        parent = {node_id: None for node_id in self.nodes}
        
        def dfs(node_id: str) -> Optional[List[str]]:
            """DFS to find cycle"""
            color[node_id] = GRAY
            
            for dep_id in self.nodes[node_id].depends_on:
                if color[dep_id] == GRAY:
                    # Found cycle, reconstruct path
                    cycle = [dep_id]
                    current = node_id
                    while current != dep_id:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(dep_id)  # Complete the cycle
                    return cycle[::-1]
                
                if color[dep_id] == WHITE:
                    parent[dep_id] = node_id
                    result = dfs(dep_id)
                    if result:
                        return result
            
            color[node_id] = BLACK
            return None
        
        for node_id in self.nodes:
            if color[node_id] == WHITE:
                result = dfs(node_id)
                if result:
                    return result
        
        return None
    
    def topological_sort(self) -> List[str]:
    # Validate dependencies first
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise GraphError(f"Dependency '{dep}' doesn't exist")

        if self.has_cycle():
            raise CycleDetectedError("Graph contains a cycle")
            
            # Calculate in-degree (number of dependencies)
        in_degree = {node_id: 0 for node_id in self.nodes}
            
        for node_id, node in self.nodes.items():
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    raise GraphError(
                        f"Node '{node_id}' depends on '{dep_id}' which doesn't exist"
                    )
                in_degree[node_id] += 1
            
            # Start with nodes that have no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
            
            # Process nodes
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
                
                # "Remove" edges by decreasing in-degree
            for dependent_id in self.get_dependents(node_id):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
            
            # If we didn't process all nodes, there's a cycle
            # (This shouldn't happen since we checked earlier, but safety check)
        if len(result) != len(self.nodes):
            raise CycleDetectedError(["unknown"])
            
        return result
    
    def get_execution_order(self, force_recompute: bool = False) -> List[str]:
        """
        Get cached execution order.
        
        Args:
            force_recompute: If True, recompute even if cached
            
        Returns:
            List of node IDs in execution order
        """
        if self._execution_order is None or force_recompute:
            self._execution_order = self.topological_sort()
        
        return self._execution_order.copy()
    
    def reset(self):
        """Reset all nodes to pending state"""
        for node in self.nodes.values():
            node.reset()
        self._cache.clear()
    
    def __len__(self) -> int:
        """Number of nodes in graph"""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists"""
        return node_id in self.nodes
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Graph(nodes={len(self.nodes)})"