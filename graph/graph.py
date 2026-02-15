"""
Graph class for managing dependency graphs.

A Graph manages nodes and their dependencies, executing them in
topologically sorted order.
"""

from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from graph.node import Node, NodeStatus
from core.exceptions import FramewormError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Literal
import threading

def _execute_node_process(node_id: str, node_fn, inputs: dict):
    """
    Top-level worker function for ProcessPoolExecutor.
    Must be module-level to be pickleable on Windows.
    """
    try:
        if inputs:
            return node_fn(**inputs)
        return node_fn()
    except Exception as e:
        raise e


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

        cycle = self.find_cycle()
        if cycle:
            raise CycleDetectedError(cycle)

            
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
    # Add execute method to Graph class for convenience
# Add this method to the Graph class:

    def execute(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = False,
        skip_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the graph.
        
        Convenience method that creates an ExecutionEngine and runs it.
        
        Args:
            initial_inputs: Initial values
            continue_on_error: Continue even if nodes fail
            skip_nodes: Nodes to skip
            
        Returns:
            Dictionary of results
        """
        engine = ExecutionEngine(self)
        return engine.execute(
            initial_inputs=initial_inputs,
            continue_on_error=continue_on_error,
            skip_nodes=skip_nodes
        )
    
    def get_last_execution_summary(self) -> Dict[str, Any]:
        """Get summary of last execution"""
        engine = ExecutionEngine(self)
        # Copy results from nodes
        for node_id, node in self.nodes.items():
            if node.result is not None:
                engine.results[node_id] = node.result
            if node.error is not None:
                engine.errors[node_id] = node.error
        
        return engine.get_execution_summary()

    def execute_parallel(
        self,
        max_workers: int = 4,
        executor_type: Literal['thread', 'process'] = 'thread',
        initial_inputs: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute graph in parallel.
        
        Args:
            max_workers: Maximum parallel workers
            executor_type: 'thread' or 'process'
            initial_inputs: Initial values
            progress_callback: Progress callback
            **kwargs: Additional execution arguments
            
        Returns:
            Dictionary of results
        """
        engine = ParallelExecutionEngine(
            self,
            max_workers=max_workers,
            executor_type=executor_type
        )
        
        return engine.execute(
            initial_inputs=initial_inputs,
            progress_callback=progress_callback,
            **kwargs
        )
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        Get nodes grouped by execution level.
        
        Returns:
            List of levels, each level can execute in parallel
        """
        engine = ParallelExecutionEngine(self)
        return engine.compute_execution_levels()
    
        
class ExecutionEngine:
    """
    Engine for executing dependency graphs.
    
    Handles node execution, result passing, and error management.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}

    def execute(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = False,
        skip_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute all nodes in the graph.
        
        Args:
            initial_inputs: Initial values for nodes (for nodes with no dependencies)
            continue_on_error: If True, continue execution even if nodes fail
            skip_nodes: List of node IDs to skip
            
        Returns:
            Dictionary mapping node_id → result
            
        Raises:
            GraphError: If execution fails and continue_on_error=False
        """
        initial_inputs = initial_inputs or {}
        skip_nodes = skip_nodes or []
        
        # Reset state
        self.results = {}
        self.errors = {}
        self.graph.reset()
        
        # Get execution order
        try:
            execution_order = self.graph.get_execution_order()
        except CycleDetectedError as e:
            raise GraphError(f"Cannot execute graph with cycle: {e}")
        
        # Execute nodes in order
        for node_id in execution_order:
            node = self.graph.get_node(node_id)

            # If node has no dependencies and initial input is provided,
            # use it instead of executing the node
            if not node.depends_on and node_id in initial_inputs:
                self.results[node_id] = initial_inputs[node_id]
                node.status = NodeStatus.COMPLETED
                continue
            # Check if should skip

            if node_id in skip_nodes:
                self.graph.get_node(node_id).status = NodeStatus.SKIPPED
                continue
            
            # Check if dependencies failed
            node = self.graph.get_node(node_id)
            failed_deps = [dep for dep in node.depends_on if dep in self.errors]
            skipped_deps = [dep for dep in node.depends_on if self.graph.get_node(dep).status == NodeStatus.SKIPPED]

            if failed_deps:
                node.status = NodeStatus.SKIPPED
                self.errors[node_id] = GraphError(
                    f"Dependencies failed: {failed_deps}"
                )
                if not continue_on_error:
                    raise self.errors[node_id]
                continue

            if skipped_deps:
                # Downstream of a skipped node → skip silently
                node.status = NodeStatus.SKIPPED
                continue

            
            # Prepare inputs
            inputs = self._prepare_inputs(node, initial_inputs)
            
            # Execute node
            try:
                result = node.execute(inputs)
                self.results[node_id] = result
            except Exception as e:
                self.errors[node_id] = e
                
                if not continue_on_error:
                    raise GraphError(
                        f"Node '{node_id}' failed: {e}",
                        node_id=node_id,
                        original_error=e
                    )
        
        return self.results
    
    
    def _prepare_inputs(
        self,
        node: Node,
        initial_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for node execution.
        
        Args:
            node: Node to prepare inputs for
            initial_inputs: Initial values provided by user
            
        Returns:
            Dictionary of inputs for the node
        """
        inputs = {}
        
        for dep_id in node.depends_on:
            # Check if dependency result is available
            if dep_id in self.results:
                inputs[dep_id] = self.results[dep_id]
            elif dep_id in initial_inputs:
                inputs[dep_id] = initial_inputs[dep_id]
            else:
                # This shouldn't happen if topological sort is correct
                raise GraphError(
                    f"Missing input for dependency '{dep_id}' of node '{node.node_id}'"
                )
        
        return inputs
    
    def get_failed_nodes(self) -> List[str]:
        """Get list of failed node IDs"""
        return list(self.errors.keys())
    
    def get_completed_nodes(self) -> List[str]:
        """Get list of successfully completed node IDs"""
        return list(self.results.keys())
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution.
        
        Returns:
            Dictionary with execution statistics
        """
        total_nodes = len(self.graph)
        completed = len(self.results)
        failed = len(self.errors)
        skipped = total_nodes - completed - failed
        
        # Calculate total duration
        total_duration = sum(
            node.get_duration() or 0
            for node in self.graph.nodes.values()
            if node.status == NodeStatus.COMPLETED
        )
        
        return {
            'total_nodes': total_nodes,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'total_duration': total_duration,
            'errors': {node_id: str(err) for node_id, err in self.errors.items()}
        }

class CachedGraph(Graph):
    """
    Graph with result caching.
    
    Caches node results based on function hash and input hash.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".graph_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, node: Node, inputs: Dict[str, Any]) -> str:
        """Generate cache key for node execution"""
        import hashlib
        import pickle
        
        # Hash node function
        node_hash = node.get_hash()
        
        # Hash inputs
        try:
            input_bytes = pickle.dumps(sorted(inputs.items()))
            input_hash = hashlib.sha256(input_bytes).hexdigest()[:16]
        except:
            # If inputs can't be pickled, no caching
            return None
        
        return f"{node.node_id}_{node_hash}_{input_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load result from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, cache_key: str, result: Any):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            # If result can't be pickled, skip caching
            pass
    
    def execute(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute with caching"""
        if not use_cache:
            return super().execute(initial_inputs, **kwargs)
        
        # Custom execution with cache checks
        initial_inputs = initial_inputs or {}
        results = {}
        
        execution_order = self.get_execution_order()
        
        for node_id in execution_order:
            node = self.get_node(node_id)
            
            if not node.depends_on and node_id in initial_inputs:
                results[node_id] = initial_inputs[node_id]
                node.result = initial_inputs[node_id]
                node.status = NodeStatus.COMPLETED
                continue

            # Prepare inputs
            inputs = {}
            for dep_id in node.depends_on:
                if dep_id in results:
                    inputs[dep_id] = results[dep_id]
                elif dep_id in initial_inputs:
                    inputs[dep_id] = initial_inputs[dep_id]
            
            # Check cache
            cache_key = self._get_cache_key(node, inputs)
            
            if cache_key and node.cache_result:
                cached_result = self._load_from_cache(cache_key)
                
                if cached_result is not None:
                    node.result = cached_result
                    node.status = NodeStatus.COMPLETED
                    results[node_id] = cached_result
                    continue
            
            # Execute node
            result = node.execute(inputs)
            results[node_id] = result
            
            # Save to cache
            if cache_key and node.cache_result:
                self._save_to_cache(cache_key, result)
        
        return results
    
    def clear_cache(self):
        """Clear all cached results"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()


class ParallelExecutionEngine(ExecutionEngine):
    """
    Engine for parallel graph execution.
    
    Executes independent nodes in parallel while maintaining
    topological order.
    """
    
    def __init__(
        self,
        graph: Graph,
        max_workers: int = 4,
        executor_type: Literal['thread', 'process'] = 'thread'
    ):
        super().__init__(graph)
        self.max_workers = max_workers
        self.executor_type = executor_type
        self._lock = threading.Lock()
    
    def compute_execution_levels(self) -> List[List[str]]:
        """
        Group nodes into levels for parallel execution.
        
        Level 0: Nodes with no dependencies
        Level 1: Nodes depending only on Level 0
        Level 2: Nodes depending on Level 0 or 1
        ...
        
        Returns:
            List of levels, each level is a list of node IDs
        """
        levels = []
        remaining = set(self.graph.nodes.keys())
        completed = set()
        
        while remaining:
            # Find nodes with all dependencies satisfied
            level = [
                node_id for node_id in remaining
                if all(dep in completed for dep in self.graph.nodes[node_id].depends_on)
            ]
            
            if not level:
                # This shouldn't happen if topological sort succeeded
                raise GraphError("Cannot compute execution levels")
            
            levels.append(level)
            completed.update(level)
            remaining -= set(level)
        
        return levels
    
    def execute(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = False,
        skip_nodes: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute graph in parallel.
        
        Args:
            initial_inputs: Initial values
            continue_on_error: Continue if nodes fail
            skip_nodes: Nodes to skip
            progress_callback: Called on each node completion: callback(node_id, completed, total)
            
        Returns:
            Dictionary of results
        """
        initial_inputs = initial_inputs or {}
        skip_nodes = skip_nodes or []
        
        # Reset state
        self.results = {}
        self.errors = {}
        self.graph.reset()
        
        # Check for cycles
        try:
            self.graph.get_execution_order()
        except CycleDetectedError as e:
            raise GraphError(f"Cannot execute graph with cycle: {e}")
        
        # Compute execution levels
        levels = self.compute_execution_levels()
        
        # Choose executor
        ExecutorClass = ThreadPoolExecutor if self.executor_type == 'thread' else ProcessPoolExecutor
        
        total_nodes = len(self.graph)
        completed_count = 0
        
        # Execute level by level
        for level_idx, level_nodes in enumerate(levels):
            # Filter out skipped nodes
            level_nodes = [n for n in level_nodes if n not in skip_nodes]
            
            if not level_nodes:
                continue
            
            # Execute level in parallel
            with ExecutorClass(max_workers=min(self.max_workers, len(level_nodes))) as executor:
                # Submit all nodes in this level
                futures = {}
                for node_id in level_nodes:
                    # Check if dependencies failed
                    node = self.graph.get_node(node_id)
                    failed_deps = [dep for dep in node.depends_on if dep in self.errors]
                    
                    if failed_deps and not continue_on_error:
                        node.status = NodeStatus.SKIPPED
                        continue
                    
                    # Prepare inputs
                    inputs = self._prepare_inputs(node, initial_inputs)
                    
                    # Submit for execution
                    node = self.graph.get_node(node_id)

                    if self.executor_type == "process":
                        # 🔥 DO NOT send self into process pool
                        future = executor.submit(
                            _execute_node_process,
                            node_id,
                            node.fn,
                            inputs
                        )
                    else:
                        # Thread mode can use bound method safely
                        future = executor.submit(
                            self._execute_node_safe,
                            node_id,
                            inputs
                        )

                    futures[future] = node_id
                
                # Collect results as they complete
                for future in as_completed(futures):
                    node_id = futures[future]
                    
                    try:
                        result = future.result()
                        
                        # Store result (thread-safe)
                        with self._lock:
                            self.results[node_id] = result
                            completed_count += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(node_id, completed_count, total_nodes)
                    
                    except Exception as e:
                        with self._lock:
                            self.errors[node_id] = e
                            completed_count += 1
                        
                        if not continue_on_error:
                            # Cancel remaining futures in this level
                            for f in futures:
                                f.cancel()
                            
                            raise GraphError(
                                f"Node '{node_id}' failed: {e}",
                                node_id=node_id,
                                original_error=e
                            )
        
        return self.results
    
    def _execute_node_safe(self, node_id: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute node with error handling.
        
        This method is called in parallel, so it needs to be thread-safe.
        """
        node = self.graph.get_node(node_id)
        
        try:
            result = node.execute(inputs)
            return result
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = e
            raise