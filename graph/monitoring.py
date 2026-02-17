"""
Performance monitoring for graph execution.

Tracks execution time, memory usage, and bottlenecks.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from graph import Graph


@dataclass
class NodeMetrics:
    """Metrics for a single node execution"""

    node_id: str
    start_time: float
    end_time: float
    duration: float
    memory_delta: float  # MB
    status: str


@dataclass
class GraphMetrics:
    """Metrics for entire graph execution"""

    total_duration: float
    peak_memory: float  # MB
    node_metrics: List[NodeMetrics] = field(default_factory=list)
    parallelism_factor: float = 0.0  # Actual speedup from parallelism

    def get_bottlenecks(self, top_n: int = 5) -> List[NodeMetrics]:
        """Get slowest nodes"""
        return sorted(self.node_metrics, key=lambda m: m.duration, reverse=True)[:top_n]

    def get_critical_path(self, graph: Graph) -> List[str]:
        """
        Get critical path (longest path through graph).

        This is the minimum time for sequential execution.
        """
        # Build duration map
        durations = {m.node_id: m.duration for m in self.node_metrics}

        # Compute longest path to each node
        longest_path = {}

        for node_id in graph.get_execution_order():
            node = graph.get_node(node_id)

            # Maximum of all dependency paths + this node's duration
            if not node.depends_on:
                longest_path[node_id] = durations.get(node_id, 0)
            else:
                max_dep_path = max(longest_path[dep] for dep in node.depends_on)
                longest_path[node_id] = max_dep_path + durations.get(node_id, 0)

        # Find node with longest path (end of critical path)
        end_node = max(longest_path, key=longest_path.get)

        # Backtrack to build critical path
        critical_path = [end_node]
        current = end_node

        while graph.get_node(current).depends_on:
            # Find dependency with longest path
            deps = graph.get_node(current).depends_on
            prev = max(deps, key=lambda d: longest_path[d])
            critical_path.append(prev)
            current = prev

        return critical_path[::-1]

    def print_summary(self, graph: Optional[Graph] = None):
        """Print execution summary"""
        print("\nGraph Execution Metrics")
        print("=" * 60)
        print(f"Total Duration: {self.total_duration:.2f}s")
        print(f"Peak Memory: {self.peak_memory:.2f} MB")
        print(f"Nodes Executed: {len(self.node_metrics)}")

        if self.parallelism_factor > 1:
            print(f"Parallelism Speedup: {self.parallelism_factor:.2f}x")

        # Bottlenecks
        print("\nTop 5 Slowest Nodes:")
        for i, metric in enumerate(self.get_bottlenecks(5), 1):
            print(f"  {i}. {metric.node_id}: {metric.duration:.2f}s")

        # Critical path
        if graph:
            critical = self.get_critical_path(graph)
            print(f"\nCritical Path ({len(critical)} nodes):")
            print("  " + " → ".join(critical))

        print("=" * 60)


class MonitoredExecutionEngine:
    """
    Execution engine with performance monitoring.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.metrics = GraphMetrics(total_duration=0, peak_memory=0)
        self._start_memory = 0
        self._peak_memory = 0

    def execute(self, **kwargs) -> tuple[Dict[str, Any], GraphMetrics]:
        """
        Execute with monitoring.

        Returns:
            (results, metrics)
        """
        # Record initial state
        process = psutil.Process()
        self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        # Execute (override node execute to track metrics)
        self._wrap_node_execution()

        results = self.graph.execute(**kwargs)

        # Record final state
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024

        self.metrics.total_duration = end_time - start_time
        self.metrics.peak_memory = max(self._peak_memory, end_memory) - self._start_memory

        return results, self.metrics

    def _wrap_node_execution(self):
        """Wrap node execution to track metrics"""
        process = psutil.Process()

        for node in self.graph.nodes.values():
            original_execute = node.execute

            def tracked_execute(inputs, node=node, orig=original_execute):
                start_mem = process.memory_info().rss / 1024 / 1024
                start_time = time.time()

                result = orig(inputs)

                end_time = time.time()
                end_mem = process.memory_info().rss / 1024 / 1024

                # Track peak
                self._peak_memory = max(self._peak_memory, end_mem)

                # Record metrics
                metric = NodeMetrics(
                    node_id=node.node_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    memory_delta=end_mem - start_mem,
                    status=node.status.value,
                )
                self.metrics.node_metrics.append(metric)

                return result

            node.execute = tracked_execute


def profile_graph(graph: Graph, **kwargs) -> GraphMetrics:
    """
    Profile graph execution.

    Args:
        graph: Graph to profile
        **kwargs: Execution arguments

    Returns:
        GraphMetrics with performance data
    """
    engine = MonitoredExecutionEngine(graph)
    _, metrics = engine.execute(**kwargs)
    return metrics
