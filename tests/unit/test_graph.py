"""Tests for dependency graph system"""

import pytest
from graph.node import Node, NodeStatus
from graph.graph import Graph, CycleDetectedError, GraphError
from graph.graph import CachedGraph

class TestNode:
    """Test Node class"""
    
    def test_node_creation(self):
        """Should create node"""
        node = Node("test", fn=lambda: 42)
        assert node.node_id == "test"
        assert node.status == NodeStatus.PENDING
        assert node.depends_on == []
    
    def test_node_with_dependencies(self):
        """Should create node with dependencies"""
        node = Node("test", fn=lambda x: x * 2, depends_on=["dep1"])
        assert node.depends_on == ["dep1"]
    
    def test_node_execute(self):
        """Should execute node function"""
        node = Node("test", fn=lambda: 42)
        result = node.execute({})
        
        assert result == 42
        assert node.result == 42
        assert node.status == NodeStatus.COMPLETED
    
    def test_node_execute_with_inputs(self):
        """Should execute with dependency inputs"""
        node = Node("test", fn=lambda x, y: x + y, depends_on=["a", "b"])
        result = node.execute({"a": 10, "b": 20})
        
        assert result == 30
    
    def test_node_execute_failure(self):
        """Should handle execution failure"""
        def failing_fn():
            raise ValueError("Test error")
        
        node = Node("test", fn=failing_fn)
        
        with pytest.raises(ValueError):
            node.execute({})
        
        assert node.status == NodeStatus.FAILED
        assert node.error is not None
    
    def test_node_duration(self):
        """Should track execution duration"""
        import time
        
        def slow_fn():
            time.sleep(0.01)
            return 42
        
        node = Node("test", fn=slow_fn)
        node.execute({})
        
        duration = node.get_duration()
        assert duration is not None
        assert duration >= 0.01
    
    def test_node_reset(self):
        """Should reset node state"""
        node = Node("test", fn=lambda: 42)
        node.execute({})
        
        assert node.status == NodeStatus.COMPLETED
        
        node.reset()
        assert node.status == NodeStatus.PENDING
        assert node.result is None


class TestGraph:
    """Test Graph class"""
    
    def test_graph_creation(self):
        """Should create empty graph"""
        graph = Graph()
        assert len(graph) == 0
    
    def test_add_node(self):
        """Should add node to graph"""
        graph = Graph()
        node = Node("test", fn=lambda: 42)
        graph.add_node(node)
        
        assert len(graph) == 1
        assert "test" in graph
    
    def test_add_duplicate_node(self):
        """Should raise error for duplicate node ID"""
        graph = Graph()
        graph.add_node(Node("test", fn=lambda: 1))
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(Node("test", fn=lambda: 2))
    
    def test_get_node(self):
        """Should get node by ID"""
        graph = Graph()
        node = Node("test", fn=lambda: 42)
        graph.add_node(node)
        
        retrieved = graph.get_node("test")
        assert retrieved == node
    
    def test_get_nonexistent_node(self):
        """Should raise error for missing node"""
        graph = Graph()
        
        with pytest.raises(KeyError):
            graph.get_node("nonexistent")
    
    def test_simple_topological_sort(self):
        """Should sort simple linear graph"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 1))
        graph.add_node(Node("b", fn=lambda x: x + 1, depends_on=["a"]))
        graph.add_node(Node("c", fn=lambda x: x * 2, depends_on=["b"]))
        
        order = graph.topological_sort()
        
        assert order == ["a", "b", "c"]
    
    def test_complex_topological_sort(self):
        """Should sort complex DAG"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 1))
        graph.add_node(Node("b", fn=lambda: 2))
        graph.add_node(Node("c", fn=lambda x, y: x + y, depends_on=["a", "b"]))
        graph.add_node(Node("d", fn=lambda x: x * 2, depends_on=["c"]))
        
        order = graph.topological_sort()
        
        # a and b can be in any order, but must come before c
        # c must come before d
        a_idx = order.index("a")
        b_idx = order.index("b")
        c_idx = order.index("c")
        d_idx = order.index("d")
        
        assert a_idx < c_idx
        assert b_idx < c_idx
        assert c_idx < d_idx
    
    def test_cycle_detection(self):
        """Should detect cycle in graph"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda x: x, depends_on=["b"]))
        graph.add_node(Node("b", fn=lambda x: x, depends_on=["a"]))
        
        assert graph.has_cycle()
    
    def test_cycle_error(self):
        """Should raise CycleDetectedError"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda x: x, depends_on=["b"]))
        graph.add_node(Node("b", fn=lambda x: x, depends_on=["a"]))
        
        with pytest.raises(CycleDetectedError):
            graph.topological_sort()
    
    def test_missing_dependency(self):
        """Should error on missing dependency"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda x: x, depends_on=["nonexistent"]))
        
        with pytest.raises(GraphError, match="doesn't exist"):
            graph.topological_sort()
    
    def test_get_dependencies(self):
        """Should get node dependencies"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 1))
        graph.add_node(Node("b", fn=lambda x: x, depends_on=["a"]))
        
        deps = graph.get_dependencies("b")
        assert deps == ["a"]
    
    def test_get_dependents(self):
        """Should get nodes that depend on given node"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 1))
        graph.add_node(Node("b", fn=lambda x: x, depends_on=["a"]))
        graph.add_node(Node("c", fn=lambda x: x, depends_on=["a"]))
        
        dependents = graph.get_dependents("a")
        assert set(dependents) == {"b", "c"}


class TestExecutionEngine:
    """Test graph execution"""
    
    def test_simple_execution(self):
        """Should execute simple graph"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 10))
        graph.add_node(Node("b", fn=lambda x: x * 2, depends_on=["a"]))
        
        results = graph.execute()
        
        assert results["a"] == 10
        assert results["b"] == 20
    
    def test_complex_execution(self):
        """Should execute complex graph"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 5))
        graph.add_node(Node("b", fn=lambda: 3))
        graph.add_node(Node("c", fn=lambda x, y: x + y, depends_on=["a", "b"]))
        graph.add_node(Node("d", fn=lambda x: x ** 2, depends_on=["c"]))
        
        results = graph.execute()
        
        assert results["a"] == 5
        assert results["b"] == 3
        assert results["c"] == 8  # 5 + 3
        assert results["d"] == 64  # 8^2
    
    def test_execution_with_initial_inputs(self):
        """Should use initial inputs for root nodes"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: None))  # Will be replaced by initial input
        graph.add_node(Node("b", fn=lambda x: x * 2, depends_on=["a"]))
        
        results = graph.execute(initial_inputs={"a": 15})
        
        # Note: Node "a" still executes, but "b" uses the result
        assert results["b"] == 30  # 15 * 2
    
    def test_execution_failure(self):
        """Should handle node failure"""
        def failing_fn():
            raise ValueError("Test error")
        
        graph = Graph()
        graph.add_node(Node("a", fn=failing_fn))
        graph.add_node(Node("b", fn=lambda x: x, depends_on=["a"]))
        
        with pytest.raises(GraphError):
            graph.execute()
    
    def test_continue_on_error(self):
        """Should continue execution after failure"""
        def failing_fn():
            raise ValueError("Test error")
        
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 10))
        graph.add_node(Node("b", fn=failing_fn))
        graph.add_node(Node("c", fn=lambda x: x * 2, depends_on=["a"]))
        
        results = graph.execute(continue_on_error=True)
        
        # Node "a" and "c" should succeed
        assert results["a"] == 10
        assert results["c"] == 20
        # Node "b" failed, so not in results
        assert "b" not in results
    
    def test_skip_nodes(self):
        """Should skip specified nodes"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 10))
        graph.add_node(Node("b", fn=lambda x: x * 2, depends_on=["a"]))
        graph.add_node(Node("c", fn=lambda x: x + 1, depends_on=["b"]))
        
        results = graph.execute(skip_nodes=["b"])
        
        assert results["a"] == 10
        assert "b" not in results
        # "c" skipped because dependency "b" was skipped
        assert "c" not in results
    
    def test_execution_summary(self):
        """Should provide execution summary"""
        graph = Graph()
        graph.add_node(Node("a", fn=lambda: 10))
        graph.add_node(Node("b", fn=lambda x: x * 2, depends_on=["a"]))
        
        graph.execute()
        summary = graph.get_last_execution_summary()
        
        assert summary['total_nodes'] == 2
        assert summary['completed'] == 2
        assert summary['failed'] == 0
        assert summary['total_duration'] >= 0

class TestConditionalNode:
    """Test conditional execution"""
    
    def test_conditional_node_execute(self):
        """Should execute when condition is True"""
        from graph.node import ConditionalNode
        
        node = ConditionalNode(
            "test",
            fn=lambda x: x * 2,
            condition=lambda x: x > 5,
            depends_on=["a"]
        )
        
        result = node.execute({"a": 10})
        assert result == 20
        assert node.status == NodeStatus.COMPLETED
    
    def test_conditional_node_skip(self):
        """Should skip when condition is False"""
        from graph.node import ConditionalNode
        
        node = ConditionalNode(
            "test",
            fn=lambda x: x * 2,
            condition=lambda x: x > 5,
            depends_on=["a"]
        )
        
        result = node.execute({"a": 3})
        assert result is None
        assert node.status == NodeStatus.SKIPPED


class TestCachedGraph:
    """Test result caching"""
    
    def test_cache_simple(self, tmp_path):
        from graph.graph import CachedGraph

        call_count = [0]

        def expensive_fn():
            call_count[0] += 1
            return 42

        graph = CachedGraph(cache_dir=str(tmp_path))
        graph.add_node(Node("a", fn=expensive_fn))  # track call_count

        results1 = graph.execute()
        assert results1["a"] == 42
        assert call_count[0] == 1

        results2 = graph.execute()
        assert results2["a"] == 42
        assert call_count[0] == 1  # cached, not called again

    
    def test_cache_invalidation(self, tmp_path):
        from graph.graph import CachedGraph

        call_count = [0]

        def fn(x):
            call_count[0] += 1
            return x * 2

        graph = CachedGraph(cache_dir=str(tmp_path))
        graph.add_node(Node("a", fn=lambda: 10))  # root node with fixed value
        graph.add_node(Node("b", fn=fn, depends_on=["a"]))

        # First execution
        graph.execute()
        assert call_count[0] == 1

        # Cached execution → call_count should not increase
        graph.execute()
        assert call_count[0] == 1


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])