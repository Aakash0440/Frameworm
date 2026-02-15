"""Tests for dependency graph system"""

import pytest
from graph.node import Node, NodeStatus
from graph.graph import Graph, CycleDetectedError, GraphError


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


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])