
"""Base class for pipelines"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from core import Config
from graph import Graph, Node


class BasePipeline(ABC):
    """
    Abstract base class for all pipelines.
    
    Pipelines orchestrate multiple models and processing steps.
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the pipeline.
        
        Must be implemented by subclasses.
        """
        pass
    
    def setup(self):
        """Setup pipeline components"""
        pass
    
    def teardown(self):
        """Cleanup pipeline resources"""
        pass


class GraphPipeline(BasePipeline):
    """
    Pipeline that uses dependency graph for execution.
    
    Allows defining complex pipelines with dependencies.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.graph = Graph()
    
    def add_step(
        self,
        step_id: str,
        fn: Callable,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add a step to the pipeline.
        
        Args:
            step_id: Step identifier
            fn: Function to execute
            depends_on: Steps this depends on
            **kwargs: Additional node arguments
        """
        node = Node(step_id, fn, depends_on=depends_on, **kwargs)
        self.graph.add_node(node)
    
    def run(self, **initial_inputs) -> Dict[str, Any]:
        """
        Execute pipeline.
        
        Args:
            **initial_inputs: Initial values for root nodes
            
        Returns:
            Dictionary of step results
        """
        self.setup()
        
        try:
            results = self.graph.execute(initial_inputs=initial_inputs)
            return results
        finally:
            self.teardown()
    
    def get_execution_plan(self) -> List[str]:
        """Get execution order"""
        return self.graph.get_execution_order()