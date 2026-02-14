
"""Base class for pipelines"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from core import Config


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
