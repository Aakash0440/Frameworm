from .base import BasePipeline

# pipelines/__init__.py
from graph import Graph as GraphPipeline

__all__ = ["BasePipeline", "GraphPipeline"]
