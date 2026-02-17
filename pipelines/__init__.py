# pipelines/__init__.py
from graph import Graph as GraphPipeline

from .base import BasePipeline

__all__ = ["BasePipeline", "GraphPipeline"]
