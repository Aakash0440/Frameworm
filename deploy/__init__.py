"""
FRAMEWORM DEPLOY — One command from trained model to production API.

Quick start:
    from frameworm.deploy import ModelExporter, ModelRegistry, ModelStage

    # 1. Export
    exporter = ModelExporter()
    manifest = exporter.export("experiments/checkpoints/best.pt", "my_model")

    # 2. Register
    registry = ModelRegistry()
    record   = registry.register("my_model", manifest)

    # 3. Promote to production
    registry.transition(record.id, ModelStage.STAGING)
    registry.transition(record.id, ModelStage.PRODUCTION)

    # 4. Serve (Steps 6-7)
    # frameworm deploy start --model my_model
"""

from deploy.core import (
    ModelExporter,
    ExportManifest,
    ModelRegistry,
    ModelStage,
    DeploymentRecord,
    LatencyTracker,
    LatencySnapshot,
    get_tracker,
)

__version__ = "0.1.0"

__all__ = [
    "ModelExporter",
    "ExportManifest",
    "ModelRegistry",
    "ModelStage",
    "DeploymentRecord",
    "LatencyTracker",
    "LatencySnapshot",
    "get_tracker",
]