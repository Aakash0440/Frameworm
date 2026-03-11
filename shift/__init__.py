"""
FRAMEWORM SHIFT — Distribution Shift Detection

Drop-in drift monitoring for any ML model in production.

Quick start:
    from frameworm.shift import ShiftMonitor

    # At training time — save reference distribution
    monitor = ShiftMonitor("my_model")
    monitor.profile_reference(X_train, feature_names=[...])

    # At inference time — check for drift
    result = monitor.check(X_batch)
    # Alerts fire automatically if drift is detected
"""

from shift.core import (AlertManager, DatasetProfile, DriftEngine, DriftResult,
                        DriftSeverity, FeatureProfiler, ReferenceStore,
                        load_reference, save_reference)
from shift.sdk.monitor import ShiftMonitor

__version__ = "0.1.0"
__all__ = [
    "ShiftMonitor",
    "FeatureProfiler",
    "DatasetProfile",
    "ReferenceStore",
    "DriftEngine",
    "DriftResult",
    "DriftSeverity",
    "AlertManager",
    "save_reference",
    "load_reference",
]
