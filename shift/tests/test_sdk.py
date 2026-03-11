"""End-to-end SDK tests: profile → save → load → check → alert."""

import tempfile
import numpy as np
import sys

sys.path.insert(0, ".")

from shift.sdk.monitor import ShiftMonitor
from shift.core.drift_engine import DriftSeverity

rng = np.random.default_rng(7)


def test_monitor_profile_and_check_no_drift():
    X_train = rng.normal(0, 1, (1000, 4))
    X_live = rng.normal(0, 1, (100, 4))
    with tempfile.TemporaryDirectory() as d:
        monitor = ShiftMonitor("test", store_dir=d, auto_alert=False)
        monitor.profile_reference(X_train, feature_names=["a", "b", "c", "d"])
        result = monitor.check(X_live, feature_names=["a", "b", "c", "d"])
        assert result.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)


def test_monitor_profile_and_check_with_drift():
    X_train = rng.normal(0, 1, (1000, 3))
    X_live = rng.normal(15, 3, (100, 3))  # massive shift
    with tempfile.TemporaryDirectory() as d:
        monitor = ShiftMonitor("test", store_dir=d, auto_alert=False)
        monitor.profile_reference(X_train, feature_names=["x", "y", "z"])
        result = monitor.check(X_live, feature_names=["x", "y", "z"])
        assert result.overall_drifted


def test_monitor_from_reference_classmethod():
    X_train = rng.normal(0, 1, (500, 2))
    with tempfile.TemporaryDirectory() as d:
        # Save reference
        m1 = ShiftMonitor("mymodel", store_dir=d, auto_alert=False)
        m1.profile_reference(X_train, feature_names=["a", "b"])
        # Load via classmethod
        import os

        path = os.path.join(d, "mymodel")
        m2 = ShiftMonitor.from_reference(path, auto_alert=False)
        assert m2._reference_profile is not None
        assert m2._reference_profile.n_samples == 500


def test_check_count_tracked():
    X = rng.normal(0, 1, (200, 2))
    with tempfile.TemporaryDirectory() as d:
        monitor = ShiftMonitor("counter", store_dir=d, auto_alert=False)
        monitor.profile_reference(X, ["a", "b"])
        for _ in range(3):
            monitor.check(rng.normal(0, 1, (50, 2)), ["a", "b"])
        assert monitor._check_count == 3


def test_datapoint_window_accumulation():
    X_train = rng.normal(0, 1, (500, 2))
    with tempfile.TemporaryDirectory() as d:
        monitor = ShiftMonitor("windowed", store_dir=d, auto_alert=False)
        monitor.profile_reference(X_train, ["a", "b"])
        results = []
        for _ in range(120):
            r = monitor.check_datapoint(
                rng.normal(0, 1, 2).tolist(),
                feature_names=["a", "b"],
                window_size=100,
            )
            if r is not None:
                results.append(r)
        # Should have fired exactly once (at 100 points)
        assert len(results) == 1


def test_status_dict_structure():
    X = rng.normal(0, 1, (200, 2))
    with tempfile.TemporaryDirectory() as d:
        monitor = ShiftMonitor("s", store_dir=d, auto_alert=False)
        monitor.profile_reference(X, ["a", "b"])
        s = monitor.status()
        assert "checks_run" in s
        assert "drift_rate" in s
        assert s["reference_loaded"] is True
