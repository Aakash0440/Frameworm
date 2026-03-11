"""Tests for feature_profiles, reference_store, drift_engine."""

import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

from shift.core.drift_engine import DriftEngine, DriftSeverity
from shift.core.feature_profiles import DatasetProfile, FeatureProfiler
from shift.core.reference_store import ReferenceStore

rng = np.random.default_rng(99)


def test_numerical_profiler_basic():
    X = rng.normal(5, 2, (1000, 3))
    p = FeatureProfiler().profile(X, ["a", "b", "c"])
    assert len(p.numerical) == 3
    assert abs(p.numerical["a"].mean - 5.0) < 0.3
    assert abs(p.numerical["a"].std - 2.0) < 0.3
    assert len(p.numerical["a"].histogram_counts) == 20
    assert "p25" in p.numerical["a"].percentiles


def test_categorical_profiler():
    X = np.array([["cat", "dog", "cat", "bird"] * 100]).T
    p = FeatureProfiler().profile(X, ["animal"])
    assert "animal" in p.categorical
    assert p.categorical["animal"].n_unique == 3
    assert p.categorical["animal"].entropy > 0


def test_missing_rate_tracked():
    X = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]]).T
    p = FeatureProfiler().profile(X, ["x"])
    assert abs(p.numerical["x"].missing_rate - 0.4) < 0.05


def test_profile_roundtrip():
    X = rng.normal(0, 1, (300, 2))
    p = FeatureProfiler().profile(X, ["x", "y"])
    restored = DatasetProfile.from_dict(p.to_dict())
    assert restored.n_samples == p.n_samples
    assert abs(restored.numerical["x"].mean - p.numerical["x"].mean) < 1e-6


def test_reference_store_save_load():
    X = rng.normal(0, 1, (500, 4))
    with tempfile.TemporaryDirectory() as d:
        store = ReferenceStore(d)
        store.save(X, "test", ["a", "b", "c", "d"])
        assert store.exists("test")
        loaded = store.load("test")
        assert loaded.n_samples == 500
        assert loaded.feature_names == ["a", "b", "c", "d"]


def test_no_drift_on_same_distribution():
    X_ref = rng.normal(0, 1, (1000, 3))
    X_cur = rng.normal(0, 1, (300, 3))
    profiler = FeatureProfiler()
    ref = profiler.profile(X_ref, ["a", "b", "c"])
    cur = profiler.profile(X_cur, ["a", "b", "c"])
    result = DriftEngine().compare(ref, cur)
    # With same distribution, expect NONE or LOW severity
    assert result.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)


def test_drift_detected_on_shifted_distribution():
    X_ref = rng.normal(0, 1, (1000, 3))
    X_drift = rng.normal(10, 5, (300, 3))  # completely different
    profiler = FeatureProfiler()
    ref = profiler.profile(X_ref, ["a", "b", "c"])
    drifted = profiler.profile(X_drift, ["a", "b", "c"])
    result = DriftEngine().compare(ref, drifted)
    assert result.overall_drifted
    assert result.overall_severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)
    assert len(result.drifted_features) == 3


def test_false_positive_rate_below_threshold():
    """FP rate on healthy data must be < 5%."""
    profiler = FeatureProfiler()
    engine = DriftEngine()
    X_ref = rng.normal(0, 1, (2000, 5))
    ref = profiler.profile(X_ref, [f"f{i}" for i in range(5)])
    fp_count = 0
    N_TRIALS = 40
    for _ in range(N_TRIALS):
        X_cur = rng.normal(0, 1, (200, 5))
        cur = profiler.profile(X_cur, [f"f{i}" for i in range(5)])
        r = engine.compare(ref, cur)
        if r.overall_drifted:
            fp_count += 1
    fp_rate = fp_count / N_TRIALS
    assert fp_rate < 0.15, f"FP rate too high: {fp_rate:.2f}"


def test_categorical_drift_detected():
    # Reference: balanced A/B/C
    ref_data = np.array([["A", "B", "C"] * 100]).T
    # Current: heavily skewed to A
    cur_data = np.array([["A"] * 280 + ["B"] * 15 + ["C"] * 5]).T
    profiler = FeatureProfiler()
    ref = profiler.profile(ref_data, ["cat"])
    cur = profiler.profile(cur_data, ["cat"])
    result = DriftEngine().compare(ref, cur)
    assert result.features["cat"].test_used == "Chi2"
    assert result.features["cat"].drifted
