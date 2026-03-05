"""
CLI integration tests.
Tests the core functions directly (no subprocess needed).
"""

import sys
import tempfile
import numpy as np
import os
sys.path.insert(0, ".")

rng = np.random.default_rng(21)


def test_load_data_npy():
    from shift.cli.commands import _load_data
    X = rng.normal(0, 1, (100, 3))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.npy")
        np.save(path, X)
        loaded = _load_data(path)
        assert loaded.shape == (100, 3)

def test_load_data_csv():
    from shift.cli.commands import _load_data
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.csv")
        with open(path, "w") as f:
            f.write("a,b,c\n")
            for _ in range(50):
                row = rng.normal(0, 1, 3)
                f.write(",".join(str(x) for x in row) + "\n")
        loaded = _load_data(path)
        assert loaded is not None

def test_report_generator_html():
    from shift.core.feature_profiles import FeatureProfiler
    from shift.core.drift_engine import DriftEngine
    from shift.report.report_generator import ReportGenerator
    profiler = FeatureProfiler()
    engine   = DriftEngine()
    X_ref  = rng.normal(0, 1, (500, 3))
    X_cur  = rng.normal(5, 2, (200, 3))
    ref    = profiler.profile(X_ref, ["x","y","z"])
    cur    = profiler.profile(X_cur, ["x","y","z"])
    result = engine.compare(ref, cur)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.html")
        gen = ReportGenerator()
        path = gen.generate_html(result, ref, cur, out, model_name="test_model")
        assert os.path.exists(path)
        html = open(path).read()
        assert "FRAMEWORM SHIFT" in html
        assert "test_model" in html
        assert "NONE" in html or "LOW" in html or "MEDIUM" in html or "HIGH" in html

def test_report_generator_json():
    from shift.core.feature_profiles import FeatureProfiler
    from shift.core.drift_engine import DriftEngine
    from shift.report.report_generator import ReportGenerator
    import json
    profiler = FeatureProfiler()
    X = rng.normal(0, 1, (200, 2))
    ref = profiler.profile(X, ["a","b"])
    cur = profiler.profile(X, ["a","b"])
    result = DriftEngine().compare(ref, cur)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        ReportGenerator().generate_json(result, out)
        assert os.path.exists(out)
        data = json.load(open(out))
        assert "features" in data
        assert "overall_drifted" in data
