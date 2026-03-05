"""
FastAPI middleware integration tests.
Requires: pip install fastapi httpx
Skips gracefully if not installed.
"""

import sys
import json
import numpy as np
import tempfile
import time
sys.path.insert(0, ".")

rng = np.random.default_rng(13)


def test_middleware_passes_request_through():
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from shift.middleware.fastapi_middleware import ShiftMiddleware
        from shift.sdk.monitor import ShiftMonitor
    except ImportError:
        print("  [SKIP] fastapi/httpx not installed")
        return

    X_train = rng.normal(0, 1, (500, 3))
    with tempfile.TemporaryDirectory() as d:
        m = ShiftMonitor("mw_test", store_dir=d, auto_alert=False)
        m.profile_reference(X_train, ["a","b","c"])

        import os
        ref_path = os.path.join(d, "mw_test")

        app = FastAPI()

        @app.post("/predict")
        def predict(data: dict):
            return {"result": "ok"}

        app.add_middleware(
            ShiftMiddleware,
            reference=ref_path,
            feature_names=["a","b","c"],
            window_size=5,
            async_check=False,
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/predict",
            json={"features": [0.1, 0.2, 0.3]},
        )
        # Middleware must never block the response
        assert response.status_code == 200
        assert response.json() == {"result": "ok"}


def test_middleware_feature_extraction():
    from shift.middleware.fastapi_middleware import ShiftMiddleware

    class FakeApp:
        async def __call__(self, scope, receive, send): pass

    mw = ShiftMiddleware.__new__(ShiftMiddleware)
    mw._input_key = "features"

    body = json.dumps({"features": [1.0, 2.0, 3.0]}).encode()
    arr = mw._extract_features(body)
    assert arr is not None
    assert arr.shape == (1, 3)

    body_bad = b"not json at all"
    assert mw._extract_features(body_bad) is None

