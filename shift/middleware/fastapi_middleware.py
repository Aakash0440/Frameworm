"""
FRAMEWORM SHIFT FastAPI Middleware.

Drop-in drift monitoring for any FastAPI model endpoint.
Intercepts every request, extracts input, runs drift check async.
Zero impact on response latency — check runs in background thread.

Usage:
    from fastapi import FastAPI
    from frameworm.shift.middleware import ShiftMiddleware

    app = FastAPI()
    app.add_middleware(
        ShiftMiddleware,
        reference="fraud_classifier",           # name or .shift file path
        feature_names=["age", "income", ...],   # optional
        alert_channels=["slack", "log"],
    )

    @app.post("/predict")
    def predict(data: dict):
        ...
"""

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("frameworm.shift.middleware")


class ShiftMiddleware:
    """
    Starlette/FastAPI middleware that checks incoming request payloads for drift.

    How it works:
    1.  Request comes in → middleware extracts the body
    2.  Passes request through to your endpoint unchanged
    3.  In a background thread, runs drift check on extracted features
    4.  Fires alert if drift detected — response is never blocked

    Configuration:
        reference:        model name or path to .shift file
        feature_names:    list of feature names to extract from request body
        input_key:        key in JSON body that holds feature array (default: "features")
        alert_channels:   ["slack","webhook","log","stdout"]
        min_severity:     minimum severity to alert on
        window_size:      accumulate N requests before checking (default: 50)
        async_check:      run check in background thread (default: True)
    """

    def __init__(
        self,
        app,
        reference: str,
        feature_names: Optional[List[str]] = None,
        input_key: str = "features",
        alert_channels: Optional[List[str]] = None,
        min_severity: str = "MEDIUM",
        window_size: int = 50,
        async_check: bool = True,
    ):
        self.app = app

        # Lazy import to avoid hard dependency when middleware isn't used
        from shift.core.drift_engine import DriftSeverity
        from shift.sdk.monitor import ShiftMonitor

        severity_map = {
            "NONE": DriftSeverity.NONE,
            "LOW": DriftSeverity.LOW,
            "MEDIUM": DriftSeverity.MEDIUM,
            "HIGH": DriftSeverity.HIGH,
        }

        self._monitor = ShiftMonitor.from_reference(
            reference,
            alert_channels=alert_channels or ["log", "stdout"],
            min_severity=severity_map.get(min_severity.upper(), DriftSeverity.MEDIUM),
            auto_alert=True,
        )

        self._feature_names = feature_names
        self._input_key = input_key
        self._window_size = window_size
        self._async_check = async_check
        self._buffer: List = []
        self._lock = threading.Lock()

        logger.info(
            f"[SHIFT] Middleware active — monitoring '{reference}' "
            f"(window={window_size}, severity>={min_severity})"
        )

    async def __call__(self, scope, receive, send):
        """ASGI interface — called for every request."""
        if scope["type"] == "http":
            # Read body without consuming it (re-inject for actual endpoint)
            body = await self._read_body(receive)
            receive = self._make_receive(body)

            # Extract features and buffer them (non-blocking)
            features = self._extract_features(body)
            if features is not None:
                self._buffer_and_check(features)

        await self.app(scope, receive, send)

    # ──────────────────────────────────────────────── body handling

    async def _read_body(self, receive) -> bytes:
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)
        return body

    def _make_receive(self, body: bytes):
        """Re-create receive callable so the endpoint can read the body normally."""

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    # ──────────────────────────────────────────────── feature extraction

    def _extract_features(self, body: bytes) -> Optional[np.ndarray]:
        """
        Extract feature array from request body.
        Tries JSON → looks for self._input_key → falls back to full body.
        """
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        # Try configured key first
        if self._input_key in payload:
            raw = payload[self._input_key]
        elif "data" in payload:
            raw = payload["data"]
        elif "inputs" in payload:
            raw = payload["inputs"]
        else:
            # Try to extract all numeric values
            raw = [v for v in payload.values() if isinstance(v, (int, float))]

        if not raw:
            return None

        try:
            arr = np.array(raw, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except (ValueError, TypeError):
            return None

    # ──────────────────────────────────────────────── buffering + check

    def _buffer_and_check(self, features: np.ndarray):
        """Buffer features and trigger drift check when window is full."""
        with self._lock:
            self._buffer.append(features)
            if len(self._buffer) >= self._window_size:
                batch = np.vstack(self._buffer)
                self._buffer = []
            else:
                return  # window not full yet

        # Run check outside the lock
        if self._async_check:
            t = threading.Thread(
                target=self._run_check,
                args=(batch,),
                daemon=True,
            )
            t.start()
        else:
            self._run_check(batch)

    def _run_check(self, batch: np.ndarray):
        try:
            result = self._monitor.check(batch, self._feature_names)
            if result.overall_drifted:
                logger.warning(
                    f"[SHIFT] Drift detected in live traffic — "
                    f"{result.overall_severity.value} "
                    f"({len(result.drifted_features)} features)"
                )
        except Exception as e:
            logger.error(f"[SHIFT] Middleware drift check failed: {e}")
