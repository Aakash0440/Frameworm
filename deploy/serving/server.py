"""
Runtime FastAPI server loaded from a generated server spec.
This is NOT hand-written — it's the template that server_builder.py generates
and saves to deploy/generated/<name>/server.py.

This file provides the base classes and utilities the generated server imports.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("frameworm.deploy.server")


class FramewormModelServer:
    """
    Base server class that generated servers inherit from.
    Handles model loading, SHIFT monitor attachment, health tracking.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        model_version: str,
        model_type: str,
        shift_reference: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        self.device = device
        self.model = None
        self._shift_monitor = None

        # Health + latency — imported from deploy/core/
        from deploy.core.latency_tracker import LatencyTracker
        from deploy.serving.health import HealthChecker

        self._latency = LatencyTracker(model_name=model_name)
        self._health = HealthChecker(model_name, model_version)

        # SHIFT drift monitor — auto-attached if reference exists
        if shift_reference:
            self._attach_shift(shift_reference)

    # ──────────────────────────────────────────────── lifecycle

    def load(self):
        """Load model from checkpoint — TorchScript or state dict."""
        logger.info(f"[DEPLOY] Loading {self.model_type} model from {self.model_path}")
        try:
            # Try TorchScript first
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            self._health.mark_ready()
            logger.info(
                f"[DEPLOY] TorchScript model loaded — {self.model_name} v{self.model_version}"
            )
        except Exception:
            try:
                # Fall back to state dict checkpoint
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
                # Build model from registry
                from core.config import Config
                from core.registry import get_model

                cfg = ckpt.get("config", None)
                if cfg is None:
                    # Minimal stub model that just holds the weights
                    self.model = ckpt.get("model_state_dict", ckpt)
                else:
                    self.model = get_model(self.model_type)(cfg)
                    state = ckpt.get("model_state_dict", ckpt)
                    self.model.load_state_dict(state, strict=False)
                    self.model.eval()
                self._health.mark_ready()
                logger.info(f"[DEPLOY] State dict model loaded — {self.model_name}")
            except Exception as e:
                self._health.mark_not_ready(str(e))
                raise RuntimeError(f"[DEPLOY] Failed to load model: {e}")

    def load_onnx(self):
        """Load model from ONNX file using onnxruntime."""
        try:
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._health.mark_ready()
            logger.info(f"[DEPLOY] ONNX model loaded — {self.model_name}")
        except ImportError:
            raise RuntimeError(
                "[DEPLOY] onnxruntime not installed. " "pip install onnxruntime or onnxruntime-gpu"
            )

    # ──────────────────────────────────────────────── inference helpers

    def _preprocess(self, data: Any) -> torch.Tensor:
        """Convert input to tensor. Override in generated server."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, (list, np.ndarray)):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        raise ValueError(f"Unsupported input type: {type(data)}")

    def _postprocess(self, output: torch.Tensor) -> Any:
        """Convert tensor output to JSON-serialisable format."""
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().tolist()
        return output

    def _check_drift(self, data: Any):
        """Run SHIFT drift check if monitor is attached."""
        if self._shift_monitor is None:
            return
        try:
            arr = np.array(data)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self._shift_monitor.check_datapoint(
                arr[0].tolist(),
                window_size=100,
            )
        except Exception as e:
            logger.warning(f"[DEPLOY] SHIFT check failed: {e}")

    # ──────────────────────────────────────────────── health routes

    def health_route(self) -> dict:
        return self._health.health()

    def ready_route(self):
        readiness = self._health.readiness()
        if not readiness["ready"]:
            raise HTTPException(status_code=503, detail="Model not ready")
        return readiness

    def metrics_route(self) -> dict:
        return {
            "latency": self._latency.snapshot(),
            "health": self._health.health(),
        }

    # ──────────────────────────────────────────────── SHIFT

    def _attach_shift(self, reference: str):
        try:
            from shift.sdk.monitor import ShiftMonitor

            self._shift_monitor = ShiftMonitor.from_reference(
                reference,
                auto_alert=True,
                alert_channels=["slack", "log"],
            )
            logger.info(f"[DEPLOY] SHIFT drift monitor attached — reference: {reference}")
        except Exception as e:
            logger.warning(f"[DEPLOY] Could not attach SHIFT monitor: {e}")


def build_app(server: FramewormModelServer) -> FastAPI:
    """
    Wrap a FramewormModelServer in a FastAPI app with all standard routes.
    Generated servers call this to get a fully wired app.
    """
    from deploy.serving.middleware import LatencyMiddleware

    app = FastAPI(
        title=f"FRAMEWORM DEPLOY — {server.model_name}",
        version=server.model_version,
        description=f"Serving {server.model_type} model via FRAMEWORM DEPLOY",
    )

    # Standard routes
    app.get("/health")(server.health_route)
    app.get("/ready")(server.ready_route)
    app.get("/metrics")(server.metrics_route)

    # Attach latency + error-rate middleware
    app.add_middleware(
        LatencyMiddleware,
        latency_tracker=server._latency,
        health_checker=server._health,
    )

    return app
