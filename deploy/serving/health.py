"""
FRAMEWORM DEPLOY — Health and readiness checker.
Tracks model server health state, error rate, and readiness.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Optional

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


class HealthChecker:
    """
    Tracks readiness and request health for a deployed model endpoint.

    Usage:
        hc = HealthChecker("dcgan_v1", "1.0.0")
        hc.mark_ready()
        hc.record_request(success=True)
        hc.record_request(success=False)
        print(hc.health())
        print(hc.readiness())
    """

    WINDOW = 200

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.started_at = datetime.utcnow().isoformat()
        self._ready = False
        self._model_loaded = False

        self._lock = threading.Lock()
        self._window = deque(maxlen=self.WINDOW)  # True=success, False=error
        self._request_count = 0
        self._error_count = 0
        self._last_request_at: Optional[str] = None

    # ─── state ────────────────────────────────────────────────────────────────

    def mark_ready(self):
        self._ready = True
        self._model_loaded = True

    def mark_not_ready(self, reason: str = ""):
        self._ready = False

    def record_request(self, success: bool = True):
        with self._lock:
            self._window.append(success)
            self._request_count += 1
            if not success:
                self._error_count += 1
            self._last_request_at = datetime.utcnow().isoformat()

    # ─── metrics ──────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error_rate(self) -> float:
        """Fraction of recent requests that failed (rolling window)."""
        with self._lock:
            window = list(self._window)
        if not window:
            return 0.0
        return sum(1 for s in window if not s) / len(window)

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    # ─── endpoint payloads ────────────────────────────────────────────────────

    def health(self) -> dict:
        """/health response — always returns a dict, never raises."""
        cpu, mem = -1.0, -1.0
        if _PSUTIL:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
            except Exception:
                pass

        gpu_mem_mb = -1
        if _TORCH:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            except Exception:
                pass

        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "started_at": self.started_at,
            "uptime_seconds": self._uptime(),
            "requests": self.request_count,
            "errors": self._error_count,
            "error_rate": self.error_rate,
            "last_request": self._last_request_at,
            "system": {
                "cpu_percent": cpu,
                "mem_percent": mem,
                "gpu_mem_mb": gpu_mem_mb,
            },
        }

    def readiness(self) -> dict:
        """/ready response."""
        return {
            "ready": self._ready,
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
        }

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _uptime(self) -> float:
        try:
            started = datetime.fromisoformat(self.started_at)
            return (datetime.utcnow() - started).total_seconds()
        except Exception:
            return 0.0
