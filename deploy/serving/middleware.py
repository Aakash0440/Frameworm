"""
Latency + error-rate tracking middleware for generated FRAMEWORM servers.
Feeds p50/p95/p99 data to the rollback monitor.
Reuses deploy/core/latency_tracker.py.
"""

import logging
import time
from typing import Callable

logger = logging.getLogger("frameworm.deploy")


class LatencyMiddleware:
    """
    ASGI middleware — records request latency and error rate.
    Attaches to every generated FastAPI server automatically.

    Also feeds HealthChecker so rollback monitor can act on error rate.
    """

    def __init__(self, app, latency_tracker, health_checker):
        self.app = app
        self._tracker = latency_tracker
        self._health = health_checker

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Skip health/ready endpoints — don't track them as real requests
        if path in ("/health", "/ready", "/metrics"):
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        success = True

        async def send_wrapper(message):
            nonlocal success
            if message["type"] == "http.response.start":
                status = message.get("status", 200)
                if status >= 500:
                    success = False
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            success = False
            logger.error(f"[DEPLOY] Unhandled error in request: {e}")
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._tracker.record(elapsed_ms)
            self._health.record_request(success=success)
