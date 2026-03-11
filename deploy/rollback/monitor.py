"""
FRAMEWORM DEPLOY — Degradation monitor.
Watches a running deployment for latency spikes and error rate degradation.
Triggers rollback callback when thresholds are exceeded.
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("frameworm.deploy.rollback")


class DegradationMonitor:
    """
    Watches a LatencyTracker + HealthChecker and calls on_degradation(reason)
    when performance degrades past configured thresholds.

    Usage:
        monitor = DegradationMonitor(
            latency_tracker=tracker,
            health_checker=health,
            on_degradation=lambda reason: controller.rollback(reason),
            p95_threshold_ms=2000.0,
            check_interval_s=30,
            consecutive_failures=3,
        )
        monitor.start()          # background loop
        monitor._check()         # or trigger manually in tests
    """

    DEFAULT_P95_THRESHOLD_MS = 2000.0
    DEFAULT_ERROR_RATE = 0.10
    DEFAULT_CHECK_INTERVAL_S = 30
    DEFAULT_CONSECUTIVE = 3

    def __init__(
        self,
        latency_tracker,
        health_checker,
        on_degradation: Callable[[str], None],
        p95_threshold_ms: float = DEFAULT_P95_THRESHOLD_MS,
        error_rate_threshold: float = DEFAULT_ERROR_RATE,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
        consecutive_failures: int = DEFAULT_CONSECUTIVE,
    ):
        self._latency = latency_tracker
        self._health = health_checker
        self._on_degradation = on_degradation
        self._p95_threshold = p95_threshold_ms
        self._error_rate_threshold = error_rate_threshold
        self._check_interval = check_interval_s
        self._consecutive_required = consecutive_failures

        self._failures = 0
        self._triggered = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="frameworm-deploy-monitor"
        )
        self._thread.start()
        logger.info(
            f"[DEPLOY] DegradationMonitor started — "
            f"p95>{self._p95_threshold}ms or "
            f"err>{self._error_rate_threshold*100:.0f}% triggers after "
            f"{self._consecutive_required} checks"
        )

    def stop(self):
        self._running = False
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ─── check logic ──────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_evt.wait(self._check_interval):
            try:
                self._check()
            except Exception as e:
                logger.error(f"[DEPLOY] Monitor check error: {e}")

    def _check(self):
        """
        Run one degradation check.
        Uses snapshot() — NOT summary() which doesn't exist on LatencyTracker.
        Fires on_degradation once consecutive_failures threshold is reached.
        """
        if self._triggered:
            return

        reasons = []

        # ── latency check — uses snapshot(), not summary() ──
        snap = self._latency.snapshot()
        if snap is not None:
            if snap.p95_ms >= self._p95_threshold:
                reasons.append(
                    f"p95 latency {snap.p95_ms:.0f}ms exceeds "
                    f"threshold {self._p95_threshold:.0f}ms"
                )

        # ── error rate check ──
        err = self._health.error_rate
        if err >= self._error_rate_threshold:
            reasons.append(
                f"error rate {err*100:.1f}% exceeds "
                f"threshold {self._error_rate_threshold*100:.1f}%"
            )

        if reasons:
            self._failures += 1
            reason_str = "; ".join(reasons)
            logger.warning(
                f"[DEPLOY] Degradation check failed "
                f"({self._failures}/{self._consecutive_required}): {reason_str}"
            )
            if self._failures >= self._consecutive_required:
                self._triggered = True
                self._running = False
                logger.error(f"[DEPLOY] DEGRADATION DETECTED — {reason_str}")
                try:
                    self._on_degradation(reason_str)
                except Exception as e:
                    logger.error(f"[DEPLOY] on_degradation callback raised: {e}")
                self._failures = 0
        else:
            if self._failures > 0:
                logger.info("[DEPLOY] Check passed — resetting failure count")
            self._failures = 0
            self._triggered = False
