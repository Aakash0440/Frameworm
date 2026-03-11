"""Tests for serving layer: health checker, latency middleware."""

import sys
import time

sys.path.insert(0, ".")


def test_health_checker_initial_state():
    from deploy.serving.health import HealthChecker

    hc = HealthChecker("test_model", "v1.0")
    assert not hc.is_ready
    assert hc.error_rate == 0.0
    h = hc.health()
    assert h["status"] == "ok"
    assert h["model_name"] == "test_model"


def test_health_checker_mark_ready():
    from deploy.serving.health import HealthChecker

    hc = HealthChecker("m", "v1")
    hc.mark_ready()
    assert hc.is_ready
    assert hc.readiness()["ready"] is True


def test_health_checker_error_rate():
    from deploy.serving.health import HealthChecker

    hc = HealthChecker("m", "v1")
    hc.record_request(success=True)
    hc.record_request(success=True)
    hc.record_request(success=False)
    assert abs(hc.error_rate - (1 / 3)) < 0.01


def test_degradation_monitor_triggers():
    from deploy.serving.health import HealthChecker
    from deploy.core.latency_tracker import LatencyTracker
    from deploy.rollback.monitor import DegradationMonitor

    tracker = LatencyTracker("test")
    health = HealthChecker("test", "v1")

    # Simulate 20 requests with high latency
    for _ in range(20):
        tracker.record(3000.0)  # 3000ms — above 2000ms threshold
        health.record_request(success=True)

    triggered_reasons = []

    monitor = DegradationMonitor(
        latency_tracker=tracker,
        health_checker=health,
        on_degradation=lambda r: triggered_reasons.append(r),
        p95_threshold_ms=2000.0,
        check_interval_s=999,  # don't auto-run loop
        consecutive_failures=1,  # trigger on first check
    )

    monitor._check()  # manually trigger one check
    assert len(triggered_reasons) == 1
    assert "p95" in triggered_reasons[0].lower()


def test_degradation_monitor_no_trigger_on_healthy():
    from deploy.serving.health import HealthChecker
    from deploy.core.latency_tracker import LatencyTracker
    from deploy.rollback.monitor import DegradationMonitor

    tracker = LatencyTracker("healthy")
    health = HealthChecker("healthy", "v1")

    for _ in range(20):
        tracker.record(50.0)  # fast — well below threshold
        health.record_request(success=True)

    triggered = []
    monitor = DegradationMonitor(
        latency_tracker=tracker,
        health_checker=health,
        on_degradation=lambda r: triggered.append(r),
        p95_threshold_ms=2000.0,
        check_interval_s=999,
        consecutive_failures=1,
    )
    monitor._check()
    assert len(triggered) == 0
