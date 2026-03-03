"""
Tests for agent/observer/ — mirrors your existing tests/ structure.
Run with: pytest tests/agent/test_observer.py -v
"""

import numpy as np
import pytest
from agent.observer.rolling_window import RollingWindow, MetricSnapshot
from agent.observer.signal_extractor import SignalExtractor, SignalSnapshot


class TestRollingWindow:
    def test_push_and_len(self):
        w = RollingWindow(size=10)
        for i in range(5):
            w.push(MetricSnapshot(step=i, loss=float(i), grad_norm=1.0, lr=0.001))
        assert len(w) == 5

    def test_eviction(self):
        w = RollingWindow(size=5)
        for i in range(10):
            w.push(MetricSnapshot(step=i, loss=float(i), grad_norm=1.0, lr=0.001))
        assert len(w) == 5
        assert w.latest().step == 9

    def test_losses_array(self):
        w = RollingWindow(size=100)
        for i in range(20):
            w.push(MetricSnapshot(step=i, loss=float(i), grad_norm=1.0, lr=0.001))
        losses = w.losses()
        assert losses.shape == (20,)
        assert losses[-1] == 19.0

    def test_is_ready(self):
        w = RollingWindow(size=100)
        for i in range(9):
            w.push(MetricSnapshot(step=i, loss=1.0, grad_norm=1.0, lr=0.001))
        assert not w.is_ready
        w.push(MetricSnapshot(step=9, loss=1.0, grad_norm=1.0, lr=0.001))
        assert w.is_ready

    def test_thread_safety(self):
        import threading
        w = RollingWindow(size=200)
        errors = []

        def push_worker(offset):
            try:
                for i in range(50):
                    w.push(MetricSnapshot(step=offset+i, loss=1.0, grad_norm=1.0, lr=0.001))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push_worker, args=(i*100,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"


class TestSignalExtractor:
    def _make_window(self, n=120, anomaly=False):
        w = RollingWindow(size=300)
        for i in range(n):
            if anomaly and i == n - 1:
                loss = 10.0
            else:
                loss = 1.0 - i * 0.005 + np.random.normal(0, 0.01)
            w.push(MetricSnapshot(step=i, loss=loss, grad_norm=2.0, lr=0.0002))
        return w

    def test_returns_none_when_not_ready(self):
        w = RollingWindow(size=100)
        for i in range(5):
            w.push(MetricSnapshot(step=i, loss=1.0, grad_norm=1.0, lr=0.001))
        e = SignalExtractor()
        assert e.extract(w) is None

    def test_returns_snapshot_when_ready(self):
        w = self._make_window(n=120)
        e = SignalExtractor()
        sig = e.extract(w)
        assert sig is not None
        assert isinstance(sig, SignalSnapshot)

    def test_z_score_spikes_on_anomaly(self):
        np.random.seed(0)
        w = self._make_window(n=120, anomaly=True)
        e = SignalExtractor()
        sig = e.extract(w)
        assert sig.loss_z_score > 3.0, \
            f"Expected z_score > 3, got {sig.loss_z_score}"

    def test_plateau_score_low_when_stuck(self):
        w = RollingWindow(size=300)
        for i in range(200):
            w.push(MetricSnapshot(step=i, loss=0.5 + np.random.normal(0, 0.0005),
                                  grad_norm=0.1, lr=0.0001))
        e = SignalExtractor()
        sig = e.extract(w)
        assert sig.plateau_score < 0.1

    def test_divergence_score_high_when_rising(self):
        w = RollingWindow(size=300)
        for i in range(150):
            loss = 0.5 + i * 0.01   # always increasing
            w.push(MetricSnapshot(step=i, loss=loss, grad_norm=1.0, lr=0.001))
        e = SignalExtractor()
        sig = e.extract(w)
        assert sig.divergence_score > 0.7

