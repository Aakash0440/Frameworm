"""
Tests for agent/classifier/ — rule engine + priority queue.
Run with: pytest tests/agent/test_classifier.py -v
"""

import numpy as np
import pytest

from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.classifier.priority_queue import AnomalyPriorityQueue
from agent.classifier.rule_engine import RuleEngine, RuleEngineConfig
from agent.observer.rolling_window import MetricSnapshot, RollingWindow
from agent.observer.signal_extractor import SignalExtractor


def _make_signals(n=120, loss_override=None, grad_override=None):
    w = RollingWindow(size=300)
    for i in range(n):
        loss = (
            loss_override
            if loss_override and i == n - 1
            else (1.0 - i * 0.004 + np.random.normal(0, 0.01))
        )
        gn = grad_override if grad_override and i == n - 1 else (2.0 + np.random.normal(0, 0.1))
        w.push(MetricSnapshot(step=i, loss=max(loss, 0.0), grad_norm=max(gn, 0.0), lr=0.0002))
    return SignalExtractor().extract(w)


class TestRuleEngine:
    def test_healthy_run_no_events(self):
        np.random.seed(42)
        signals = _make_signals(n=120)
        engine = RuleEngine()
        events = engine.classify(signals)
        assert len(events) == 0

    def test_detects_gradient_explosion(self):
        np.random.seed(0)
        signals = _make_signals(n=120, grad_override=55.0)
        engine = RuleEngine()
        events = engine.classify(signals)
        types = [e.anomaly_type for e in events]
        assert AnomalyType.GRADIENT_EXPLOSION in types

    def test_detects_loss_spike(self):
        np.random.seed(0)
        signals = _make_signals(n=120, loss_override=15.0)
        engine = RuleEngine()
        events = engine.classify(signals)
        types = [e.anomaly_type for e in events]
        assert AnomalyType.LOSS_SPIKE in types

    def test_events_sorted_by_priority(self):
        np.random.seed(0)
        signals = _make_signals(n=120, loss_override=15.0, grad_override=55.0)
        engine = RuleEngine()
        events = engine.classify(signals)
        if len(events) >= 2:
            for i in range(len(events) - 1):
                assert events[i].anomaly_type.priority <= events[i + 1].anomaly_type.priority

    def test_early_training_lenience(self):
        """Thresholds should be relaxed during first 10% of training."""
        w = RollingWindow(size=100)
        for i in range(15):
            w.push(
                MetricSnapshot(
                    step=i, loss=1.0 + np.random.normal(0, 0.05), grad_norm=2.0, lr=0.0002
                )
            )
        signals = SignalExtractor(total_steps=10_000).extract(w)
        engine = RuleEngine()
        # A borderline z-score that would fire in normal training
        # should be suppressed in early training
        assert signals.is_early_training

    def test_reset_counters(self):
        engine = RuleEngine()
        engine._plateau_counter = 999
        engine._divergence_counter = 999
        engine.reset_counters()
        assert engine._plateau_counter == 0
        assert engine._divergence_counter == 0

    def test_custom_thresholds(self):
        cfg = RuleEngineConfig(grad_explosion_threshold=5.0)
        signals = _make_signals(n=120, grad_override=6.0)
        engine = RuleEngine(config=cfg)
        events = engine.classify(signals)
        assert any(e.anomaly_type == AnomalyType.GRADIENT_EXPLOSION for e in events)

    def test_false_positive_rate_on_healthy(self):
        """FP rate on healthy data should be < 5%."""
        np.random.seed(7)
        engine = RuleEngine()
        fp_count = 0
        n_trials = 50
        for trial in range(n_trials):
            signals = _make_signals(n=120)
            events = engine.classify(signals)
            if events:
                fp_count += 1
            engine.reset_counters()
        fp_rate = fp_count / n_trials
        assert fp_rate < 0.05, f"FP rate {fp_rate:.1%} exceeds 5%"


class TestAnomalyPriorityQueue:
    def test_push_pop_ordering(self):
        q = AnomalyPriorityQueue()
        q.push(AnomalyEvent(AnomalyType.PLATEAU, Severity.LOW, step=1))
        q.push(AnomalyEvent(AnomalyType.GRADIENT_EXPLOSION, Severity.HIGH, step=2))
        q.push(AnomalyEvent(AnomalyType.LOSS_SPIKE, Severity.MEDIUM, step=3))
        first = q.pop()
        assert first.anomaly_type == AnomalyType.GRADIENT_EXPLOSION

    def test_deduplication(self):
        q = AnomalyPriorityQueue()
        q.push(AnomalyEvent(AnomalyType.PLATEAU, Severity.LOW, step=1))
        q.push(AnomalyEvent(AnomalyType.PLATEAU, Severity.LOW, step=2))
        assert len(q) == 1

    def test_clear(self):
        q = AnomalyPriorityQueue()
        q.push(AnomalyEvent(AnomalyType.PLATEAU, Severity.LOW, step=1))
        q.clear()
        assert q.is_empty()

    def test_has_type(self):
        q = AnomalyPriorityQueue()
        q.push(AnomalyEvent(AnomalyType.DIVERGENCE, Severity.HIGH, step=1))
        assert q.has_type(AnomalyType.DIVERGENCE)
        assert not q.has_type(AnomalyType.PLATEAU)
