"""Tests for frameworm-cost core components."""

import pytest
import time
from cost.calculator import CostCalculator, CostBreakdown
from cost.tracker import CostTracker
from cost.store import CostStore
from cost.report import CostReport


class TestCostCalculator:
    def test_basic_calculation(self):
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        cost = calc.calculate(latency_ms=38.0)
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost_usd > 0
        assert cost.latency_ms == 38.0

    def test_batch_reduces_per_request_cost(self):
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        single = calc.calculate(latency_ms=38.0, batch_size=1)
        batched = calc.calculate(latency_ms=38.0, batch_size=8)
        assert batched.total_cost_usd < single.total_cost_usd

    def test_ddpm_more_expensive_than_vae(self):
        calc_vae = CostCalculator(hardware="t4", architecture="vae")
        calc_ddpm = CostCalculator(hardware="t4", architecture="ddpm")
        vae_cost = calc_vae.calculate(latency_ms=20.0)
        ddpm_cost = calc_ddpm.calculate(latency_ms=20.0)
        assert ddpm_cost.total_cost_usd > vae_cost.total_cost_usd

    def test_compare_architectures(self):
        calc = CostCalculator(hardware="t4")
        results = calc.compare_architectures(latency_ms=50.0)
        assert len(results) > 0
        # Should be sorted cheapest first
        costs = [r.total_cost_usd for r in results]
        assert costs == sorted(costs)

    def test_optimization_hint_for_batching(self):
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        cost = calc.calculate(latency_ms=80.0, batch_size=1)
        assert cost.optimization_hint is not None
        assert "batch" in cost.optimization_hint.lower()

    def test_monthly_cost_projection(self):
        calc = CostCalculator(hardware="t4", architecture="vae")
        cost = calc.calculate(latency_ms=20.0)
        monthly = cost.monthly_cost_at_rps
        assert "1_rps" in monthly
        assert "10_rps" in monthly
        assert monthly["10_rps"] > monthly["1_rps"]

    def test_to_dict(self):
        calc = CostCalculator(hardware="t4", architecture="dcgan", model_name="test")
        cost = calc.calculate(latency_ms=38.0)
        d = cost.to_dict()
        assert "total_cost_usd" in d
        assert "latency_ms" in d
        assert "architecture" in d


class TestCostTracker:
    def test_context_manager(self):
        tracker = CostTracker(architecture="vae")
        with tracker.track():
            time.sleep(0.01)
        assert tracker.last_cost is not None
        assert tracker.last_cost.latency_ms > 5
        assert tracker.total_requests == 1

    def test_multiple_tracks(self):
        tracker = CostTracker(architecture="dcgan")
        for _ in range(5):
            with tracker.track():
                time.sleep(0.005)
        assert tracker.total_requests == 5
        assert tracker.total_cost > 0
        assert tracker.average_cost > 0

    def test_track_call(self):
        tracker = CostTracker(architecture="vae")
        result, cost = tracker.track_call(lambda x: x * 2, 5)
        assert result == 10
        assert cost.total_cost_usd > 0

    def test_summary(self):
        tracker = CostTracker(model_name="test-model", architecture="vae")
        with tracker.track():
            time.sleep(0.01)
        s = tracker.summary()
        assert s["model_name"] == "test-model"
        assert s["total_requests"] == 1


class TestCostStore:
    def test_record_and_retrieve(self):
        store = CostStore()
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        cost = calc.calculate(38.0)
        store.record(cost)
        records = store.get_all()
        assert len(records) == 1
        assert records[0]["architecture"] == "dcgan"

    def test_summary(self):
        store = CostStore()
        calc = CostCalculator(hardware="t4", architecture="vae")
        for _ in range(10):
            store.record(calc.calculate(20.0))
        s = store.summary()
        assert s["total_requests"] == 10
        assert s["total_cost_usd"] > 0

    def test_by_model(self):
        store = CostStore()
        calc_a = CostCalculator(model_name="model-a", architecture="vae")
        calc_b = CostCalculator(model_name="model-b", architecture="dcgan")
        store.record(calc_a.calculate(20.0))
        store.record(calc_b.calculate(38.0))
        a_records = store.get_by_model("model-a")
        assert len(a_records) == 1
        assert a_records[0]["model_name"] == "model-a"

    def test_persistence(self, tmp_path):
        path = tmp_path / "costs.json"
        store = CostStore(path=path)
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        store.record(calc.calculate(38.0))
        # Reload
        store2 = CostStore(path=path)
        assert len(store2.get_all()) == 1


class TestCostReport:
    def test_to_dict(self):
        store = CostStore()
        calc = CostCalculator(hardware="t4", architecture="dcgan", model_name="test")
        for _ in range(20):
            store.record(calc.calculate(80.0))
        report = CostReport(store)
        d = report.to_dict()
        assert "summary" in d
        assert "savings_opportunities" in d
        assert "hints" in d

    def test_savings_opportunities_populated(self):
        store = CostStore()
        calc = CostCalculator(hardware="t4", architecture="ddpm")
        for _ in range(10):
            store.record(calc.calculate(200.0))
        report = CostReport(store)
        d = report.to_dict()
        assert len(d["savings_opportunities"]) > 0
