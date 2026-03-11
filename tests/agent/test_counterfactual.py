"""
Tests for agent/counterfactual/ — delta tracker + eval report.
Run with: pytest tests/agent/test_counterfactual.py -v
"""

from pathlib import Path

import pytest

from agent.classifier.anomaly_types import AnomalyType
from agent.counterfactual.delta_tracker import DeltaTracker
from agent.counterfactual.eval_report import EvalReportGenerator
from agent.counterfactual.twin_runner import ShadowRun

TMP = Path("/tmp/fw_test_cf")


class TestDeltaTracker:
    def setup_method(self):
        self.tracker = DeltaTracker(log_dir=TMP / "deltas")

    def test_record_intervention(self):
        d = self.tracker.record_intervention(
            AnomalyType.LOSS_SPIKE,
            "ADJUST_LR(factor=0.5)",
            200,
            run_a_loss=0.4,
            run_a_grad_norm=2.0,
        )
        assert d.intervention_id is not None
        assert not d.shadow_available

    def test_record_shadow_result(self):
        d = self.tracker.record_intervention(
            AnomalyType.GRADIENT_EXPLOSION, "ROLLBACK", 300, run_a_loss=0.3, run_a_grad_norm=1.5
        )
        shadow = ShadowRun(
            run_id="sh1",
            spawn_step=280,
            seed=300,
            n_steps=200,
            completed=True,
            final_loss=0.9,
            final_grad_norm=4.0,
        )
        updated = self.tracker.record_shadow_result(d.intervention_id, shadow)
        assert updated.shadow_available
        assert updated.loss_delta == pytest.approx(0.3 - 0.9, abs=1e-4)
        assert updated.agent_helped  # A loss < B loss

    def test_unknown_intervention_id(self):
        result = self.tracker.record_shadow_result(
            "nonexistent", ShadowRun(run_id="x", spawn_step=0, seed=0, n_steps=0, completed=True)
        )
        assert result is None

    def test_incomplete_shadow_skipped(self):
        d = self.tracker.record_intervention(
            AnomalyType.PLATEAU, "WATCH", 100, run_a_loss=0.5, run_a_grad_norm=1.0
        )
        shadow = ShadowRun(run_id="sh_inc", spawn_step=80, seed=0, n_steps=0, completed=False)
        updated = self.tracker.record_shadow_result(d.intervention_id, shadow)
        assert not updated.shadow_available

    def test_success_rate(self):
        for i in range(5):
            d = self.tracker.record_intervention(
                AnomalyType.OSCILLATING, "ADJUST_LR", i * 100, run_a_loss=0.3, run_a_grad_norm=1.0
            )
            shadow = ShadowRun(
                run_id=f"sh{i}",
                spawn_step=i * 100 - 20,
                seed=i,
                n_steps=200,
                completed=True,
                final_loss=0.8,  # worse than agent
                final_grad_norm=3.0,
            )
            self.tracker.record_shadow_result(d.intervention_id, shadow)
        assert self.tracker.success_rate == 1.0  # all deltas negative


class TestEvalReportGenerator:
    def test_empty_tracker_returns_report(self):
        tracker = DeltaTracker(log_dir=TMP / "empty_deltas")
        gen = EvalReportGenerator(tracker, log_dir=TMP / "empty_reports")
        report = gen.generate()
        assert report.n_total_interventions == 0
        assert report.overall_success_rate == 0.0

    def test_markdown_table_generated(self):
        tracker = DeltaTracker(log_dir=TMP / "md_deltas")
        for i in range(6):
            d = tracker.record_intervention(
                AnomalyType.LOSS_SPIKE,
                "ADJUST_LR",
                i * 50,
                run_a_loss=0.3 + i * 0.01,
                run_a_grad_norm=1.0,
            )
            shadow = ShadowRun(
                run_id=f"s{i}",
                spawn_step=i * 50,
                seed=i,
                n_steps=100,
                completed=True,
                final_loss=0.7,
            )
            tracker.record_shadow_result(d.intervention_id, shadow)
        gen = EvalReportGenerator(tracker, log_dir=TMP / "md_reports")
        report = gen.generate()
        md = report.to_markdown_table()
        assert "LOSS_SPIKE" in md or "FRAMEWORM" in md
        assert "success rate" in md.lower() or "%" in md
