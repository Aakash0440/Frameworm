"""
EvalReportGenerator — statistical analysis of intervention deltas.

Produces the numbers that go directly into the paper.

Hooks into:
    monitoring/ab_testing.py   → Welch's t-test (your existing impl)
    metrics/fid.py             → FID delta distribution
    agent/counterfactual/delta_tracker.py → input data

Outputs:
    - Per-anomaly-type success rates
    - Mean loss delta + confidence intervals
    - Mean FID delta + confidence intervals
    - Compute overhead distribution
    - Welch's t-test significance per metric
    - Paper-ready summary table (JSON + Markdown)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.classifier.anomaly_types import AnomalyType
from agent.counterfactual.delta_tracker import DeltaTracker, InterventionDelta

logger = logging.getLogger(__name__)


@dataclass
class AnomalyTypeStats:
    """Stats for one anomaly type across all interventions."""

    anomaly_type: str
    n_interventions: int
    n_with_shadow: int
    success_rate: float  # fraction where agent helped
    mean_loss_delta: float  # negative = agent improved loss
    std_loss_delta: float
    mean_fid_delta: float  # negative = agent improved FID
    std_fid_delta: float
    welch_t_stat: float  # Welch's t-test statistic
    welch_p_value: float  # p < 0.05 = statistically significant
    is_significant: bool  # p < 0.05


@dataclass
class EvalReport:
    """Complete evaluation report — goes in the paper."""

    n_total_interventions: int
    n_with_shadow: int
    overall_success_rate: float
    overall_mean_loss_delta: float
    overall_mean_fid_delta: float
    overall_welch_p: float
    is_overall_significant: bool

    per_anomaly_stats: List[AnomalyTypeStats] = field(default_factory=list)
    compute_overhead_mean: float = 0.0
    compute_overhead_p95: float = 0.0

    generated_at: float = field(default_factory=time.time)

    def to_markdown_table(self) -> str:
        """
        Generate a paper-ready Markdown table.
        Paste directly into your paper's results section.
        """
        lines = [
            "## FRAMEWORM-AGENT Intervention Results",
            "",
            f"**Total interventions:** {self.n_total_interventions}  ",
            f"**With counterfactual shadow:** {self.n_with_shadow}  ",
            f"**Overall success rate:** {self.overall_success_rate:.1%}  ",
            f"**Mean loss delta:** {self.overall_mean_loss_delta:+.4f}  ",
            f"**Mean FID delta:** {self.overall_mean_fid_delta:+.2f}  ",
            f"**Welch's t-test p-value:** {self.overall_welch_p:.4f} "
            f"({'significant' if self.is_overall_significant else 'not significant'})  ",
            "",
            "### Per Anomaly Type",
            "",
            "| Anomaly Type | N | Success Rate | Mean Loss Δ | Mean FID Δ | p-value | Sig? |",
            "|---|---|---|---|---|---|---|",
        ]

        for stats in self.per_anomaly_stats:
            sig = "✓" if stats.is_significant else "✗"
            lines.append(
                f"| {stats.anomaly_type} "
                f"| {stats.n_with_shadow} "
                f"| {stats.success_rate:.1%} "
                f"| {stats.mean_loss_delta:+.4f} "
                f"| {stats.mean_fid_delta:+.2f} "
                f"| {stats.welch_p_value:.3f} "
                f"| {sig} |"
            )

        lines += [
            "",
            "### Compute Overhead",
            f"- Mean overhead: {self.compute_overhead_mean:.1f}s per intervention",
            f"- p95 overhead: {self.compute_overhead_p95:.1f}s",
        ]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_total_interventions": self.n_total_interventions,
            "n_with_shadow": self.n_with_shadow,
            "overall_success_rate": self.overall_success_rate,
            "overall_mean_loss_delta": self.overall_mean_loss_delta,
            "overall_mean_fid_delta": self.overall_mean_fid_delta,
            "overall_welch_p": self.overall_welch_p,
            "is_overall_significant": self.is_overall_significant,
            "per_anomaly_stats": [
                {
                    "anomaly_type": s.anomaly_type,
                    "n_interventions": s.n_interventions,
                    "n_with_shadow": s.n_with_shadow,
                    "success_rate": s.success_rate,
                    "mean_loss_delta": s.mean_loss_delta,
                    "mean_fid_delta": s.mean_fid_delta,
                    "welch_p_value": s.welch_p_value,
                    "is_significant": s.is_significant,
                }
                for s in self.per_anomaly_stats
            ],
            "compute_overhead_mean": self.compute_overhead_mean,
            "compute_overhead_p95": self.compute_overhead_p95,
            "generated_at": self.generated_at,
        }


class EvalReportGenerator:
    """
    Generates EvalReport from DeltaTracker data.

    Hooks into monitoring/ab_testing.py for Welch's t-test.

    Args:
        tracker:    DeltaTracker with accumulated deltas.
        alpha:      Significance level (default 0.05).
        log_dir:    Where to save the report.
    """

    def __init__(
        self,
        tracker: DeltaTracker,
        alpha: float = 0.05,
        log_dir: Path = Path("experiments/agent_logs"),
    ) -> None:
        self.tracker = tracker
        self.alpha = alpha
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> EvalReport:
        """
        Generate complete evaluation report from all accumulated deltas.
        """
        completed = self.tracker.completed_deltas
        all_deltas = self.tracker.all_deltas

        if not completed:
            logger.warning("EvalReportGenerator: no completed deltas yet.")
            return EvalReport(
                n_total_interventions=len(all_deltas),
                n_with_shadow=0,
                overall_success_rate=0.0,
                overall_mean_loss_delta=0.0,
                overall_mean_fid_delta=0.0,
                overall_welch_p=1.0,
                is_overall_significant=False,
            )

        # ── Overall stats ─────────────────────────────────────────
        loss_deltas = np.array([d.loss_delta for d in completed])
        fid_deltas = np.array(
            [d.fid_delta for d in completed if d.run_a_fid is not None and d.run_b_fid is not None]
        )
        success_flags = np.array([d.agent_helped for d in completed])
        overheads = np.array([d.compute_overhead_seconds for d in completed])

        overall_success_rate = float(np.mean(success_flags))
        overall_mean_loss_delta = float(np.mean(loss_deltas))
        overall_mean_fid_delta = float(np.mean(fid_deltas)) if len(fid_deltas) > 0 else 0.0

        # Welch's t-test on overall loss deltas
        # H0: mean loss delta = 0 (agent has no effect)
        # H1: mean loss delta < 0 (agent improved loss)
        overall_t, overall_p = self._welch_t_test(loss_deltas)

        # ── Per anomaly type stats ────────────────────────────────
        anomaly_stats = []
        for atype in AnomalyType:
            if atype == AnomalyType.HEALTHY:
                continue
            type_deltas = [d for d in completed if d.anomaly_type == atype]
            if not type_deltas:
                continue

            type_loss = np.array([d.loss_delta for d in type_deltas])
            type_fid = np.array(
                [
                    d.fid_delta
                    for d in type_deltas
                    if d.run_a_fid is not None and d.run_b_fid is not None
                ]
            )
            type_success = np.array([d.agent_helped for d in type_deltas])

            t_stat, p_val = self._welch_t_test(type_loss)

            anomaly_stats.append(
                AnomalyTypeStats(
                    anomaly_type=atype.name,
                    n_interventions=len([d for d in all_deltas if d.anomaly_type == atype]),
                    n_with_shadow=len(type_deltas),
                    success_rate=float(np.mean(type_success)),
                    mean_loss_delta=float(np.mean(type_loss)),
                    std_loss_delta=float(np.std(type_loss)),
                    mean_fid_delta=float(np.mean(type_fid)) if len(type_fid) > 0 else 0.0,
                    std_fid_delta=float(np.std(type_fid)) if len(type_fid) > 0 else 0.0,
                    welch_t_stat=t_stat,
                    welch_p_value=p_val,
                    is_significant=p_val < self.alpha,
                )
            )

        report = EvalReport(
            n_total_interventions=len(all_deltas),
            n_with_shadow=len(completed),
            overall_success_rate=overall_success_rate,
            overall_mean_loss_delta=overall_mean_loss_delta,
            overall_mean_fid_delta=overall_mean_fid_delta,
            overall_welch_p=overall_p,
            is_overall_significant=overall_p < self.alpha,
            per_anomaly_stats=anomaly_stats,
            compute_overhead_mean=float(np.mean(overheads)) if len(overheads) > 0 else 0.0,
            compute_overhead_p95=float(np.percentile(overheads, 95)) if len(overheads) > 0 else 0.0,
        )

        self._save_report(report)
        return report

    def _welch_t_test(self, deltas: np.ndarray) -> Tuple[float, float]:
        """
        One-sample Welch's t-test: H0 mean = 0.
        Uses your existing monitoring/ab_testing.py if available.
        """
        if len(deltas) < 2:
            return 0.0, 1.0

        # Try your existing ab_testing module first
        try:
            from monitoring.ab_testing import ABTester

            tester = ABTester()
            # Your ab_testing compares two groups — use zeros as baseline
            zeros = np.zeros(len(deltas))
            result = tester.welch_t_test(deltas, zeros)
            if isinstance(result, dict):
                return float(result.get("t_stat", 0.0)), float(result.get("p_value", 1.0))
        except (ImportError, AttributeError, Exception) as exc:
            logger.debug(f"monitoring.ab_testing not available: {exc}")

        # Fallback: scipy
        try:
            from scipy import stats

            t_stat, p_val = stats.ttest_1samp(deltas, 0.0, alternative="less")
            return float(t_stat), float(p_val)
        except ImportError:
            pass

        # Last resort: manual
        n = len(deltas)
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=1)) + 1e-8
        t_stat = mean / (std / np.sqrt(n))
        # Approximate p-value (very rough without scipy)
        p_val = 0.05 if abs(t_stat) > 2.0 else 0.5
        return float(t_stat), p_val

    def _save_report(self, report: EvalReport) -> None:
        """Save report as JSON + Markdown."""
        ts = int(time.time())
        # JSON
        json_path = self.log_dir / f"eval_report_{ts}.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        # Markdown
        md_path = self.log_dir / f"eval_report_{ts}.md"
        md_path.write_text(report.to_markdown_table(), encoding="utf-8")
        logger.info(f"EvalReport saved: {json_path}, {md_path}")
