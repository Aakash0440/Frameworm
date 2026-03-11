"""
ResultsAnalyzer — turns SuiteResult into paper-ready tables.

Generates:
    - Detection rate table (per baseline × anomaly type)
    - Resolution rate table
    - Mean time-to-recovery table
    - Compute overhead table
    - Full Markdown results section
    - LaTeX table (paste into paper)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from agent.benchmark.run_suite import ScenarioResult, SuiteResult
from agent.benchmark.inject_failures import SCENARIO_REGISTRY

logger = logging.getLogger(__name__)

BASELINES = ["HUMAN", "RULE_BASED", "LLM_ONLY", "FULL_AGENT"]
ANOMALY_TYPES = ["GRADIENT_EXPLOSION", "LOSS_SPIKE", "PLATEAU", "DIVERGENCE"]


class ResultsAnalyzer:

    def __init__(
        self,
        suite_result: SuiteResult,
        output_dir: Path = Path("experiments/benchmark"),
    ) -> None:
        self.result = suite_result
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict:
        """Aggregate stats by baseline."""
        out = {}
        for b in BASELINES:
            out[b] = {
                "detection_rate": self.result.detection_rate(b),
                "resolution_rate": self.result.resolution_rate(b),
                "mean_detection_latency": self.result.mean_detection_latency(b),
                "mean_time_to_recovery": self.result.mean_time_to_recovery(b),
            }
        return out

    def to_markdown(self) -> str:
        lines = [
            "# FRAMEWORM-AGENT Benchmark Results",
            "",
            "## Overall Performance",
            "",
            "| Baseline | Detection Rate | Resolution Rate "
            "| Mean Detection Latency | Mean Recovery Steps |",
            "|---|---|---|---|---|",
        ]

        for b in BASELINES:
            dr = self.result.detection_rate(b)
            rr = self.result.resolution_rate(b)
            dl = self.result.mean_detection_latency(b)
            rec = self.result.mean_time_to_recovery(b)
            dl_str = f"{dl:.0f}" if dl < float("inf") else "N/A"
            rec_str = f"{rec:.0f}" if rec < float("inf") else "N/A"
            lines.append(f"| {b} | {dr:.1%} | {rr:.1%} | {dl_str} steps | {rec_str} steps |")

        lines += [
            "",
            "## Per Anomaly Type — Resolution Rate",
            "",
            "| Anomaly Type | HUMAN | RULE_BASED | LLM_ONLY | FULL_AGENT |",
            "|---|---|---|---|---|",
        ]

        for atype in ANOMALY_TYPES:
            row = [atype]
            for b in BASELINES:
                r = [x for x in self.result.results if x.baseline == b and x.anomaly_type == atype]
                rate = float(np.mean([x.resolved for x in r])) if r else 0.0
                row.append(f"{rate:.1%}")
            lines.append("| " + " | ".join(row) + " |")

        lines += [
            "",
            "## Per Severity — Detection Rate",
            "",
            "| Severity | HUMAN | RULE_BASED | LLM_ONLY | FULL_AGENT |",
            "|---|---|---|---|---|",
        ]

        for sev in ["mild", "moderate", "severe"]:
            row = [sev.upper()]
            for b in BASELINES:
                r = [x for x in self.result.results if x.baseline == b and x.severity == sev]
                rate = float(np.mean([x.detected for x in r])) if r else 0.0
                row.append(f"{rate:.1%}")
            lines.append("| " + " | ".join(row) + " |")

        lines += [
            "",
            f"*Generated from {len(self.result.results)} benchmark runs. "
            f"Total duration: {self.result.total_duration_seconds:.1f}s*",
        ]

        return "\n".join(lines)

    def to_latex_table(self) -> str:
        """LaTeX table for the paper. Paste into results section."""
        rows = []
        for b in BASELINES:
            dr = self.result.detection_rate(b)
            rr = self.result.resolution_rate(b)
            dl = self.result.mean_detection_latency(b)
            rec = self.result.mean_time_to_recovery(b)
            dl_str = f"{dl:.0f}" if dl < float("inf") else "---"
            rec_str = f"{rec:.0f}" if rec < float("inf") else "---"
            rows.append(f"        {b} & {dr:.1%} & {rr:.1%} & {dl_str} & {rec_str} \\\\")

        return (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\begin{tabular}{lcccc}\n"
            "\\hline\n"
            "Baseline & Detection Rate & Resolution Rate "
            "& Detect Latency & Recovery Steps \\\\\n"
            "\\hline\n" + "\n".join(rows) + "\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\caption{FRAMEWORM-AGENT benchmark results "
            "across 12 failure scenarios.}\n"
            "\\label{tab:benchmark}\n"
            "\\end{table}"
        )

    def save(self) -> None:
        md = self.to_markdown()
        latex = self.to_latex_table()
        summary = self.summary()

        (self.output_dir / "results_table.md").write_text(md)
        (self.output_dir / "results_table.tex").write_text(latex)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Results saved to {self.output_dir}/")
        print(md)
