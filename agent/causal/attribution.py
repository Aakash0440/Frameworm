"""
AttributionEngine — combines CausalGraph + DoIntervention
into a complete AttributionReport.

The report is:
    1. Appended to the LLM prompt context (richer, better decisions)
    2. Logged to the experiment snapshot (for paper analysis)
    3. Stored in the experience buffer (for offline RL training)

Two modes:
    FAST   → graph traversal only (< 1 second, always runs)
    FULL   → graph traversal + do-intervention replay (30–120 seconds,
             only for CRITICAL/HIGH severity anomalies)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

from agent.causal.causal_graph import CausalGraph, CausalNode, NodeStatus
from agent.causal.do_intervention import DoIntervention, FreezeVariable, ReplayResult
from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalSnapshot

logger = logging.getLogger(__name__)


class AttributionMode(Enum):
    FAST = auto()  # graph only — always runs
    FULL = auto()  # graph + do-intervention — CRITICAL/HIGH only


@dataclass
class RootCause:
    """One identified root cause node."""

    node_name: str
    node_description: str
    z_score: float
    first_deviation_step: Optional[int]
    causal_path_to_loss: List[str]  # e.g. ["batch_quality", "gradient_dist", "loss"]
    confirmed_by_replay: bool = False  # True if do-intervention confirmed it

    @property
    def confidence(self) -> str:
        if self.confirmed_by_replay:
            return "HIGH (confirmed by replay)"
        if abs(self.z_score) > 4.0:
            return "MEDIUM (strong z-score)"
        return "LOW (graph traversal only)"


@dataclass
class AttributionReport:
    """
    Full attribution report for one anomaly event.
    Logged to disk and appended to LLM prompt.
    """

    anomaly_type: str
    anomaly_step: int
    mode: str  # "FAST" or "FULL"

    root_causes: List[RootCause] = field(default_factory=list)
    replay_results: List[ReplayResult] = field(default_factory=list)

    graph_summary: str = ""  # human-readable graph status
    attribution_summary: str = ""  # one-sentence for LLM prompt

    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def has_confirmed_root_cause(self) -> bool:
        return any(rc.confirmed_by_replay for rc in self.root_causes)

    @property
    def top_root_cause(self) -> Optional[RootCause]:
        if not self.root_causes:
            return None
        confirmed = [rc for rc in self.root_causes if rc.confirmed_by_replay]
        return confirmed[0] if confirmed else self.root_causes[0]

    def to_prompt_text(self) -> str:
        """
        Compact text appended to LLM prompt for richer context.
        Kept short to conserve tokens.
        """
        if not self.root_causes:
            return "Root cause: unknown (insufficient data for attribution)"

        lines = ["ROOT CAUSE ANALYSIS:"]
        for rc in self.root_causes[:2]:  # max 2 root causes in prompt
            conf = "✓ confirmed" if rc.confirmed_by_replay else "~ inferred"
            path = " → ".join(rc.causal_path_to_loss)
            lines.append(f"  {conf}: {rc.node_name} (z={rc.z_score:+.2f})")
            lines.append(f"  Causal path: {path}")
        lines.append(f"  Summary: {self.attribution_summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type,
            "anomaly_step": self.anomaly_step,
            "mode": self.mode,
            "root_causes": [
                {
                    "node_name": rc.node_name,
                    "description": rc.node_description,
                    "z_score": rc.z_score,
                    "first_deviation_step": rc.first_deviation_step,
                    "causal_path": rc.causal_path_to_loss,
                    "confirmed": rc.confirmed_by_replay,
                    "confidence": rc.confidence,
                }
                for rc in self.root_causes
            ],
            "attribution_summary": self.attribution_summary,
            "has_confirmed_root_cause": self.has_confirmed_root_cause,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


class AttributionEngine:
    """
    Orchestrates CausalGraph + DoIntervention into AttributionReports.

    Args:
        graph:          CausalGraph instance (shared, calibrated once).
        do_intervention: DoIntervention instance (optional — for FULL mode).
        log_dir:        Where to write attribution logs.
        full_mode_severities: Severity levels that trigger FULL mode
                              (do-intervention replay).
    """

    def __init__(
        self,
        graph: Optional[CausalGraph] = None,
        do_intervention: Optional[DoIntervention] = None,
        log_dir: Path = Path("experiments/agent_logs/attribution"),
        full_mode_severities: Optional[set] = None,
    ) -> None:
        self.graph = graph or CausalGraph()
        self.do_intervention = do_intervention
        self.log_dir = log_dir
        self.full_mode_severities = full_mode_severities or {Severity.CRITICAL, Severity.HIGH}
        self._is_calibrated = False
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, window: RollingWindow, healthy_steps: int = 100) -> None:
        """Calibrate graph baselines. Call once after first 100 training steps."""
        self.graph.calibrate_baseline(window, healthy_steps=healthy_steps)
        self._is_calibrated = True
        logger.info("AttributionEngine: baseline calibrated.")

    def attribute(
        self,
        event: AnomalyEvent,
        snapshot,  # MetricSnapshot
        signals: SignalSnapshot,
        window: RollingWindow,
        pre_anomaly_checkpoint_step: Optional[int] = None,
    ) -> AttributionReport:
        """
        Generate an AttributionReport for an anomaly event.

        FAST mode (always):
            - Evaluate all causal nodes
            - Find root causes via graph traversal
            - Build summary

        FULL mode (CRITICAL/HIGH severity only):
            - Everything in FAST mode, plus:
            - Run do-intervention replays for top candidate variables
            - Confirm root cause

        Args:
            event:                      The detected AnomalyEvent.
            snapshot:                   Current MetricSnapshot.
            signals:                    Current SignalSnapshot.
            window:                     Current RollingWindow.
            pre_anomaly_checkpoint_step: Step to roll back to for replay.
        """
        start = time.monotonic()

        # Calibrate if not done yet
        if not self._is_calibrated and len(window) >= 100:
            self.calibrate(window)

        # Determine mode
        mode = (
            AttributionMode.FULL
            if (
                event.severity in self.full_mode_severities
                and self.do_intervention is not None
                and pre_anomaly_checkpoint_step is not None
                and self._is_calibrated
            )
            else AttributionMode.FAST
        )

        # ── FAST: graph evaluation ────────────────────────────────
        node_statuses = self.graph.evaluate_at(snapshot, signals, event.step)
        root_cause_nodes = self.graph.find_root_causes()

        root_causes = []
        for node in root_cause_nodes:
            path = self.graph.get_causal_path(node.name, "loss")
            root_causes.append(
                RootCause(
                    node_name=node.name,
                    node_description=node.description,
                    z_score=node.current_z_score,
                    first_deviation_step=node.first_deviation_step,
                    causal_path_to_loss=path,
                    confirmed_by_replay=False,
                )
            )

        graph_summary = self.graph.summary()
        replay_results = []

        # ── FULL: do-intervention replay ──────────────────────────
        if mode == AttributionMode.FULL and root_causes:
            logger.info(
                f"AttributionEngine: running FULL mode replay for "
                f"{event.anomaly_type.name} at step {event.step}"
            )

            # Map root cause nodes to freeze variables
            variable_map = {
                "batch_quality": FreezeVariable.BATCH_SEQUENCE,
                "data_stats": FreezeVariable.BATCH_SEQUENCE,
                "gradient_dist": FreezeVariable.GRADIENT_CLIP,
                "lr_schedule": FreezeVariable.LR_FREEZE,
                "layer_grad_norms": FreezeVariable.LAYER_FREEZE,
                "optimizer_state": FreezeVariable.LR_FREEZE,
            }

            candidate_vars = []
            for rc in root_causes[:2]:  # test top 2 candidates
                var = variable_map.get(rc.node_name)
                if var and var not in candidate_vars:
                    candidate_vars.append(var)

            if candidate_vars:
                replay_results = self.do_intervention.run(
                    anomaly_step=event.step,
                    pre_anomaly_checkpoint_step=pre_anomaly_checkpoint_step,
                    candidate_variables=candidate_vars,
                )

                # Update root causes with replay confirmation
                for replay in replay_results:
                    if replay.confirmed_root_cause:
                        # Find matching root cause and mark confirmed
                        var_to_node = {v: k for k, v in variable_map.items()}
                        confirmed_node = var_to_node.get(replay.freeze_variable)
                        for rc in root_causes:
                            if rc.node_name == confirmed_node:
                                rc.confirmed_by_replay = True

        # ── Build summary ─────────────────────────────────────────
        attribution_summary = self._build_summary(event, root_causes, replay_results)

        duration = time.monotonic() - start
        report = AttributionReport(
            anomaly_type=event.anomaly_type.name,
            anomaly_step=event.step,
            mode=mode.name,
            root_causes=root_causes,
            replay_results=replay_results,
            graph_summary=graph_summary,
            attribution_summary=attribution_summary,
            duration_seconds=duration,
        )

        logger.info(
            f"AttributionEngine: {mode.name} attribution in {duration:.1f}s. "
            f"Root causes: {[rc.node_name for rc in root_causes]}"
        )

        self._save_report(report, event.step)
        return report

    def _build_summary(
        self,
        event: AnomalyEvent,
        root_causes: List[RootCause],
        replay_results: List[ReplayResult],
    ) -> str:
        """Build one-sentence attribution summary for LLM prompt."""
        if not root_causes:
            return (
                f"{event.anomaly_type.name} at step {event.step}: "
                "root cause unknown — insufficient signal deviation."
            )

        top = root_causes[0]
        confirmed = any(r.confirmed_root_cause for r in replay_results)

        path_str = " → ".join(top.causal_path_to_loss) if top.causal_path_to_loss else top.node_name
        conf_str = "confirmed by replay" if confirmed else "inferred from causal graph"

        return (
            f"{event.anomaly_type.name} traced to '{top.node_name}' "
            f"({top.node_description}, z={top.z_score:+.2f}), "
            f"propagating via {path_str} ({conf_str})."
        )

    def _save_report(self, report: AttributionReport, step: int) -> None:
        """Save attribution report to JSON log."""
        try:
            path = self.log_dir / f"attribution_step_{step}_{int(time.time())}.json"
            path.write_text(json.dumps(report.to_dict(), indent=2))
        except Exception as exc:
            logger.debug(f"Could not save attribution report: {exc}")
