"""
DeltaTracker — records and manages intervention deltas.

For each agent intervention:
    - Stores the Run A (with agent) metrics
    - Stores the Run B (shadow, no agent) metrics
    - Computes the delta: how much better/worse A was vs B

These deltas are the primary result for the paper.
EvalReportGenerator reads from DeltaTracker to produce
statistical analysis and plots.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agent.classifier.anomaly_types import AnomalyType
from agent.counterfactual.twin_runner import ShadowRun

logger = logging.getLogger(__name__)


@dataclass
class InterventionDelta:
    """
    The measured delta between agent run (A) and shadow run (B).

    Positive FID delta = agent run had LOWER FID = agent helped.
    Negative loss delta = agent run had LOWER loss = agent helped.
    """
    intervention_id: str
    intervention_step: int
    anomaly_type: AnomalyType
    action_taken: str               # e.g. "ADJUST_LR(factor=0.5)"

    # Run A (with agent) metrics
    run_a_final_loss: float
    run_a_final_grad_norm: float
    run_a_fid: Optional[float] = None

    # Run B (shadow, no agent) metrics
    run_b_final_loss: float = float("inf")
    run_b_final_grad_norm: float = 0.0
    run_b_fid: Optional[float] = None

    # Deltas (A - B, so negative loss delta = agent improved)
    loss_delta: float = 0.0         # run_a_loss - run_b_loss
    fid_delta: float = 0.0          # run_a_fid - run_b_fid
    grad_norm_delta: float = 0.0    # run_a_grad - run_b_grad

    # Compute overhead (extra GPU time due to agent)
    compute_overhead_seconds: float = 0.0

    # Was the shadow run available for comparison?
    shadow_available: bool = False
    shadow_run_id: str = ""

    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Compute deltas from A and B values."""
        if self.shadow_available:
            self.loss_delta = self.run_a_final_loss - self.run_b_final_loss
            self.grad_norm_delta = self.run_a_final_grad_norm - self.run_b_final_grad_norm
            if self.run_a_fid is not None and self.run_b_fid is not None:
                self.fid_delta = self.run_a_fid - self.run_b_fid

    @property
    def agent_helped(self) -> bool:
        """True if agent run was meaningfully better than shadow."""
        return self.loss_delta < -0.01  # A loss lower than B by at least 0.01

    @property
    def agent_helped_fid(self) -> bool:
        """True if agent improved FID score."""
        if self.run_a_fid is None or self.run_b_fid is None:
            return False
        return self.fid_delta < -0.5  # A FID lower than B by at least 0.5

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "intervention_step": self.intervention_step,
            "anomaly_type": self.anomaly_type.name,
            "action_taken": self.action_taken,
            "run_a_final_loss": self.run_a_final_loss,
            "run_b_final_loss": self.run_b_final_loss,
            "loss_delta": self.loss_delta,
            "run_a_fid": self.run_a_fid,
            "run_b_fid": self.run_b_fid,
            "fid_delta": self.fid_delta,
            "agent_helped": self.agent_helped,
            "shadow_available": self.shadow_available,
            "compute_overhead_seconds": self.compute_overhead_seconds,
            "timestamp": self.timestamp,
        }


class DeltaTracker:
    """
    Accumulates InterventionDeltas across a training run (or many runs).

    Integrates with TwinRunner: when a shadow run completes,
    call record_shadow_result() to update the delta.

    Usage:
        tracker = DeltaTracker()

        # When agent intervenes:
        delta = tracker.record_intervention(event, action, run_a_metrics)

        # When shadow run completes:
        tracker.record_shadow_result(delta.intervention_id, shadow_run)

        # After many runs, generate report:
        report_gen = EvalReportGenerator(tracker)
        report = report_gen.generate()
    """

    def __init__(
        self,
        log_dir: Path = Path("experiments/agent_logs/deltas"),
    ) -> None:
        self.log_dir = log_dir
        self._deltas: Dict[str, InterventionDelta] = {}
        self._counter = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_intervention(
        self,
        anomaly_type: AnomalyType,
        action_taken: str,
        intervention_step: int,
        run_a_loss: float,
        run_a_grad_norm: float,
        run_a_fid: Optional[float] = None,
    ) -> InterventionDelta:
        """
        Record a new intervention (Run A side).
        Call this immediately when the agent takes an action.
        Shadow result is added later via record_shadow_result().
        """
        self._counter += 1
        intervention_id = f"intervention_{self._counter}_step{intervention_step}"

        delta = InterventionDelta(
            intervention_id=intervention_id,
            intervention_step=intervention_step,
            anomaly_type=anomaly_type,
            action_taken=action_taken,
            run_a_final_loss=run_a_loss,
            run_a_final_grad_norm=run_a_grad_norm,
            run_a_fid=run_a_fid,
            shadow_available=False,
        )

        self._deltas[intervention_id] = delta
        logger.info(
            f"DeltaTracker: recorded intervention {intervention_id} "
            f"({anomaly_type.name}, {action_taken})"
        )
        return delta

    def record_shadow_result(
        self,
        intervention_id: str,
        shadow: ShadowRun,
    ) -> Optional[InterventionDelta]:
        """
        Update an existing delta with shadow run (Run B) results.
        Call when TwinRunner.spawn_shadow() completes.
        """
        if intervention_id not in self._deltas:
            logger.warning(f"DeltaTracker: unknown intervention_id {intervention_id}")
            return None

        delta = self._deltas[intervention_id]

        if not shadow.completed or shadow.error:
            logger.warning(
                f"DeltaTracker: shadow run {shadow.run_id} not completed "
                f"or errored — skipping delta update"
            )
            return delta

        delta.run_b_final_loss = shadow.final_loss
        delta.run_b_final_grad_norm = shadow.final_grad_norm
        delta.run_b_fid = shadow.fid_score
        delta.shadow_available = True
        delta.shadow_run_id = shadow.run_id

        # Recompute deltas
        delta.loss_delta = delta.run_a_final_loss - delta.run_b_final_loss
        delta.grad_norm_delta = delta.run_a_final_grad_norm - delta.run_b_final_grad_norm
        if delta.run_a_fid is not None and delta.run_b_fid is not None:
            delta.fid_delta = delta.run_a_fid - delta.run_b_fid

        logger.info(
            f"DeltaTracker: updated {intervention_id}. "
            f"loss_delta={delta.loss_delta:+.4f}, "
            f"fid_delta={delta.fid_delta:+.2f}, "
            f"agent_helped={delta.agent_helped}"
        )

        self._save_delta(delta)
        return delta

    @property
    def all_deltas(self) -> List[InterventionDelta]:
        return list(self._deltas.values())

    @property
    def completed_deltas(self) -> List[InterventionDelta]:
        """Only deltas with shadow run results available."""
        return [d for d in self._deltas.values() if d.shadow_available]

    @property
    def success_rate(self) -> float:
        """Fraction of interventions where agent helped."""
        completed = self.completed_deltas
        if not completed:
            return 0.0
        return float(np.mean([d.agent_helped for d in completed]))

    def _save_delta(self, delta: InterventionDelta) -> None:
        path = self.log_dir / f"{delta.intervention_id}.json"
        try:
            path.write_text(json.dumps(delta.to_dict(), indent=2))
        except Exception as exc:
            logger.debug(f"Could not save delta: {exc}")

    def save_all(self) -> Path:
        """Save all deltas to a single JSONL file."""
        path = self.log_dir / f"all_deltas_{int(time.time())}.jsonl"
        with open(path, "w") as f:
            for delta in self._deltas.values():
                f.write(json.dumps(delta.to_dict()) + "\n")
        logger.info(f"DeltaTracker: saved {len(self._deltas)} deltas to {path}")
        return path
