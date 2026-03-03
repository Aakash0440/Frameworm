
"""
Anomaly type definitions and event dataclasses.
Pure Python — no FRAMEWORM or numpy dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class AnomalyType(Enum):
    """
    All detectable training failure modes.

    Priority order (highest = most urgent, resolve first):
        GRADIENT_EXPLOSION  — training will diverge immediately
        DIVERGENCE          — loss trending up consistently
        LOSS_SPIKE          — sudden large jump in loss
        VANISHING_GRAD      — gradients dying, no learning
        OSCILLATING         — LR too high, loss bouncing
        PLATEAU             — no progress for many steps
        HEALTHY             — everything fine, no action needed
    """
    # Ordered from most to least urgent
    GRADIENT_EXPLOSION = auto()
    DIVERGENCE = auto()
    LOSS_SPIKE = auto()
    VANISHING_GRAD = auto()
    OSCILLATING = auto()
    PLATEAU = auto()
    HEALTHY = auto()

    @property
    def priority(self) -> int:
        """Lower number = higher priority."""
        return {
            AnomalyType.GRADIENT_EXPLOSION: 1,
            AnomalyType.DIVERGENCE: 2,
            AnomalyType.LOSS_SPIKE: 3,
            AnomalyType.VANISHING_GRAD: 4,
            AnomalyType.OSCILLATING: 5,
            AnomalyType.PLATEAU: 6,
            AnomalyType.HEALTHY: 99,
        }[self]

    @property
    def is_actionable(self) -> bool:
        """True for anything that should trigger the ReAct agent."""
        return self != AnomalyType.HEALTHY

    def __lt__(self, other: AnomalyType) -> bool:
        return self.priority < other.priority


class Severity(Enum):
    """How bad is it? Used for prompt context and action selection."""
    LOW = "low"         # Borderline threshold, monitor closely
    MEDIUM = "medium"   # Clear anomaly, soft intervention
    HIGH = "high"       # Aggressive intervention needed
    CRITICAL = "critical"  # Immediate rollback or pause


# Human-readable descriptions sent to LLM in prompts
ANOMALY_DESCRIPTIONS: Dict[AnomalyType, str] = {
    AnomalyType.GRADIENT_EXPLOSION: (
        "Gradient norm has exceeded the explosion threshold. "
        "Weights are receiving extreme updates and training will diverge."
    ),
    AnomalyType.DIVERGENCE: (
        "Loss has been consistently increasing over many steps. "
        "Training is moving away from a good solution."
    ),
    AnomalyType.LOSS_SPIKE: (
        "Loss jumped sharply above the rolling mean by several standard deviations. "
        "May be a bad batch, unstable LR, or optimizer issue."
    ),
    AnomalyType.VANISHING_GRAD: (
        "Gradient norm is near zero. Gradients are vanishing — "
        "the network is not learning. Check architecture and LR."
    ),
    AnomalyType.OSCILLATING: (
        "Loss is bouncing up and down without making net progress. "
        "Learning rate is likely too high."
    ),
    AnomalyType.PLATEAU: (
        "Loss has not meaningfully improved over many steps. "
        "Training is stuck — LR may be too low or model has converged."
    ),
    AnomalyType.HEALTHY: "Training is proceeding normally.",
}

# Suggested actions per anomaly type (for LLM prompt context)
SUGGESTED_ACTIONS: Dict[AnomalyType, list] = {
    AnomalyType.GRADIENT_EXPLOSION: ["ADJUST_LR", "ROLLBACK", "PAUSE"],
    AnomalyType.DIVERGENCE: ["ROLLBACK", "ADJUST_LR", "SWAP_SCHEDULER"],
    AnomalyType.LOSS_SPIKE: ["WATCH", "ADJUST_LR", "ROLLBACK"],
    AnomalyType.VANISHING_GRAD: ["ADJUST_LR", "SWAP_SCHEDULER", "ALERT"],
    AnomalyType.OSCILLATING: ["ADJUST_LR", "WATCH"],
    AnomalyType.PLATEAU: ["SWAP_SCHEDULER", "ADJUST_LR", "ALERT"],
    AnomalyType.HEALTHY: ["WATCH"],
}
# Ordered list of failure modes (excludes HEALTHY) — used for one-hot encoding
FAILURE_TYPES = [
    AnomalyType.GRADIENT_EXPLOSION,
    AnomalyType.DIVERGENCE,
    AnomalyType.LOSS_SPIKE,
    AnomalyType.VANISHING_GRAD,
    AnomalyType.OSCILLATING,
    AnomalyType.PLATEAU,
]

@dataclass
class AnomalyEvent:
    """
    A detected anomaly, ready to be sent to the ReAct agent.

    Created by RuleEngine, enqueued in AnomalyPriorityQueue,
    consumed by the ReAct agent loop.
    """
    anomaly_type: AnomalyType
    severity: Severity
    step: int
    detected_at: float = field(default_factory=time.monotonic)

    # Raw signal values at detection time (included in LLM prompt)
    loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    loss_z_score: float = 0.0
    grad_norm_z_score: float = 0.0
    plateau_score: float = 0.0
    divergence_score: float = 0.0

    # Threshold that was breached (for context in prompt)
    triggered_rule: str = ""
    triggered_value: float = 0.0
    threshold_value: float = 0.0

    # Human-readable description for LLM
    @property
    def description(self) -> str:
        return ANOMALY_DESCRIPTIONS[self.anomaly_type]

    @property
    def suggested_actions(self) -> list:
        return SUGGESTED_ACTIONS[self.anomaly_type]

    def __lt__(self, other: AnomalyEvent) -> bool:
        """Priority queue ordering — highest priority first."""
        return self.anomaly_type.priority < other.anomaly_type.priority

    def to_prompt_dict(self) -> dict:
        """Serialize for inclusion in LLM prompt."""
        return {
            "anomaly_type": self.anomaly_type.name,
            "severity": self.severity.value,
            "step": self.step,
            "description": self.description,
            "loss": round(self.loss, 6),
            "grad_norm": round(self.grad_norm, 4),
            "lr": self.lr,
            "loss_z_score": round(self.loss_z_score, 2),
            "triggered_rule": self.triggered_rule,
            "triggered_value": round(self.triggered_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "suggested_actions": self.suggested_actions,
        }