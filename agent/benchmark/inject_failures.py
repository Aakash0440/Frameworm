"""
Synthetic failure injection for controlled benchmarking.

Injects deterministic, reproducible anomalies into a
training run so the benchmark suite can measure:
    - Does the agent detect it? (detection rate)
    - Does the agent fix it?   (resolution rate)
    - How long does recovery take? (time-to-recovery)
    - How much compute is wasted? (overhead)

12 scenarios = 4 anomaly types × 3 severity levels:

    GRADIENT_EXPLOSION × {MILD, MODERATE, SEVERE}
    LOSS_SPIKE         × {MILD, MODERATE, SEVERE}
    PLATEAU            × {MILD, MODERATE, SEVERE}
    DIVERGENCE         × {MILD, MODERATE, SEVERE}

Each scenario is a function that modifies the trainer's
state at a specific step to create the failure condition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnomalySeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class FailureScenario:
    """
    One reproducible failure scenario.

    name:           Unique identifier (used in results table)
    anomaly_type:   Which failure type this injects
    severity:       How bad the injected failure is
    inject_at_step: Which training step to trigger the failure
    inject_fn:      Function that modifies trainer/metrics to create failure
    expected_detection_steps: How many steps after injection should
                              the classifier detect it (used to measure
                              detection latency)
    """

    name: str
    anomaly_type: str
    severity: AnomalySeverity
    inject_at_step: int
    inject_fn: Callable  # inject_fn(trainer, metrics_dict) -> None
    expected_detection_steps: int = 10
    description: str = ""


class FailureInjector:
    """
    Injects synthetic failures into training for benchmarking.

    Usage:
        injector = FailureInjector(trainer)
        injector.register(SCENARIO_REGISTRY["gradient_explosion_severe"])
        # In your training loop at step N:
        injector.check_and_inject(step, current_metrics)
    """

    def __init__(self, trainer_ref=None) -> None:
        self.trainer_ref = trainer_ref
        self._scheduled: List[FailureScenario] = []
        self._injected: List[str] = []

    def register(self, scenario: FailureScenario) -> None:
        self._scheduled.append(scenario)

    def register_all(self, scenarios: List[FailureScenario]) -> None:
        self._scheduled.extend(scenarios)

    def check_and_inject(self, step: int, metrics: dict) -> Optional[FailureScenario]:
        """
        Check if any scenario should fire at this step.
        Returns the fired scenario or None.
        """
        for scenario in self._scheduled:
            if step == scenario.inject_at_step and scenario.name not in self._injected:
                logger.info(f"[Benchmark] Injecting {scenario.name} at step {step}")
                try:
                    scenario.inject_fn(self.trainer_ref, metrics)
                    self._injected.append(scenario.name)
                    return scenario
                except Exception as exc:
                    logger.warning(f"[Benchmark] Injection failed for {scenario.name}: {exc}")
        return None

    @property
    def injected_count(self) -> int:
        return len(self._injected)


# ── Injection functions ───────────────────────────────────────────
# Each returns a function(trainer, metrics) -> None
# that modifies the training state to create the failure.


def _inject_grad_explosion(multiplier: float):
    """Multiply gradient norms by multiplier to simulate explosion."""

    def fn(trainer, metrics):
        metrics["grad_norm"] = metrics.get("grad_norm", 2.0) * multiplier
        if trainer is not None and hasattr(trainer, "model"):
            try:
                import torch

                for param in trainer.model.parameters():
                    if param.grad is not None:
                        param.grad.data *= multiplier
            except Exception:
                pass

    return fn


def _inject_loss_spike(spike_magnitude: float):
    """Add a large positive value to loss."""

    def fn(trainer, metrics):
        metrics["loss"] = metrics.get("loss", 1.0) + spike_magnitude
        if trainer is not None:
            trainer._last_loss = metrics["loss"]

    return fn


def _inject_plateau(flatness: float):
    """Freeze loss at current value (simulate stuck training)."""
    cached = {}

    def fn(trainer, metrics):
        if "loss" not in cached:
            cached["loss"] = metrics.get("loss", 0.5)
        metrics["loss"] = cached["loss"] + np.random.normal(0, flatness)
        metrics["grad_norm"] = 0.001 + np.random.uniform(0, 0.01)
        if trainer is not None:
            trainer._last_loss = metrics["loss"]
            trainer._last_grad_norm = metrics["grad_norm"]

    return fn


def _inject_divergence(lr_multiplier: float):
    """Spike LR to cause sustained divergence."""

    def fn(trainer, metrics):
        if trainer is not None and hasattr(trainer, "optimizer"):
            try:
                for g in trainer.optimizer.param_groups:
                    g["lr"] *= lr_multiplier
                metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
            except Exception:
                pass
        # Simulate rising loss
        metrics["loss"] = metrics.get("loss", 0.5) * (1.0 + 0.1 * lr_multiplier)

    return fn


# ── Scenario registry ─────────────────────────────────────────────
# 12 scenarios: 4 types × 3 severities

SCENARIO_REGISTRY: Dict[str, FailureScenario] = {
    # Gradient explosion
    "grad_explosion_mild": FailureScenario(
        name="grad_explosion_mild",
        anomaly_type="GRADIENT_EXPLOSION",
        severity=AnomalySeverity.MILD,
        inject_at_step=500,
        inject_fn=_inject_grad_explosion(multiplier=6.0),
        expected_detection_steps=3,
        description="Gradient norm ×6 — borderline explosion",
    ),
    "grad_explosion_moderate": FailureScenario(
        name="grad_explosion_moderate",
        anomaly_type="GRADIENT_EXPLOSION",
        severity=AnomalySeverity.MODERATE,
        inject_at_step=500,
        inject_fn=_inject_grad_explosion(multiplier=15.0),
        expected_detection_steps=1,
        description="Gradient norm ×15 — clear explosion",
    ),
    "grad_explosion_severe": FailureScenario(
        name="grad_explosion_severe",
        anomaly_type="GRADIENT_EXPLOSION",
        severity=AnomalySeverity.SEVERE,
        inject_at_step=500,
        inject_fn=_inject_grad_explosion(multiplier=50.0),
        expected_detection_steps=1,
        description="Gradient norm ×50 — catastrophic explosion",
    ),
    # Loss spike
    "loss_spike_mild": FailureScenario(
        name="loss_spike_mild",
        anomaly_type="LOSS_SPIKE",
        severity=AnomalySeverity.MILD,
        inject_at_step=300,
        inject_fn=_inject_loss_spike(spike_magnitude=0.5),
        expected_detection_steps=5,
        description="Loss +0.5 above baseline — borderline spike",
    ),
    "loss_spike_moderate": FailureScenario(
        name="loss_spike_moderate",
        anomaly_type="LOSS_SPIKE",
        severity=AnomalySeverity.MODERATE,
        inject_at_step=300,
        inject_fn=_inject_loss_spike(spike_magnitude=2.0),
        expected_detection_steps=2,
        description="Loss +2.0 — clear spike",
    ),
    "loss_spike_severe": FailureScenario(
        name="loss_spike_severe",
        anomaly_type="LOSS_SPIKE",
        severity=AnomalySeverity.SEVERE,
        inject_at_step=300,
        inject_fn=_inject_loss_spike(spike_magnitude=8.0),
        expected_detection_steps=1,
        description="Loss +8.0 — catastrophic spike",
    ),
    # Plateau
    "plateau_mild": FailureScenario(
        name="plateau_mild",
        anomaly_type="PLATEAU",
        severity=AnomalySeverity.MILD,
        inject_at_step=200,
        inject_fn=_inject_plateau(flatness=0.005),
        expected_detection_steps=110,
        description="Very slow plateau — tiny noise around fixed loss",
    ),
    "plateau_moderate": FailureScenario(
        name="plateau_moderate",
        anomaly_type="PLATEAU",
        severity=AnomalySeverity.MODERATE,
        inject_at_step=200,
        inject_fn=_inject_plateau(flatness=0.001),
        expected_detection_steps=105,
        description="Moderate plateau — minimal variance",
    ),
    "plateau_severe": FailureScenario(
        name="plateau_severe",
        anomaly_type="PLATEAU",
        severity=AnomalySeverity.SEVERE,
        inject_at_step=200,
        inject_fn=_inject_plateau(flatness=0.0001),
        expected_detection_steps=100,
        description="Severe plateau — completely frozen loss",
    ),
    # Divergence
    "divergence_mild": FailureScenario(
        name="divergence_mild",
        anomaly_type="DIVERGENCE",
        severity=AnomalySeverity.MILD,
        inject_at_step=400,
        inject_fn=_inject_divergence(lr_multiplier=3.0),
        expected_detection_steps=55,
        description="LR ×3 causing slow divergence",
    ),
    "divergence_moderate": FailureScenario(
        name="divergence_moderate",
        anomaly_type="DIVERGENCE",
        severity=AnomalySeverity.MODERATE,
        inject_at_step=400,
        inject_fn=_inject_divergence(lr_multiplier=8.0),
        expected_detection_steps=52,
        description="LR ×8 causing clear divergence",
    ),
    "divergence_severe": FailureScenario(
        name="divergence_severe",
        anomaly_type="DIVERGENCE",
        severity=AnomalySeverity.SEVERE,
        inject_at_step=400,
        inject_fn=_inject_divergence(lr_multiplier=20.0),
        expected_detection_steps=50,
        description="LR ×20 causing immediate divergence",
    ),
}
