"""
Rule-based anomaly classifier.

Fast, deterministic, zero LLM cost.
Runs every tick. Only escalates to the ReAct agent when non-HEALTHY.

Thresholds are configurable via your YAML config system.
Add this block to configs/base.yaml:

    agent:
      grad_explosion_threshold: 10.0
      vanishing_grad_threshold: 0.001
      loss_spike_z_score: 3.0
      plateau_score_threshold: 0.05
      plateau_min_steps: 100
      divergence_score_threshold: 0.75
      divergence_min_steps: 50
      oscillation_score_threshold: 0.01
      early_training_lenience: true
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.observer.signal_extractor import SignalSnapshot

logger = logging.getLogger(__name__)


@dataclass
class RuleEngineConfig:
    """
    All thresholds for the rule engine.
    Load from your YAML config or use defaults.

    To load from FRAMEWORM config:
        from core.config import load_config
        cfg = load_config("configs/base.yaml")
        agent_cfg = cfg.get("agent", {})
        rule_config = RuleEngineConfig(**agent_cfg)
    """
    # Gradient explosion: grad_norm > this → GRADIENT_EXPLOSION
    grad_explosion_threshold: float = 10.0

    # Vanishing gradients: grad_norm < this → VANISHING_GRAD
    vanishing_grad_threshold: float = 0.001

    # Loss spike: z_score > this → LOSS_SPIKE
    loss_spike_z_score: float = 3.0

    # Plateau: plateau_score < this for min_steps → PLATEAU
    plateau_score_threshold: float = 0.05
    plateau_min_steps: int = 100

    # Divergence: divergence_score > this for min_steps → DIVERGENCE
    divergence_score_threshold: float = 0.75
    divergence_min_steps: int = 50

    # Oscillation: oscillation_score > this → OSCILLATING
    oscillation_score_threshold: float = 0.01

    # If True, relax thresholds during early training (first 10%)
    early_training_lenience: bool = True

    # Lenience multiplier applied during early training
    early_lenience_factor: float = 2.0


class RuleEngine:
    """
    Classifies a SignalSnapshot into an AnomalyEvent using
    deterministic threshold rules.

    Rules are evaluated in priority order. First match wins.
    Multiple anomalies firing simultaneously are enqueued
    in priority order by AnomalyPriorityQueue.

    Args:
        config:         RuleEngineConfig with thresholds.
                        If None, loads defaults.

    Usage:
        engine = RuleEngine()
        signals = extractor.extract(window)
        events = engine.classify(signals)
        # events is a list — usually 0 or 1 items, occasionally 2+
        if events:
            print(events[0])  # highest priority anomaly
    """

    def __init__(self, config: Optional[RuleEngineConfig] = None) -> None:
        self.config = config or RuleEngineConfig()
        self._plateau_counter: int = 0
        self._divergence_counter: int = 0

    def classify(self, signals: SignalSnapshot) -> List[AnomalyEvent]:
        """
        Run all rules against signals.
        Returns list of AnomalyEvents, sorted by priority (highest first).
        Empty list means HEALTHY.
        """
        cfg = self.config
        lenience = cfg.early_lenience_factor if (
            signals.is_early_training and cfg.early_training_lenience
        ) else 1.0

        events: List[AnomalyEvent] = []

        # ── Rule 1: Gradient Explosion ─────────────────────────────
        threshold = cfg.grad_explosion_threshold
        if signals.grad_norm_current > threshold:
            sev = self._grad_explosion_severity(signals.grad_norm_current, threshold)
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                severity=sev,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                loss_z_score=signals.loss_z_score,
                grad_norm_z_score=signals.grad_norm_z_score,
                triggered_rule="grad_norm > grad_explosion_threshold",
                triggered_value=signals.grad_norm_current,
                threshold_value=threshold,
            ))
            logger.warning(
                f"[Step {signals.step}] GRADIENT_EXPLOSION detected: "
                f"grad_norm={signals.grad_norm_current:.3f} > {threshold}"
            )

        # ── Rule 2: Vanishing Gradients ────────────────────────────
        vanish_threshold = cfg.vanishing_grad_threshold * lenience
        if signals.grad_norm_current < vanish_threshold and not signals.is_early_training:
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.VANISHING_GRAD,
                severity=Severity.HIGH,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                triggered_rule="grad_norm < vanishing_grad_threshold",
                triggered_value=signals.grad_norm_current,
                threshold_value=vanish_threshold,
            ))
            logger.warning(
                f"[Step {signals.step}] VANISHING_GRAD detected: "
                f"grad_norm={signals.grad_norm_current:.6f} < {vanish_threshold}"
            )

        # ── Rule 3: Loss Spike ─────────────────────────────────────
        spike_z = cfg.loss_spike_z_score * lenience
        if signals.loss_z_score > spike_z:
            sev = self._spike_severity(signals.loss_z_score, spike_z)
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.LOSS_SPIKE,
                severity=sev,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                loss_z_score=signals.loss_z_score,
                triggered_rule="loss_z_score > loss_spike_z_score",
                triggered_value=signals.loss_z_score,
                threshold_value=spike_z,
            ))
            logger.warning(
                f"[Step {signals.step}] LOSS_SPIKE detected: "
                f"z={signals.loss_z_score:.2f} > {spike_z}"
            )

        # ── Rule 4: Divergence (sustained loss increase) ───────────
        div_threshold = cfg.divergence_score_threshold
        if signals.divergence_score > div_threshold:
            self._divergence_counter += 1
        else:
            self._divergence_counter = 0

        if self._divergence_counter >= cfg.divergence_min_steps:
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.DIVERGENCE,
                severity=Severity.HIGH,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                divergence_score=signals.divergence_score,
                triggered_rule="divergence_score > threshold for min_steps",
                triggered_value=signals.divergence_score,
                threshold_value=div_threshold,
            ))
            logger.warning(
                f"[Step {signals.step}] DIVERGENCE detected: "
                f"score={signals.divergence_score:.2f} for "
                f"{self._divergence_counter} steps"
            )

        # ── Rule 5: Oscillation ────────────────────────────────────
        osc_threshold = cfg.oscillation_score_threshold * lenience
        if (signals.oscillation_score > osc_threshold
                and abs(signals.loss_delta) < signals.loss_rolling_std * 0.5):
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.OSCILLATING,
                severity=Severity.MEDIUM,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                triggered_rule="oscillation_score > threshold",
                triggered_value=signals.oscillation_score,
                threshold_value=osc_threshold,
            ))

        # ── Rule 6: Plateau (no progress) ─────────────────────────
        plateau_threshold = cfg.plateau_score_threshold * lenience
        if signals.plateau_score < plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        if self._plateau_counter >= cfg.plateau_min_steps:
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.PLATEAU,
                severity=Severity.MEDIUM,
                step=signals.step,
                loss=signals.loss_raw,
                grad_norm=signals.grad_norm_current,
                lr=signals.lr_current,
                plateau_score=signals.plateau_score,
                triggered_rule="plateau_score < threshold for min_steps",
                triggered_value=signals.plateau_score,
                threshold_value=plateau_threshold,
            ))
            logger.info(
                f"[Step {signals.step}] PLATEAU detected: "
                f"no progress for {self._plateau_counter} steps"
            )

        # Sort by priority (highest urgency first)
        events.sort()
        return events

    def reset_counters(self) -> None:
        """Reset persistent counters — call after a rollback."""
        self._plateau_counter = 0
        self._divergence_counter = 0

    # ──────────────────────────────────────────────
    # Severity helpers
    # ──────────────────────────────────────────────

    def _grad_explosion_severity(self, value: float, threshold: float) -> Severity:
        ratio = value / threshold
        if ratio > 10:
            return Severity.CRITICAL
        if ratio > 5:
            return Severity.HIGH
        if ratio > 2:
            return Severity.MEDIUM
        return Severity.LOW

    def _spike_severity(self, z: float, threshold: float) -> Severity:
        if z > threshold * 4:
            return Severity.CRITICAL
        if z > threshold * 2.5:
            return Severity.HIGH
        if z > threshold * 1.5:
            return Severity.MEDIUM
        return Severity.LOW
