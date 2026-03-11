"""
do-Intervention: replay training with one variable frozen.

Approximates Pearl's do-calculus for neural network training.

The idea:
    1. Anomaly detected at step N
    2. Roll back to checkpoint at step N-K
    3. Replay K steps with variable X held fixed at its baseline
    4. If anomaly disappears → X is confirmed root cause
    5. Report: "Root cause confirmed: {X} caused {anomaly}"

Variables that can be frozen:
    - batch_sequence:   replay with different (shuffled) batches
    - gradient_clip:    add aggressive clipping to isolate grad explosion
    - lr_freeze:        hold LR constant to test scheduler effect
    - layer_freeze:     freeze specific layer weights to test contribution

This is computationally expensive — only runs on CONFIRMED anomalies
(not predictions). Typically takes 30–120 seconds per intervention.

Hooks into:
    training/trainer.py    → load_checkpoint + replay loop
    training/callbacks.py  → temporary callback injection
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FreezeVariable(Enum):
    """Which variable to hold fixed during do-intervention replay."""

    BATCH_SEQUENCE = auto()  # reshuffle batches to replace suspicious batch
    GRADIENT_CLIP = auto()  # add aggressive gradient clipping
    LR_FREEZE = auto()  # hold LR constant at its value before anomaly
    LAYER_FREEZE = auto()  # freeze weights of a specific layer
    NONE = auto()  # baseline replay (no freeze) for comparison


@dataclass
class ReplayResult:
    """
    Outcome of one do-intervention replay.

    anomaly_present:  Did the anomaly still appear in the replay?
    If False with freeze X → X is confirmed root cause.
    """

    freeze_variable: FreezeVariable
    freeze_value: Any  # what value was held fixed

    anomaly_present: bool  # did anomaly reoccur in replay?
    replay_loss_mean: float  # mean loss during replay
    replay_grad_norm_mean: float  # mean grad norm during replay
    replay_steps: int  # how many steps were replayed
    replay_duration_seconds: float

    # Comparison to baseline replay (no freeze)
    baseline_loss_mean: float = 0.0
    loss_delta_from_baseline: float = 0.0  # positive = replay was worse

    confirmed_root_cause: bool = False

    def __repr__(self) -> str:
        status = "CONFIRMED ROOT CAUSE" if self.confirmed_root_cause else "not root cause"
        return (
            f"ReplayResult(freeze={self.freeze_variable.name}, "
            f"anomaly={'YES' if self.anomaly_present else 'NO'}, "
            f"{status})"
        )


class DoIntervention:
    """
    Runs do-calculus style interventions on a live training job.

    Rolls back to a pre-anomaly checkpoint, replays with one
    variable frozen, and observes whether the anomaly disappears.

    Args:
        trainer_ref:        Reference to the FRAMEWORM Trainer.
        replay_steps:       How many steps to replay (default 50).
        checkpoint_dir:     Where checkpoints are stored.
        max_replay_seconds: Timeout per replay (default 120s).
    """

    def __init__(
        self,
        trainer_ref=None,
        replay_steps: int = 50,
        checkpoint_dir: str = "checkpoints",
        max_replay_seconds: float = 120.0,
    ) -> None:
        self.trainer_ref = trainer_ref
        self.replay_steps = replay_steps
        self.checkpoint_dir = checkpoint_dir
        self.max_replay_seconds = max_replay_seconds

    def run(
        self,
        anomaly_step: int,
        pre_anomaly_checkpoint_step: int,
        candidate_variables: Optional[List[FreezeVariable]] = None,
    ) -> List[ReplayResult]:
        """
        Run do-interventions for each candidate variable.

        Args:
            anomaly_step:                   Step where anomaly was detected.
            pre_anomaly_checkpoint_step:    Step to roll back to before replay.
            candidate_variables:            Which variables to freeze.
                                            If None, tests all by default.

        Returns:
            List of ReplayResults, one per variable tested.
            Sorted so confirmed root causes appear first.
        """
        if self.trainer_ref is None:
            logger.warning("DoIntervention: no trainer_ref — cannot replay")
            return []

        variables = candidate_variables or [
            FreezeVariable.BATCH_SEQUENCE,
            FreezeVariable.GRADIENT_CLIP,
            FreezeVariable.LR_FREEZE,
        ]

        results = []

        # First: baseline replay (no freeze) to get reference loss
        baseline = self._replay(
            checkpoint_step=pre_anomaly_checkpoint_step,
            freeze=FreezeVariable.NONE,
            freeze_value=None,
        )
        results.append(baseline)

        # Then: one replay per candidate variable
        for variable in variables:
            freeze_value = self._get_freeze_value(variable)
            result = self._replay(
                checkpoint_step=pre_anomaly_checkpoint_step,
                freeze=variable,
                freeze_value=freeze_value,
            )

            # Confirmed root cause = anomaly disappears when variable frozen
            result.baseline_loss_mean = baseline.replay_loss_mean
            result.loss_delta_from_baseline = result.replay_loss_mean - baseline.replay_loss_mean
            result.confirmed_root_cause = not result.anomaly_present and baseline.anomaly_present

            results.append(result)
            logger.info(f"DoIntervention: {result}")

            if result.confirmed_root_cause:
                logger.info(f"ROOT CAUSE CONFIRMED: {variable.name} " f"at step {anomaly_step}")
                # Stop after first confirmed root cause (efficiency)
                break

        # Restore original checkpoint after all replays
        self._restore_checkpoint(pre_anomaly_checkpoint_step)

        return results

    # ── Private replay engine ─────────────────────────────────────

    def _replay(
        self,
        checkpoint_step: int,
        freeze: FreezeVariable,
        freeze_value: Any,
    ) -> ReplayResult:
        """Run one replay episode."""
        start_time = time.monotonic()

        # Load pre-anomaly checkpoint
        loaded = self._load_checkpoint(checkpoint_step)
        if not loaded:
            return ReplayResult(
                freeze_variable=freeze,
                freeze_value=freeze_value,
                anomaly_present=True,  # assume worst case
                replay_loss_mean=float("inf"),
                replay_grad_norm_mean=float("inf"),
                replay_steps=0,
                replay_duration_seconds=0.0,
            )

        # Apply freeze
        restore_fn = self._apply_freeze(freeze, freeze_value)

        # Run replay steps
        losses = []
        grad_norms = []
        anomaly_detected = False

        try:
            from agent.classifier.rule_engine import RuleEngine
            from agent.observer.rolling_window import (MetricSnapshot,
                                                       RollingWindow)
            from agent.observer.signal_extractor import SignalExtractor

            replay_window = RollingWindow(size=self.replay_steps + 10)
            replay_extractor = SignalExtractor()
            replay_engine = RuleEngine()

            for step_offset in range(self.replay_steps):
                elapsed = time.monotonic() - start_time
                if elapsed > self.max_replay_seconds:
                    logger.warning("DoIntervention: timeout during replay")
                    break

                # Run one training step
                step_metrics = self._run_single_step(step_offset)
                if step_metrics is None:
                    break

                loss = step_metrics.get("loss", 0.0)
                grad_norm = step_metrics.get("grad_norm", 0.0)
                lr = step_metrics.get("lr", 0.0)

                losses.append(loss)
                grad_norms.append(grad_norm)

                replay_window.push(
                    MetricSnapshot(
                        step=checkpoint_step + step_offset,
                        loss=loss,
                        grad_norm=grad_norm,
                        lr=lr,
                    )
                )

                if replay_window.is_ready:
                    signals = replay_extractor.extract(replay_window)
                    if signals:
                        events = replay_engine.classify(signals)
                        if events:
                            anomaly_detected = True
                            break

        except Exception as exc:
            logger.error(f"DoIntervention._replay error: {exc}", exc_info=True)
            anomaly_detected = True  # conservative assumption

        finally:
            # Restore frozen variable
            if restore_fn:
                try:
                    restore_fn()
                except Exception as exc:
                    logger.warning(f"DoIntervention: restore_fn failed: {exc}")

        duration = time.monotonic() - start_time
        return ReplayResult(
            freeze_variable=freeze,
            freeze_value=freeze_value,
            anomaly_present=anomaly_detected,
            replay_loss_mean=float(np.mean(losses)) if losses else float("inf"),
            replay_grad_norm_mean=float(np.mean(grad_norms)) if grad_norms else 0.0,
            replay_steps=len(losses),
            replay_duration_seconds=duration,
        )

    def _run_single_step(self, step_offset: int) -> Optional[Dict[str, float]]:
        """
        Run one training step on the trainer and return metrics.
        Hooks into your trainer's step method.
        """
        trainer = self.trainer_ref
        try:
            # Try your trainer's step interface
            if hasattr(trainer, "_train_step"):
                # Need a batch — get next from dataloader
                if hasattr(trainer, "train_dataloader"):
                    try:
                        batch = next(iter(trainer.train_dataloader))
                        loss_val = trainer._train_step(batch, step_offset)
                        return {
                            "loss": float(loss_val) if loss_val else trainer._last_loss,
                            "grad_norm": getattr(trainer, "_last_grad_norm", 0.0),
                            "lr": trainer.optimizer.param_groups[0]["lr"],
                        }
                    except StopIteration:
                        return None
            # Fallback: read current metrics without stepping
            return {
                "loss": getattr(trainer, "_last_loss", 0.0),
                "grad_norm": getattr(trainer, "_last_grad_norm", 0.0),
                "lr": (
                    trainer.optimizer.param_groups[0]["lr"]
                    if hasattr(trainer, "optimizer")
                    else 0.0
                ),
            }
        except Exception as exc:
            logger.debug(f"_run_single_step failed: {exc}")
            return None

    def _get_freeze_value(self, variable: FreezeVariable) -> Any:
        """Get the current value of a variable before freezing it."""
        trainer = self.trainer_ref
        if variable == FreezeVariable.LR_FREEZE:
            if hasattr(trainer, "optimizer"):
                return trainer.optimizer.param_groups[0]["lr"]
        elif variable == FreezeVariable.GRADIENT_CLIP:
            return 1.0  # aggressive clip value
        elif variable == FreezeVariable.BATCH_SEQUENCE:
            return "reshuffled"
        return None

    def _apply_freeze(self, variable: FreezeVariable, value: Any) -> Optional[Callable]:
        """
        Apply freeze to the variable. Returns a restore function.
        The restore function is called after replay completes.
        """
        trainer = self.trainer_ref
        restore_fn = None

        try:
            if variable == FreezeVariable.LR_FREEZE and hasattr(trainer, "optimizer"):
                # Save current LR, set to freeze value
                original_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
                for g in trainer.optimizer.param_groups:
                    g["lr"] = value

                def restore_fn():
                    for g, lr in zip(trainer.optimizer.param_groups, original_lrs):
                        g["lr"] = lr

            elif variable == FreezeVariable.GRADIENT_CLIP:
                # Inject aggressive gradient clipping callback
                original_clip = getattr(trainer, "_gradient_clip_val", None)
                trainer._gradient_clip_val = 0.5  # very aggressive

                def restore_fn():
                    trainer._gradient_clip_val = original_clip

            elif variable == FreezeVariable.BATCH_SEQUENCE:
                # Reseed the dataloader for different batches
                if hasattr(trainer, "train_dataloader"):
                    import torch

                    torch.manual_seed(42)  # deterministic reseed

                def restore_fn():
                    pass  # dataloader state naturally resets

        except Exception as exc:
            logger.warning(f"DoIntervention._apply_freeze failed: {exc}")

        return restore_fn

    def _load_checkpoint(self, step: int) -> bool:
        """Load checkpoint from the given step."""
        import os

        trainer = self.trainer_ref

        # Try step-specific checkpoint first
        paths = [
            f"{self.checkpoint_dir}/step_{step}.pt",
            f"{self.checkpoint_dir}/latest.pt",
            f"{self.checkpoint_dir}/best.pt",
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    if hasattr(trainer, "load_checkpoint"):
                        trainer.load_checkpoint(path)
                    else:
                        import torch

                        state = torch.load(path, map_location="cpu")
                        if hasattr(trainer, "model") and "model" in state:
                            trainer.model.load_state_dict(state["model"])
                    logger.debug(f"DoIntervention: loaded {path}")
                    return True
                except Exception as exc:
                    logger.warning(f"DoIntervention: could not load {path}: {exc}")

        logger.warning(f"DoIntervention: no checkpoint found for step {step}")
        return False

    def _restore_checkpoint(self, step: int) -> None:
        """Restore original state after all replays complete."""
        self._load_checkpoint(step)
