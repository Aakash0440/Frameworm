"""
Low-level action implementations.

Each action is a function that takes the training state
(accessed via AgentControlPlugin) and applies a change.

Hooks into:
    training/trainer.py     → get/set lr, load checkpoint
    training/schedulers.py  → swap scheduler
    integrations/notifications.py → Slack
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Outcome of executing one action."""

    success: bool
    action_name: str
    params: Dict[str, Any]
    message: str = ""
    exception: Optional[str] = None

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ActionResult({status} {self.action_name}({self.params}): {self.message})"


class ActionExecutor:
    """
    Executes agent actions against the live training state.

    All methods are safe — they catch exceptions and return
    ActionResult with success=False rather than crashing the agent.

    Args:
        trainer_ref: Reference to the FRAMEWORM Trainer instance.
                     Set by AgentControlPlugin when training starts.
    """

    def __init__(self, trainer_ref=None) -> None:
        self.trainer_ref = trainer_ref

    # ──────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────

    def adjust_lr(self, factor: float) -> ActionResult:
        """
        Multiply current LR by factor.
        Hooks into training/schedulers.py via trainer.optimizer.
        """
        if self.trainer_ref is None:
            return ActionResult(
                False,
                "adjust_lr",
                {"factor": factor},
                "No trainer reference — training may not have started yet.",
            )
        try:
            optimizer = self.trainer_ref.optimizer
            old_lrs = []
            new_lrs = []
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                new_lr = old_lr * factor
                param_group["lr"] = new_lr
                old_lrs.append(old_lr)
                new_lrs.append(new_lr)

            msg = (
                f"LR adjusted by factor {factor:.2f}: "
                f"{[f'{lr:.2e}' for lr in old_lrs]} → "
                f"{[f'{lr:.2e}' for lr in new_lrs]}"
            )
            logger.info(f"[AgentAction] {msg}")
            return ActionResult(True, "adjust_lr", {"factor": factor}, msg)

        except Exception as exc:
            logger.error(f"adjust_lr failed: {exc}")
            return ActionResult(False, "adjust_lr", {"factor": factor}, exception=str(exc))

    def rollback_checkpoint(self, step: Optional[int] = None) -> ActionResult:
        """
        Load checkpoint from a specific step.
        Uses your existing checkpoint system in training/trainer.py.

        If step=None, loads the most recent checkpoint.
        """
        if self.trainer_ref is None:
            return ActionResult(False, "rollback", {"step": step}, "No trainer reference.")
        try:
            # Try your existing trainer's checkpoint loading
            # FRAMEWORM trainer uses load_checkpoint(path) or resume_from
            checkpoint_dir = "checkpoints"

            if step is None:
                # Load latest.pt — your existing checkpoint
                ckpt_path = f"{checkpoint_dir}/latest.pt"
            else:
                ckpt_path = f"{checkpoint_dir}/step_{step}.pt"

            # Try trainer.load_checkpoint first (your existing method)
            if hasattr(self.trainer_ref, "load_checkpoint"):
                self.trainer_ref.load_checkpoint(ckpt_path)
                msg = f"Rolled back to checkpoint: {ckpt_path}"
            elif hasattr(self.trainer_ref, "resume_from"):
                self.trainer_ref.resume_from(ckpt_path)
                msg = f"Resumed from checkpoint: {ckpt_path}"
            else:
                # Manual fallback — load state dict directly
                import torch

                state = torch.load(ckpt_path, map_location="cpu")
                if hasattr(self.trainer_ref, "model"):
                    self.trainer_ref.model.load_state_dict(state.get("model", state))
                msg = f"Loaded model weights from {ckpt_path}"

            logger.info(f"[AgentAction] {msg}")
            return ActionResult(True, "rollback", {"step": step, "path": ckpt_path}, msg)

        except FileNotFoundError:
            msg = f"Checkpoint not found for step {step}. No rollback performed."
            logger.warning(f"[AgentAction] {msg}")
            return ActionResult(False, "rollback", {"step": step}, msg)
        except Exception as exc:
            logger.error(f"rollback failed: {exc}")
            return ActionResult(False, "rollback", {"step": step}, exception=str(exc))

    def swap_scheduler(self, name: str) -> ActionResult:
        """
        Replace the active scheduler.
        Hooks into training/schedulers.py — your existing scheduler factory.
        """
        if self.trainer_ref is None:
            return ActionResult(False, "swap_scheduler", {"name": name}, "No trainer reference.")
        try:
            from training.schedulers import build_scheduler

            new_scheduler = build_scheduler(
                name=name,
                optimizer=self.trainer_ref.optimizer,
            )
            # Replace in trainer
            if hasattr(self.trainer_ref, "scheduler"):
                self.trainer_ref.scheduler = new_scheduler
            elif hasattr(self.trainer_ref, "lr_scheduler"):
                self.trainer_ref.lr_scheduler = new_scheduler

            msg = f"Scheduler swapped to: {name}"
            logger.info(f"[AgentAction] {msg}")
            return ActionResult(True, "swap_scheduler", {"name": name}, msg)

        except ImportError:
            # Fallback: try torch schedulers directly
            try:
                import torch.optim.lr_scheduler as torch_sched

                scheduler_map = {
                    "cosine": torch_sched.CosineAnnealingLR,
                    "step": torch_sched.StepLR,
                    "plateau": torch_sched.ReduceLROnPlateau,
                }
                cls = scheduler_map.get(name)
                if cls and hasattr(self.trainer_ref, "optimizer"):
                    new_sched = cls(self.trainer_ref.optimizer, T_max=1000)
                    self.trainer_ref.scheduler = new_sched
                    msg = f"Scheduler swapped to {name} (torch fallback)"
                    logger.info(f"[AgentAction] {msg}")
                    return ActionResult(True, "swap_scheduler", {"name": name}, msg)
            except Exception as exc2:
                pass
            return ActionResult(
                False, "swap_scheduler", {"name": name}, "training.schedulers not importable."
            )
        except Exception as exc:
            logger.error(f"swap_scheduler failed: {exc}")
            return ActionResult(False, "swap_scheduler", {"name": name}, exception=str(exc))

    def pause_training(self) -> ActionResult:
        """
        Pause the training loop.
        Sets a flag on the trainer that check_agent_commands() reads.
        """
        if self.trainer_ref is None:
            return ActionResult(False, "pause", {}, "No trainer reference.")
        try:
            self.trainer_ref._agent_pause_requested = True
            msg = "Training pause requested. Waiting for human intervention."
            logger.warning(f"[AgentAction] {msg}")
            return ActionResult(True, "pause", {}, msg)
        except Exception as exc:
            return ActionResult(False, "pause", {}, exception=str(exc))

    def send_alert(self, message: str) -> ActionResult:
        """
        Send Slack alert via your existing integrations/notifications.py.
        """
        try:
            from integrations.notifications import NotificationManager

            notifier = NotificationManager()
            notifier.send(
                title="🤖 FRAMEWORM-AGENT Alert",
                message=message,
                level="warning",
            )
            logger.info(f"[AgentAction] Alert sent: {message}")
            return ActionResult(True, "alert", {"message": message}, "Slack alert sent.")
        except ImportError:
            logger.warning(
                f"integrations.notifications not available. " f"Alert message: {message}"
            )
            return ActionResult(
                False, "alert", {"message": message}, "notifications module not available."
            )
        except Exception as exc:
            logger.error(f"send_alert failed: {exc}")
            return ActionResult(False, "alert", {"message": message}, exception=str(exc))

    def watch(self, steps: int = 50) -> ActionResult:
        """No-op — just monitor for N more steps."""
        msg = f"Watching for {steps} more steps before next action."
        logger.info(f"[AgentAction] {msg}")
        return ActionResult(True, "watch", {"steps": steps}, msg)
