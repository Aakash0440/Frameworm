"""
AgentControlPlugin — extends your existing plugin system.

Drop this in frameworm_plugins/ OR the agent registers it
directly when the agent starts. Either works.

Responsibilities:
    1. Hold a reference to the live Trainer instance
       (set by the training loop via check_agent_commands hook)
    2. Write metrics to /tmp/frameworm_agent_metrics.json
       every N steps (for LOCAL mode MetricStream)
    3. Receive action commands from the ReAct agent
    4. Execute them via ActionExecutor
    5. Track last checkpoint step/loss for prompt context

The command channel uses a simple thread-safe dict in memory.
No sockets, no Redis — the agent and training loop are in the
same process or the agent is a daemon thread.

Hooks into:
    plugins/hooks.py        → FramewormPlugin base class
    training/trainer.py     → 4-line hook (see FILE 10 below)
    integrations/notifications.py → Slack
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from agent.control.actions import ActionExecutor, ActionResult
from agent.control.cooldown import CooldownManager
from agent.react.action_parser import ActionType, ParsedAction
from agent.classifier.anomaly_types import AnomalyEvent

logger = logging.getLogger(__name__)

# Metrics file path — read by MetricStream in LOCAL mode
METRICS_FILE = Path("/tmp/frameworm_agent_metrics.json")


class AgentControlPlugin:
    """
    Control interface between the ReAct agent and the training loop.

    Extends your plugin interface pattern.
    Does NOT extend FramewormPlugin directly to avoid import
    cycle issues — but follows the same interface so it can
    be registered manually or auto-discovered.

    Usage in training loop (see FILE 10):
        # In training/trainer.py, end of _train_step():
        if hasattr(self, '_agent_plugin') and step % 50 == 0:
            self._agent_plugin.check_commands()

    Usage from agent:
        plugin = AgentControlPlugin(cooldown=CooldownManager())
        agent = FramewormAgent(..., control=plugin)
        agent.start()
        # Plugin gets trainer reference when training registers it
    """

    def __init__(self, cooldown: Optional[CooldownManager] = None) -> None:
        self.cooldown = cooldown or CooldownManager()
        self.executor: Optional[ActionExecutor] = None

        # Checkpoint tracking (updated by write_metrics)
        self.last_checkpoint_step: int = 0
        self.last_checkpoint_loss: float = float("inf")

        # Pause flag (set by pause action, read by check_commands)
        self._pause_requested = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────
    # Called from training loop (the 4-line hook)
    # ──────────────────────────────────────────────

    def register_trainer(self, trainer) -> None:
        """
        Called once when training starts to bind the trainer.
        The trainer calls this on itself if _agent_plugin is set.
        """
        self.executor = ActionExecutor(trainer_ref=trainer)
        # Also make executor's trainer_ref available for pause check
        trainer._agent_pause_requested = False
        logger.info("AgentControlPlugin: trainer registered.")

    def check_commands(self) -> None:
        """
        Called from training loop every N steps.
        Writes current metrics to local file for MetricStream.
        Checks for pause flag.

        Add to training/trainer.py (see FILE 10).
        """
        # Write metrics to file (for LOCAL mode MetricStream)
        self._write_metrics()

        # Handle pause
        if self._pause_requested:
            logger.warning("AgentControlPlugin: training paused by agent. Waiting...")
            import time
            while self._pause_requested:
                time.sleep(2.0)
            logger.info("AgentControlPlugin: training resumed.")

    def _write_metrics(self) -> None:
        """Write current training metrics to shared JSON file."""
        if self.executor is None or self.executor.trainer_ref is None:
            return

        trainer = self.executor.trainer_ref
        try:
            metrics = {
                "step": getattr(trainer, "global_step", 0),
                "loss": float(getattr(trainer, "_last_loss", 0.0)),
                "grad_norm": float(getattr(trainer, "_last_grad_norm", 0.0)),
                "lr": self._get_current_lr(trainer),
                "epoch": getattr(trainer, "current_epoch", 0),
                "weight_update_ratio": float(getattr(trainer, "_weight_update_ratio", 0.0)),
            }
            METRICS_FILE.write_text(json.dumps(metrics))
        except Exception as exc:
            logger.debug(f"_write_metrics failed: {exc}")

    def _get_current_lr(self, trainer) -> float:
        """Safely get current LR from trainer's optimizer."""
        try:
            if hasattr(trainer, "optimizer"):
                return float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        return 0.0

    # ──────────────────────────────────────────────
    # Called from ReAct agent
    # ──────────────────────────────────────────────

    def execute(self, action: ParsedAction, event: AnomalyEvent) -> bool:
        """
        Execute a ParsedAction on the training loop.
        Returns True if execution succeeded.
        """
        if self.executor is None:
            logger.warning(
                "AgentControlPlugin: no executor — "
                "trainer not registered yet. Skipping action."
            )
            return False

        result = self._dispatch(action)
        logger.info(f"Action result: {result}")
        return result.success

    def _dispatch(self, action: ParsedAction) -> ActionResult:
        """Route action to correct executor method."""
        t = action.action_type
        p = action.params

        if t == ActionType.WATCH:
            return self.executor.watch(steps=p.get("steps", 50))

        elif t == ActionType.ADJUST_LR:
            return self.executor.adjust_lr(factor=float(p.get("factor", 0.5)))

        elif t == ActionType.ROLLBACK:
            step = p.get("step")
            if step == "None" or step is None:
                step = self.last_checkpoint_step or None
            return self.executor.rollback_checkpoint(step=step)

        elif t == ActionType.SWAP_SCHEDULER:
            return self.executor.swap_scheduler(name=str(p.get("name", "cosine")))

        elif t == ActionType.PAUSE:
            with self._lock:
                self._pause_requested = True
            return self.executor.pause_training()

        elif t == ActionType.ALERT:
            msg = str(p.get("message", "FRAMEWORM-AGENT: anomaly detected."))
            return self.executor.send_alert(msg)

        else:
            logger.warning(f"Unknown action type: {t}")
            return self.executor.watch()

    def send_alert(self, message: str) -> None:
        """Direct alert shortcut — used for escalation."""
        if self.executor:
            self.executor.send_alert(message)

    def resume_training(self) -> None:
        """Unpause training — can be called externally."""
        with self._lock:
            self._pause_requested = False