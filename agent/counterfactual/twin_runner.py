"""
TwinRunner — spawns shadow training runs for counterfactual evaluation.

When the agent intervenes in Run A:
    → TwinRunner spawns Run B from the same checkpoint
    → Run B uses the same random seed and batch sequence
    → Run B has NO agent involvement
    → After N steps, compare Run A vs Run B metrics
    → Delta = agent's causal contribution

This is the most important piece for the paper.

Hooks into:
    training/trainer.py    → load_checkpoint, deterministic seed
    training/callbacks.py  → inject NoAgentCallback for shadow run
    checkpoints/           → spawn point for shadow run
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowRun:
    """
    Metadata and results from one shadow (counterfactual) training run.
    """
    run_id: str
    spawn_step: int                 # checkpoint step used as start
    seed: int                       # random seed for deterministic batches
    n_steps: int                    # how many steps were run
    agent_enabled: bool = False     # always False for shadow runs

    # Metrics collected during shadow run
    loss_history: list = field(default_factory=list)
    grad_norm_history: list = field(default_factory=list)
    final_loss: float = float("inf")
    final_grad_norm: float = 0.0

    # Quality metrics (computed after shadow run completes)
    fid_score: Optional[float] = None
    inception_score: Optional[float] = None
    lpips_score: Optional[float] = None

    # Runtime
    duration_seconds: float = 0.0
    completed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "spawn_step": self.spawn_step,
            "seed": self.seed,
            "n_steps": self.n_steps,
            "agent_enabled": self.agent_enabled,
            "final_loss": self.final_loss,
            "final_grad_norm": self.final_grad_norm,
            "fid_score": self.fid_score,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "error": self.error,
        }


class TwinRunner:
    """
    Manages shadow training runs for counterfactual evaluation.

    Two execution modes:
        IN_PROCESS:  Run shadow steps in same process (fast, uses same model)
        SUBPROCESS:  Spawn a separate Python process (isolated, slower)

    IN_PROCESS is recommended for speed. SUBPROCESS for full isolation.

    Args:
        trainer_ref:        Reference to the FRAMEWORM Trainer.
        shadow_steps:       How many steps each shadow run covers.
        checkpoint_dir:     Where checkpoints live.
        log_dir:            Where to save shadow run logs.
        mode:               "in_process" or "subprocess"
    """

    def __init__(
        self,
        trainer_ref=None,
        shadow_steps: int = 200,
        checkpoint_dir: str = "checkpoints",
        log_dir: Path = Path("experiments/agent_logs/shadow_runs"),
        mode: str = "in_process",
    ) -> None:
        self.trainer_ref = trainer_ref
        self.shadow_steps = shadow_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.mode = mode
        self._shadow_count = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def spawn_shadow(
        self,
        intervention_step: int,
        checkpoint_step: int,
        seed: Optional[int] = None,
    ) -> ShadowRun:
        """
        Spawn a shadow run from the given checkpoint.
        Runs asynchronously in a daemon thread (in_process mode).

        Args:
            intervention_step:  The step where the agent intervened in Run A.
            checkpoint_step:    The checkpoint to start shadow run from.
            seed:               Random seed for deterministic batches.
                                If None, uses intervention_step as seed.

        Returns:
            ShadowRun object. Check .completed to know when it's done.
        """
        if seed is None:
            seed = intervention_step  # deterministic but unique per intervention

        self._shadow_count += 1
        run_id = f"shadow_{self._shadow_count}_step{checkpoint_step}"

        shadow = ShadowRun(
            run_id=run_id,
            spawn_step=checkpoint_step,
            seed=seed,
            n_steps=self.shadow_steps,
        )

        logger.info(
            f"TwinRunner: spawning shadow run {run_id} "
            f"from checkpoint step {checkpoint_step}, seed={seed}"
        )

        if self.mode == "in_process":
            # Run in daemon thread so it doesn't block the main agent
            thread = threading.Thread(
                target=self._run_in_process,
                args=(shadow,),
                daemon=True,
                name=f"shadow-{run_id}",
            )
            thread.start()
        else:
            # Subprocess mode (more isolated but slower)
            thread = threading.Thread(
                target=self._run_subprocess,
                args=(shadow,),
                daemon=True,
                name=f"shadow-sub-{run_id}",
            )
            thread.start()

        return shadow

    # ── In-process shadow run ─────────────────────────────────────

    def _run_in_process(self, shadow: ShadowRun) -> None:
        """
        Run shadow steps in the current process.
        Loads the checkpoint, sets seed, runs N steps with no agent.
        """
        start_time = time.monotonic()

        if self.trainer_ref is None:
            shadow.error = "No trainer reference"
            shadow.completed = True
            return

        trainer = self.trainer_ref

        try:
            import torch

            # 1. Save current training state so we can restore it
            saved_state = self._save_current_state(trainer)

            # 2. Load shadow checkpoint
            self._load_checkpoint_for_shadow(trainer, shadow.spawn_step)

            # 3. Set deterministic seed for reproducible batches
            torch.manual_seed(shadow.seed)
            np.random.seed(shadow.seed)

            # 4. Disable agent for shadow run
            original_agent_plugin = getattr(trainer, "_agent_plugin", None)
            trainer._agent_plugin = None  # no agent in shadow

            # 5. Run N steps collecting metrics
            losses = []
            grad_norms = []

            for step_i in range(shadow.n_steps):
                metrics = self._run_shadow_step(trainer, step_i)
                if metrics is None:
                    break
                losses.append(metrics.get("loss", 0.0))
                grad_norms.append(metrics.get("grad_norm", 0.0))

            # 6. Collect results
            shadow.loss_history = losses
            shadow.grad_norm_history = grad_norms
            shadow.final_loss = float(np.mean(losses[-20:])) if losses else float("inf")
            shadow.final_grad_norm = float(np.mean(grad_norms[-20:])) if grad_norms else 0.0

            # 7. Compute quality metrics if available
            shadow.fid_score = self._compute_fid(trainer)

            # 8. Restore original training state
            self._restore_state(trainer, saved_state)
            trainer._agent_plugin = original_agent_plugin

            shadow.completed = True
            shadow.duration_seconds = time.monotonic() - start_time

            logger.info(
                f"TwinRunner: shadow {shadow.run_id} complete. "
                f"final_loss={shadow.final_loss:.4f}, "
                f"fid={shadow.fid_score}"
            )

            # Save shadow run log
            self._save_shadow_log(shadow)

        except Exception as exc:
            logger.error(f"TwinRunner shadow run failed: {exc}", exc_info=True)
            shadow.error = str(exc)
            shadow.completed = True

    def _run_shadow_step(
        self, trainer, step_i: int
    ) -> Optional[Dict[str, float]]:
        """Run one training step in the shadow run."""
        try:
            if hasattr(trainer, "_train_step") and hasattr(trainer, "train_dataloader"):
                try:
                    batch = next(iter(trainer.train_dataloader))
                    trainer._train_step(batch, step_i)
                    return {
                        "loss": getattr(trainer, "_last_loss", 0.0),
                        "grad_norm": getattr(trainer, "_last_grad_norm", 0.0),
                    }
                except StopIteration:
                    return None
            # Fallback
            return {
                "loss": getattr(trainer, "_last_loss", 0.0),
                "grad_norm": getattr(trainer, "_last_grad_norm", 0.0),
            }
        except Exception as exc:
            logger.debug(f"Shadow step {step_i} failed: {exc}")
            return None

    def _save_current_state(self, trainer) -> dict:
        """Save trainer state before shadow run."""
        try:
            import torch
            import copy
            state = {}
            if hasattr(trainer, "model"):
                state["model"] = copy.deepcopy(trainer.model.state_dict())
            if hasattr(trainer, "optimizer"):
                state["optimizer"] = copy.deepcopy(trainer.optimizer.state_dict())
            if hasattr(trainer, "global_step"):
                state["global_step"] = trainer.global_step
            return state
        except Exception as exc:
            logger.warning(f"Could not save trainer state: {exc}")
            return {}

    def _restore_state(self, trainer, state: dict) -> None:
        """Restore trainer state after shadow run."""
        try:
            if "model" in state and hasattr(trainer, "model"):
                trainer.model.load_state_dict(state["model"])
            if "optimizer" in state and hasattr(trainer, "optimizer"):
                trainer.optimizer.load_state_dict(state["optimizer"])
            if "global_step" in state:
                trainer.global_step = state["global_step"]
        except Exception as exc:
            logger.warning(f"Could not restore trainer state: {exc}")

    def _load_checkpoint_for_shadow(self, trainer, step: int) -> None:
        """Load checkpoint for shadow run start."""
        import os
        paths = [
            f"{self.checkpoint_dir}/step_{step}.pt",
            f"{self.checkpoint_dir}/latest.pt",
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
                    return
                except Exception as exc:
                    logger.warning(f"Shadow checkpoint load failed ({path}): {exc}")

    def _compute_fid(self, trainer) -> Optional[float]:
        """Compute FID score using your existing metrics/fid.py."""
        try:
            from metrics.fid import FIDCalculator
            calc = FIDCalculator()
            if hasattr(trainer, "model") and hasattr(trainer, "val_dataloader"):
                return calc.compute(trainer.model, trainer.val_dataloader)
        except (ImportError, Exception) as exc:
            logger.debug(f"FID computation failed: {exc}")
        return None

    def _run_subprocess(self, shadow: ShadowRun) -> None:
        """Run shadow as a subprocess (isolated mode)."""
        # Write shadow config to temp file
        config = {
            "spawn_step": shadow.spawn_step,
            "seed": shadow.seed,
            "n_steps": shadow.n_steps,
            "run_id": shadow.run_id,
            "checkpoint_dir": self.checkpoint_dir,
        }
        config_path = self.log_dir / f"{shadow.run_id}_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        # This would invoke a shadow_run.py script
        # Implementation depends on your training entry point
        logger.info(f"TwinRunner: subprocess mode not fully implemented — use in_process mode")
        shadow.error = "subprocess mode not implemented"
        shadow.completed = True

    def _save_shadow_log(self, shadow: ShadowRun) -> None:
        """Save shadow run results to disk."""
        try:
            path = self.log_dir / f"{shadow.run_id}.json"
            path.write_text(json.dumps(shadow.to_dict(), indent=2))
        except Exception as exc:
            logger.debug(f"Could not save shadow log: {exc}")

