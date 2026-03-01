
"""
Metric stream — polls W&B or reads from a local shared file.

Hooks into:
    integrations/wandb.py   (your existing W&B client)
    monitoring/metrics.py   (fallback metric reader)

Two modes:
    WANDB   — polls wandb.Api() for the active run history
    LOCAL   — reads from a JSON file written by training loop
              (used when W&B is not configured)

The LOCAL mode file is written by AgentControlPlugin every N steps.
Format: {"step": 100, "loss": 0.42, "grad_norm": 1.3, "lr": 0.0002}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from agent.observer.rolling_window import MetricSnapshot, RollingWindow
from agent.observer.signal_extractor import SignalExtractor, SignalSnapshot

logger = logging.getLogger(__name__)


class StreamMode(Enum):
    WANDB = auto()
    LOCAL = auto()


class MetricStream:
    """
    Pulls metric snapshots from a running FRAMEWORM training job
    and feeds them into a RollingWindow + SignalExtractor.

    Args:
        run_id:         W&B run ID (e.g. "abc123"). Required for WANDB mode.
        local_path:     Path to shared JSON file. Required for LOCAL mode.
        poll_every:     How many seconds between polls (default 10s).
        window_size:    Rolling window capacity (default 500 steps).
        total_steps:    Total training steps (for early_training flag).
        ema_alpha:      EMA smoothing for signal extractor.

    Usage:
        # W&B mode
        stream = MetricStream(run_id="abc123def")
        snapshot = stream.tick()
        signals = stream.signals()  # derived SignalSnapshot

        # Local file mode (no W&B)
        stream = MetricStream(local_path="/tmp/fw_metrics.json")
        snapshot = stream.tick()
    """

    # Path written by AgentControlPlugin during training
    DEFAULT_LOCAL_PATH = Path("/tmp/frameworm_agent_metrics.json")

    def __init__(
        self,
        run_id: Optional[str] = None,
        local_path: Optional[Path] = None,
        poll_every: float = 10.0,
        window_size: int = 500,
        total_steps: int = 10_000,
        ema_alpha: float = 0.05,
    ) -> None:
        self.run_id = run_id
        self.local_path = Path(local_path) if local_path else self.DEFAULT_LOCAL_PATH
        self.poll_every = poll_every

        # Determine mode
        if run_id is not None:
            self.mode = StreamMode.WANDB
        else:
            self.mode = StreamMode.LOCAL
            logger.info(
                "No run_id provided — using LOCAL mode. "
                f"Reading from {self.local_path}"
            )

        # Core components
        self.window = RollingWindow(size=window_size)
        self.extractor = SignalExtractor(
            ema_alpha=ema_alpha,
            total_steps=total_steps,
        )

        # W&B client (lazy import so FRAMEWORM works without wandb)
        self._wandb_api = None
        self._wandb_run = None
        self._last_step_seen: int = -1
        self._last_poll_time: float = 0.0

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def tick(self) -> Optional[MetricSnapshot]:
        """
        Poll for new data. Returns the latest MetricSnapshot if new
        data was available, or None if no new steps since last poll.

        Call this in a loop — it respects poll_every timing internally.
        """
        now = time.monotonic()
        if now - self._last_poll_time < self.poll_every:
            return None
        self._last_poll_time = now

        try:
            if self.mode == StreamMode.WANDB:
                snapshot = self._poll_wandb()
            else:
                snapshot = self._poll_local()
        except Exception as exc:
            logger.warning(f"MetricStream poll failed: {exc}. Retrying next tick.")
            return None

        if snapshot is None:
            return None

        # Skip if we've already seen this step
        if snapshot.step <= self._last_step_seen:
            return None

        self._last_step_seen = snapshot.step
        self.window.push(snapshot)
        return snapshot

    def signals(self) -> Optional[SignalSnapshot]:
        """
        Compute and return derived signals from current window.
        Returns None if window is not ready yet (< 10 steps).
        """
        return self.extractor.extract(self.window)

    def latest(self) -> Optional[MetricSnapshot]:
        """Latest raw snapshot from the window."""
        return self.window.latest()

    def is_ready(self) -> bool:
        """True once we have enough history to compute signals."""
        return self.window.is_ready

    # ──────────────────────────────────────────────
    # W&B polling
    # ──────────────────────────────────────────────

    def _poll_wandb(self) -> Optional[MetricSnapshot]:
        """Poll W&B API for latest training step."""
        api = self._get_wandb_api()
        if api is None:
            logger.warning("W&B not available — falling back to LOCAL mode.")
            self.mode = StreamMode.LOCAL
            return self._poll_local()

        try:
            run = api.run(self.run_id)
            # Get last row from run history
            history = run.history(samples=1, pandas=False)
            if not history:
                return None
            row = history[-1]
            return self._row_to_snapshot(row)
        except Exception as exc:
            logger.warning(f"W&B poll error: {exc}")
            return None

    def _get_wandb_api(self):
        """Lazy-init W&B API client."""
        if self._wandb_api is not None:
            return self._wandb_api
        try:
            import wandb
            self._wandb_api = wandb.Api()
            return self._wandb_api
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return None
        except Exception as exc:
            logger.warning(f"W&B API init failed: {exc}")
            return None

    def _row_to_snapshot(self, row: dict) -> Optional[MetricSnapshot]:
        """Convert a W&B history row to a MetricSnapshot."""
        step = row.get("_step", row.get("step", 0))
        loss = row.get("loss", row.get("train/loss", row.get("train_loss")))
        grad_norm = row.get("grad_norm", row.get("train/grad_norm", 0.0))
        lr = row.get("lr", row.get("learning_rate", 0.0))

        if loss is None:
            logger.debug(f"No loss found in W&B row at step {step}. Keys: {list(row.keys())}")
            return None

        # Layer-wise grad norms (if logged)
        layer_grads = {
            k.replace("grad/", ""): float(v)
            for k, v in row.items()
            if k.startswith("grad/") and isinstance(v, (int, float))
        }

        return MetricSnapshot(
            step=int(step),
            loss=float(loss),
            grad_norm=float(grad_norm),
            lr=float(lr),
            epoch=int(row.get("epoch", 0)),
            layer_grad_norms=layer_grads,
            weight_update_ratio=float(row.get("weight_update_ratio", 0.0)),
        )

    # ──────────────────────────────────────────────
    # Local file polling
    # ──────────────────────────────────────────────

    def _poll_local(self) -> Optional[MetricSnapshot]:
        """
        Read latest metrics from shared JSON file.
        This file is written by AgentControlPlugin every N steps.
        """
        if not self.local_path.exists():
            logger.debug(
                f"Local metrics file not found at {self.local_path}. "
                "Waiting for training to start..."
            )
            return None

        try:
            with open(self.local_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(f"Could not read local metrics file: {exc}")
            return None

        # Support both single snapshot and list of snapshots
        if isinstance(data, list):
            data = data[-1]  # take the latest

        loss = data.get("loss")
        if loss is None:
            return None

        return MetricSnapshot(
            step=int(data.get("step", 0)),
            loss=float(loss),
            grad_norm=float(data.get("grad_norm", 0.0)),
            lr=float(data.get("lr", 0.0)),
            epoch=int(data.get("epoch", 0)),
            layer_grad_norms=data.get("layer_grad_norms", {}),
            weight_update_ratio=float(data.get("weight_update_ratio", 0.0)),
        )

    # ──────────────────────────────────────────────
    # Context manager support
    # ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass