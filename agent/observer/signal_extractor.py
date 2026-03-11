"""
Derives diagnostic signals from the rolling window.

All signals are computed from raw metric arrays.
Pure numpy — no FRAMEWORM dependencies.

Signals produced:
    loss_ema            Exponential moving average of loss
    loss_delta          Rate of change over last 10 steps
    loss_z_score        How many std devs from rolling mean
    grad_norm_mean      Rolling mean of gradient norms
    grad_norm_var       Rolling variance (instability indicator)
    plateau_score       Abs(loss_delta) / std — near 0 = plateau
    divergence_score    Fraction of recent steps with loss increasing
    oscillation_score   Variance of loss deltas (bouncing)
    lr_stability        Whether LR has changed recently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.observer.rolling_window import RollingWindow


@dataclass
class SignalSnapshot:
    """
    All derived signals for a single tick.
    This is what the classifier operates on.
    """

    step: int

    # Loss signals
    loss_raw: float
    loss_ema: float
    loss_delta: float  # positive = increasing, negative = improving
    loss_z_score: float  # how many σ from rolling mean
    loss_rolling_mean: float
    loss_rolling_std: float

    # Gradient signals
    grad_norm_current: float
    grad_norm_mean: float  # rolling mean
    grad_norm_var: float  # rolling variance
    grad_norm_z_score: float  # how many σ from rolling mean

    # Composite scores (used directly by rule engine)
    plateau_score: float  # near 0 = stuck, high = moving
    divergence_score: float  # 0–1, fraction of steps getting worse
    oscillation_score: float  # high = bouncing loss

    # LR info
    lr_current: float
    lr_changed: bool  # did LR change in last 10 steps?

    # Context
    window_size: int  # how many steps we have
    is_early_training: bool  # first 10% of training (noisy, be lenient)

    def __repr__(self) -> str:
        return (
            f"SignalSnapshot(step={self.step}, "
            f"loss_ema={self.loss_ema:.4f}, "
            f"z={self.loss_z_score:.2f}, "
            f"grad_norm={self.grad_norm_current:.3f}, "
            f"plateau={self.plateau_score:.4f}, "
            f"divergence={self.divergence_score:.2f})"
        )


class SignalExtractor:
    """
    Computes diagnostic signals from a RollingWindow.

    Args:
        ema_alpha:      EMA smoothing factor (smaller = smoother)
        short_window:   Steps for short-term signals (z-score, delta)
        long_window:    Steps for long-term signals (divergence trend)
        total_steps:    Total training steps (for early_training flag)
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        short_window: int = 50,
        long_window: int = 100,
        total_steps: int = 10_000,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.short_window = short_window
        self.long_window = long_window
        self.total_steps = total_steps
        self._ema_cache: Optional[float] = None

    # ──────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────

    def extract(self, window: RollingWindow) -> Optional[SignalSnapshot]:
        """
        Compute all signals from the current window state.
        Returns None if window has fewer than 10 steps (not ready).
        """
        if not window.is_ready:
            return None

        latest = window.latest()
        losses = window.losses()
        grad_norms = window.grad_norms()
        lrs = window.lrs()

        # ── Loss signals ──────────────────────────
        loss_ema = self._compute_ema(losses)
        short_losses = losses[-self.short_window :] if len(losses) >= self.short_window else losses
        loss_rolling_mean = float(np.mean(short_losses))
        loss_rolling_std = float(np.std(short_losses)) + 1e-8

        loss_z_score = (latest.loss - loss_rolling_mean) / loss_rolling_std
        loss_delta = self._compute_delta(losses, n=10)

        # ── Gradient signals ──────────────────────
        short_grads = (
            grad_norms[-self.short_window :] if len(grad_norms) >= self.short_window else grad_norms
        )
        grad_norm_mean = float(np.mean(short_grads))
        grad_norm_var = float(np.var(short_grads))
        grad_norm_std = float(np.std(short_grads)) + 1e-8
        grad_norm_z_score = (latest.grad_norm - grad_norm_mean) / grad_norm_std

        # ── Composite scores ──────────────────────
        plateau_score = abs(loss_delta) / loss_rolling_std
        divergence_score = self._compute_divergence_score(losses, self.long_window)
        oscillation_score = self._compute_oscillation_score(losses, n=20)

        # ── LR change detection ───────────────────
        recent_lrs = lrs[-10:] if len(lrs) >= 10 else lrs
        lr_changed = bool(np.max(recent_lrs) - np.min(recent_lrs) > 1e-10)

        # ── Early training flag ───────────────────
        early_cutoff = max(100, int(self.total_steps * 0.1))
        is_early = latest.step < early_cutoff

        return SignalSnapshot(
            step=latest.step,
            loss_raw=latest.loss,
            loss_ema=loss_ema,
            loss_delta=loss_delta,
            loss_z_score=loss_z_score,
            loss_rolling_mean=loss_rolling_mean,
            loss_rolling_std=loss_rolling_std,
            grad_norm_current=latest.grad_norm,
            grad_norm_mean=grad_norm_mean,
            grad_norm_var=grad_norm_var,
            grad_norm_z_score=grad_norm_z_score,
            plateau_score=plateau_score,
            divergence_score=divergence_score,
            oscillation_score=oscillation_score,
            lr_current=latest.lr,
            lr_changed=lr_changed,
            window_size=len(window),
            is_early_training=is_early,
        )

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _compute_ema(self, values: np.ndarray) -> float:
        """Incremental EMA — reuses cached value for efficiency."""
        if self._ema_cache is None:
            self._ema_cache = float(values[0])
        alpha = self.ema_alpha
        for v in values:
            self._ema_cache = alpha * v + (1 - alpha) * self._ema_cache
        return self._ema_cache

    def _compute_delta(self, values: np.ndarray, n: int = 10) -> float:
        """
        Rate of change: mean of last n values minus mean of prior n values.
        Positive = getting worse, negative = improving.
        """
        if len(values) < n * 2:
            return 0.0
        recent = float(np.mean(values[-n:]))
        prior = float(np.mean(values[-n * 2 : -n]))
        return recent - prior

    def _compute_divergence_score(self, losses: np.ndarray, n: int) -> float:
        """
        Fraction of consecutive step-pairs where loss increased.
        0.0 = always improving, 1.0 = always getting worse.
        """
        window = losses[-n:] if len(losses) >= n else losses
        if len(window) < 2:
            return 0.0
        deltas = np.diff(window)
        return float(np.mean(deltas > 0))

    def _compute_oscillation_score(self, losses: np.ndarray, n: int) -> float:
        """
        Variance of consecutive loss deltas.
        High = loss is bouncing up and down (LR too high).
        """
        window = losses[-n:] if len(losses) >= n else losses
        if len(window) < 3:
            return 0.0
        deltas = np.diff(window)
        return float(np.var(deltas))
