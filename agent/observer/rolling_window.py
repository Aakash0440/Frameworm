
"""
Rolling window buffer for metric history.
Pure numpy — no FRAMEWORM dependencies.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class MetricSnapshot:
    """
    A single point-in-time reading from the training loop.
    Raw values — no derived signals yet.
    """
    step: int
    loss: float
    grad_norm: float
    lr: float
    epoch: int = 0
    # Optional per-layer grad norms — dict of {layer_name: grad_norm}
    layer_grad_norms: Dict[str, float] = field(default_factory=dict)
    # Optional activation statistics — dict of {layer_name: mean_activation}
    activation_stats: Dict[str, float] = field(default_factory=dict)
    # Optional weight update ratio — how much weights changed this step
    weight_update_ratio: float = 0.0
    # Raw loss before any smoothing
    raw_loss: float = 0.0

    def __post_init__(self):
        if self.raw_loss == 0.0:
            self.raw_loss = self.loss


class RollingWindow:
    """
    Thread-safe circular buffer storing MetricSnapshot history.

    Maintains separate deques for scalar metrics so numpy ops
    are fast and don't require iterating over full snapshots.

    Args:
        size: Maximum number of steps to retain (default 500)
    """

    def __init__(self, size: int = 500) -> None:
        self.size = size
        self._lock = threading.Lock()

        # Raw snapshots (full objects)
        self._snapshots: Deque[MetricSnapshot] = deque(maxlen=size)

        # Scalar arrays for fast numpy slicing
        self._losses: Deque[float] = deque(maxlen=size)
        self._grad_norms: Deque[float] = deque(maxlen=size)
        self._lrs: Deque[float] = deque(maxlen=size)
        self._steps: Deque[int] = deque(maxlen=size)
        self._weight_update_ratios: Deque[float] = deque(maxlen=size)

    # ──────────────────────────────────────────────
    # Mutation
    # ──────────────────────────────────────────────

    def push(self, snapshot: MetricSnapshot) -> None:
        """Add a new snapshot to the window."""
        with self._lock:
            self._snapshots.append(snapshot)
            self._losses.append(snapshot.loss)
            self._grad_norms.append(snapshot.grad_norm)
            self._lrs.append(snapshot.lr)
            self._steps.append(snapshot.step)
            self._weight_update_ratios.append(snapshot.weight_update_ratio)

    def clear(self) -> None:
        """Reset all buffers."""
        with self._lock:
            self._snapshots.clear()
            self._losses.clear()
            self._grad_norms.clear()
            self._lrs.clear()
            self._steps.clear()
            self._weight_update_ratios.clear()

    # ──────────────────────────────────────────────
    # Accessors — return numpy arrays (copies, thread-safe)
    # ──────────────────────────────────────────────

    def losses(self, n: Optional[int] = None) -> np.ndarray:
        with self._lock:
            arr = np.array(self._losses)
        return arr[-n:] if n else arr

    def grad_norms(self, n: Optional[int] = None) -> np.ndarray:
        with self._lock:
            arr = np.array(self._grad_norms)
        return arr[-n:] if n else arr

    def lrs(self, n: Optional[int] = None) -> np.ndarray:
        with self._lock:
            arr = np.array(self._lrs)
        return arr[-n:] if n else arr

    def steps(self, n: Optional[int] = None) -> np.ndarray:
        with self._lock:
            arr = np.array(self._steps)
        return arr[-n:] if n else arr

    def snapshots(self, n: Optional[int] = None) -> List[MetricSnapshot]:
        with self._lock:
            snaps = list(self._snapshots)
        return snaps[-n:] if n else snaps

    def latest(self) -> Optional[MetricSnapshot]:
        """Most recent snapshot or None if empty."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    # ──────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._snapshots)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def is_ready(self) -> bool:
        """True once we have at least 10 steps of history."""
        return len(self) >= 10
