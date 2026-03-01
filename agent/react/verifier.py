"""
Post-intervention verifier.

After the agent takes an action, this watches the next N steps
and uses your existing KS test (monitoring/drift.py) to determine
whether the intervention actually helped.

This is the closed-loop mechanism that makes FRAMEWORM-AGENT a
real agent rather than a fire-and-forget rule system.

Hooks into:
    monitoring/drift.py     → KS test (your existing implementation)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.observer.rolling_window import RollingWindow

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Outcome of a post-intervention verification watch.
    Logged to experience buffer for offline RL training.
    """
    resolved: bool
    pre_loss_mean: float
    post_loss_mean: float
    loss_delta: float           # negative = improved
    ks_statistic: float         # 0–1, higher = more different distributions
    ks_pvalue: float            # < 0.05 = statistically significant change
    steps_watched: int
    watch_duration_seconds: float

    @property
    def improved(self) -> bool:
        """True if loss went down AND the change is statistically significant."""
        return self.resolved and self.loss_delta < 0

    def __repr__(self) -> str:
        status = "✓ RESOLVED" if self.resolved else "✗ UNRESOLVED"
        return (
            f"VerificationResult({status}, "
            f"delta={self.loss_delta:+.4f}, "
            f"ks_p={self.ks_pvalue:.3f}, "
            f"steps={self.steps_watched})"
        )


class Verifier:
    """
    Watches post-intervention training for N steps and determines
    whether the intervention resolved the anomaly.

    Uses KS test to compare pre-intervention vs post-intervention
    loss distributions. A statistically significant downward shift
    = intervention worked.

    Args:
        watch_steps:    Steps to observe after intervention (default 50)
        ks_alpha:       Significance level for KS test (default 0.05)
        improvement_threshold: Minimum loss_delta to count as resolved
    """

    def __init__(
        self,
        watch_steps: int = 50,
        ks_alpha: float = 0.05,
        improvement_threshold: float = 0.0,
    ) -> None:
        self.watch_steps = watch_steps
        self.ks_alpha = ks_alpha
        self.improvement_threshold = improvement_threshold

    def verify(
        self,
        window: RollingWindow,
        pre_intervention_step: int,
        timeout_seconds: float = 300.0,
    ) -> VerificationResult:
        """
        Watch the window until enough post-intervention steps accumulate,
        then run KS test.

        In practice this is called after the agent acts — it polls the
        window (which MetricStream keeps updating) until watch_steps
        new snapshots have been added.

        Args:
            window:                   The live RollingWindow from MetricStream.
            pre_intervention_step:    Step number when action was taken.
            timeout_seconds:          Give up after this many seconds.

        Returns:
            VerificationResult with resolved=False if timeout reached.
        """
        start_time = time.monotonic()

        # Capture pre-intervention loss distribution
        pre_losses = window.losses(n=self.watch_steps)
        pre_mean = float(np.mean(pre_losses))

        # Wait for watch_steps new snapshots
        initial_len = len(window)
        target_len = initial_len + self.watch_steps

        logger.info(
            f"Verifier: watching {self.watch_steps} steps "
            f"post-intervention (from step {pre_intervention_step})"
        )

        while len(window) < target_len:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Verifier: timeout after {elapsed:.0f}s — "
                    "marking as unresolved"
                )
                return VerificationResult(
                    resolved=False,
                    pre_loss_mean=pre_mean,
                    post_loss_mean=pre_mean,
                    loss_delta=0.0,
                    ks_statistic=0.0,
                    ks_pvalue=1.0,
                    steps_watched=len(window) - initial_len,
                    watch_duration_seconds=elapsed,
                )
            time.sleep(1.0)

        # Capture post-intervention loss distribution
        post_losses = window.losses(n=self.watch_steps)
        post_mean = float(np.mean(post_losses))
        loss_delta = post_mean - pre_mean

        # KS test via your existing monitoring/drift.py
        ks_stat, ks_pvalue = self._run_ks_test(pre_losses, post_losses)

        elapsed = time.monotonic() - start_time

        # Resolved = loss improved AND statistically significant
        resolved = (
            loss_delta < self.improvement_threshold
            and ks_pvalue < self.ks_alpha
        )

        result = VerificationResult(
            resolved=resolved,
            pre_loss_mean=pre_mean,
            post_loss_mean=post_mean,
            loss_delta=loss_delta,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            steps_watched=self.watch_steps,
            watch_duration_seconds=elapsed,
        )

        logger.info(f"Verifier: {result}")
        return result

    def _run_ks_test(
        self, pre: np.ndarray, post: np.ndarray
    ) -> tuple:
        """
        Run KS test. Uses your existing monitoring/drift.py if available,
        falls back to scipy.
        """
        # Try your existing drift detection first
        try:
            from monitoring.drift import DriftDetector
            detector = DriftDetector()
            result = detector.ks_test(pre, post)
            # Your drift.py returns a dict with statistic and pvalue
            if isinstance(result, dict):
                return result.get("statistic", 0.0), result.get("pvalue", 1.0)
        except (ImportError, AttributeError, Exception) as exc:
            logger.debug(f"monitoring.drift not available ({exc}), using scipy")

        # Fallback: scipy
        try:
            from scipy import stats
            stat, pvalue = stats.ks_2samp(pre, post)
            return float(stat), float(pvalue)
        except ImportError:
            logger.warning("scipy not installed — KS test unavailable, using simple mean comparison")
            # Last resort: just compare means
            stat = abs(float(np.mean(post)) - float(np.mean(pre)))
            pvalue = 0.01 if stat > 0.01 else 0.5
            return stat, pvalue
