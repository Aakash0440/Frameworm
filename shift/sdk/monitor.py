"""
ShiftMonitor — the 2-line public SDK.

Usage:
    from frameworm.shift import ShiftMonitor

    # Training time
    monitor = ShiftMonitor("fraud_classifier")
    monitor.profile_reference(X_train, feature_names=["age", "income", ...])

    # Inference time
    result = monitor.check(X_batch)   # alerts fire automatically
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from shift.core.reference_store import ReferenceStore
from shift.core.feature_profiles import FeatureProfiler
from shift.core.drift_engine import DriftEngine, DriftResult, DriftSeverity
from shift.core.alert_manager import AlertManager

logger = logging.getLogger("frameworm.shift")


class ShiftMonitor:
    """
    One object that owns the full drift monitoring lifecycle.

    Args:
        name:            model/pipeline identifier — used for file naming + alerts
        store_dir:       where .shift files live (default: experiments/shift_profiles)
        alert_channels:  list of ["slack","webhook","log","stdout"]
        min_severity:    minimum severity that triggers an alert
        slack_webhook:   override FRAMEWORM_SLACK_WEBHOOK env var
        auto_alert:      if True, check() fires alerts automatically (default True)
    """

    def __init__(
        self,
        name: str,
        store_dir: Optional[str] = None,
        alert_channels: Optional[List[str]] = None,
        min_severity: DriftSeverity = DriftSeverity.MEDIUM,
        slack_webhook: Optional[str] = None,
        auto_alert: bool = True,
    ):
        self.name = name
        self.auto_alert = auto_alert

        self._store     = ReferenceStore(store_dir)
        self._profiler  = FeatureProfiler()
        self._engine    = DriftEngine()
        self._alerter   = AlertManager(
            channels=alert_channels,
            slack_webhook=slack_webhook,
            min_severity=min_severity,
        )

        self._reference_profile = None
        self._check_count       = 0
        self._drift_count       = 0

        # Eagerly load reference if it already exists
        if self._store.exists(name):
            self._reference_profile = self._store.load(name)
            logger.info(f"[SHIFT] Loaded existing reference profile for '{name}'")

    # ──────────────────────────────────────────────── training-time API

    def profile_reference(
        self,
        data,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> "ShiftMonitor":
        """
        Profile training data and save as the reference distribution.
        Call this once after training, before deployment.

        Returns self for chaining.
        """
        self._store.save(data, self.name, feature_names, metadata)
        self._reference_profile = self._store.load(self.name)
        return self

    # ──────────────────────────────────────────────── inference-time API

    def check(
        self,
        data,
        feature_names: Optional[List[str]] = None,
    ) -> DriftResult:
        """
        Compare incoming data against the reference distribution.
        Fires alerts automatically if auto_alert=True.

        Args:
            data:          numpy array (n_samples, n_features) or DataFrame
            feature_names: optional — uses reference feature names if omitted

        Returns:
            DriftResult with per-feature breakdown
        """
        if self._reference_profile is None:
            raise RuntimeError(
                f"[SHIFT] No reference profile for '{self.name}'. "
                f"Call monitor.profile_reference(X_train) first."
            )

        # Use reference feature names if not provided
        if feature_names is None:
            feature_names = self._reference_profile.feature_names

        current_profile = self._profiler.profile(data, feature_names)
        result = self._engine.compare(self._reference_profile, current_profile)

        self._check_count += 1
        if result.overall_drifted:
            self._drift_count += 1

        self._log_check(result)

        if self.auto_alert:
            self._alerter.alert_if_needed(result, model_name=self.name)

        return result

    def check_datapoint(
        self,
        datapoint: Union[list, np.ndarray],
        feature_names: Optional[List[str]] = None,
        window_size: int = 100,
    ) -> Optional[DriftResult]:
        """
        Accumulates single datapoints into a rolling window.
        Only runs drift check once window is full.

        Useful for low-throughput APIs where batches are small.
        """
        if not hasattr(self, "_window_buffer"):
            self._window_buffer: List = []
            self._window_size = window_size

        self._window_buffer.append(datapoint)

        if len(self._window_buffer) >= self._window_size:
            batch = np.array(self._window_buffer)
            self._window_buffer = []
            return self.check(batch, feature_names)

        return None   # window not yet full

    # ──────────────────────────────────────────────── status / reporting

    def status(self) -> dict:
        """Return current monitor stats."""
        return {
            "model_name":        self.name,
            "checks_run":        self._check_count,
            "drift_detections":  self._drift_count,
            "drift_rate":        (
                self._drift_count / self._check_count
                if self._check_count > 0 else 0.0
            ),
            "reference_loaded":  self._reference_profile is not None,
            "reference_samples": (
                self._reference_profile.n_samples
                if self._reference_profile else 0
            ),
        }

    def print_status(self):
        s = self.status()
        print(f"\n[SHIFT] Monitor: {s['model_name']}")
        print(f"  Checks run:       {s['checks_run']}")
        print(f"  Drift detections: {s['drift_detections']}  "
              f"({s['drift_rate']*100:.1f}% drift rate)")
        print(f"  Reference:        {s['reference_samples']} samples\n")

    # ──────────────────────────────────────────────── private

    def _log_check(self, result: DriftResult):
        """Append check result to experiments/shift_logs/<name>_checks.jsonl"""
        log_dir = Path("experiments/shift_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.name}_checks.jsonl"
        entry = result.to_dict()
        entry["checked_at"] = datetime.utcnow().isoformat()
        entry["check_number"] = self._check_count
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ──────────────────────────────────────────────── convenience classmethods

    @classmethod
    def from_reference(cls, name_or_path: str, **kwargs) -> "ShiftMonitor":
        """
        Construct a monitor from an existing .shift file.
        Accepts short name ("fraud_classifier") or full/partial path.
        """
        from pathlib import Path
        p = Path(name_or_path)
        name = p.stem
        store_dir = str(p.parent) if str(p.parent) not in (".", "") else None
        return cls(name, store_dir=store_dir, **kwargs)
