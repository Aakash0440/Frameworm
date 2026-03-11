"""
FailurePredictor — wraps GradForecaster, drives proactive interventions.

Every N steps, it:
    1. Extracts features from current rolling window
    2. Runs forward pass through GradForecaster (~0.5ms on CPU)
    3. If P(failure) > confidence_threshold → trigger soft proactive
       intervention BEFORE the anomaly fires

Proactive interventions are softer than reactive ones (e.g. reduce LR
by 10% not 50%) and are logged separately so paper comparisons work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agent.classifier.anomaly_types import AnomalyType
from agent.forecaster.grad_forecaster import ForecasterConfig, GradForecaster
from agent.forecaster.training_data import (
    FAILURE_MODES,
    HORIZONS,
    N_FAILURE_MODES,
    N_FEATURES,
    SEQ_LEN,
    DataCollector,
)
from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalExtractor

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path("agent/forecaster/weights/grad_forecaster.pt")


@dataclass
class PredictionResult:
    step: int
    probs: np.ndarray  # (N_FAILURE_MODES, N_HORIZONS)
    high_confidence: List[Dict] = field(default_factory=list)
    triggered_proactive: bool = False

    @property
    def max_prob(self) -> float:
        return float(self.probs.max())

    @property
    def is_alarming(self) -> bool:
        return len(self.high_confidence) > 0

    def top_prediction(self) -> Optional[Dict]:
        if not self.high_confidence:
            return None
        return max(self.high_confidence, key=lambda x: x["prob"])

    def summary(self) -> str:
        if not self.is_alarming:
            return f"Step {self.step}: No predicted failures (max_p={self.max_prob:.3f})"
        top = self.top_prediction()
        return (
            f"Step {self.step}: PREDICTED {top['failure_mode'].upper()} "
            f"in next {top['horizon']} steps (p={top['prob']:.3f})"
        )


@dataclass
class ProactiveIntervention:
    step: int
    predicted_failure_mode: str
    predicted_horizon: int
    confidence: float
    action_taken: str
    action_params: dict


class FailurePredictor:
    """
    Runs GradForecaster on the live rolling window and triggers
    proactive interventions when confidence exceeds threshold.

    Args:
        model:               Trained GradForecaster instance
        confidence_threshold: P(failure) threshold for proactive action
        run_every:           Run inference every N steps
        weights_path:        Path to saved model weights
    """

    FAILURE_MODE_TO_ANOMALY = {
        "gradient_explosion": AnomalyType.GRADIENT_EXPLOSION,
        "divergence": AnomalyType.DIVERGENCE,
        "loss_spike": AnomalyType.LOSS_SPIKE,
        "vanishing_grad": AnomalyType.VANISHING_GRAD,
        "oscillating": AnomalyType.OSCILLATING,
        "plateau": AnomalyType.PLATEAU,
    }

    # Softer proactive actions vs reactive ones
    PROACTIVE_ACTIONS = {
        "gradient_explosion": ("adjust_lr", {"factor": 0.8}),
        "divergence": ("adjust_lr", {"factor": 0.9}),
        "loss_spike": ("watch", {"steps": 50}),
        "vanishing_grad": ("adjust_lr", {"factor": 1.2}),
        "oscillating": ("adjust_lr", {"factor": 0.85}),
        "plateau": ("swap_scheduler", {"name": "cosine"}),
    }

    def __init__(
        self,
        model: Optional[GradForecaster] = None,
        extractor: Optional[SignalExtractor] = None,
        confidence_threshold: float = 0.80,
        run_every: int = 25,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.run_every = run_every
        self.weights_path = Path(weights_path)
        self.model = model or GradForecaster.load_or_init(self.weights_path)
        self.extractor = extractor or SignalExtractor()
        self.proactive_log: List[ProactiveIntervention] = []
        self._last_run_step = -1

    def tick(
        self,
        window: RollingWindow,
        current_step: int,
        control=None,
    ) -> Optional[PredictionResult]:
        """
        Called every step from agent.py.
        Only runs inference every self.run_every steps.
        Returns PredictionResult if inference ran, else None.
        """
        if current_step - self._last_run_step < self.run_every:
            return None
        if not self.model.is_ready or not window.is_ready:
            return None

        feature_window = self._build_feature_window(window)
        if feature_window is None:
            return None

        self._last_run_step = current_step
        probs = self.model.predict(feature_window)
        high_confidence = self._find_high_confidence(probs, current_step)

        result = PredictionResult(step=current_step, probs=probs, high_confidence=high_confidence)

        if result.is_alarming:
            logger.info(f"[Forecaster] {result.summary()}")
            if control is not None:
                self._trigger_proactive(result, control, current_step)
                result.triggered_proactive = True

        return result

    def _build_feature_window(self, window: RollingWindow) -> Optional[np.ndarray]:
        losses = window.losses()
        grad_norms = window.grad_norms()
        lrs = window.lrs()

        if len(losses) < 20:
            return None

        n = len(losses)
        alpha = 0.05
        ema = np.zeros(n)
        ema[0] = losses[0]
        for i in range(1, n):
            ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]

        w = min(50, n)
        rolling_mean = np.array([np.mean(losses[max(0, i - w) : i + 1]) for i in range(n)])
        rolling_std = np.array([np.std(losses[max(0, i - w) : i + 1]) + 1e-8 for i in range(n)])
        loss_delta = np.zeros(n)
        for i in range(10, n):
            loss_delta[i] = np.mean(losses[i - 5 : i]) - np.mean(losses[max(0, i - 10) : i - 5])
        grad_var = np.array([np.var(grad_norms[max(0, i - w) : i + 1]) for i in range(n)])

        features = np.stack(
            [
                losses,
                ema,
                loss_delta,
                (losses - rolling_mean) / rolling_std,
                grad_norms,
                grad_var,
                lrs,
                np.abs(loss_delta) / rolling_std,
            ],
            axis=1,
        ).astype(np.float32)

        # Normalize
        for j in range(N_FEATURES):
            col = features[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 1e-8:
                features[:, j] = (col - col_min) / (col_max - col_min)

        if n < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - n, N_FEATURES), dtype=np.float32)
            features = np.vstack([pad, features])
        else:
            features = features[-SEQ_LEN:]

        return features

    def _find_high_confidence(self, probs: np.ndarray, step: int) -> List[Dict]:
        hits = []
        for mode_idx, mode_name in enumerate(FAILURE_MODES):
            for h_idx, horizon in enumerate(HORIZONS):
                p = float(probs[mode_idx, h_idx])
                if p >= self.confidence_threshold:
                    hits.append(
                        {
                            "failure_mode": mode_name,
                            "horizon": horizon,
                            "prob": p,
                            "step": step,
                        }
                    )
        hits.sort(key=lambda x: -x["prob"])
        return hits

    def _trigger_proactive(self, result: PredictionResult, control, current_step: int) -> None:
        top = result.top_prediction()
        if top is None:
            return
        mode = top["failure_mode"]
        action_name, action_params = self.PROACTIVE_ACTIONS.get(mode, ("watch", {"steps": 50}))
        logger.info(
            f"[Forecaster] PROACTIVE: {action_name}({action_params}) "
            f"to prevent predicted {mode.upper()} in {top['horizon']} steps (p={top['prob']:.3f})"
        )
        try:
            from agent.classifier.anomaly_types import AnomalyEvent, Severity
            from agent.react.action_parser import ActionType, ParsedAction

            fake_action = ParsedAction(
                action_type=ActionType[action_name.upper()],
                params=action_params,
                think=f"Proactive: preventing predicted {mode}",
                reason=f"P({mode}) = {top['prob']:.3f} in next {top['horizon']} steps",
            )
            fake_event = AnomalyEvent(
                anomaly_type=self.FAILURE_MODE_TO_ANOMALY.get(mode, AnomalyType.HEALTHY),
                severity=Severity.LOW,
                step=current_step,
            )
            control.execute(fake_action, fake_event)
        except Exception as exc:
            logger.warning(f"Proactive intervention failed: {exc}")
            return

        self.proactive_log.append(
            ProactiveIntervention(
                step=current_step,
                predicted_failure_mode=mode,
                predicted_horizon=top["horizon"],
                confidence=top["prob"],
                action_taken=action_name,
                action_params=action_params,
            )
        )

    def train_forecaster(
        self,
        experiments_dir: Path = Path("experiments"),
        save_path: Optional[Path] = None,
    ) -> dict:
        """Collect data from past runs and train/retrain the forecaster."""
        save_path = save_path or self.weights_path
        collector = DataCollector(experiments_dir=experiments_dir)
        dataset = collector.collect()
        history = self.model.fit(dataset)
        self.model.save(save_path)
        logger.info(
            f"Forecaster trained. Best epoch: {history['best_epoch']}. Saved to {save_path}"
        )
        return history

    @property
    def is_ready(self) -> bool:
        return self.model.is_ready
