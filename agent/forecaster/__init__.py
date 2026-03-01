
"""
agent.forecaster
================
Proactive failure prediction via gradient trajectory forecasting.

Instead of waiting for an anomaly to fire (reactive), the forecaster
predicts the PROBABILITY of each failure mode occurring in the
next N steps and triggers a soft pre-emptive intervention
when confidence is high enough.

Architecture:
    training_data.py     → collect + format sequences from past runs
    grad_forecaster.py   → LSTM model: window → P(failure) per horizon
    failure_predictor.py → wraps forecaster, drives proactive actions

Research question this answers:
    "Does proactive intervention outperform reactive monitoring,
     and by how much?"
"""

from agent.forecaster.training_data import (
    ForecasterDataset,
    ForecasterSample,
    DataCollector,
)
from agent.forecaster.grad_forecaster import GradForecaster, ForecasterConfig
from agent.forecaster.failure_predictor import (
    FailurePredictor,
    PredictionResult,
    ProactiveIntervention,
)

__all__ = [
    "ForecasterDataset",
    "ForecasterSample",
    "DataCollector",
    "GradForecaster",
    "ForecasterConfig",
    "FailurePredictor",
    "PredictionResult",
    "ProactiveIntervention",
]