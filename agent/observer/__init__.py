
"""
agent.observer
==============
Metric observation layer.

Responsibilities:
- Poll W&B API or read from local shared file every N steps
- Maintain a rolling window of raw metric history
- Extract derived signals (z-score, EMA, grad SNR, etc.)
- Return a clean MetricSnapshot to the classifier each tick
"""

from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalExtractor, SignalSnapshot
from agent.observer.metric_stream import MetricStream, MetricSnapshot

__all__ = [
    "RollingWindow",
    "SignalExtractor",
    "SignalSnapshot",
    "MetricStream",
    "MetricSnapshot",
]
