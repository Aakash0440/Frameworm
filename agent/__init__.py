"""
FRAMEWORM-AGENT
===============
Self-healing training pipeline agent.

Monitors live training runs, detects anomalies, and autonomously
intervenes to recover — adjusting LR, rolling back checkpoints,
swapping schedulers, or alerting the user via Slack.

Architecture:
    observer/       → reads metric stream from W&B / local file
    classifier/     → fast rule-based anomaly detection
    react/          → LLM ReAct loop (reason → act → verify)
    control/        → AgentControlPlugin, action execution
    forecaster/     → LSTM gradient trajectory forecasting
    causal/         → root cause attribution engine
    counterfactual/ → twin runner for intervention evaluation
    policy/         → offline RL meta-policy (CQL)
    pomdp/          → formal POMDP state space definition
    benchmark/      → evaluation suite, paper results

Usage:
    from agent import FramewormAgent

    agent = FramewormAgent(run_id="my-wandb-run-id")
    agent.start()   # runs as daemon alongside training
"""

from agent.observer.metric_stream import MetricStream
from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalExtractor
from agent.classifier.anomaly_types import AnomalyType, AnomalyEvent
from agent.classifier.rule_engine import RuleEngine
from agent.classifier.priority_queue import AnomalyPriorityQueue

__version__ = "0.1.0"
__all__ = [
    "MetricStream",
    "RollingWindow",
    "SignalExtractor",
    "AnomalyType",
    "AnomalyEvent",
    "RuleEngine",
    "AnomalyPriorityQueue",
]