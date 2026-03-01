
"""
agent.classifier
================
Fast rule-based anomaly classification.

Runs before the LLM — cheap, no API calls, microsecond latency.
Only escalates to the ReAct agent when something non-HEALTHY fires.

Pipeline:
    SignalSnapshot → RuleEngine → AnomalyEvent → AnomalyPriorityQueue
"""

from agent.classifier.anomaly_types import AnomalyType, AnomalyEvent, Severity
from agent.classifier.rule_engine import RuleEngine, RuleEngineConfig
from agent.classifier.priority_queue import AnomalyPriorityQueue

__all__ = [
    "AnomalyType",
    "AnomalyEvent",
    "Severity",
    "RuleEngine",
    "RuleEngineConfig",
    "AnomalyPriorityQueue",
]
