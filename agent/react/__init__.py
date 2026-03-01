
"""
agent.react
===========
The ReAct (Reasoning + Acting) loop.

Flow per anomaly event:
    1. prompts.py      → build structured LLM prompt from AnomalyEvent
    2. agent.py        → call LLM, parse response
    3. action_parser.py → extract action + params from LLM text
    4. control/        → execute the action on the training loop
    5. verifier.py     → watch next N steps, KS test to confirm fix
    6. agent.py        → log outcome, escalate if unresolved

The LLM is only called when classifier returns non-HEALTHY.
All other ticks are free (rule engine only).
"""

from agent.react.prompts import PromptBuilder
from agent.react.action_parser import ActionParser, ParsedAction, ActionType
from agent.react.verifier import Verifier, VerificationResult
from agent.react.agent import FramewormAgent

__all__ = [
    "PromptBuilder",
    "ActionParser",
    "ParsedAction",
    "ActionType",
    "Verifier",
    "VerificationResult",
    "FramewormAgent",
]