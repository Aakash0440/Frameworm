"""
agent.control
=============
Action execution layer.

Receives ParsedAction from the ReAct loop and applies it
to the running training job via AgentControlPlugin.

Hooks into:
    plugins/hooks.py        → base plugin class
    plugins/loader.py       → auto-discovery (drop in /plugins)
    training/trainer.py     → check_agent_commands() hook (4 lines)
    training/schedulers.py  → LR adjustment
    integrations/notifications.py → Slack alerts
    checkpoints/            → rollback target
"""

from agent.control.cooldown import CooldownManager
from agent.control.actions import ActionExecutor, ActionResult
from agent.control.control_plugin import AgentControlPlugin

__all__ = [
    "CooldownManager",
    "ActionExecutor",
    "ActionResult",
    "AgentControlPlugin",
]
