"""
Cooldown manager — prevents action spam.

If the agent fires on the same anomaly type twice within
cooldown_steps steps, the second event is suppressed.

This prevents feedback loops where:
    1. Agent reduces LR
    2. Loss spikes briefly (normal noise after LR change)
    3. Agent reduces LR again → training stalls completely
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from agent.classifier.anomaly_types import AnomalyType

logger = logging.getLogger(__name__)


class CooldownManager:
    """
    Per-anomaly-type cooldown tracker.

    Args:
        cooldown_steps: Minimum steps between actions on same anomaly type.
                        Configured via agent.cooldown_steps in base.yaml.
    """

    def __init__(self, cooldown_steps: int = 200) -> None:
        self.cooldown_steps = cooldown_steps
        self._last_action_step: Dict[AnomalyType, int] = {}
        self._lock = threading.Lock()

    def is_blocked(self, anomaly_type: AnomalyType) -> bool:
        """
        Returns True if this anomaly type is in cooldown.
        The agent should skip acting if this returns True.
        """
        with self._lock:
            return anomaly_type in self._last_action_step

    def register(self, anomaly_type: AnomalyType, step: int) -> None:
        """
        Record that an action was taken at this step for this anomaly type.
        Call this AFTER executing an action.
        """
        with self._lock:
            self._last_action_step[anomaly_type] = step
            logger.debug(
                f"Cooldown registered: {anomaly_type.name} at step {step}. "
                f"Blocked until step {step + self.cooldown_steps}."
            )

    def update(self, current_step: int) -> None:
        """
        Clear expired cooldowns.
        Call this every tick from the agent loop.
        """
        with self._lock:
            expired = [
                atype
                for atype, registered_step in self._last_action_step.items()
                if current_step - registered_step >= self.cooldown_steps
            ]
            for atype in expired:
                del self._last_action_step[atype]
                logger.debug(f"Cooldown cleared: {atype.name} at step {current_step}")

    def clear_all(self) -> None:
        """Clear all cooldowns — call after a rollback."""
        with self._lock:
            self._last_action_step.clear()
            logger.debug("All cooldowns cleared.")

    def status(self) -> Dict[str, int]:
        """Return current cooldown state for logging."""
        with self._lock:
            return {k.name: v for k, v in self._last_action_step.items()}
