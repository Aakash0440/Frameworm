
"""
Parses LLM output into a structured ParsedAction.

The LLM is prompted to output exactly:
    THINK: <reasoning>
    ACT: <action>
    REASON: <justification>

This parser is defensive — if the LLM produces malformed output,
it defaults to WATCH (never crashes, never takes a destructive action
on a parse failure).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ActionType(Enum):
    WATCH = auto()
    ADJUST_LR = auto()
    ROLLBACK = auto()
    SWAP_SCHEDULER = auto()
    PAUSE = auto()
    ALERT = auto()


# Maps string names from LLM output → ActionType enum
ACTION_MAP = {
    "watch": ActionType.WATCH,
    "adjust_lr": ActionType.ADJUST_LR,
    "adjustlr": ActionType.ADJUST_LR,
    "rollback": ActionType.ROLLBACK,
    "swap_scheduler": ActionType.SWAP_SCHEDULER,
    "swapscheduler": ActionType.SWAP_SCHEDULER,
    "pause": ActionType.PAUSE,
    "alert": ActionType.ALERT,
}

# Valid schedulers the agent can request
VALID_SCHEDULERS = {"cosine", "polynomial", "warmup", "plateau", "step"}

# Factor bounds for ADJUST_LR
LR_FACTOR_MIN = 0.05
LR_FACTOR_MAX = 2.0


@dataclass
class ParsedAction:
    """
    Structured output of the LLM response.
    Always valid — defaults to WATCH on parse failure.
    """
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)

    # LLM reasoning (for logging + experience buffer)
    think: str = ""
    reason: str = ""

    # True if this was a parse fallback, not genuine LLM output
    is_fallback: bool = False

    # Raw LLM text (for debugging)
    raw_text: str = ""

    def __repr__(self) -> str:
        p = f"({self.params})" if self.params else ""
        return f"ParsedAction({self.action_type.name}{p})"

    @classmethod
    def watch_fallback(cls, raw_text: str = "", reason: str = "") -> "ParsedAction":
        """Safe fallback when parsing fails."""
        return cls(
            action_type=ActionType.WATCH,
            params={"steps": 50},
            think="Parse failed — defaulting to WATCH.",
            reason=reason or "Could not parse LLM output. Monitoring.",
            is_fallback=True,
            raw_text=raw_text,
        )


class ActionParser:
    """
    Parses raw LLM text into a ParsedAction.

    Usage:
        parser = ActionParser()
        action = parser.parse(llm_response_text)
    """

    def parse(self, text: str) -> ParsedAction:
        """
        Parse LLM response text.
        Always returns a valid ParsedAction — never raises.
        """
        if not text or not text.strip():
            logger.warning("ActionParser: empty LLM response — defaulting to WATCH")
            return ParsedAction.watch_fallback(raw_text=text)

        think = self._extract_field(text, "THINK")
        act_raw = self._extract_field(text, "ACT")
        reason = self._extract_field(text, "REASON")

        if not act_raw:
            logger.warning(
                f"ActionParser: no ACT field found in LLM response. "
                f"Raw text: {text[:200]!r}"
            )
            return ParsedAction.watch_fallback(raw_text=text)

        action_type, params = self._parse_act_field(act_raw)

        return ParsedAction(
            action_type=action_type,
            params=params,
            think=think,
            reason=reason,
            is_fallback=False,
            raw_text=text,
        )

    # ──────────────────────────────────────────────
    # Field extraction
    # ──────────────────────────────────────────────

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract content after FIELD_NAME: on a line."""
        pattern = rf"^{field_name}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_act_field(self, act_raw: str) -> tuple:
        """
        Parse the ACT field into (ActionType, params dict).

        Handles formats like:
            WATCH
            ADJUST_LR(factor=0.5)
            ROLLBACK(step=4000)
            SWAP_SCHEDULER(name=cosine)
            ALERT(message="LR too high, manual check needed")
        """
        act_clean = act_raw.strip()

        # Extract action name (everything before first '(' or end)
        name_match = re.match(r"([A-Za-z_]+)", act_clean)
        if not name_match:
            logger.warning(f"ActionParser: cannot extract action name from {act_raw!r}")
            return ActionType.WATCH, {"steps": 50}

        action_name = name_match.group(1).strip().lower()
        action_type = ACTION_MAP.get(action_name)

        if action_type is None:
            logger.warning(
                f"ActionParser: unknown action '{action_name}' — defaulting to WATCH"
            )
            return ActionType.WATCH, {"steps": 50}

        # Extract params from parentheses
        params = self._parse_params(act_clean, action_type)

        return action_type, params

    def _parse_params(self, act_raw: str, action_type: ActionType) -> Dict[str, Any]:
        """Parse key=value pairs from inside parentheses."""
        params_match = re.search(r"\((.+?)\)", act_raw, re.DOTALL)
        if not params_match:
            return self._default_params(action_type)

        params_str = params_match.group(1)
        params = {}

        # Split on commas (not inside quotes)
        pairs = re.split(r",\s*(?=\w+=)", params_str)
        for pair in pairs:
            kv = pair.strip()
            if "=" not in kv:
                continue
            k, _, v = kv.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            params[k] = self._coerce_value(v)

        # Validate and clamp
        params = self._validate_params(params, action_type)
        return params

    def _coerce_value(self, v: str) -> Any:
        """Try to coerce string to int or float."""
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v  # keep as string

    def _validate_params(
        self, params: Dict[str, Any], action_type: ActionType
    ) -> Dict[str, Any]:
        """Clamp params to safe ranges."""
        if action_type == ActionType.ADJUST_LR:
            factor = params.get("factor", 0.5)
            try:
                factor = float(factor)
            except (TypeError, ValueError):
                factor = 0.5
            params["factor"] = max(LR_FACTOR_MIN, min(LR_FACTOR_MAX, factor))

        elif action_type == ActionType.ROLLBACK:
            if "step" not in params:
                params["step"] = None  # control plugin uses last checkpoint

        elif action_type == ActionType.SWAP_SCHEDULER:
            name = str(params.get("name", "cosine")).lower()
            if name not in VALID_SCHEDULERS:
                logger.warning(
                    f"ActionParser: unknown scheduler '{name}' — defaulting to cosine"
                )
                name = "cosine"
            params["name"] = name

        elif action_type == ActionType.WATCH:
            params.setdefault("steps", 50)

        elif action_type == ActionType.ALERT:
            params.setdefault("message", "Agent detected anomaly — manual check recommended.")

        return params

    def _default_params(self, action_type: ActionType) -> Dict[str, Any]:
        """Default params when none are specified."""
        defaults = {
            ActionType.WATCH: {"steps": 50},
            ActionType.ADJUST_LR: {"factor": 0.5},
            ActionType.ROLLBACK: {"step": None},
            ActionType.SWAP_SCHEDULER: {"name": "cosine"},
            ActionType.PAUSE: {},
            ActionType.ALERT: {"message": "Agent detected anomaly."},
        }
        return defaults.get(action_type, {})
