"""
Builds structured prompts for the ReAct agent LLM call.

The prompt is the most important piece of the entire agent.
It must give the LLM:
    - Exactly what is happening (anomaly type, severity, signals)
    - Full context (step, scheduler position, model type)
    - History of what the agent already tried (avoid loops)
    - Available actions with clear descriptions
    - Output format to parse deterministically

Design principles:
    - Tokens are money. Keep prompts tight.
    - Force structured output so action_parser never fails.
    - Include action history so the agent doesn't repeat itself.
    - Be conservative by default — WATCH > ALERT > ADJUST > ROLLBACK.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from agent.classifier.anomaly_types import AnomalyEvent
from agent.observer.rolling_window import RollingWindow


# ──────────────────────────────────────────────────────────────────
# System prompt — sent once per session
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FRAMEWORM-AGENT, an autonomous monitor for neural network training runs.

Your job is to detect training anomalies and take corrective action to recover the run — \
adjusting learning rate, rolling back to a checkpoint, swapping schedulers, or alerting the user.

CORE PRINCIPLES:
1. Be CONSERVATIVE. Prefer WATCH over irreversible actions.
2. Never repeat the same action type twice in a row for the same anomaly.
3. If unsure, ALERT the user rather than taking destructive action.
4. A loss spike may be a bad batch — always WATCH first unless severity is CRITICAL.
5. ROLLBACK is a last resort. Prefer ADJUST_LR first.

AVAILABLE ACTIONS:
  WATCH                        — monitor for N more steps, no intervention
  ADJUST_LR(factor=0.5)        — multiply current LR by factor (0.1 to 2.0)
  ROLLBACK(step=N)             — restore checkpoint from step N
  SWAP_SCHEDULER(name=cosine)  — switch to: cosine | polynomial | warmup | plateau
  PAUSE                        — pause training, wait for human
  ALERT(message="...")         — send Slack alert to user, continue training

OUTPUT FORMAT (strict — no other text):
THINK: <one sentence explaining what you observe and why>
ACT: <action name and params exactly as shown above>
REASON: <one sentence justifying this specific action>"""


# ──────────────────────────────────────────────────────────────────
# PromptBuilder
# ──────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds the user-turn prompt for each ReAct LLM call.

    Args:
        total_steps:    Total planned training steps (for % progress)
        model_name:     Model architecture name (e.g. "DCGAN")
        config_name:    Active config file name (e.g. "gan_highres.yaml")
    """

    def __init__(
        self,
        total_steps: int = 10_000,
        model_name: str = "unknown",
        config_name: str = "base.yaml",
    ) -> None:
        self.total_steps = total_steps
        self.model_name = model_name
        self.config_name = config_name

    def build(
        self,
        event: AnomalyEvent,
        window: RollingWindow,
        action_history: List[dict],
        last_checkpoint_step: int,
        last_checkpoint_loss: float,
    ) -> str:
        """
        Build the full user-turn prompt for one ReAct call.

        Args:
            event:                  The AnomalyEvent from the classifier.
            window:                 Current RollingWindow (for loss history).
            action_history:         List of previous agent actions this run.
                                    Each dict: {step, action, outcome, resolved}
            last_checkpoint_step:   Most recent checkpoint step number.
            last_checkpoint_loss:   Loss value at last checkpoint.

        Returns:
            Formatted string prompt (user turn only — system prompt separate).
        """
        losses = window.losses(n=20)
        grad_norms = window.grad_norms(n=10)
        schedule_pct = round(event.step / max(self.total_steps, 1) * 100, 1)

        # Format loss window as compact list
        loss_str = "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
        grad_str = "[" + ", ".join(f"{g:.3f}" for g in grad_norms) + "]"

        # Format action history (last 3 only to save tokens)
        history_str = self._format_history(action_history[-3:])

        # Rollback options (last checkpoint + 2 steps before that)
        rollback_options = self._format_rollback_options(
            last_checkpoint_step, event.step
        )

        prompt = f"""═══════════════════════════════════════
TRAINING ANOMALY REPORT
═══════════════════════════════════════

MODEL:      {self.model_name}
CONFIG:     {self.config_name}
STEP:       {event.step:,} / {self.total_steps:,}  ({schedule_pct}% complete)

ANOMALY:    {event.anomaly_type.name}
SEVERITY:   {event.severity.value.upper()}
DESCRIPTION: {event.description}

───────────────────────────────────────
CURRENT METRICS
───────────────────────────────────────
Loss (last 20 steps): {loss_str}
Grad norms (last 10): {grad_str}
Current LR:           {event.lr:.2e}
Loss z-score:         {event.loss_z_score:.2f}σ
Plateau score:        {event.plateau_score:.4f}
Divergence score:     {event.divergence_score:.2f}

Triggered rule:       {event.triggered_rule}
Triggered value:      {event.triggered_value:.4f}
Threshold:            {event.threshold_value:.4f}

───────────────────────────────────────
CHECKPOINT INFO
───────────────────────────────────────
Last checkpoint:      step {last_checkpoint_step:,} (loss: {last_checkpoint_loss:.4f})
Rollback options:     {rollback_options}

───────────────────────────────────────
AGENT HISTORY (this run)
───────────────────────────────────────
{history_str}

───────────────────────────────────────
SUGGESTED ACTIONS for {event.anomaly_type.name}:
  {', '.join(event.suggested_actions)}
───────────────────────────────────────

Respond with THINK / ACT / REASON only."""

        return prompt

    def _format_history(self, history: List[dict]) -> str:
        if not history:
            return "  (no prior actions this run)"
        lines = []
        for h in history:
            resolved = "✓ resolved" if h.get("resolved") else "✗ unresolved"
            lines.append(
                f"  Step {h['step']:,}: {h['action']} → {resolved}"
            )
        return "\n".join(lines)

    def _format_rollback_options(self, last_ckpt: int, current_step: int) -> str:
        options = [last_ckpt]
        # Approximate earlier checkpoint
        earlier = last_ckpt - max(500, (current_step - last_ckpt))
        if earlier > 0:
            options.append(earlier)
        return ", ".join(f"step {s:,}" for s in options)
