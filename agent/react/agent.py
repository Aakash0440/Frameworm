"""
FramewormAgent — the main ReAct loop.

Runs as a daemon thread alongside the training loop.
Never imports torch or any heavy dependency — stays lightweight.

Full loop per tick:
    1. MetricStream.tick()          → new MetricSnapshot
    2. SignalExtractor.extract()    → SignalSnapshot
    3. RuleEngine.classify()        → List[AnomalyEvent]
    4. AnomalyPriorityQueue.push()  → prioritized queue
    5. Cooldown.is_blocked()?       → skip if too soon
    6. PromptBuilder.build()        → LLM prompt string
    7. LLM call                     → raw text response
    8. ActionParser.parse()         → ParsedAction
    9. AgentControlPlugin.execute() → action on training loop
   10. Verifier.verify()            → was it resolved?
   11. Log outcome                  → action_history + JSON log

Usage:
    agent = FramewormAgent.from_config("configs/base.yaml", run_id="abc123")
    agent.start()       # non-blocking, runs as daemon thread

    # Or block current thread:
    agent.run()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType
from agent.classifier.priority_queue import AnomalyPriorityQueue
from agent.classifier.rule_engine import RuleEngine, RuleEngineConfig
from agent.observer.metric_stream import MetricStream
from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalExtractor
from agent.react.action_parser import ActionParser, ActionType, ParsedAction
from agent.react.prompts import PromptBuilder, SYSTEM_PROMPT
from agent.react.verifier import Verifier, VerificationResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Decision log entry (written to JSON after every action)
# ──────────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """One complete agent decision cycle — saved to disk."""
    step: int
    anomaly_type: str
    severity: str
    action: str
    action_params: dict
    think: str
    reason: str
    resolved: bool
    loss_delta: float
    ks_pvalue: float
    timestamp: float = field(default_factory=time.time)
    is_fallback: bool = False


# ──────────────────────────────────────────────────────────────────
# LLM client wrapper (thin, swap out any provider)
# ──────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around an LLM API.
    Defaults to OpenAI-compatible API (works with OpenAI, Together, Ollama).

    Set FRAMEWORM_AGENT_LLM_API_KEY env var.
    Set FRAMEWORM_AGENT_LLM_BASE_URL for non-OpenAI providers.
    Set FRAMEWORM_AGENT_LLM_MODEL for model name (default: gpt-4o-mini).

    For Ollama (free, local):
        FRAMEWORM_AGENT_LLM_BASE_URL=http://localhost:11434/v1
        FRAMEWORM_AGENT_LLM_MODEL=llama3.2
        FRAMEWORM_AGENT_LLM_API_KEY=ollama
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.2,   # low temp for deterministic structured output
    ) -> None:
        import os
        self.model = os.environ.get("FRAMEWORM_AGENT_LLM_MODEL", model)
        self.base_url = os.environ.get("FRAMEWORM_AGENT_LLM_BASE_URL", base_url)
        self.api_key = os.environ.get("FRAMEWORM_AGENT_LLM_API_KEY", api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key or "dummy"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            return self._client
        except ImportError:
            raise RuntimeError(
                "openai package not installed. "
                "Run: pip install openai\n"
                "Or use Ollama (free): pip install openai && "
                "set FRAMEWORM_AGENT_LLM_BASE_URL=http://localhost:11434/v1"
            )

    def call(self, system: str, user: str) -> str:
        """Make one LLM call. Returns raw text response."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return ""


# ──────────────────────────────────────────────────────────────────
# Main agent
# ──────────────────────────────────────────────────────────────────

class FramewormAgent:
    """
    The main FRAMEWORM-AGENT ReAct loop.

    Args:
        stream:         MetricStream (W&B or local mode).
        rule_engine:    RuleEngine with configured thresholds.
        llm:            LLMClient.
        control:        AgentControlPlugin (imported in Part 2).
        verifier:       Verifier for post-intervention watch.
        prompt_builder: PromptBuilder with model/config context.
        log_dir:        Where to write decision JSON logs.
        max_consecutive_actions: Stop agent if this many actions
                                 fire without a HEALTHY period.
    """

    def __init__(
        self,
        stream: MetricStream,
        rule_engine: RuleEngine,
        llm: LLMClient,
        control,                # AgentControlPlugin — imported at runtime
        verifier: Verifier,
        prompt_builder: PromptBuilder,
        log_dir: Path = Path("experiments/agent_logs"),
        max_consecutive_actions: int = 5,
    ) -> None:
        self.stream = stream
        self.rule_engine = rule_engine
        self.llm = llm
        self.control = control
        self.verifier = verifier
        self.prompt_builder = prompt_builder
        self.log_dir = log_dir
        self.max_consecutive_actions = max_consecutive_actions

        # Internal state
        self.parser = ActionParser()
        self.queue = AnomalyPriorityQueue()
        self.action_history: List[dict] = []
        self.decision_log: List[DecisionRecord] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consecutive_actions = 0

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Start agent as a daemon thread — non-blocking."""
        if self._running:
            logger.warning("FramewormAgent already running.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self.run,
            name="frameworm-agent",
            daemon=True,
        )
        self._thread.start()
        logger.info("FramewormAgent started (daemon thread).")

    def stop(self) -> None:
        """Signal the agent to stop after current tick."""
        self._running = False
        logger.info("FramewormAgent stopping...")

    def run(self) -> None:
        """
        Main loop. Runs until self._running = False or training ends.
        Call this directly for blocking usage.
        """
        logger.info("FramewormAgent loop started.")
        self._running = True

        while self._running:
            try:
                self._tick()
            except Exception as exc:
                logger.error(f"FramewormAgent tick error: {exc}", exc_info=True)
                time.sleep(5.0)  # back off on unexpected errors

        logger.info("FramewormAgent loop ended.")
        self._save_decision_log()

    # ──────────────────────────────────────────────
    # Core tick
    # ──────────────────────────────────────────────

    def _tick(self) -> None:
        """One iteration of the observe → classify → act → verify loop."""

        # 1. Observe
        snapshot = self.stream.tick()
        if snapshot is None:
            time.sleep(1.0)
            return

        # 2. Extract signals
        signals = self.stream.signals()
        if signals is None:
            return  # window not ready yet

        # 3. Classify
        events = self.rule_engine.classify(signals)
        if not events:
            self._consecutive_actions = 0
            return  # HEALTHY — nothing to do

        # 4. Enqueue
        self.queue.push_all(events)

        # 5. Check consecutive action limit (safety valve)
        if self._consecutive_actions >= self.max_consecutive_actions:
            logger.warning(
                f"FramewormAgent: {self._consecutive_actions} consecutive actions "
                "without HEALTHY period — sending alert and pausing agent."
            )
            self.control.send_alert(
                f"Agent has taken {self._consecutive_actions} actions without recovery. "
                "Manual intervention recommended."
            )
            self._running = False
            return

        # 6. Pop highest priority event
        event = self.queue.pop()
        if event is None:
            return

        # 7. Check cooldown
        if self.control.cooldown.is_blocked(event.anomaly_type):
            logger.info(
                f"[Step {event.step}] {event.anomaly_type.name} blocked by cooldown."
            )
            return

        # 8. Build prompt + call LLM
        logger.info(
            f"[Step {event.step}] Escalating to LLM: "
            f"{event.anomaly_type.name} ({event.severity.value})"
        )

        prompt = self.prompt_builder.build(
            event=event,
            window=self.stream.window,
            action_history=self.action_history,
            last_checkpoint_step=self.control.last_checkpoint_step,
            last_checkpoint_loss=self.control.last_checkpoint_loss,
        )

        raw_response = self.llm.call(system=SYSTEM_PROMPT, user=prompt)
        logger.debug(f"LLM raw response:\n{raw_response}")

        # 9. Parse action
        action = self.parser.parse(raw_response)
        logger.info(f"[Step {event.step}] Action: {action}")

        # 10. Execute
        success = self.control.execute(action, event)

        # 11. Register cooldown
        self.control.cooldown.register(event.anomaly_type, event.step)
        self._consecutive_actions += 1

        # 12. Verify (skip for WATCH and ALERT — no physical change made)
        resolved = True
        verification: Optional[VerificationResult] = None

        if action.action_type not in (ActionType.WATCH, ActionType.ALERT):
            verification = self.verifier.verify(
                window=self.stream.window,
                pre_intervention_step=event.step,
            )
            resolved = verification.resolved

            if not resolved:
                logger.warning(
                    f"[Step {event.step}] Intervention unresolved — escalating to user."
                )
                self.control.send_alert(
                    f"FRAMEWORM-AGENT: {action.action_type.name} did not resolve "
                    f"{event.anomaly_type.name} at step {event.step}. "
                    "Manual check recommended."
                )

        # 13. Log decision
        record = DecisionRecord(
            step=event.step,
            anomaly_type=event.anomaly_type.name,
            severity=event.severity.value,
            action=action.action_type.name,
            action_params=action.params,
            think=action.think,
            reason=action.reason,
            resolved=resolved,
            loss_delta=verification.loss_delta if verification else 0.0,
            ks_pvalue=verification.ks_pvalue if verification else 1.0,
            is_fallback=action.is_fallback,
        )
        self.decision_log.append(record)
        self.action_history.append({
            "step": event.step,
            "action": f"{action.action_type.name}({action.params})",
            "resolved": resolved,
        })

        # Reset counter if resolved
        if resolved:
            self._consecutive_actions = 0
            self.rule_engine.reset_counters()

    # ──────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────

    def _save_decision_log(self) -> None:
        """Write full decision log to JSON at end of run."""
        if not self.decision_log:
            return
        log_path = self.log_dir / f"agent_decisions_{int(time.time())}.json"
        with open(log_path, "w") as f:
            json.dump([asdict(r) for r in self.decision_log], f, indent=2)
        logger.info(f"Decision log saved to {log_path}")

    # ──────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/base.yaml",
        run_id: Optional[str] = None,
        model_name: str = "unknown",
        total_steps: int = 10_000,
    ) -> "FramewormAgent":
        """
        Build FramewormAgent from your FRAMEWORM YAML config.

        Usage:
            agent = FramewormAgent.from_config(
                config_path="configs/models/gan/dcgan.yaml",
                run_id="wandb-run-abc123",
                model_name="DCGAN",
                total_steps=50_000,
            )
            agent.start()
        """
        # Load config via your existing core.config
        agent_cfg = {}
        try:
            from core.config import load_config
            full_cfg = load_config(config_path)
            agent_cfg = full_cfg.get("agent", {})
        except Exception as exc:
            logger.warning(f"Could not load config from {config_path}: {exc}. Using defaults.")

        # Build components
        stream = MetricStream(
            run_id=run_id,
            poll_every=float(agent_cfg.get("poll_every", 10.0)),
            window_size=int(agent_cfg.get("window_size", 500)),
            total_steps=total_steps,
        )

        rule_config = RuleEngineConfig(
            grad_explosion_threshold=float(agent_cfg.get("grad_explosion_threshold", 10.0)),
            vanishing_grad_threshold=float(agent_cfg.get("vanishing_grad_threshold", 0.001)),
            loss_spike_z_score=float(agent_cfg.get("loss_spike_z_score", 3.0)),
            plateau_score_threshold=float(agent_cfg.get("plateau_score_threshold", 0.05)),
            plateau_min_steps=int(agent_cfg.get("plateau_min_steps", 100)),
            divergence_score_threshold=float(agent_cfg.get("divergence_score_threshold", 0.75)),
            divergence_min_steps=int(agent_cfg.get("divergence_min_steps", 50)),
            oscillation_score_threshold=float(agent_cfg.get("oscillation_score_threshold", 0.01)),
            early_training_lenience=bool(agent_cfg.get("early_training_lenience", True)),
        )

        # Import control plugin (defined later in this part)
        from agent.control.control_plugin import AgentControlPlugin
        from agent.control.cooldown import CooldownManager

        cooldown = CooldownManager(
            cooldown_steps=int(agent_cfg.get("cooldown_steps", 200))
        )
        control = AgentControlPlugin(cooldown=cooldown)

        return cls(
            stream=stream,
            rule_engine=RuleEngine(config=rule_config),
            llm=LLMClient(),
            control=control,
            verifier=Verifier(),
            prompt_builder=PromptBuilder(
                total_steps=total_steps,
                model_name=model_name,
                config_name=Path(config_path).name,
            ),
        )
