"""
PolicyEvaluator — compares CQL policy vs LLM baseline.

This generates the central result curve of the paper:
    X axis: number of training runs seen
    Y axis: policy win rate vs LLM

At run 0 the policy loses every comparison (untrained).
At some crossover point N* the policy starts winning.
The shape and position of N* is the research contribution.

How comparison works:
    For each test state (from held-out transitions),
    ask both the policy and the LLM what action they'd take.
    Compare the resulting rewards using the buffer's ground truth.
    Win = policy reward > LLM reward on same state.

Hooks into:
    metrics/evaluator.py    → your existing eval infrastructure
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from agent.policy.cql_policy import CQLPolicy
from agent.policy.experience_buffer import (
    ExperienceBuffer,
    Transition,
    INDEX_ACTION,
    N_ACTIONS,
)
from agent.classifier.anomaly_types import AnomalyType
from agent.react.action_parser import ActionType

logger = logging.getLogger(__name__)


@dataclass
class EvalComparison:
    """
    Head-to-head comparison of policy vs LLM on one state.
    """
    state_idx: int
    anomaly_type: str

    # Policy
    policy_action: str
    policy_q_value: float
    policy_confidence: float

    # LLM (ground truth from buffer — what the LLM chose historically)
    llm_action: str
    llm_reward: float               # actual reward the LLM got

    # Simulated policy reward (from Q-network estimate)
    policy_est_reward: float

    # Did policy win?
    policy_wins: bool

    @property
    def reward_delta(self) -> float:
        return self.policy_est_reward - self.llm_reward


@dataclass
class PolicyEvalResult:
    """
    Aggregate policy vs LLM evaluation results.
    One of these is generated per training checkpoint.
    """
    n_transitions_trained_on: int   # size of buffer at eval time
    n_test_states: int

    overall_win_rate: float         # fraction policy won
    mean_reward_delta: float        # positive = policy better on average

    # Per anomaly type
    win_rate_per_type: dict = field(default_factory=dict)

    # For plotting the learning curve
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "n_transitions_trained_on": self.n_transitions_trained_on,
            "n_test_states": self.n_test_states,
            "overall_win_rate": self.overall_win_rate,
            "mean_reward_delta": self.mean_reward_delta,
            "win_rate_per_type": self.win_rate_per_type,
            "timestamp": self.timestamp,
        }


class PolicyEvaluator:
    """
    Evaluates CQL policy against the LLM baseline using held-out
    transitions from the experience buffer.

    Args:
        policy:         Trained CQLPolicy.
        buffer:         ExperienceBuffer (used for test states).
        test_frac:      Fraction of buffer to hold out for eval.
        log_dir:        Where to save eval results (learning curve).
    """

    def __init__(
        self,
        policy: CQLPolicy,
        buffer: ExperienceBuffer,
        test_frac: float = 0.2,
        log_dir: Path = Path("experiments/policy"),
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.test_frac = test_frac
        self.log_dir = log_dir
        self._eval_history: List[PolicyEvalResult] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> PolicyEvalResult:
        """
        Run one evaluation checkpoint.
        Call this every N training epochs to build the learning curve.
        """
        data = self.buffer._get_all()
        if len(data) < 10:
            logger.warning("PolicyEvaluator: insufficient data")
            return PolicyEvalResult(
                n_transitions_trained_on=len(data),
                n_test_states=0,
                overall_win_rate=0.0,
                mean_reward_delta=0.0,
            )

        # Hold out test set (last test_frac fraction, chronologically)
        n_test = max(1, int(len(data) * self.test_frac))
        test_transitions = data[-n_test:]

        comparisons = []
        for i, t in enumerate(test_transitions):
            comp = self._compare_on_transition(i, t)
            if comp is not None:
                comparisons.append(comp)

        if not comparisons:
            return PolicyEvalResult(
                n_transitions_trained_on=len(data),
                n_test_states=0,
                overall_win_rate=0.0,
                mean_reward_delta=0.0,
            )

        win_flags = [c.policy_wins for c in comparisons]
        reward_deltas = [c.reward_delta for c in comparisons]

        # Per anomaly type win rates
        win_per_type: dict = {}
        for c in comparisons:
            atype = c.anomaly_type
            if atype not in win_per_type:
                win_per_type[atype] = []
            win_per_type[atype].append(c.policy_wins)

        result = PolicyEvalResult(
            n_transitions_trained_on=len(data),
            n_test_states=len(comparisons),
            overall_win_rate=float(np.mean(win_flags)),
            mean_reward_delta=float(np.mean(reward_deltas)),
            win_rate_per_type={
                k: float(np.mean(v)) for k, v in win_per_type.items()
            },
        )

        self._eval_history.append(result)
        self._save_result(result)

        logger.info(
            f"PolicyEvaluator: win_rate={result.overall_win_rate:.1%}, "
            f"mean_delta={result.mean_reward_delta:+.3f}, "
            f"n={len(comparisons)}"
        )

        return result

    def _compare_on_transition(
        self, idx: int, t: Transition
    ) -> Optional[EvalComparison]:
        """Compare policy vs LLM on one held-out transition."""
        try:
            anomaly_type = AnomalyType[t.anomaly_type] \
                if t.anomaly_type in AnomalyType.__members__ \
                else AnomalyType.HEALTHY

            # Policy choice
            policy_action, confidence = self.policy.select_action(
                t.state, anomaly_type
            )

            if policy_action is None:
                # Policy deferred to LLM — count as LLM win
                return EvalComparison(
                    state_idx=idx,
                    anomaly_type=t.anomaly_type,
                    policy_action="DEFERRED_TO_LLM",
                    policy_q_value=0.0,
                    policy_confidence=0.0,
                    llm_action=t.action_name,
                    llm_reward=t.reward,
                    policy_est_reward=t.reward * 0.8,  # slight penalty for deferral
                    policy_wins=False,
                )

            # Estimate policy reward from Q-network
            policy_est_reward = self._estimate_q_value(t.state, policy_action)

            # LLM "reward" = actual reward from buffer (what happened)
            llm_reward = t.reward
            llm_action = t.action_name

            # Policy wins if estimated reward > LLM actual reward
            policy_wins = policy_est_reward > llm_reward

            return EvalComparison(
                state_idx=idx,
                anomaly_type=t.anomaly_type,
                policy_action=policy_action.name,
                policy_q_value=confidence,
                policy_confidence=confidence,
                llm_action=llm_action,
                llm_reward=llm_reward,
                policy_est_reward=policy_est_reward,
                policy_wins=policy_wins,
            )

        except Exception as exc:
            logger.debug(f"PolicyEvaluator comparison failed: {exc}")
            return None

    def _estimate_q_value(
        self, state: np.ndarray, action: ActionType
    ) -> float:
        """Get Q-value estimate for a state-action pair."""
        from agent.policy.experience_buffer import ACTION_INDEX
        try:
            import torch
            self.policy._q_network.eval()
            state_t = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(self.policy._device)
            with torch.no_grad():
                q_values = self.policy._q_network(state_t).squeeze(0)
                action_idx = ACTION_INDEX.get(action, 0)
                return float(q_values[action_idx].item())
        except Exception:
            return 0.0

    def learning_curve(self) -> List[dict]:
        """
        Return the full learning curve — win rate at each eval checkpoint.
        This is the central figure of the paper.

        X: n_transitions_trained_on (proxy for number of runs)
        Y: overall_win_rate
        """
        return [r.to_dict() for r in self._eval_history]

    def _save_result(self, result: PolicyEvalResult) -> None:
        path = self.log_dir / f"policy_eval_{int(time.time())}.json"
        try:
            path.write_text(json.dumps(result.to_dict(), indent=2))
        except Exception as exc:
            logger.debug(f"Could not save eval result: {exc}")

    def save_learning_curve(self) -> Path:
        """Save full learning curve to JSON for plotting."""
        path = self.log_dir / "learning_curve.json"
        path.write_text(json.dumps(self.learning_curve(), indent=2))
        logger.info(f"Learning curve saved to {path}")
        return path
