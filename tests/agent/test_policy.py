"""
Tests for agent/policy/ + agent/pomdp/
Run with: pytest tests/agent/test_policy.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from agent.classifier.anomaly_types import AnomalyType
from agent.observer.rolling_window import MetricSnapshot, RollingWindow
from agent.observer.signal_extractor import SignalExtractor, SignalSnapshot
from agent.policy.cql_policy import CQLConfig, CQLPolicy
from agent.policy.experience_buffer import (
    N_ACTIONS,
    STATE_DIM,
    ExperienceBuffer,
    Transition,
    compute_reward,
    encode_state,
)
from agent.pomdp.belief_updater import BeliefUpdater
from agent.pomdp.state_space import POMDPSpec
from agent.react.action_parser import ActionType

TMP = Path("/tmp/fw_test_policy")


class TestExperienceBuffer:
    def test_add_and_len(self):
        buf = ExperienceBuffer(db_path=TMP / "exp.db")
        for _ in range(10):
            buf.add(
                Transition(
                    state=np.zeros(STATE_DIM, dtype=np.float32),
                    action=0,
                    reward=1.0,
                    next_state=np.zeros(STATE_DIM, dtype=np.float32),
                    done=False,
                )
            )
        assert len(buf) == 10

    def test_is_ready_threshold(self):
        buf = ExperienceBuffer(db_path=TMP / "exp2.db")
        assert not buf.is_ready
        for _ in range(100):
            buf.add(
                Transition(
                    state=np.zeros(STATE_DIM, dtype=np.float32),
                    action=1,
                    reward=0.5,
                    next_state=np.zeros(STATE_DIM, dtype=np.float32),
                    done=False,
                )
            )
        assert buf.is_ready

    def test_state_dim_enforced(self):
        with pytest.raises(AssertionError):
            Transition(
                state=np.zeros(5, dtype=np.float32),  # wrong dim
                action=0,
                reward=1.0,
                next_state=np.zeros(STATE_DIM, dtype=np.float32),
                done=False,
            )

    def test_sample_returns_batch(self):
        buf = ExperienceBuffer(db_path=TMP / "exp3.db")
        for _ in range(50):
            buf.add(
                Transition(
                    state=np.random.randn(STATE_DIM).astype(np.float32),
                    action=np.random.randint(N_ACTIONS),
                    reward=1.0,
                    next_state=np.random.randn(STATE_DIM).astype(np.float32),
                    done=False,
                )
            )
        batch = buf.sample(16)
        assert batch is not None
        assert len(batch) == 16

    def test_to_arrays(self):
        buf = ExperienceBuffer(db_path=TMP / "exp4.db")
        for _ in range(20):
            buf.add(
                Transition(
                    state=np.random.randn(STATE_DIM).astype(np.float32),
                    action=0,
                    reward=1.0,
                    next_state=np.random.randn(STATE_DIM).astype(np.float32),
                    done=False,
                )
            )
        arrays = buf.to_arrays()
        assert arrays is not None
        states, actions, rewards, next_states, dones = arrays
        assert states.shape == (20, STATE_DIM)
        assert actions.shape == (20,)


class TestComputeReward:
    def test_resolved_positive(self):
        r = compute_reward(True, -0.1, None, ActionType.WATCH, False)
        assert r > 0

    def test_unresolved_negative(self):
        r = compute_reward(False, 0.2, None, ActionType.WATCH, False)
        assert r < 0

    def test_rollback_penalized(self):
        r_rollback = compute_reward(True, -0.1, None, ActionType.ROLLBACK, False)
        r_adjust = compute_reward(True, -0.1, None, ActionType.ADJUST_LR, False)
        assert r_rollback < r_adjust

    def test_pause_most_penalized(self):
        r_pause = compute_reward(True, -0.1, None, ActionType.PAUSE, False)
        r_watch = compute_reward(True, -0.1, None, ActionType.WATCH, False)
        assert r_pause < r_watch


class TestCQLPolicyUntrained:
    def test_untrained_defers_to_llm(self):
        policy = CQLPolicy()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action, conf = policy.select_action(state, AnomalyType.LOSS_SPIKE)
        assert action is None  # untrained → defer

    def test_should_use_policy_false_when_untrained(self):
        policy = CQLPolicy()
        assert not policy.should_use_policy(AnomalyType.PLATEAU)


class TestPOMDPSpec:
    def test_spec_creation(self):
        spec = POMDPSpec()
        assert spec.S.obs_dim == STATE_DIM
        assert spec.A.n_actions == N_ACTIONS
        assert spec.gamma == 0.99

    def test_paper_text_is_latex(self):
        spec = POMDPSpec()
        text = spec.to_paper_text()
        assert "\\section" in text
        assert "POMDP" in text
        assert "mathcal" in text

    def test_json_serializable(self):
        import json

        spec = POMDPSpec()
        j = spec.to_json()
        parsed = json.loads(j)
        assert "S" in parsed
        assert "A" in parsed
        assert "gamma" in parsed


class TestBeliefUpdater:
    def _make_signals(self, z=0.0, divergence=0.0):
        return SignalSnapshot(
            step=100,
            loss_raw=0.5,
            loss_ema=0.5,
            loss_delta=0.0,
            loss_z_score=z,
            loss_rolling_mean=0.5,
            loss_rolling_std=0.1,
            grad_norm_current=2.0,
            grad_norm_mean=2.0,
            grad_norm_var=0.1,
            grad_norm_z_score=0.0,
            plateau_score=0.5,
            divergence_score=divergence,
            oscillation_score=0.01,
            lr_current=0.0002,
            lr_changed=False,
            window_size=100,
            is_early_training=False,
        )

    def test_initial_belief_healthy(self):
        updater = BeliefUpdater(use_gp=False)
        b = updater.initial_belief()
        assert b.p_healthy > 0.8
        assert b.most_likely_state == "HEALTHY"

    def test_anomaly_increases_p_anomalous(self):
        updater = BeliefUpdater(use_gp=False)
        b = updater.initial_belief()
        signals_ok = self._make_signals(z=0.0, divergence=0.0)
        signals_bad = self._make_signals(z=5.0, divergence=0.9)
        b_after = updater.update(b, signals_bad, n_anomaly_events=2)
        assert b_after.p_anomalous > b.p_anomalous

    def test_entropy_computed(self):
        updater = BeliefUpdater(use_gp=False)
        b = updater.initial_belief()
        assert b.entropy > 0
