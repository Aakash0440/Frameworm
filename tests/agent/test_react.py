"""
Tests for agent/react/ — prompts, parser, verifier, cooldown.
Run with: pytest tests/agent/test_react.py -v
"""

import numpy as np
import pytest
from agent.react.action_parser import ActionParser, ActionType, ParsedAction
from agent.react.prompts import PromptBuilder
from agent.control.cooldown import CooldownManager
from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.observer.rolling_window import RollingWindow, MetricSnapshot


class TestActionParser:
    def setup_method(self):
        self.parser = ActionParser()

    def test_watch(self):
        a = self.parser.parse("THINK: ok\nACT: WATCH\nREASON: ok")
        assert a.action_type == ActionType.WATCH

    def test_adjust_lr(self):
        a = self.parser.parse("THINK: t\nACT: ADJUST_LR(factor=0.3)\nREASON: r")
        assert a.action_type == ActionType.ADJUST_LR
        assert a.params["factor"] == pytest.approx(0.3)

    def test_rollback(self):
        a = self.parser.parse("THINK: t\nACT: ROLLBACK(step=4000)\nREASON: r")
        assert a.action_type == ActionType.ROLLBACK
        assert a.params["step"] == 4000

    def test_swap_scheduler(self):
        a = self.parser.parse("THINK: t\nACT: SWAP_SCHEDULER(name=cosine)\nREASON: r")
        assert a.action_type == ActionType.SWAP_SCHEDULER
        assert a.params["name"] == "cosine"

    def test_alert(self):
        a = self.parser.parse('THINK: t\nACT: ALERT(message="check it")\nREASON: r')
        assert a.action_type == ActionType.ALERT

    def test_garbage_falls_back_to_watch(self):
        a = self.parser.parse("gibberish %%% @@@")
        assert a.action_type == ActionType.WATCH
        assert a.is_fallback

    def test_empty_falls_back_to_watch(self):
        a = self.parser.parse("")
        assert a.action_type == ActionType.WATCH
        assert a.is_fallback

    def test_lr_factor_clamped_max(self):
        a = self.parser.parse("THINK: t\nACT: ADJUST_LR(factor=999)\nREASON: r")
        assert a.params["factor"] <= 2.0

    def test_lr_factor_clamped_min(self):
        a = self.parser.parse("THINK: t\nACT: ADJUST_LR(factor=0.0)\nREASON: r")
        assert a.params["factor"] >= 0.05

    def test_unknown_scheduler_defaults_to_cosine(self):
        a = self.parser.parse("THINK: t\nACT: SWAP_SCHEDULER(name=unicorn)\nREASON: r")
        assert a.params["name"] == "cosine"

    def test_think_reason_extracted(self):
        a = self.parser.parse("THINK: grad exploded\nACT: ADJUST_LR(factor=0.5)\nREASON: reduce lr")
        assert "grad" in a.think.lower()
        assert "reduce" in a.reason.lower()


class TestPromptBuilder:
    def test_contains_model_name(self):
        w = RollingWindow(size=100)
        for i in range(50):
            w.push(MetricSnapshot(step=i, loss=1.0, grad_norm=2.0, lr=0.0002))
        event = AnomalyEvent(
            AnomalyType.LOSS_SPIKE,
            Severity.HIGH,
            step=50,
            loss=3.0,
            grad_norm=5.0,
            lr=0.0002,
        )
        builder = PromptBuilder(model_name="DCGAN")
        prompt = builder.build(event, w, [], 100, 0.5)
        assert "DCGAN" in prompt
        assert "LOSS_SPIKE" in prompt

    def test_history_included(self):
        w = RollingWindow(size=100)
        for i in range(50):
            w.push(MetricSnapshot(step=i, loss=1.0, grad_norm=2.0, lr=0.0002))
        event = AnomalyEvent(AnomalyType.PLATEAU, Severity.MEDIUM, step=50)
        builder = PromptBuilder()
        history = [{"step": 20, "action": "ADJUST_LR", "resolved": True}]
        prompt = builder.build(event, w, history, 100, 0.5)
        assert "ADJUST_LR" in prompt


class TestCooldownManager:
    def test_not_blocked_initially(self):
        cd = CooldownManager(cooldown_steps=100)
        assert not cd.is_blocked(AnomalyType.LOSS_SPIKE)

    def test_blocked_after_register(self):
        cd = CooldownManager(cooldown_steps=100)
        cd.register(AnomalyType.LOSS_SPIKE, step=500)
        assert cd.is_blocked(AnomalyType.LOSS_SPIKE)

    def test_unblocked_after_cooldown(self):
        cd = CooldownManager(cooldown_steps=100)
        cd.register(AnomalyType.LOSS_SPIKE, step=500)
        cd.update(current_step=601)
        assert not cd.is_blocked(AnomalyType.LOSS_SPIKE)

    def test_still_blocked_within_cooldown(self):
        cd = CooldownManager(cooldown_steps=100)
        cd.register(AnomalyType.LOSS_SPIKE, step=500)
        cd.update(current_step=599)
        assert cd.is_blocked(AnomalyType.LOSS_SPIKE)

    def test_clear_all(self):
        cd = CooldownManager(cooldown_steps=100)
        cd.register(AnomalyType.LOSS_SPIKE, step=500)
        cd.register(AnomalyType.PLATEAU, step=600)
        cd.clear_all()
        assert not cd.is_blocked(AnomalyType.LOSS_SPIKE)
        assert not cd.is_blocked(AnomalyType.PLATEAU)

    def test_multiple_types_independent(self):
        cd = CooldownManager(cooldown_steps=100)
        cd.register(AnomalyType.LOSS_SPIKE, step=500)
        assert cd.is_blocked(AnomalyType.LOSS_SPIKE)
        assert not cd.is_blocked(AnomalyType.PLATEAU)
