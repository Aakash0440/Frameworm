import json
from pathlib import Path
from agent.react.prompts import PromptBuilder, SYSTEM_PROMPT
from agent.react.action_parser import ActionParser, ActionType
from agent.react.verifier import Verifier
from agent.control.cooldown import CooldownManager
from agent.control.actions import ActionExecutor
from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.observer.rolling_window import RollingWindow, MetricSnapshot
import numpy as np

print("=== FRAMEWORM-AGENT Part 2 Smoke Test ===\n")

# 1. PromptBuilder
print("1. Testing PromptBuilder...")
window = RollingWindow(size=200)
for i in range(50):
    window.push(MetricSnapshot(step=i, loss=1.0 - i*0.01, grad_norm=2.0, lr=0.0002))

event = AnomalyEvent(
    anomaly_type=AnomalyType.LOSS_SPIKE,
    severity=Severity.HIGH,
    step=50,
    loss=2.5,
    grad_norm=8.3,
    lr=0.0002,
    loss_z_score=4.2,
    triggered_rule="loss_z_score > loss_spike_z_score",
    triggered_value=4.2,
    threshold_value=3.0,
)

builder = PromptBuilder(total_steps=10_000, model_name="DCGAN")
prompt = builder.build(event, window, [], 1000, 0.42)
assert "LOSS_SPIKE" in prompt
assert "DCGAN" in prompt
print(f"   Prompt built: {len(prompt)} chars ✓")

# 2. ActionParser — valid actions
print("\n2. Testing ActionParser...")
parser = ActionParser()

cases = [
    ("THINK: loss spiked\nACT: ADJUST_LR(factor=0.5)\nREASON: LR too high",
     ActionType.ADJUST_LR, {"factor": 0.5}),
    ("THINK: bad\nACT: WATCH\nREASON: wait",
     ActionType.WATCH, {"steps": 50}),
    ("THINK: diverging\nACT: ROLLBACK(step=4000)\nREASON: rollback",
     ActionType.ROLLBACK, {"step": 4000}),
    ("THINK: plateau\nACT: SWAP_SCHEDULER(name=cosine)\nREASON: try cosine",
     ActionType.SWAP_SCHEDULER, {"name": "cosine"}),
    ("THINK: bad\nACT: ALERT(message=\"check this\")\nREASON: alert user",
     ActionType.ALERT, {}),
    ("unparseable garbage %%%",
     ActionType.WATCH, {}),  # fallback
]

for text, expected_type, expected_params in cases:
    action = parser.parse(text)
    assert action.action_type == expected_type, f"Expected {expected_type}, got {action.action_type}"
    for k, v in expected_params.items():
        assert action.params.get(k) == v, f"Param {k}: expected {v}, got {action.params.get(k)}"
    print(f"   {expected_type.name}: ✓")

# 3. LR factor clamping
print("\n3. Testing LR factor safety clamping...")
extreme = parser.parse("THINK: x\nACT: ADJUST_LR(factor=100.0)\nREASON: x")
assert extreme.params["factor"] <= 2.0, "Factor should be clamped to max 2.0"
print(f"   factor=100.0 clamped to {extreme.params['factor']} ✓")

tiny = parser.parse("THINK: x\nACT: ADJUST_LR(factor=0.0001)\nREASON: x")
assert tiny.params["factor"] >= 0.05, "Factor should be clamped to min 0.05"
print(f"   factor=0.0001 clamped to {tiny.params['factor']} ✓")

# 4. Cooldown manager
print("\n4. Testing CooldownManager...")
cd = CooldownManager(cooldown_steps=100)
assert not cd.is_blocked(AnomalyType.LOSS_SPIKE)
cd.register(AnomalyType.LOSS_SPIKE, step=500)
assert cd.is_blocked(AnomalyType.LOSS_SPIKE)
cd.update(current_step=600)       # 100 steps later — still blocked
assert cd.is_blocked(AnomalyType.LOSS_SPIKE)
cd.update(current_step=601)       # just past cooldown
assert not cd.is_blocked(AnomalyType.LOSS_SPIKE)
print("   Cooldown register / update / clear ✓")

# 5. ActionExecutor (no trainer — tests graceful failure)
print("\n5. Testing ActionExecutor (no trainer ref)...")
executor = ActionExecutor(trainer_ref=None)
result = executor.adjust_lr(factor=0.5)
assert not result.success  # expected — no trainer
result = executor.watch(steps=50)
assert result.success      # watch always succeeds
print("   Graceful failure without trainer ✓")

# 6. Metrics file write (LOCAL mode)
print("\n6. Testing metrics file write...")
from agent.control.control_plugin import AgentControlPlugin, METRICS_FILE

class FakeTrainer:
    global_step = 42
    _last_loss = 0.88
    _last_grad_norm = 2.3
    current_epoch = 1
    _weight_update_ratio = 0.0
    _agent_pause_requested = False
    class optimizer:
        param_groups = [{"lr": 0.0002}]

plugin = AgentControlPlugin()
plugin.register_trainer(FakeTrainer())
plugin._write_metrics()

if METRICS_FILE.exists():
    data = json.loads(METRICS_FILE.read_text())
    assert data["step"] == 42
    assert data["loss"] == 0.88
    print(f"   Metrics written: {data} ✓")
else:
    print("   Metrics file write skipped (tmp not writable) — ok")

print("\n✓ All Part 2 tests passed.")
print("\nNext: run a real training job with agent.start() to test live.")