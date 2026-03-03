
import numpy as np
from pathlib import Path
from agent.policy.experience_buffer import (
    ExperienceBuffer, Transition, encode_state,
    compute_reward, STATE_DIM, N_ACTIONS, ACTION_INDEX
)
from agent.policy.cql_policy import CQLPolicy, CQLConfig, train_cql_policy
from agent.policy.policy_eval import PolicyEvaluator
from agent.pomdp.state_space import POMDPSpec
from agent.pomdp.belief_updater import BeliefUpdater
from agent.classifier.anomaly_types import AnomalyType, Severity
from agent.react.action_parser import ActionType
from agent.observer.signal_extractor import SignalExtractor, SignalSnapshot
from agent.observer.rolling_window import RollingWindow, MetricSnapshot

print("=== FRAMEWORM-AGENT Part 5 Smoke Test ===\n")

# 1. State encoding
print("1. Testing state encoding...")
window = RollingWindow(size=200)
for i in range(120):
    window.push(MetricSnapshot(step=i, loss=1.0-i*0.005, grad_norm=2.0, lr=0.0002))
extractor = SignalExtractor()
signals = extractor.extract(window)
state = encode_state(signals, AnomalyType.LOSS_SPIKE)
assert state.shape == (STATE_DIM,)
assert state.dtype == np.float32
print(f"   State vector shape: {state.shape} ✓")
print(f"   One-hot section (last 6): {state[-6:]} ✓")

# 2. Reward computation
print("\n2. Testing reward computation...")
r_good = compute_reward(True, -0.1, -5.0, ActionType.ADJUST_LR, False)
r_bad  = compute_reward(False, 0.2, 2.0, ActionType.ROLLBACK, True)
assert r_good > r_bad, f"Good reward {r_good} should exceed bad {r_bad}"
print(f"   Good intervention reward: {r_good:.2f} ✓")
print(f"   Bad intervention reward:  {r_bad:.2f} ✓")

# 3. ExperienceBuffer
print("\n3. Testing ExperienceBuffer...")
buffer = ExperienceBuffer(
    db_path=Path("/tmp/fw_test_experience.db"),
    max_memory=1000,
)
for i in range(120):
    t = Transition(
        state=np.random.randn(STATE_DIM).astype(np.float32),
        action=np.random.randint(N_ACTIONS),
        reward=np.random.uniform(-2, 3),
        next_state=np.random.randn(STATE_DIM).astype(np.float32),
        done=False,
        step=i,
        anomaly_type=np.random.choice(["LOSS_SPIKE","PLATEAU","GRADIENT_EXPLOSION"]),
        action_name=np.random.choice(["ADJUST_LR","WATCH","ROLLBACK"]),
        loss_delta=np.random.uniform(-0.3, 0.3),
        resolved=bool(np.random.randint(2)),
    )
    buffer.add(t)
assert len(buffer) == 120
assert buffer.is_ready
print(f"   Buffer size: {len(buffer)} ✓")
stats = buffer.stats()
print(f"   Stats: {stats}")

# 4. CQL Policy (mini training)
print("\n4. Testing CQL Policy mini training...")
try:
    cfg = CQLConfig(
        max_epochs=3, batch_size=16, patience=5,
        save_dir="/tmp/fw_test_policy"
    )
    policy = CQLPolicy(config=cfg, min_samples=5)
    history = train_cql_policy(policy, buffer, verbose=False)
    assert policy.is_trained
    assert len(history["td_loss"]) > 0
    print(f"   Training epochs: {len(history['td_loss'])}, "
          f"final loss: {history['td_loss'][-1]:.4f} ✓")

    # Action selection
    action, conf = policy.select_action(state, AnomalyType.LOSS_SPIKE)
    if action is not None:
        print(f"   Policy action: {action.name} (conf={conf:.2f}) ✓")
    else:
        print("   Policy deferred to LLM (not enough per-type samples) ✓")

    # Policy evaluation
    evaluator = PolicyEvaluator(policy, buffer, log_dir=Path("/tmp/fw_test_policy"))
    result = evaluator.evaluate()
    print(f"   Win rate vs LLM: {result.overall_win_rate:.1%} ✓")

except ImportError:
    print("   PyTorch not installed — skipping CQL training test")

# 5. POMDP spec
print("\n5. Testing POMDPSpec...")
spec = POMDPSpec()
paper_text = spec.to_paper_text()
assert "POMDP" in paper_text
assert "\\section" in paper_text
json_spec = spec.to_json()
print(f"   POMDPSpec JSON: {len(json_spec)} chars ✓")
print(f"   Paper text: {len(paper_text)} chars (paste into Section 3) ✓")

# 6. BeliefUpdater
print("\n6. Testing BeliefUpdater...")
updater = BeliefUpdater(use_gp=False)  # no GP for test speed
belief = updater.initial_belief()
assert belief.p_healthy > 0.8
print(f"   Initial belief: {belief} ✓")

# Simulate healthy steps
for i in range(120):
    window.push(MetricSnapshot(step=i+120, loss=0.5, grad_norm=1.5, lr=0.0002))
signals2 = extractor.extract(window)
for _ in range(10):
    belief = updater.update(belief, signals2, n_anomaly_events=0)
print(f"   After 10 healthy updates: {belief} ✓")
assert belief.p_healthy > 0.5

# Inject anomaly
signals2_spike = SignalSnapshot(
    step=200, loss_raw=3.0, loss_ema=1.5, loss_delta=0.8,
    loss_z_score=5.0, loss_rolling_mean=0.5, loss_rolling_std=0.1,
    grad_norm_current=15.0, grad_norm_mean=2.0, grad_norm_var=1.0,
    grad_norm_z_score=6.5, plateau_score=0.5, divergence_score=0.9,
    oscillation_score=0.05, lr_current=0.0002, lr_changed=False,
    window_size=200, is_early_training=False,
)
belief_anomaly = updater.update(belief, signals2_spike, n_anomaly_events=2)
print(f"   After anomaly injection: {belief_anomaly} ✓")
assert belief_anomaly.p_anomalous > belief.p_anomalous

print("\n✓ All Part 5 tests passed.")
print("\nSystem summary:")
print("  Parts 1-2: Reactive agent (observer + classifier + ReAct + control)")
print("  Part 3:    Proactive forecaster (predict failures before they happen)")
print("  Part 4:    Causal attribution + counterfactual evaluation")
print("  Part 5:    CQL meta-policy (learns from history) + POMDP formalization")
print("\nOne part left: benchmark/ + tests/ + CLI (Part 6)")
print("Then your agent is complete and paper-ready.")