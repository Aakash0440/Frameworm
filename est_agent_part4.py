import numpy as np
from agent.causal.causal_graph import CausalGraph, NodeStatus
from agent.causal.attribution import AttributionEngine, AttributionMode
from agent.causal.do_intervention import DoIntervention, FreezeVariable
from agent.counterfactual.twin_runner import TwinRunner, ShadowRun
from agent.counterfactual.delta_tracker import DeltaTracker
from agent.counterfactual.eval_report import EvalReportGenerator
from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType, Severity
from agent.observer.rolling_window import RollingWindow, MetricSnapshot
from agent.observer.signal_extractor import SignalExtractor

print("=== FRAMEWORM-AGENT Part 4 Smoke Test ===\n")

# 1. CausalGraph construction + Kahn's sort
print("1. Testing CausalGraph structure...")
graph = CausalGraph()
assert "loss" in graph.nodes
assert "gradient_dist" in graph.nodes
assert len(graph._topo_order) == len(graph.nodes)
assert graph._topo_order[-1] == "loss"   # loss is always the sink
print(f"   Nodes: {list(graph.nodes.keys())} ✓")
print(f"   Topo order: {graph._topo_order} ✓")

# 2. Baseline calibration
print("\n2. Testing baseline calibration...")
window = RollingWindow(size=500)
for i in range(150):
    loss = 1.0 - i * 0.005 + np.random.normal(0, 0.01)
    window.push(MetricSnapshot(step=i, loss=loss, grad_norm=2.0, lr=0.0002))

graph.calibrate_baseline(window, healthy_steps=100)
assert graph.nodes["loss"].baseline_n >= 10
print(f"   loss baseline: mean={graph.nodes['loss'].baseline_mean:.4f} ✓")

# 3. Node evaluation + root cause detection
print("\n3. Testing root cause detection...")
extractor = SignalExtractor()
signals = extractor.extract(window)

# Inject a spike snapshot
from agent.observer.rolling_window import MetricSnapshot
spike_snap = MetricSnapshot(step=150, loss=5.0, grad_norm=25.0, lr=0.0002)

statuses = graph.evaluate_at(spike_snap, signals, step=150)
print(f"   Node statuses: {statuses}")

root_causes = graph.find_root_causes()
print(f"   Root causes: {[rc.name for rc in root_causes]}")
print(f"   Graph summary:\n{graph.summary()}")

# 4. AttributionEngine (FAST mode — no trainer needed)
print("\n4. Testing AttributionEngine FAST mode...")
engine = AttributionEngine(graph=graph)

event = AnomalyEvent(
    anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
    severity=Severity.HIGH,
    step=150,
    loss=5.0,
    grad_norm=25.0,
    lr=0.0002,
    loss_z_score=8.5,
    triggered_rule="grad_norm > threshold",
    triggered_value=25.0,
    threshold_value=10.0,
)

report = engine.attribute(event, spike_snap, signals, window)
print(f"   Mode: {report.mode}")
print(f"   Root causes: {[rc.node_name for rc in report.root_causes]}")
print(f"   Summary: {report.attribution_summary}")
print(f"   Prompt text:\n{report.to_prompt_text()}")

# 5. DeltaTracker
print("\n5. Testing DeltaTracker...")
from pathlib import Path
tracker = DeltaTracker(log_dir=Path("/tmp/fw_test_deltas"))

delta = tracker.record_intervention(
    anomaly_type=AnomalyType.LOSS_SPIKE,
    action_taken="ADJUST_LR(factor=0.5)",
    intervention_step=200,
    run_a_loss=0.42,
    run_a_grad_norm=2.1,
    run_a_fid=12.3,
)
print(f"   Recorded: {delta.intervention_id} ✓")

# Simulate shadow run completing
shadow = ShadowRun(
    run_id="shadow_1_step180",
    spawn_step=180,
    seed=200,
    n_steps=200,
    completed=True,
    final_loss=0.81,
    final_grad_norm=4.2,
    fid_score=31.7,
)
updated = tracker.record_shadow_result(delta.intervention_id, shadow)
assert updated.shadow_available
assert updated.loss_delta == 0.42 - 0.81   # negative = agent improved
assert updated.agent_helped                  # agent helped
print(f"   Delta: loss_delta={updated.loss_delta:+.4f}, fid_delta={updated.fid_delta:+.2f} ✓")
print(f"   Agent helped: {updated.agent_helped} ✓")

# 6. EvalReportGenerator
print("\n6. Testing EvalReportGenerator...")
# Add a few more synthetic deltas
for i in range(5):
    d = tracker.record_intervention(
        AnomalyType.PLATEAU, "SWAP_SCHEDULER(name=cosine)",
        intervention_step=300 + i*100,
        run_a_loss=0.5 + i * 0.02,
        run_a_grad_norm=1.5,
    )
    s = ShadowRun(
        run_id=f"shadow_test_{i}",
        spawn_step=280 + i*100,
        seed=i,
        n_steps=200,
        completed=True,
        final_loss=0.7 + i * 0.02,   # shadow worse than agent
        final_grad_norm=2.0,
    )
    tracker.record_shadow_result(d.intervention_id, s)

gen = EvalReportGenerator(tracker, log_dir=Path("/tmp/fw_test_reports"))
report = gen.generate()
print(f"   Total interventions: {report.n_total_interventions}")
print(f"   With shadow: {report.n_with_shadow}")
print(f"   Success rate: {report.overall_success_rate:.1%}")
print(f"   Mean loss delta: {report.overall_mean_loss_delta:+.4f}")
print(f"   Is significant: {report.is_overall_significant}")
print(f"\n   Markdown table preview:")
print(report.to_markdown_table()[:500])

print("\n✓ All Part 4 tests passed.")
print("\nKey next step: run 10+ training experiments with the agent active,")
print("then call EvalReportGenerator.generate() to get paper-ready numbers.")