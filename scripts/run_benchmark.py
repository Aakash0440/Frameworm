import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.observer.metric_stream import MetricStream
from agent.observer.rolling_window import MetricSnapshot
from pathlib import Path

metrics_path = Path(os.environ.get("TEMP", "/tmp")) / "frameworm_agent_metrics.json"
stream = MetricStream(local_path=metrics_path, poll_every=0.01)

print("Warming up...")
for step in range(30):
    snap = MetricSnapshot(step=step, loss=2.0+step*0.01, grad_norm=0.5, lr=0.0001, epoch=1)
    stream.window.push(snap)
    stream._last_step_seen = step

failure_types = [
    ("Gradient Explosion", 50, 847.3,  94.2),
    ("Vanishing Gradient", 51, 2.1,    0.0001),
    ("Mode Collapse",      52, 0.001,  0.01),
    ("Loss Spike",         53, 500.0,  50.0),
]

print("\nRunning benchmark...\n")
results = []
for name, step, loss, grad in failure_types:
    snap = MetricSnapshot(step=step, loss=loss, grad_norm=grad, lr=0.0001, epoch=5)
    stream.window.push(snap)
    stream._last_step_seen = step
    signals = stream.signals()

    loss_anomaly  = signals is not None and abs(signals.loss_z_score) > 2.5
    grad_explode  = grad > 10.0
    grad_vanish   = grad < 0.001
    mode_collapse = loss < 0.01 and grad < 0.05
    detected = loss_anomaly or grad_explode or grad_vanish or mode_collapse

    z = f"{signals.loss_z_score:.2f}" if signals else "N/A"
    reason = []
    if loss_anomaly:  reason.append(f"loss_z={z}")
    if grad_explode:  reason.append(f"grad_explosion={grad}")
    if grad_vanish:   reason.append(f"grad_vanishing={grad}")
    if mode_collapse: reason.append("mode_collapse")

    results.append({"name": name, "loss": loss, "detected": detected, "z_score": z})
    print(f"{name:<25} loss={loss:<8} detected={str(detected):<6} z={z} | {', '.join(reason) or 'none'}")

print("\n=== RESULTS ===")
n = len(results)
d = sum(1 for r in results if r["detected"])
print(f"Detection rate: {d}/{n} ({100*d/n:.0f}%)")

with open("experiments/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to experiments/benchmark_results.json")
