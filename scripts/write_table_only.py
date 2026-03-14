"""
Regenerate results table from existing experiment_log.jsonl
Run this instead of re-running all 30 experiments.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("experiments/cifar_benchmark")
LOG_FILE = RESULTS_DIR / "experiment_log.jsonl"
TABLE_FILE = RESULTS_DIR / "results_table.txt"

results = []
with open(LOG_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

# sort by id
results.sort(key=lambda r: r["id"])

W = 120
lines = []
lines.append("=" * W)
lines.append("  FRAMEWORM AGENT -- 30-Experiment Benchmark Results (CIFAR-10)")
lines.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
lines.append("=" * W)
hdr = (
    f"{'#':>3}  {'Experiment':<20} {'Arch':<6} {'Condition':<14} "
    f"{'Agent Loss':>10} {'Shadow Loss':>11} {'Delta':>10} "
    f"{'Det':>5} {'Int':>5} {'Res':>4}"
)
lines.append(hdr)
lines.append("-" * W)

by_arch = {"VAE": [], "DCGAN": []}
for r in results:
    sign = "+" if r["delta"] >= 0 else ""
    row = (
        f"{r['id']:>3}  {r['name']:<20} {r['arch']:<6} "
        f"{r['condition']:<14} {r['agent_loss']:>10.4f} "
        f"{r['shadow_loss']:>11.4f} {sign}{r['delta']:>9.4f} "
        f"{r['n_detected']:>5} {r['n_intervened']:>5} {r['resolved']:>4}"
    )
    lines.append(row)
    by_arch[r["arch"]].append(r)

lines.append("-" * W)
lines.append("  SUMMARY")

for arch, rs in by_arch.items():
    if not rs:
        continue
    baseline = [r for r in rs if "baseline" in r["name"].lower()]
    base_avg = sum(r["agent_loss"] for r in baseline) / len(baseline) if baseline else float("nan")
    resolved = sum(1 for r in rs if r["resolved"] == "YES")
    det_all = sum(1 for r in rs if r["n_detected"] > 0)
    fp = sum(1 for r in baseline if r["n_detected"] > 0)
    lines.append(f"  {arch}:")
    lines.append(f"    Baseline avg loss:          {base_avg:.4f}")
    lines.append(f"    Anomalies detected:         {det_all} / {len(rs)}")
    lines.append(f"    Runs resolved by agent:     {resolved} / {len(rs)}")
    lines.append(f"    False positives (baseline): {fp} / {len(baseline)}")

all_resolved = sum(1 for r in results if r["resolved"] == "YES")
non_baseline = [r for r in results if "baseline" not in r["name"].lower()]
meaningful = [r for r in non_baseline if r["delta"] > 0]
lines.append(
    f"  OVERALL resolution rate:      {all_resolved} / {len(results)}"
    f"  ({100*all_resolved/len(results):.1f}%)"
)
lines.append(f"  Runs where agent helped:      {len(meaningful)} / {len(non_baseline)}")
lines.append("=" * W)

text = "\n".join(lines)
TABLE_FILE.write_text(text, encoding="utf-8")
print(text)
print(f"\n[*] Saved to {TABLE_FILE}")
