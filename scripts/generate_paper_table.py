import json

with open("experiments/benchmark_results.json") as f:
    results = json.load(f)

print()
print("=" * 65)
print("TABLE 1: FRAMEWORM-AGENT Anomaly Detection Benchmark Results")
print("=" * 65)
print(f"{'Failure Type':<25} {'Loss':<10} {'Z-Score':<10} {'Detected'}")
print("-" * 65)

for r in results:
    detected = "YES" if r["detected"] else "NO"
    print(f"{r['name']:<25} {r['loss']:<10} {str(r['z_score']):<10} {detected}")

n = len(results)
d = sum(1 for r in results if r["detected"])
print("-" * 65)
print(f"Overall Detection Rate: {d}/{n} ({100*d/n:.0f}%)")
print("=" * 65)
