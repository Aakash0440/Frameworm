# FRAMEWORM SHIFT

> Drop-in distribution drift detection for any ML model in production.

## What it does

SHIFT detects when the data your model receives in production no longer
matches the data it was trained on — before silent performance degradation
becomes a real problem.

It runs three statistical tests (KS, Chi-squared, MMD) per feature,
assigns severity levels (NONE / LOW / MEDIUM / HIGH), and alerts you
through Slack, webhooks, or log files.

---

## Installation

SHIFT lives inside FRAMEWORM. No extra install needed.

```python
from frameworm.shift import ShiftMonitor
```

---

## Quick Start

### 1. Save reference distribution (training time)

```python
from frameworm.shift import ShiftMonitor

monitor = ShiftMonitor("fraud_classifier")
monitor.profile_reference(
    X_train,
    feature_names=["age", "income", "num_transactions", "country"],
    metadata={"model_version": "v2.1", "dataset": "train_2024_q4.csv"}
)
# Saved to experiments/shift_profiles/fraud_classifier.shift
```

### 2. Check for drift (inference time)

```python
# Check a batch of live data
result = monitor.check(X_live_batch)

# result.overall_drifted  →  True / False
# result.overall_severity →  NONE / LOW / MEDIUM / HIGH
# result.drifted_features →  ["income", "num_transactions"]
result.print_summary()
```

### 3. FastAPI middleware (zero code change)

```python
from fastapi import FastAPI
from frameworm.shift.middleware import ShiftMiddleware

app = FastAPI()
app.add_middleware(
    ShiftMiddleware,
    reference="fraud_classifier",
    feature_names=["age", "income", "num_transactions", "country"],
    alert_channels=["slack", "log"],
    window_size=100,
)

@app.post("/predict")
def predict(data: dict):
    return model.predict(data)
# SHIFT runs in background. Your endpoint is unchanged.
```

### 4. CLI

```bash
# Save reference from CSV
frameworm shift profile --data train.csv --name fraud_classifier

# Check live data
frameworm shift check --name fraud_classifier --current live.csv

# Generate HTML + JSON report
frameworm shift report \
  --name fraud_classifier \
  --current live.csv \
  --output reports/drift_2024_q1.html

# List all saved profiles
frameworm shift list
```

---

## Configuration

`configs/shift_config.yaml`

```yaml
shift:
  ks_threshold:        0.05   # p-value cutoff for KS test
  chi2_threshold:      0.05   # p-value cutoff for Chi-squared
  severity_high:       0.01   # p < 0.01  → HIGH
  severity_medium:     0.05   # p < 0.05  → MEDIUM
  severity_low:        0.10   # p < 0.10  → LOW
  alert_on:            ["slack", "log"]
  min_alert_severity:  "MEDIUM"
  log_path:            "experiments/shift_logs"
  profile_dir:         "experiments/shift_profiles"
```

---

## Architecture

```
Your Model
    │
    ├── Training time
    │       X_train ──→ FeatureProfiler ──→ .shift file
    │
    └── Inference time
            X_live  ──→ FeatureProfiler ──→ current profile
                                │
                        DriftEngine.compare(reference, current)
                                │
                        FeatureDriftReport × N features
                                │
                        AlertManager ──→ Slack / webhook / log
```

---

## Severity Levels

| Severity | p-value     | Meaning                         | Action               |
|----------|-------------|----------------------------------|----------------------|
| NONE     | ≥ 0.10      | No meaningful drift              | Nothing              |
| LOW      | 0.05–0.10   | Worth watching                   | Monitor              |
| MEDIUM   | 0.01–0.05   | Real drift, investigate          | Investigate          |
| HIGH     | < 0.01      | Significant drift, act now       | Retrain / alert      |

---

## How SHIFT reuses FRAMEWORM

| FRAMEWORM component              | SHIFT usage                          |
|----------------------------------|--------------------------------------|
| `monitoring/drift_detector.py`   | KS + Chi-squared core tests          |
| `monitoring/ab_testing.py`       | Welch's t-test for severity          |
| Slack webhook integration        | Alert delivery                       |
| `experiments/` folder            | Profiles + logs storage              |
| Plugin system                    | ShiftMiddleware as ASGI plugin       |
| Config system (YAML inheritance) | `shift_config.yaml` block            |

---

## Single-datapoint mode (low-throughput APIs)

```python
# Accumulates datapoints into a 100-point window before checking
for request in incoming_requests:
    result = monitor.check_datapoint(
        request.features,
        feature_names=["age", "income"],
        window_size=100,
    )
    if result:   # fires once per window
        result.print_summary()
```

---

## Tests

```bash
python test_shift_steps1_5.py   # core foundation
python test_shift_steps6_12.py  # full suite (25 tests)
```


================================================================================
WIRE INTO EXISTING CLI (2 lines in cli/main.py)
================================================================================

Add to your existing cli/main.py:

    from shift.cli.commands import register_shift_commands

    # At the bottom, after your existing CLI group is defined:
    register_shift_commands(main)   # replace 'main' with your root Click group name