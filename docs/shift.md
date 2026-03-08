# FRAMEWORM SHIFT

Drop-in distribution drift detection for any ML model in production. Detects when real-world data no longer matches your training distribution — before silent performance degradation becomes a real problem.

---

## Installation

SHIFT lives inside FRAMEWORM. No extra install needed.

```python
from frameworm.shift import ShiftMonitor
```

---

## Quick Start

### 1. Save reference distribution at training time

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

### 2. Check for drift at inference time

```python
result = monitor.check(X_live_batch)

# result.overall_drifted  →  True / False
# result.overall_severity →  NONE / LOW / MEDIUM / HIGH
# result.drifted_features →  ["income", "num_transactions"]
result.print_summary()
```

### 3. FastAPI middleware — zero changes to your endpoint

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
# SHIFT runs in a background thread. Your endpoint response is never touched.
```

### 4. CLI

```bash
# Save reference from CSV
frameworm shift profile --data train.csv --name fraud_classifier

# Check live data against reference
frameworm shift check --name fraud_classifier --current live.csv

# Generate HTML + JSON drift report
frameworm shift report \
  --name fraud_classifier \
  --current live.csv \
  --output reports/drift_report.html

# List all saved profiles
frameworm shift list
```

---

## Detection Methods

| Feature Type | Test | Output |
|---|---|---|
| Numerical | Kolmogorov-Smirnov | p-value + severity |
| High-dimensional | Maximum Mean Discrepancy | drift score |
| Categorical | Chi-squared | p-value + severity |

---

## Severity Levels

| Severity | p-value | Meaning | Action |
|---|---|---|---|
| NONE | ≥ 0.10 | No meaningful drift | Nothing |
| LOW | 0.05–0.10 | Worth watching | Monitor |
| MEDIUM | 0.01–0.05 | Real drift, investigate | Investigate |
| HIGH | < 0.01 | Significant drift, act now | Retrain / alert |

---

## Single-Datapoint Mode

For low-throughput APIs that receive one request at a time:

```python
# Accumulates datapoints into a window before checking
for request in incoming_requests:
    result = monitor.check_datapoint(
        request.features,
        feature_names=["age", "income"],
        window_size=100,
    )
    if result:   # fires once per complete window
        result.print_summary()
```

---

## Configuration

`configs/shift_config.yaml`

```yaml
shift:
  ks_threshold:        0.05
  chi2_threshold:      0.05
  severity_high:       0.01
  severity_medium:     0.05
  severity_low:        0.10
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
    │       X_train ──> FeatureProfiler ──> .shift file
    │
    └── Inference time
            X_live  ──> FeatureProfiler ──> current profile
                                │
                        DriftEngine.compare(reference, current)
                                │
                        FeatureDriftReport × N features
                                │
                        AlertManager ──> Slack / webhook / log
```

---

## How SHIFT Reuses FRAMEWORM

| FRAMEWORM component | SHIFT usage |
|---|---|
| `monitoring/drift_detector.py` | KS + Chi-squared core tests |
| `monitoring/ab_testing.py` | Welch's t-test for severity |
| Slack webhook integration | Alert delivery |
| `experiments/` folder | Profiles + logs storage |
| Plugin system | ShiftMiddleware as ASGI plugin |
| Config system | `shift_config.yaml` block |

---

## Wire Into CLI

Add 2 lines to `cli/main.py`:

```python
from shift.cli.commands import register_shift_commands
register_shift_commands(main)   # replace 'main' with your root Click group
```

---

## Tests

```bash
python test_shift_steps1_5.py   # core foundation (steps 1-5)
python test_shift_steps6_12.py  # full suite — 25 tests (steps 6-12)
```