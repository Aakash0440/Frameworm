# FRAMEWORM
### Production-Grade Generative AI Framework

[![PyPI](https://img.shields.io/pypi/v/frameworm)](https://pypi.org/project/frameworm/)
[![Python](https://img.shields.io/pypi/pyversions/frameworm)](https://pypi.org/project/frameworm/)
[![Tests](https://github.com/Aakash0440/Frameworm/actions/workflows/tests.yml/badge.svg)](https://github.com/Aakash0440/Frameworm/actions)
[![License](https://img.shields.io/github/license/Aakash0440/frameworm)](LICENSE)

---

## The Product Family

FRAMEWORM is not a single tool. It is five products that cover the full ML lifecycle — from training to production.

```
FRAMEWORM          → core ML infrastructure
FRAMEWORM AGENT    → autonomous training monitor
FRAMEWORM SHIFT    → distribution drift detection
FRAMEWORM DEPLOY   → one-command model deployment
FRAMEWORM COST     → per-request inference cost tracking
```

---

## FRAMEWORM (Core)

Production-grade ML infrastructure. Dependency graph engine, Bayesian HPO, mixed precision training, TorchScript + ONNX export, and 6 built-in generative architectures.

**Built-in architectures:** VAE · DCGAN · DDPM · VQ-VAE-2 · ViT-GAN · CFG-DDPM

```bash
pip install frameworm
frameworm init my-project
cd my-project
frameworm train --config config.yaml
```

| Feature | Details |
|---|---|
| Dependency graph engine | Kahn's topological sort, DFS cycle detection, parallel execution |
| Experiment tracking | Git hash + dataset checksum snapshots, SQLite backend |
| Hyperparameter search | Grid, random, Bayesian — no external server needed |
| Config system | YAML inheritance: `base.yaml` → `model.yaml` → `experiment.yaml` |
| Plugin system | Drop in new architectures without touching core |
| Export | TorchScript + ONNX, optional INT8 quantization |

---

## FRAMEWORM AGENT

Autonomous training monitor. Watches loss curves and gradient norms in real time, detects anomalies, and intervenes automatically.

```bash
frameworm agent start --experiment my_run
```

| Anomaly | Detection | Agent Action |
|---|---|---|
| Gradient explosion | Grad norm threshold | Cut LR 95%, reinit if dead |
| Loss divergence | Rolling window comparison | Reduce LR 85% |
| Training plateau | Near-zero variance at high loss | Full model reinit + healthy LR |
| Oscillation | Coefficient of variation | Reduce LR 90% |
| Loss spike | Statistical outlier | Watch mode |

**Benchmark — 30 Experiments on CIFAR-10 across VAE + DCGAN**

Each run paired with a shadow run (no agent) as counterfactual comparison.

```
Architecture    Condition          Agent Loss   Shadow Loss    Δ Loss   Resolved
────────────────────────────────────────────────────────────────────────────────
VAE             Baseline           ~74.9        ~70.4          clean       -
DCGAN           High LR            2.75         87.70         +84.99      YES
DCGAN           High LR            3.20        100.00         +96.79      YES
DCGAN           Agent Intervenes   2.10         21.40         +19.30      YES
────────────────────────────────────────────────────────────────────────────────
Resolution rate:      33.3% overall  |  100% on catastrophic divergence
False positives:      0 / 3 DCGAN baseline runs
```

Key result: every GAN run that would have fully diverged (shadow loss 87–100) was stabilized by the agent (final loss 2–3). Paper on arXiv.

---

## FRAMEWORM SHIFT

Drop-in distribution drift detection. Two lines to attach to any model in production.

```python
from frameworm.shift import ShiftMonitor

monitor = ShiftMonitor.from_reference("experiments/my_model.shift")
monitor.check(incoming_batch)   # fires alert automatically if drift detected
```

Three integration surfaces:

```python
# SDK
monitor = ShiftMonitor("my_model")
monitor.check(batch)

# FastAPI middleware — zero changes to your endpoint
app.add_middleware(ShiftMiddleware, reference="my_model")

# CLI
frameworm shift check --reference train.csv --current live.csv
frameworm shift report --reference train.csv --current live.csv --output report.html
```

| Feature Type | Test | Output |
|---|---|---|
| Numerical | Kolmogorov-Smirnov | p-value + severity |
| High-dimensional | Maximum Mean Discrepancy | drift score |
| Categorical | Chi-squared | p-value + severity |

**Severity levels:** NONE · LOW · MEDIUM · HIGH  
**Alerts via:** Slack · webhook · log file · stdout

---

## FRAMEWORM DEPLOY

One command from trained model to production API.

```bash
frameworm deploy start --model experiments/checkpoints/best.pt --name my_model
```

What that single command does:

- Exports model to TorchScript + ONNX
- Generates a model-aware FastAPI server (correct schema per architecture)
- Builds a multi-stage Docker image with HEALTHCHECK baked in
- Starts p50/p95/p99 latency tracking on every request
- Auto-attaches FRAMEWORM SHIFT drift monitoring
- Starts a background rollback controller

**Auto-rollback:** p95 latency spike for 3 consecutive checks → stops bad container, starts previous known-good version, fires Slack alert. No human needed.

Every deployed model exposes:

```
GET /predict   → inference
GET /health    → liveness
GET /ready     → readiness
GET /metrics   → p50/p95/p99 + error rate
```

```bash
frameworm deploy stop     --name my_model
frameworm deploy status   --name my_model
frameworm deploy rollback --name my_model
frameworm deploy promote  --name my_model --version v2.0 --stage production
```

**Model-aware server generation.** FRAMEWORM DEPLOY knows all 6 built-in architectures and generates architecture-specific inference code. A DDPM server runs a 1000-step denoising loop. A VAE server unpacks three return values. A CFG-DDPM server accepts guidance scale. Generic tools can't do this.

---

## FRAMEWORM COST

Per-request inference cost tracking. Know exactly what your model costs before your cloud bill tells you.

```python
from cost import CostTracker

tracker = CostTracker(model_name="my-dcgan", architecture="dcgan", hardware="t4")

with tracker.track():
    output = model(input)

print(tracker.last_cost.total_cost_usd)   # cost of that single request
print(tracker.monthly_projection(rps=10)) # projected monthly bill
```

```bash
# CLI
frameworm cost estimate --arch dcgan --hardware t4 --latency 38
frameworm cost compare --latency 50 --hardware t4
frameworm cost report costs.json
```

**Example output:**

```
Architecture    Hardware    Latency    Cost/request    Monthly (10 rps)
────────────────────────────────────────────────────────────────────────
dcgan           t4          38ms       $0.0000078      $201/mo
ddpm            t4          380ms      $0.000078        $2,014/mo
vae             t4          12ms       $0.0000025       $64/mo
```

**Auto-alerts** when projected monthly cost crosses your threshold. Drop-in FastAPI middleware tracks every request automatically. Architecture-aware — knows the difference between a DCGAN and a DDPM inference cost.

| Feature | Details |
|---|---|
| Per-request cost | Latency × hardware rate × architecture multiplier |
| Monthly projection | Live estimate at any req/s |
| Optimization hints | Batching, quantization, architecture swap suggestions |
| FastAPI middleware | Auto-tracks every `/predict` request |
| Slack alerts | Fires when cost threshold exceeded |
| Cost dashboard | Live HTML dashboard at `/cost/dashboard` |

---

## Why FRAMEWORM?

| | FRAMEWORM | PyTorch Lightning | BentoML | MLflow |
|---|---|---|---|---|
| Training infrastructure | ✅ | ✅ | ❌ | ⚠️ |
| Autonomous training monitor | ✅ AGENT | ❌ | ❌ | ❌ |
| Drift detection | ✅ SHIFT | ❌ | ❌ | ⚠️ |
| One-command deployment | ✅ DEPLOY | ❌ | ✅ | ⚠️ |
| Auto-rollback | ✅ | ❌ | ❌ | ❌ |
| Inference cost tracking | ✅ COST | ❌ | ❌ | ❌ |
| Model-aware server generation | ✅ | ❌ | ❌ | ❌ |
| Experiment lineage | ✅ | ⚠️ | ❌ | ✅ |
| External services required | ❌ None | ⚠️ | ⚠️ | ⚠️ |

---

## Quick Start

```bash
pip install frameworm

# Train a model
frameworm train --config config.yaml

# Monitor training autonomously
frameworm agent start --experiment my_run

# Check for drift in production
frameworm shift check --reference train.csv --current live.csv

# Deploy to production
frameworm deploy start --model experiments/checkpoints/best.pt --name my_model

# Track inference costs
frameworm cost estimate --arch dcgan --hardware t4 --latency 38
```

---

## Documentation

- [Quick Start](docs/quickstart.md)
- [FRAMEWORM AGENT](docs/agent.md)
- [FRAMEWORM SHIFT](docs/shift.md)
- [FRAMEWORM DEPLOY](docs/deploy.md)
- [FRAMEWORM COST](docs/cost.md)
- [API Reference](docs/api.md)

---

## License

MIT — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@software{frameworm2026,
  title  = {FRAMEWORM: Production-Grade Generative AI Framework},
  author = {Aakash Ali},
  year   = {2026},
  url    = {https://github.com/Aakash0440/frameworm}
}
```