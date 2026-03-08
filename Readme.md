# FRAMEWORM

**Production-Grade Generative AI Framework**

[![PyPI](https://img.shields.io/pypi/v/frameworm)](https://pypi.org/project/frameworm/)
[![Python](https://img.shields.io/pypi/pyversions/frameworm)](https://python.org)
[![Tests](https://img.shields.io/github/workflow/status/Aakash0440/frameworm/tests)](https://github.com/Aakash0440/frameworm/actions)
[![Coverage](https://img.shields.io/codecov/c/github/Aakash0440/frameworm)](https://codecov.io/gh/Aakash0440/frameworm)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://frameworm.readthedocs.io)
[![License](https://img.shields.io/github/license/Aakash0440/frameworm)](https://github.com/Aakash0440/frameworm/blob/main/LICENSE)

---

## The Product Family

FRAMEWORM is not a single tool. It is four products that cover the full ML lifecycle — from training to production.

```
FRAMEWORM          → core ML infrastructure
FRAMEWORM AGENT    → autonomous training monitor
FRAMEWORM SHIFT    → distribution drift detection
FRAMEWORM DEPLOY   → one-command model deployment
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
| Config system | YAML inheritance: base.yaml → model.yaml → experiment.yaml |
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

### Benchmark — 10 Experiments

Each run paired with a shadow run (no agent) as counterfactual comparison.

```
#   Experiment           Condition        Agent Loss   Shadow Loss    D Loss   Resolved
─────────────────────────────────────────────────────────────────────────────────────────
1   Baseline-1           baseline            70.39        70.37        -0.02       -
2   Baseline-2           baseline            72.61        70.35        -2.27       -
3   Baseline-3           baseline            72.48        70.39        -2.09       -
4   HighLR-1             lr=8e-02            75.71      9999.00     +9923.29      YES
5   HighLR-2             lr=5e-02            69.99        70.37        +0.37      YES
6   GradExplosion-1      grad_explosion      75.93        70.35        -5.58      YES
7   GradExplosion-2      grad_explosion      76.07        70.35        -5.72      YES
8   LowLR-Plateau-1      lr=1e-06            94.65       283.47      +188.82      NO*
9   LowLR-Plateau-2      lr=5e-07            95.31       285.57      +190.26      NO*
10  AgentIntervenes      lr=1e-01            75.84      9999.00     +9923.16      YES
─────────────────────────────────────────────────────────────────────────────────────────
Baseline avg loss:              71.83
Runs with anomalies detected:   7 / 10
Anomalies resolved by agent:    5 / 7
Resolution rate:                71.4%
False positives on clean runs:  0 / 3
```

**Key results:**
- Experiments 4 and 10: agent loss ~75 vs shadow loss 9999 — runs that would have fully diverged were saved
- Experiments 6 and 7: gradient explosion detected and recovered, final loss within 8% of healthy baseline
- 0 false positives on clean baseline runs

*Plateau experiments (8, 9): agent detected and intervened 16 times each, reducing loss from ~285 to ~95 (+190 improvement). Full convergence within epoch budget is a known limitation and direction for future work.

---

## FRAMEWORM SHIFT

Drop-in distribution drift detection. Two lines to attach to any model in production.

```python
from frameworm.shift import ShiftMonitor

monitor = ShiftMonitor.from_reference("experiments/my_model.shift")
monitor.check(incoming_batch)   # fires alert automatically if drift detected
```

**Three integration surfaces:**

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

Severity levels: `NONE` · `LOW` · `MEDIUM` · `HIGH`
Alerts via: Slack · webhook · log file · stdout

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

## Why FRAMEWORM?

| | FRAMEWORM | PyTorch Lightning | BentoML | MLflow |
|---|---|---|---|---|
| Training infrastructure | ✅ | ✅ | ❌ | ⚠️ |
| Autonomous training monitor | ✅ AGENT | ❌ | ❌ | ❌ |
| Drift detection | ✅ SHIFT | ❌ | ❌ | ⚠️ |
| One-command deployment | ✅ DEPLOY | ❌ | ✅ | ⚠️ |
| Auto-rollback | ✅ | ❌ | ❌ | ❌ |
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
```

---

## Documentation

- [Quick Start](https://aakash0440.github.io/Frameworm/getting-started/quickstart/)
- [FRAMEWORM AGENT](https://aakash0440.github.io/Frameworm/agent/)
- [FRAMEWORM SHIFT](https://aakash0440.github.io/Frameworm/shift/)
- [FRAMEWORM DEPLOY](https://aakash0440.github.io/Frameworm/deploy/)
- [API Reference](https://aakash0440.github.io/Frameworm/api-reference/core/)

---

## License

MIT — see [LICENSE](https://github.com/Aakash0440/frameworm/blob/main/LICENSE)

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