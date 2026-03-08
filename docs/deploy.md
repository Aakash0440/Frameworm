# FRAMEWORM DEPLOY

One command from trained model to production API — with drift detection, latency monitoring, and auto-rollback built in.

---

## Quick Start

```bash
frameworm deploy start \
  --model experiments/checkpoints/best.pt \
  --name face_generator \
  --type dcgan \
  --version v1.2 \
  --shift face_generator \
  --build-docker
```

That one command:

- Exports your model to TorchScript + ONNX
- Generates a FastAPI server with the correct input/output schema for your architecture
- Builds a multi-stage Docker image (non-root user, HEALTHCHECK baked in)
- Starts p50/p95/p99 latency tracking on every request
- Auto-attaches FRAMEWORM SHIFT drift monitoring
- Starts a background rollback controller

---

## All Commands

```bash
# Deploy
frameworm deploy start --model experiments/checkpoints/best.pt --name my_model

# Check status of all versions
frameworm deploy status --name my_model

# Promote a version to production
frameworm deploy promote --name my_model --version v2.0 --stage production

# Manual rollback to previous version
frameworm deploy rollback --name my_model

# Stop and archive
frameworm deploy stop --name my_model
```

---

## Every Deployed Model Gets

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run inference |
| `/health` | GET | Liveness — always 200 while alive |
| `/ready` | GET | Readiness — 503 until model is loaded |
| `/metrics` | GET | Live p50/p95/p99 + error rate |

---

## Auto-Rollback

DEPLOY watches every deployed model in a background thread. It checks every 30 seconds.

**Triggers rollback when:**
- p95 latency exceeds threshold (default: 2000ms) for 3 consecutive checks
- Error rate exceeds threshold (default: 10%) for 3 consecutive checks

**On rollback, automatically:**
1. Looks up the previous production version in the registry
2. Stops the current Docker container
3. Starts the previous version's container
4. Promotes the old version back to production in the registry
5. Fires a Slack alert with reason, p95 value, and timestamp
6. Writes the event to `experiments/deploy_logs/` as a JSONL entry

No human needed.

---

## Model-Aware Server Generation

Generic deployment tools (BentoML, TorchServe) treat every model identically. FRAMEWORM DEPLOY knows all 6 built-in architectures and generates architecture-specific inference code.

| Architecture | Input | Output |
|---|---|---|
| VAE | Image tensor (B, C, H, W) | Reconstruction + mu + log_var |
| DCGAN | Noise vector (B, latent_dim) | Generated images |
| DDPM | batch_size + num_steps | Denoised images |
| VQ-VAE-2 | Image tensor | Reconstruction + commitment loss |
| ViT-GAN | Noise vector | Generated images |
| CFG-DDPM | batch_size + class_labels + guidance_scale | Conditional generated images |

---

## Model Lifecycle

```
dev → staging → production → archived
```

Every version is tracked in the model registry with: git hash, config snapshot, dataset checksum, and training metrics — so you always know exactly what is running in production and where it came from.

---

## Generated Server Structure

```
deploy/generated/<name>/
├── server.py            ← model-type-aware FastAPI server
├── requirements.txt     ← pinned dependencies
├── Dockerfile           ← multi-stage, non-root, HEALTHCHECK included
└── docker-compose.yml   ← one-command local deployment
```

---

## How DEPLOY Reuses FRAMEWORM

| Existing piece | DEPLOY usage |
|---|---|
| Model checkpoints | Exported to TorchScript + ONNX |
| SHIFT ShiftMonitor | Auto-attached to every deployed model |
| Slack integration | Rollback + degradation alerts |
| `experiments/` DB | Deployment history + lineage |
| p50/p95/p99 monitoring | Latency tracking per endpoint |
| Model registry | dev → staging → production → archived lifecycle |

---

## Configuration

`configs/deploy_config.yaml`

```yaml
deploy:
  export_format:         ["torchscript", "onnx"]
  quantize:              false
  latency_threshold_ms:  2000
  error_rate_threshold:  0.10
  rollback_checks:       3
  monitor_interval_s:    30
  alert_on:              ["slack", "log"]
  log_path:              "experiments/deploy_logs"
  registry_path:         "experiments/model_registry"
```

---

## Tests

```bash
python test_deploy_steps6_10.py   # 14 tests, no pytest needed
```