# FRAMEWORM DEPLOY

> One command from trained model to production API — with drift detection,
> latency monitoring, and auto-rollback built in.

## What it does

DEPLOY takes any FRAMEWORM model checkpoint and gives you:
- A generated FastAPI server with the correct input/output schema for your model type
- An optimised Docker image (multi-stage build, non-root user, health check baked in)
- p50/p95/p99 latency tracking on every request
- SHIFT drift monitoring auto-attached
- Automatic rollback if latency spikes or error rate climbs

## Quick Start

```bash
# Deploy a DCGAN model
frameworm deploy start \
  --model experiments/checkpoints/best.pt \
  --name face_generator \
  --type dcgan \
  --version v1.2 \
  --shift face_generator \
  --build-docker

# Check status
frameworm deploy status --name face_generator

# Promote to production
frameworm deploy promote --name face_generator --version v1.2 --stage production

# Manual rollback
frameworm deploy rollback --name face_generator

# Stop and archive
frameworm deploy stop --name face_generator
```

## Model Lifecycle

```
dev → staging → production → archived
```

Every version is tracked in the model registry with: git hash, config snapshot,
dataset checksum, and training metrics — so you always know exactly what's running.

## Auto-Rollback

DEPLOY watches every deployed model for:
- **p95 latency** exceeding threshold (default: 2000ms) for 3 consecutive checks
- **Error rate** exceeding threshold (default: 10%) for 3 consecutive checks

When either condition is met, DEPLOY automatically:
1. Looks up the previous production version
2. Stops the current container
3. Starts the previous version
4. Fires a Slack alert with reason and metrics

## Generated Server Structure

```
deploy/generated/<name>/
├── server.py            ← model-type-aware FastAPI server
├── requirements.txt     ← pinned dependencies
├── Dockerfile           ← multi-stage, non-root, HEALTHCHECK included
└── docker-compose.yml   ← one-command local deployment
```

## API Endpoints (every generated server)

| Endpoint   | Method | Description                        |
|------------|--------|------------------------------------|
| /predict   | POST   | Run inference                      |
| /health    | GET    | Liveness — always 200 while alive  |
| /ready     | GET    | Readiness — 503 until model loaded |
| /metrics   | GET    | p50/p95/p99 + error rate           |

## FRAMEWORM Integration

| Existing piece          | DEPLOY usage                             |
|-------------------------|------------------------------------------|
| Model checkpoints       | Exported to TorchScript/ONNX             |
| SHIFT ShiftMonitor      | Auto-attached to every deployed model    |
| Slack integration       | Rollback + degradation alerts            |
| experiments/ DB         | Deployment history + lineage             |
| Monitoring (p50/p95/p99)| Latency tracking per endpoint            |
| Model registry          | dev→staging→production→archived          |

## Tests

```bash
python test_deploy_steps6_10.py   # 14 tests, no pytest needed
```
