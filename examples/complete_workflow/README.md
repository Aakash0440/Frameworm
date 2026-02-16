# Complete Workflow Example

End-to-end example using FRAMEWORM CLI.

## Setup
```bash
# Create project
frameworm init vae-mnist --template vae
cd vae-mnist
```

## 1. Prepare Data

Download MNIST to `data/` directory.

## 2. Train Model
```bash
frameworm train \
  --config configs/config.yaml \
  --gpus 0,1,2,3 \
  --experiment vae-baseline
```

## 3. Monitor Training

In another terminal:
```bash
frameworm monitor experiments/vae-baseline
```

## 4. Hyperparameter Search
```bash
# Create search space
cat > configs/search.yaml << 'YAML'
training.lr:
  type: real
  low: 0.0001
  high: 0.01
  log: true

training.batch_size:
  type: integer
  low: 64
  high: 256
  log: true
YAML

# Run search
frameworm search \
  --config configs/config.yaml \
  --space configs/search.yaml \
  --method bayesian \
  --trials 50 \
  --parallel 4
```

## 5. Evaluate Best Model
```bash
frameworm evaluate \
  --config configs/config.yaml \
  --checkpoint experiments/best/checkpoints/best.pt \
  --metrics fid,is \
  --num-samples 50000
```

## 6. Export Model
```bash
frameworm export \
  experiments/best/checkpoints/best.pt \
  --format onnx \
  --quantize \
  --benchmark
```

## 7. Serve Model
```bash
frameworm serve exported/model.pt --port 8000
```

## 8. Test API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[...]]}'
```

## 9. Deploy with Docker
```bash
# Build
docker build -t vae-server .

# Run
docker run -p 8000:8000 vae-server
```

## 10. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get services
```