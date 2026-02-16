# Model Deployment

## Overview

Deploy FRAMEWORM models to production with export, serving, and containerization.

## Export Models

### TorchScript
```python
from frameworm.deployment import ModelExporter

exporter = ModelExporter(model, example_input)
exporter.to_torchscript('model.pt', method='trace')
```

### ONNX
```python
exporter.to_onnx('model.onnx', opset_version=14)
```

### Quantization
```python
quantized = exporter.quantize('model_quant.pt', method='dynamic')
```

## Serve Models

### FastAPI Server
```python
from frameworm.deployment import ModelServer

server = ModelServer('model.pt')
server.run(host='0.0.0.0', port=8000)
```

Or via CLI:
```bash
python -m frameworm.deployment.server --model model.pt --port 8000
```

### API Endpoints

- `POST /predict` - JSON prediction
- `POST /predict/image` - Image prediction
- `POST /predict/batch` - Batch prediction
- `GET /health` - Health check
- `GET /docs` - API documentation

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

## Docker Deployment

### Build Image
```bash
docker build -t model-server .
```

### Run Container
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models model-server
```

### Docker Compose
```bash
docker-compose up -d
```

## Kubernetes Deployment

### Deploy
```bash
kubectl apply -f k8s/deployment.yaml
```

### Check Status
```bash
kubectl get pods
kubectl get services
```

### Scale
```bash
kubectl scale deployment model-server --replicas=5
```

## Production Best Practices

1. **Use quantization** for faster inference
2. **Enable caching** for repeated requests
3. **Set up monitoring** (Prometheus/Grafana)
4. **Use load balancing** (nginx/K8s service)
5. **Implement rate limiting**
6. **Add authentication** for sensitive models

## Performance Optimization

### Batch Processing

Process multiple samples together:
```python
# Better: batch of 32
output = model(batch_of_32)

# Slower: one at a time
for sample in samples:
    output = model(sample)
```

### ONNX Runtime

2-5x faster inference:
```python
from frameworm.deployment import ONNXInferenceSession

session = ONNXInferenceSession('model.onnx')
output = session.run(input_data)
```

### GPU Inference

Enable CUDA for faster serving:
```python
server = ModelServer('model.pt', device='cuda')
```