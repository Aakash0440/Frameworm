# Production Deployment Blueprint

Complete production deployment with monitoring, scaling, and CI/CD.

**What you'll learn:**
- Kubernetes deployment
- Load balancing
- Auto-scaling
- Monitoring with Grafana
- CI/CD with GitHub Actions

**Time:** ~1 hour  
**Prerequisites:** Docker, Kubernetes, basic DevOps knowledge

---

## Architecture
GitHub → CI/CD → Docker Registry → Kubernetes Cluster
↓
[3 API Pods]
↓
Load Balancer
↓
[Prometheus + Grafana Monitoring]

---

## Step 1: Dockerize Application
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Install FRAMEWORM
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run server
CMD ["python", "-m", "frameworm.ui.api", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
# Build and push
docker build -t frameworm-api:v1.0.0 .
docker tag frameworm-api:v1.0.0 your-registry/frameworm-api:v1.0.0
docker push your-registry/frameworm-api:v1.0.0
```

---

## Step 2: Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frameworm-api
  labels:
    app: frameworm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frameworm
  template:
    metadata:
      labels:
        app: frameworm
    spec:
      containers:
      - name: api
        image: your-registry/frameworm-api:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        env:
        - name: MODEL_PATH
          value: "/models/best.pt"
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: frameworm-service
spec:
  selector:
    app: frameworm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frameworm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frameworm-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -w
```

---

## Step 3: Monitoring with Prometheus + Grafana
```yaml
# k8s/monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: frameworm-metrics
spec:
  selector:
    matchLabels:
      app: frameworm
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

Grafana Dashboard JSON:
```json
{
  "dashboard": {
    "title": "FRAMEWORM Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(frameworm_inference_requests_total[5m])"
          }
        ]
      },
      {
        "title": "p95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, frameworm_inference_latency_ms)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(frameworm_inference_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

## Step 4: CI/CD with GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t frameworm-api:${{ github.ref_name }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag frameworm-api:${{ github.ref_name }} your-registry/frameworm-api:${{ github.ref_name }}
        docker push your-registry/frameworm-api:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/frameworm-api \
          api=your-registry/frameworm-api:${{ github.ref_name }}
        kubectl rollout status deployment/frameworm-api
```

---

## Step 5: Load Testing
```python
# locustfile.py
from locust import HttpUser, task, between

class FramewormUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate_image(self):
        import numpy as np
        z = np.random.randn(1, 100).tolist()
        self.client.post("/predict", json={"input": z})
    
    @task(2)
    def health_check(self):
        self.client.get("/health")
```

Run load test:
```bash
locust -f locustfile.py --host http://your-lb-url --users 100 --spawn-rate 10
```

---

## Observability Checklist

- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards configured
- [ ] Alerts for high error rate (>1%)
- [ ] Alerts for high latency (p95 >500ms)
- [ ] Alerts for low throughput
- [ ] Log aggregation (ELK/Datadog)
- [ ] Distributed tracing (Jaeger)
- [ ] Uptime monitoring (UptimeRobot)

---

## Cost Optimization

### 1. Use Spot Instances (AWS)
```yaml
# Add node affinity for spot instances
nodeSelector:
  node.kubernetes.io/instance-type: "spot"
```

### 2. Autoscaling
```yaml
# Scale down during off-hours
minReplicas: 1  # Night
maxReplicas: 10 # Peak hours
```

### 3. Model Quantization
```bash
# Reduce model size by 4x
frameworm export model.pt --format onnx --quantize
```

---

## Disaster Recovery

### 1. Backup Strategy
```bash
# Daily model backups to S3
kubectl create cronjob model-backup \
  --image=amazon/aws-cli \
  --schedule="0 2 * * *" \
  -- aws s3 sync /models s3://frameworm-backups/models
```

### 2. Blue-Green Deployment
```bash
# Deploy new version alongside old
kubectl apply -f k8s/deployment-v2.yaml

# Gradually shift traffic
kubectl patch service frameworm-service -p '{"spec":{"selector":{"version":"v2"}}}'

# Rollback if needed
kubectl patch service frameworm-service -p '{"spec":{"selector":{"version":"v1"}}}'
```