# Day 22: MLOps & Production Monitoring

## Systems to Build

### 1. Prometheus Metrics Exporter
Expose training + inference metrics in Prometheus format.
Grafana can then visualize them in real-time dashboards.

Metrics:
- Training: loss, lr, epoch, samples/sec
- Inference: latency_ms, throughput, error_rate
- System: GPU util, memory, CPU

### 2. Model Registry
Version models with semantic versioning.
Track lineage: dataset → config → training run → model.
Compare model versions.
Promote models: dev → staging → production.

### 3. Data Drift Detector
Detect when input distribution shifts from training data.
Use statistical tests: KS test, MMD, Chi-squared.
Alert when drift detected.

### 4. A/B Testing Framework
Route inference traffic between model versions.
Collect metrics per version.
Statistical significance testing.

### 5. Model Audit Logger
Log all predictions with inputs/outputs.
Compliance and debugging.
Async buffered writes.

## Architecture
Production Traffic
↓
Load Balancer
↓
A/B Router ──────────→ Model Version A
↓                 Model Version B
Audit Logger
↓
Drift Detector
↓
Prometheus Exporter → Grafana Dashboard
