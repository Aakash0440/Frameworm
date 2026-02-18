# Frequently Asked Questions

Common questions about FRAMEWORM.

---

## General

### What is FRAMEWORM?

FRAMEWORM is a production-ready deep learning framework focused on generative models (VAE, GAN, Diffusion). It provides:
- Built-in models
- Complete MLOps stack
- Production deployment tools
- Extensive monitoring and tracking

### Who is FRAMEWORM for?

- Researchers building generative models
- ML engineers deploying models to production
- Teams needing reproducible experiments
- Anyone wanting batteries-included generative AI

### How does it compare to PyTorch Lightning?

**FRAMEWORM** is specialized for generative models with built-in MLOps.  
**Lightning** is general-purpose and requires more setup for production.

See [detailed comparison](comparisons.md).

### Is FRAMEWORM production-ready?

Yes! Includes:
- Health checks & graceful shutdown
- Rate limiting & authentication
- Model versioning & A/B testing
- Kubernetes deployment
- Monitoring with Prometheus

---

## Installation

### What Python versions are supported?

Python 3.8, 3.9, 3.10, 3.11

### Do I need a GPU?

No, but recommended for training. All features work on CPU.

### How do I install with dependencies?
```bash
pip install frameworm[all]  # Everything
pip install frameworm[wandb]  # Just W&B
pip install frameworm[deployment]  # Just deployment tools
```

---

## Training

### How do I resume training from checkpoint?
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

trainer = Trainer(model, optimizer)
trainer.train(train_loader, epochs=50, start_epoch=checkpoint['epoch'])
```

### Can I use custom optimizers?

Yes, pass any PyTorch optimizer:
```python
from torch_optimizer import Ranger

optimizer = Ranger(model.parameters())
trainer = Trainer(model, optimizer)
```

### How do I track experiments?
```python
from frameworm.experiment import Experiment

with Experiment(name='my-exp') as exp:
    trainer.set_experiment(exp)
    trainer.train(...)
```

Automatically tracks:
- Config
- Metrics
- Git commit
- System info

### How do I save/load models?
```bash
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model = get_model('vae')(config)
model.load_state_dict(torch.load('model.pt'))
```

---

## Distributed Training

### How do I use multiple GPUs?

**Single machine:**
```python
model = torch.nn.DataParallel(model)
```

**Multiple machines:**
```python
model = torch.nn.parallel.DistributedDataParallel(model)
```

See [distributed training guide](examples/distributed_training/).

### Does mixed precision work with distributed?

Yes! FRAMEWORM supports both together:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# Use in training loop
```

---

## Deployment

### How do I deploy to production?
```bash
# 1. Export
frameworm export model.pt --format onnx

# 2. Serve
frameworm serve model.pt --port 8000

# 3. Deploy to K8s
kubectl apply -f k8s/deployment.yaml
```

See [production deployment guide](examples/production_deployment/).

### Can I use Docker?

Yes, example Dockerfile:
```dockerfile
FROM python:3.10-slim
RUN pip install frameworm
COPY model.pt /app/
CMD ["frameworm", "serve", "/app/model.pt"]
```

### How do I monitor in production?

Built-in Prometheus metrics:
```python
from frameworm.monitoring import MetricsExporter

exporter = MetricsExporter(port=9090)
exporter.start()
# Metrics at http://localhost:9090/metrics
```

Connect to Grafana for visualization.

---

## Customization

### How do I create a custom model?
```python
import torch.nn as nn
from frameworm.core import register_model

@register_model('my_model')
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        return x
    
    def compute_loss(self, x, y=None):
        # Must return dict with 'loss' key
        return {'loss': ...}
```

### Can I add custom metrics?
```python
from frameworm.metrics import register_metric

@register_metric('my_score')
class MyMetric:
    def __call__(self, real, fake):
        # Compute metric
        return score
```

### How do I create plugins?
```bash
frameworm plugins create my-plugin
cd frameworm_plugins/my-plugin
# Edit __init__.py and plugin.yaml
frameworm plugins load my-plugin
```

---

## Troubleshooting

### Training is slow

1. **Check data loading:**
```python
   loader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

2. **Enable mixed precision:**
```python
   trainer.enable_mixed_precision()
```

3. **Use gradient accumulation:**
```python
   trainer.enable_gradient_accumulation(steps=4)
```

### Out of memory errors

1. **Reduce batch size**
2. **Enable gradient checkpointing:**
```python
   trainer.enable_gradient_checkpointing()
```
3. **Use gradient accumulation** to simulate larger batches

### Validation loss not decreasing

1. **Check learning rate** (try 10x smaller/larger)
2. **Add learning rate scheduler**
3. **Check data normalization**
4. **Visualize samples** to ensure data is correct

### Import errors

Make sure dependencies are installed:
```bash
pip install frameworm[all]
```

Or install specific integration:
```bash
pip install wandb  # For W&B
pip install mlflow  # For MLflow
```

---

## Performance

### How fast is FRAMEWORM?

Comparable to PyTorch Lightning (~5% faster due to optimized data pipeline).

See [benchmarks](comparisons.md#performance-benchmark).

### Can I use FP16/BF16?

Yes:
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(x)
```

Or use trainer helper:
```python
trainer.enable_mixed_precision()
```

### How do I profile my code?
```python
from frameworm.utils.profiler import TrainingProfiler

profiler = TrainingProfiler()
with profiler:
    trainer.train(...)

profiler.print_report()
```

---

## Integration

### Does FRAMEWORM work with W&B?

Yes:
```python
from frameworm.integrations import WandBIntegration

trainer.add_callback(WandBIntegration(project='my-project'))
```

### Can I use my existing Lightning code?

Partial compatibility. See [migration guide](comparisons.md#migration-guide).

### Does it support TensorBoard?

Yes, via callbacks:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# Use in custom callback
```

---

## Contributing

### How do I contribute?

1. Fork the repo
2. Create feature branch
3. Make changes + tests
4. Submit PR

See [CONTRIBUTING.md](../CONTRIBUTING.md).

### I found a bug

Please [open an issue](https://github.com/Aakash0440/frameworm/issues) with:
- Python/PyTorch versions
- Minimal reproduction code
- Error message

### Can I request features?

Yes! [Create a feature request](https://github.com/Aakash0440/frameworm/issues/new?template=feature_request.md).

---

## License & Commercial Use

### What license is FRAMEWORM?

MIT License - free for commercial use.

### Can I use FRAMEWORM in my company?

Yes! No restrictions.

### Do I need to cite FRAMEWORM?

Not required, but appreciated:
```bibtex
@software{frameworm2026,
  title={FRAMEWORM: Production-Ready Generative AI Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/Aakash0440/frameworm}
}
```

---

## Still have questions?

- **Discord**: https://discord.gg/frameworm
- **GitHub Discussions**: https://github.com/Aakash0440/frameworm/discussions
- **Email**: Aakashali0440@gmail.com