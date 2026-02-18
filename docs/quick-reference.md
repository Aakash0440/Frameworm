# Quick Reference

One-page cheat sheet for FRAMEWORM.

---

## Installation
```bash
pip install frameworm              # Core
pip install frameworm[all]         # Everything
pip install frameworm[wandb]       # W&B integration
```

---

## Basic Training
```python
from frameworm import Config, get_model, Trainer
import torch

# 1. Configure
config = Config.from_dict({
    'model': {'type': 'vae', 'latent_dim': 64}
})

# 2. Create model
model = get_model('vae')(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Train
trainer = Trainer(model, optimizer, device='cuda')
trainer.train(train_loader, val_loader, epochs=50)
```

---

## Models
```python
get_model('vae')        # Variational Autoencoder
get_model('dcgan')      # Deep Convolutional GAN
get_model('ddpm')       # Denoising Diffusion
get_model('vqvae2')     # Vector Quantized VAE
get_model('vitgan')     # Vision Transformer GAN
get_model('cfg_ddpm')   # Classifier-Free Guidance DDPM
```

---

## Callbacks
```python
from frameworm.training.callbacks import (
    EarlyStopping, ModelCheckpoint, LRScheduler
)

trainer.add_callback(EarlyStopping(patience=5))
trainer.add_callback(ModelCheckpoint('best.pt', save_best_only=True))
```

---

## Experiment Tracking
```python
from frameworm.experiment import Experiment

with Experiment(name='my-exp', tags=['vae', 'mnist']) as exp:
    trainer.set_experiment(exp)
    trainer.train(train_loader, val_loader, epochs=50)
    
    # Metrics logged automatically
    print(f"Experiment ID: {exp.experiment_id}")
```

---

## Hyperparameter Search
```python
from frameworm.search import GridSearch

search = GridSearch(
    base_config=config,
    search_space={
        'training.lr': [0.0001, 0.001, 0.01],
        'model.latent_dim': [16, 32, 64]
    },
    metric='val_loss',
    mode='min'
)

best_config, best_score = search.run(train_fn)
```

---

## Distributed Training
```python
# Single command for DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py
```

---

## Export & Deploy
```bash
# Export
frameworm export model.pt --format onnx

# Serve
frameworm serve model.pt --port 8000

# Test
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [[...]]}'
```

---

## CLI Commands
```bash
frameworm train --config config.yaml
frameworm experiment list
frameworm experiment compare exp1 exp2
frameworm export model.pt --format torchscript
frameworm serve model.pt --port 8000
frameworm plugins list
frameworm search --config search.yaml
```

---

## Integrations
```python
# W&B
from frameworm.integrations import WandBIntegration
trainer.add_callback(WandBIntegration(project='my-project'))

# S3 Storage
from frameworm.integrations import S3Storage
trainer.add_callback(S3Storage(bucket='my-bucket'))

# Slack
from frameworm.integrations import SlackNotifier
trainer.add_callback(SlackNotifier(webhook_url='...'))
```

---

## Production
```python
# Health checks
from frameworm.production import HealthChecker

health = HealthChecker()
health.add_readiness_check('model', check_model_loaded)

# Rate limiting
from frameworm.production import RateLimiter

limiter = RateLimiter(max_requests=100, window_seconds=60)
if limiter.allow(user_id):
    process_request()
```

---

## Plugin Development
```python
from frameworm.plugins.hooks import HookRegistry

@HookRegistry.on('on_epoch_end')
def my_callback(trainer, epoch, metrics):
    print(f"Epoch {epoch} done!")

# Or create plugin
frameworm plugins create my-plugin
```

---

## Configuration (YAML)
```yaml
model:
  type: vae
  in_channels: 3
  latent_dim: 128
  hidden_dim: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 128
  optimizer: adam

experiment:
  name: my-experiment
  tags: [vae, celeba]

callbacks:
  - type: early_stopping
    patience: 10
  - type: checkpoint
    filepath: best.pt
```

---

## Environment Variables
```bash
FRAMEWORM_CACHE_DIR=/path/to/cache
FRAMEWORM_LOG_LEVEL=INFO
FRAMEWORM_DEVICE=cuda
WANDB_API_KEY=your_key
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Troubleshooting

### Out of Memory
```python
trainer.enable_gradient_checkpointing()
trainer.enable_gradient_accumulation(steps=4)
```

### Slow Data Loading
```python
loader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

### Unstable Training
```python
trainer.enable_gradient_clipping(max_norm=1.0)
trainer.add_callback(LRScheduler(scheduler))
```

---

## Getting Help

- **Docs**: https://frameworm.readthedocs.io
- **Examples**: https://github.com/yourusername/frameworm/tree/main/examples
- **Issues**: https://github.com/yourusername/frameworm/issues
- **Discord**: https://discord.gg/frameworm
- **Email**: support@frameworm.ai

---

## License

MIT License - Free for commercial use
