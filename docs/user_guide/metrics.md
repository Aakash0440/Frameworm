# Advanced Metrics

## Overview

Frameworm provides production-grade metrics for evaluating generative models.

## Available Metrics

### FID (Fréchet Inception Distance)

Measures quality and diversity of generated images.

**Lower is better** (0 = identical distributions)
```python
from frameworm.metrics import FID

fid = FID(device='cuda')
score = fid.compute(real_images, generated_images)
print(f"FID: {score:.2f}")
```

### Inception Score (IS)

Measures quality and diversity based on classifier confidence.

**Higher is better** (typical range: 1-10+)
```python
from frameworm.metrics import InceptionScore

inception_score = InceptionScore(device='cuda')
score, std = inception_score.compute(generated_images)
print(f"IS: {score:.2f} ± {std:.2f}")
```

### LPIPS (Learned Perceptual Similarity)

Measures perceptual similarity between images.

**Lower is better** (0 = identical, 1 = very different)
```python
from frameworm.metrics import LPIPS

lpips = LPIPS(device='cuda')
distance = lpips.compute(image1, image2)
print(f"LPIPS: {distance:.4f}")
```

## Unified Evaluation

### MetricEvaluator

Evaluate with multiple metrics at once:
```python
from frameworm.metrics import MetricEvaluator

evaluator = MetricEvaluator(
    metrics=['fid', 'is', 'lpips'],
    real_data=real_loader,
    device='cuda'
)

results = evaluator.evaluate(model, num_samples=10000)
# {'fid': 25.3, 'is': 8.5, 'is_std': 0.3, 'lpips': 0.45}
```

### Quick Evaluation
```python
from frameworm.metrics import quick_evaluate

results = quick_evaluate(
    model,
    real_data=real_images,
    num_samples=5000,
    device='cuda'
)
```

## Integration with Training

### Automatic Evaluation
```python
from frameworm.training import Trainer
from frameworm.metrics import MetricEvaluator

evaluator = MetricEvaluator(
    metrics=['fid', 'is'],
    real_data=real_loader,
    device='cuda'
)

trainer = Trainer(model, optimizer)
trainer.set_evaluator(evaluator, eval_every=5)

# Automatically evaluates every 5 epochs
trainer.train(train_loader, val_loader, epochs=100)
```

### Manual Evaluation
```python
# Evaluate at specific points
results = evaluator.evaluate(model, num_samples=10000)

# Log to experiment
if trainer.experiment:
    for metric_name, value in results.items():
        trainer.experiment.log_metric(
            f"eval_{metric_name}",
            value,
            epoch=epoch
        )
```

## Best Practices

1. **Use enough samples** - At least 5000-10000 for FID/IS
2. **Match data distribution** - Evaluate on same distribution as training
3. **Track over time** - Monitor metrics during training
4. **Compare fairly** - Same number of samples for all models
5. **Use multiple metrics** - No single metric tells the whole story

## Interpreting Metrics

### FID
- **< 10**: Excellent quality
- **10-30**: Good quality
- **30-50**: Moderate quality
- **> 50**: Poor quality

### IS
- **> 10**: Excellent diversity and quality
- **5-10**: Good
- **< 5**: Poor

### LPIPS
- **< 0.1**: Very similar
- **0.1-0.3**: Moderately similar
- **> 0.3**: Quite different

## Common Issues

### Out of Memory
```python
# Reduce batch size
evaluator = MetricEvaluator(
    metrics=['fid'],
    real_data=real_loader,
    device='cuda',
    batch_size=50  # Reduce from default 100
)
```

### Slow Evaluation
```python
# Use fewer samples for development
results = evaluator.evaluate(model, num_samples=1000)

# Use full samples for final evaluation
results = evaluator.evaluate(model, num_samples=50000)
```

## Examples

See `examples/advanced_metrics_example.py` for complete example.