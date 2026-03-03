# Advanced Metrics Design

## Metrics for Generative Models

### 1. FID (Fréchet Inception Distance)
Measures quality and diversity of generated images.

**How it works:**
1. Extract features from real and generated images using Inception-V3
2. Model features as multivariate Gaussians
3. Compute Fréchet distance between distributions

**Lower is better** (0 = identical distributions)

### 2. IS (Inception Score)
Measures quality and diversity based on classifier confidence.

**How it works:**
1. Classify generated images with Inception-V3
2. Compute KL divergence between conditional and marginal distributions

**Higher is better** (typical range: 1-10+)

### 3. LPIPS (Learned Perceptual Image Patch Similarity)
Measures perceptual similarity between images.

**How it works:**
1. Extract features from multiple layers of a pretrained network
2. Compute weighted difference between features
3. Average across spatial dimensions

**Lower is better** (0 = identical, 1 = very different)

## Architecture
```python
from frameworm.metrics import FID, InceptionScore, LPIPS

# Compute FID
fid = FID(device='cuda')
score = fid.compute(real_images, generated_images)

# Compute IS
inception_score = InceptionScore(device='cuda')
score, std = inception_score.compute(generated_images)

# Compute LPIPS
lpips = LPIPS(device='cuda')
distance = lpips.compute(image1, image2)
```

## Integration with Experiments
```python
from frameworm.metrics import MetricEvaluator

evaluator = MetricEvaluator(
    metrics=['fid', 'is', 'lpips'],
    real_data=real_loader,
    device='cuda'
)

# Evaluate model
results = evaluator.evaluate(model, num_samples=10000)
# {'fid': 25.3, 'is': 8.5, 'is_std': 0.3, 'lpips': 0.45}
```