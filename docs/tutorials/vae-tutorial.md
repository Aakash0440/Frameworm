# VAE Tutorial

Complete guide to training a Variational Autoencoder (VAE) with FRAMEWORM.

---

## What You'll Learn

- VAE architecture and theory
- Data preparation
- Model configuration
- Training and evaluation
- Latent space exploration
- Image generation

---

## Prerequisites
```bash
pip install frameworm torchvision matplotlib
```

---

## Step 1: Understanding VAEs

A Variational Autoencoder (VAE) is a generative model that learns to:

1. **Encode** images into a latent space
2. **Sample** from the latent distribution
3. **Decode** samples back to images

### Architecture
Image → Encoder → μ, σ → Sample z → Decoder → Reconstruction

---

## Step 2: Project Setup
```bash
frameworm init vae-mnist --template vae
cd vae-mnist
```

---

## Step 3: Configuration

Edit `configs/config.yaml`:
```yaml
model:
  type: vae
  latent_dim: 128
  hidden_dim: 256
  image_channels: 1
  image_size: 64

training:
  epochs: 100
  batch_size: 128
  lr: 0.001
  device: cuda

optimizer:
  type: adam
  betas: [0.9, 0.999]
```

---

## Step 4: Data Preparation
```python
# scripts/prepare_data.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(config):
    transform = transforms.Compose([
        transforms.Resize(config.model.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        'data',
        train=False,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    return train_loader, val_loader
```

---

## Step 5: Training

=== "CLI"
```bash
    frameworm train \
      --config configs/config.yaml \
      --experiment vae-mnist-v1 \
      --gpus 0
```

=== "Python Script"
```python
    # scripts/train.py
    
    from frameworm import Trainer, Config, get_model
    from frameworm.experiment import Experiment
    import torch.optim as optim
    from prepare_data import get_mnist_loaders
    
    # Load configuration
    config = Config('configs/config.yaml')
    
    # Get data
    train_loader, val_loader = get_mnist_loaders(config)
    
    # Create model
    vae = get_model('vae')(config)
    optimizer = optim.Adam(vae.parameters(), lr=config.training.lr)
    
    # Create experiment
    experiment = Experiment(
        name='vae-mnist-v1',
        config=config,
        tags=['vae', 'mnist'],
        description='VAE on MNIST dataset'
    )
    
    # Train
    with experiment:
        trainer = Trainer(vae, optimizer, device='cuda')
        trainer.set_experiment(experiment)
        trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    
    print(f"Training complete! Experiment: {experiment.experiment_id}")
```

---

## Step 6: Monitor Training

Launch dashboard to see real-time progress:
```bash
frameworm dashboard --port 8080
```

Navigate to http://localhost:8080 and watch:

- Training/validation loss curves
- Reconstruction quality
- Resource usage

---

## Step 7: Evaluate Model
```python
# scripts/evaluate.py

from frameworm.metrics import MetricEvaluator, FID
import torch

# Load best checkpoint
checkpoint = torch.load('experiments/vae-mnist-v1/checkpoints/best.pt')
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

# Compute FID score
evaluator = MetricEvaluator(
    metrics=['fid'],
    real_data=val_loader,
    device='cuda'
)

results = evaluator.evaluate(vae, num_samples=10000)
print(f"FID Score: {results['fid']:.2f}")
```

Expected FID on MNIST: 10-30

---

## Step 8: Generate Images
```python
# scripts/generate.py

import torch
import matplotlib.pyplot as plt

vae.eval()

# Sample from latent space
with torch.no_grad():
    z = torch.randn(64, config.model.latent_dim).cuda()
    generated = vae.decode(z)
    
# Plot
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = generated[i].cpu().squeeze()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.savefig('generated_images.png')
```

---

## Step 9: Explore Latent Space
```python
# scripts/latent_space.py

import numpy as np

# Interpolate between two images
img1 = train_dataset[0][0].unsqueeze(0).cuda()
img2 = train_dataset[1][0].unsqueeze(0).cuda()

with torch.no_grad():
    z1 = vae.encode(img1)[0]  # Get mean
    z2 = vae.encode(img2)[0]
    
    # Interpolate
    alphas = np.linspace(0, 1, 10)
    interpolated = []
    
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        img = vae.decode(z)
        interpolated.append(img.cpu())

# Visualize interpolation
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(interpolated[i].squeeze(), cmap='gray')
    ax.axis('off')

plt.savefig('latent_interpolation.png')
```

---

## Step 10: Export & Deploy
```bash
# Export model
frameworm export \
  experiments/vae-mnist-v1/checkpoints/best.pt \
  --format onnx \
  --quantize

# Serve
frameworm serve exported/model.pt --port 8000
```

---

## Results

After 100 epochs, you should see:

- **Training Loss**: ~85
- **Validation Loss**: ~88
- **FID Score**: 15-25
- **Sample Quality**: Clear, recognizable digits

---

## Troubleshooting

??? question "Posterior collapse"
    KL divergence goes to zero. Solutions:
    
    - Use β-VAE: scale KL term
    - Warm-up KL weight
    - Reduce latent dimension

??? question "Blurry reconstructions"
    MSE loss causes blur. Try:
    
    - Perceptual loss
    - GAN discriminator
    - Higher capacity decoder

---

## Next Steps

- [GAN Tutorial](gan-tutorial.md) - Adversarial training
- [DDPM Tutorial](ddpm-tutorial.md) - Diffusion models
- [Hyperparameter Search](../user_guide/hyperparameter-search.md) - Optimize performance