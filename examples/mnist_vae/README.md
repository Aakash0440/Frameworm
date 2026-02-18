# MNIST VAE - Complete Tutorial

Train a Variational Autoencoder on MNIST from scratch.

**What you'll learn:**
- Data preparation
- Model configuration
- Training with callbacks
- Hyperparameter search
- Model evaluation
- Deployment

**Time:** ~30 minutes  
**Hardware:** CPU or GPU  
**Difficulty:** Beginner

---

## Setup
```bash
pip install frameworm
pip install torchvision  # For MNIST
```

---

## Step 1: Download Data
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

---

## Step 2: Create Config
```yaml
# config.yaml
model:
  type: vae
  in_channels: 1
  latent_dim: 20
  hidden_dim: 128

training:
  epochs: 20
  lr: 0.001
  optimizer: adam

experiment:
  name: mnist-vae
  tags: [mnist, vae, beginner]
```

---

## Step 3: Train
```python
from frameworm import Config, get_model, Trainer
from frameworm.training.callbacks import EarlyStopping, ModelCheckpoint
import torch

# Load config
config = Config.from_file('config.yaml')

# Create model
model = get_model('vae')(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

# Setup trainer
trainer = Trainer(model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')

# Add callbacks
trainer.add_callback(EarlyStopping(monitor='val_loss', patience=5))
trainer.add_callback(ModelCheckpoint(
    filepath='checkpoints/best.pt',
    monitor='val_loss',
    save_best_only=True
))

# Train!
trainer.train(train_loader, test_loader, epochs=config.training.epochs)
```

---

## Step 4: Evaluate
```python
# Generate reconstructions
import matplotlib.pyplot as plt
import numpy as np

model.eval()
with torch.no_grad():
    # Get test batch
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:8].to(trainer.device)
    
    # Reconstruct
    recon, mu, logvar = model(test_images)
    
    # Plot
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        # Original
        axes[0, i].imshow(test_images[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Reconstruction
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Recon')
    
    plt.tight_layout()
    plt.savefig('reconstructions.png')
    print("✓ Saved reconstructions.png")
```

---

## Step 5: Sample from Latent Space
```python
# Generate new digits
z = torch.randn(16, config.model.latent_dim).to(trainer.device)
samples = model.decode(z)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(samples[i, 0].cpu().detach(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig('generated_samples.png')
print("✓ Generated samples")
```

---

## Step 6: Hyperparameter Search
```python
from frameworm.search import GridSearch

def train_fn(config):
    model = get_model('vae')(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    trainer = Trainer(model, optimizer, device='cuda')
    trainer.train(train_loader, test_loader, epochs=5)
    return {'val_loss': trainer.state.val_metrics['loss'][-1]}

search = GridSearch(
    base_config=config,
    search_space={
        'training.lr': [0.0001, 0.001, 0.01],
        'model.latent_dim': [10, 20, 40]
    },
    metric='val_loss',
    mode='min'
)

best_config, best_score = search.run(train_fn)
print(f"Best config: {best_config}")
print(f"Best val_loss: {best_score:.4f}")
```

---

## Step 7: Deploy
```bash
# Export model
frameworm export checkpoints/best.pt --format onnx --output model.onnx

# Serve model
frameworm serve checkpoints/best.pt --port 8000
```

Test the API:
```python
import requests
import numpy as np

# Random latent vector
z = np.random.randn(1, 20).astype(np.float32)

response = requests.post(
    'http://localhost:8000/predict',
    json={'input': z.tolist()}
)

print(f"Generated image shape: {response.json()['shape']}")
```

---

## Results

Expected results after 20 epochs:
- **Reconstruction loss:** ~0.05
- **KL divergence:** ~5-10
- **Generated samples:** Clear digit shapes

---

## Troubleshooting

**Blurry reconstructions:**
- Increase `hidden_dim` to 256
- Train for more epochs
- Lower KL weight (beta-VAE)

**Training unstable:**
- Lower learning rate to 0.0001
- Add gradient clipping: `trainer.enable_gradient_clipping(max_norm=1.0)`

**Out of memory:**
- Reduce `batch_size` to 64
- Reduce `hidden_dim`