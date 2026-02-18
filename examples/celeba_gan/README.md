# CelebA Face Generation with DCGAN

Generate realistic celebrity faces using DCGAN.

**What you'll learn:**
- Large-scale GAN training
- Data augmentation
- Progressive training
- FID evaluation
- Model deployment

**Time:** ~2-4 hours (GPU)  
**Hardware:** GPU with 8GB+ VRAM  
**Difficulty:** Advanced

---

## Setup
```bash
# Download CelebA dataset
wget https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view

# Or use torchvision (easier)
pip install frameworm gdown
```

---

## Step 1: Data Preparation
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CelebA(
    root='./data',
    split='train',
    transform=transform,
    download=True
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print(f"Dataset size: {len(dataset)}")
```

---

## Step 2: Training with FRAMEWORM
```python
from frameworm import Config, get_model, Trainer
from frameworm.training.callbacks import ModelCheckpoint
from frameworm.experiment import Experiment
import torch

# Config
config = Config.from_dict({
    'model': {
        'type': 'dcgan',
        'latent_dim': 100,
        'image_size': 64,
        'image_channels': 3,
        'gen_hidden': 64,
        'disc_hidden': 64
    }
})

# Model
model = get_model('dcgan')(config)

# Separate optimizers for G and D
optimizer_g = torch.optim.Adam(
    model.generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)
optimizer_d = torch.optim.Adam(
    model.discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# Trainer (GAN requires custom training loop)
trainer = Trainer(model, optimizer_g, device='cuda')

# Track experiment
exp = Experiment(
    name='celeba-dcgan',
    config=config.to_dict(),
    tags=['celeba', 'dcgan', 'faces']
)

with exp:
    # Train for 50 epochs
    for epoch in range(50):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to('cuda')
            
            # Train discriminator
            optimizer_d.zero_grad()
            
            # Real images
            real_logits = model.discriminator(real_images)
            d_loss_real = torch.mean(torch.nn.functional.relu(1.0 - real_logits))
            
            # Fake images
            z = torch.randn(real_images.size(0), config.model.latent_dim).to('cuda')
            fake_images = model.generator(z)
            fake_logits = model.discriminator(fake_images.detach())
            d_loss_fake = torch.mean(torch.nn.functional.relu(1.0 + fake_logits))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            z = torch.randn(real_images.size(0), config.model.latent_dim).to('cuda')
            fake_images = model.generator(z)
            fake_logits = model.discriminator(fake_images)
            g_loss = -torch.mean(fake_logits)
            g_loss.backward()
            optimizer_g.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
                
                exp.log_metric('d_loss', d_loss.item(), epoch=epoch)
                exp.log_metric('g_loss', g_loss.item(), epoch=epoch)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
```

---

## Step 3: Generate Samples
```python
import matplotlib.pyplot as plt
import torchvision.utils as vutils

model.eval()
with torch.no_grad():
    # Generate 64 samples
    z = torch.randn(64, config.model.latent_dim).to('cuda')
    fake_images = model.generator(z)
    
    # Create grid
    grid = vutils.make_grid(
        fake_images.cpu(),
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig('generated_faces.png', bbox_inches='tight', dpi=150)
    print("✓ Generated faces saved")
```

---

## Step 4: Evaluate with FID
```python
from frameworm.metrics import FID

fid_calculator = FID(device='cuda')

# Compute FID between real and generated images
real_images_list = []
for batch, _ in dataloader:
    real_images_list.append(batch)
    if len(real_images_list) >= 50:  # 6400 images
        break

real_images = torch.cat(real_images_list, dim=0)[:6400]

# Generate same number of fake images
fake_images_list = []
for _ in range(50):
    with torch.no_grad():
        z = torch.randn(128, config.model.latent_dim).to('cuda')
        fakes = model.generator(z).cpu()
        fake_images_list.append(fakes)

fake_images = torch.cat(fake_images_list, dim=0)[:6400]

fid_score = fid_calculator(real_images, fake_images)
print(f"FID Score: {fid_score:.2f}")

exp.log_metric('fid', fid_score, metric_type='final')
```

---

## Step 5: Deploy API
```bash
# Export generator
frameworm export checkpoints/epoch_50.pt --format torchscript

# Serve
frameworm serve checkpoints/epoch_50.pt --port 8000
```

Test generation:
```python
import requests
import numpy as np
from PIL import Image

# Random latent vector
z = np.random.randn(1, 100).astype(np.float32)

response = requests.post(
    'http://localhost:8000/predict',
    json={'input': z.tolist()}
)

# Get generated image
image_data = np.array(response.json()['output'])
image = Image.fromarray((image_data * 255).astype(np.uint8))
image.save('api_generated.png')
```

---

## Expected Results

After 50 epochs:
- **FID Score:** 20-30 (lower is better)
- **Generated faces:** Recognizable facial features
- **Training time:** ~3-4 hours on RTX 3090

---

## Tips for Better Results

1. **Train longer:** 100+ epochs for best quality
2. **Progressive growing:** Start at 32x32, grow to 64x64
3. **Spectral normalization:** Add to discriminator for stability
4. **Self-attention:** Add attention layers at 16x16 resolution
5. **Mixed precision:** Use for 2x speedup