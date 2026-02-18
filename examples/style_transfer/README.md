# Neural Style Transfer with FRAMEWORM

Transfer artistic style from one image to another.

**What you'll learn:**
- Custom loss functions
- Perceptual losses
- Feature extraction
- Optimization-based approach

**Time:** ~20 minutes  
**Hardware:** GPU recommended  
**Difficulty:** Intermediate

---

## Implementation
```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


class StyleTransfer:
    """
    Neural Style Transfer using VGG19.
    
    Based on "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        
        # Layers for style and content
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_layers = ['conv4_2']
        
        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_features(self, image):
        """Extract style and content features"""
        features = {}
        x = image
        
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }
        
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layer_names:
                features[layer_names[name]] = x
        
        return features
    
    def gram_matrix(self, tensor):
        """Compute Gram matrix for style representation"""
        b, c, h, w = tensor.shape
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (c * h * w)
    
    def transfer(
        self,
        content_image,
        style_image,
        num_steps=300,
        style_weight=1e6,
        content_weight=1
    ):
        """
        Perform style transfer.
        
        Args:
            content_image: Content image tensor (1, 3, H, W)
            style_image: Style image tensor (1, 3, H, W)
            num_steps: Optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            
        Returns:
            Stylized image tensor
        """
        # Normalize images
        content = self.normalize(content_image).to(self.device)
        style = self.normalize(style_image).to(self.device)
        
        # Extract features
        content_features = self.get_features(content)
        style_features = self.get_features(style)
        
        # Compute style gram matrices
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers
        }
        
        # Initialize output as content image
        output = content.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.LBFGS([output], max_iter=20)
        
        run = [0]
        while run[0] <= num_steps:
            def closure():
                optimizer.zero_grad()
                
                output_features = self.get_features(output)
                
                # Content loss
                content_loss = torch.mean(
                    (output_features['conv4_2'] - content_features['conv4_2']) ** 2
                )
                
                # Style loss
                style_loss = 0
                for layer in self.style_layers:
                    output_gram = self.gram_matrix(output_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += torch.mean((output_gram - style_gram) ** 2)
                
                # Total loss
                loss = content_weight * content_loss + style_weight * style_loss
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}: Loss={loss.item():.2f}")
                
                return loss
            
            optimizer.step(closure)
        
        return output.detach().cpu()


# Usage example
def load_image(path, size=512):
    """Load and preprocess image"""
    image = Image.open(path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    
    return transform(image).unsqueeze(0)


# Run style transfer
style_transfer = StyleTransfer(device='cuda')

content = load_image('content.jpg')
style = load_image('style.jpg')

result = style_transfer.transfer(
    content,
    style,
    num_steps=300,
    style_weight=1e6
)

# Save result
transforms.ToPILImage()(result.squeeze(0)).save('stylized.jpg')
print("✓ Saved stylized.jpg")
```

---

## Integration with FRAMEWORM

Use FRAMEWORM's experiment tracking:
```python
from frameworm.experiment import Experiment

exp = Experiment(
    name='style-transfer',
    config={
        'content_image': 'content.jpg',
        'style_image': 'style.jpg',
        'num_steps': 300,
        'style_weight': 1e6
    },
    tags=['style-transfer', 'artistic']
)

with exp:
    result = style_transfer.transfer(content, style, num_steps=300)
    
    # Log artifact
    exp.log_artifact('stylized.jpg')
    
    # Log metric
    exp.log_metric('num_steps', 300, metric_type='final')

print(f"Experiment ID: {exp.experiment_id}")
```