"""
Project initialization.
"""

import shutil
from pathlib import Path

from click import echo


def create_project(project_path: Path, template: str = "basic"):
    """Create new FRAMEWORM project"""

    # Create directory structure
    project_path.mkdir(parents=True, exist_ok=True)

    (project_path / "configs").mkdir(exist_ok=True)
    (project_path / "data").mkdir(exist_ok=True)
    (project_path / "experiments").mkdir(exist_ok=True)
    (project_path / "checkpoints").mkdir(exist_ok=True)
    (project_path / "scripts").mkdir(exist_ok=True)

    # Create README
    readme = project_path / "README.md"
    readme.write_text(f"""# {project_path.name}

FRAMEWORM project created with template: {template}

## Quick Start
```bash
# Train model
frameworm train --config configs/config.yaml

# Evaluate
frameworm evaluate --checkpoint checkpoints/best.pt

# Export
frameworm export checkpoints/best.pt --format onnx

# Serve
frameworm serve exported/model.pt
```

## Structure

- `configs/` - Configuration files
- `data/` - Datasets
- `experiments/` - Experiment tracking
- `checkpoints/` - Model checkpoints
- `scripts/` - Custom scripts
""")

    # Create basic config
    config = project_path / "configs" / "config.yaml"
    config.write_text(f"""# FRAMEWORM Configuration

model:
  type: {template if template != 'basic' else 'vae'}
  latent_dim: 128
  hidden_dim: 256

training:
  epochs: 100
  batch_size: 128
  lr: 0.001
  device: cuda

data:
  root: data/
  image_size: 64
""")

    # Create .gitignore
    gitignore = project_path / ".gitignore"
    gitignore.write_text("""__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
data/
experiments/
checkpoints/
.DS_Store
""")

    echo(f"✓ Created directory structure")
    echo(f"✓ Created README.md")
    echo(f"✓ Created config.yaml")
