# Installation Guide

## Quick Install
```bash
pip install frameworm
```

## From Source
```bash
git clone https://github.com/Aakash0440/frameworm
cd frameworm
pip install -e .
```

## Verify Installation
```bash
frameworm --version
frameworm --help
```

## Optional Dependencies

### For Bayesian Optimization
```bash
pip install scikit-optimize
```

### For ONNX Export
```bash
pip install onnx onnxruntime
```

### For Deployment
```bash
pip install fastapi uvicorn
```

## Shell Completion

### Bash
```bash
frameworm completion --shell bash >> ~/.bashrc
source ~/.bashrc
```

### Zsh
```bash
frameworm completion --shell zsh >> ~/.zshrc
source ~/.zshrc
```

## GPU Support

FRAMEWORM supports CUDA out of the box with PyTorch.

Check GPU availability:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA Issues

Install appropriate PyTorch version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```