# Command Line Interface

## Quick Start
```bash
# Create new project
frameworm init my-project --template vae

# Train model
frameworm train --config config.yaml --gpus 0,1,2,3

# Evaluate
frameworm evaluate --checkpoint best.pt --metrics fid,is

# Export
frameworm export best.pt --format onnx --quantize

# Serve
frameworm serve model.pt --port 8000
```

## Commands

### `init` - Initialize Project
```bash
frameworm init PROJECT_NAME [OPTIONS]

Options:
  --template [basic|gan|vae|diffusion]  Project template
  --path TEXT                           Project directory
```

### `train` - Train Model
```bash
frameworm train [OPTIONS]

Options:
  --config TEXT       Config file path [required]
  --gpus TEXT         GPU IDs (e.g., 0,1,2,3)
  --experiment TEXT   Experiment name
  --resume TEXT       Resume from checkpoint
  --debug            Debug mode
```

### `evaluate` - Evaluate Model
```bash
frameworm evaluate [OPTIONS]

Options:
  --config TEXT        Config file [required]
  --checkpoint TEXT    Model checkpoint [required]
  --metrics TEXT       Metrics to compute (default: fid,is)
  --num-samples INT    Number of samples (default: 10000)
```

### `search` - Hyperparameter Search
```bash
frameworm search [OPTIONS]

Options:
  --config TEXT       Base config [required]
  --space TEXT        Search space YAML [required]
  --method [grid|random|bayesian]  Search method
  --trials INT        Number of trials
  --parallel INT      Parallel jobs
```

### `export` - Export Model
```bash
frameworm export CHECKPOINT [OPTIONS]

Options:
  --format [torchscript|onnx|all]  Export format
  --output TEXT                     Output directory
  --quantize                        Also quantize model
  --benchmark                       Benchmark exported model
```

### `serve` - Serve Model
```bash
frameworm serve MODEL_PATH [OPTIONS]

Options:
  --port INT       Port to serve on (default: 8000)
  --workers INT    Number of workers (default: 1)
  --host TEXT      Host to bind to (default: 0.0.0.0)
```

### `config` - Manage Configurations
```bash
frameworm config list            # List available configs
frameworm config show CONFIG     # Show config contents
frameworm config validate CONFIG # Validate config file
```

## Examples

### Complete Workflow
```bash
# 1. Create project
frameworm init gan-mnist --template gan

# 2. Prepare data (manually)
# ... download MNIST to data/

# 3. Train
frameworm train \
  --config configs/config.yaml \
  --gpus 0,1,2,3 \
  --experiment gan-baseline

# 4. Hyperparameter search
frameworm search \
  --config configs/config.yaml \
  --space configs/search_space.yaml \
  --method bayesian \
  --trials 50

# 5. Evaluate best model
frameworm evaluate \
  --checkpoint experiments/best/checkpoints/best.pt \
  --metrics fid,is \
  --num-samples 50000

# 6. Export
frameworm export \
  experiments/best/checkpoints/best.pt \
  --format onnx \
  --quantize

# 7. Serve
frameworm serve exported/model.pt --port 8000
```

## Configuration Files

See [Configuration Guide](configuration.md) for details on config files.

## Environment Variables
```bash
FRAMEWORM_DATA_DIR=/path/to/data
FRAMEWORM_EXPERIMENT_DIR=/path/to/experiments
FRAMEWORM_CHECKPOINT_DIR=/path/to/checkpoints
```