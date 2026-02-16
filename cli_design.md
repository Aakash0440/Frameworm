# FRAMEWORM CLI Design

## Commands Structure
frameworm
├── init              # Initialize new project
├── train             # Train a model
├── evaluate          # Evaluate model
├── search            # Hyperparameter search
├── export            # Export model
├── serve             # Serve model
├── deploy            # Deploy to production
└── config            # Manage configurations
frameworm train
├── --config          # Config file path
├── --model           # Model type
├── --data            # Data directory
├── --epochs          # Number of epochs
├── --gpus            # GPU IDs
└── --experiment      # Experiment name
frameworm search
├── --config          # Base config
├── --space           # Search space file
├── --method          # grid/random/bayesian
├── --trials          # Number of trials
└── --parallel        # Parallel jobs

## Implementation

Using Click for CLI:
- Hierarchical commands
- Auto-generated help
- Type validation
- Configuration files
- Progress bars

## Features

1. **Project scaffolding**
2. **Training workflows**
3. **Evaluation pipelines**
4. **Search automation**
5. **Deployment helpers**
6. **Configuration management**