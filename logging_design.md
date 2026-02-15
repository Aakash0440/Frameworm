# Experiment Logging Design

## TensorBoard
- Log scalars (loss, metrics)
- Log images (generated samples)
- Log histograms (weights, gradients)
- Log graphs (model architecture)
- Log hyperparameters

## Weights & Biases (wandb)
- Similar to TensorBoard but cloud-based
- Better collaboration
- Automatic hyperparameter tracking
- Model versioning

## Integration
```python
from frameworm.training.loggers import TensorBoardLogger

trainer = Trainer(model, optimizer)
trainer.add_logger(TensorBoardLogger('runs/experiment'))
trainer.train(train_loader, val_loader)
```