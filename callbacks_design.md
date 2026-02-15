# Callback System Design

## Purpose
Allow users to inject custom code at different training stages.

## Callback Hooks
- on_train_begin()
- on_train_end()
- on_epoch_begin(epoch)
- on_epoch_end(epoch, metrics)
- on_batch_begin(batch_idx)
- on_batch_end(batch_idx, metrics)

## Built-in Callbacks
1. EarlyStopping - Stop when no improvement
2. ModelCheckpoint - Save best/latest checkpoints
3. LearningRateMonitor - Track LR changes
4. GradientMonitor - Track gradient norms
5. CSVLogger - Log metrics to CSV

## Usage
```python
from frameworm.training.callbacks import EarlyStopping, CSVLogger

trainer = Trainer(model, optimizer)
trainer.add_callback(EarlyStopping(patience=10))
trainer.add_callback(CSVLogger('training.csv'))
trainer.train(train_loader, val_loader)
```