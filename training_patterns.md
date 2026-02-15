# Training Loop Patterns

## PyTorch Lightning
```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        return loss
```
- Pros: Very clean, handles device placement
- Cons: Opinionated, less flexibility

## Hugging Face Transformers Trainer
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
```
- Pros: Batteries included, great defaults
- Cons: Transformers-specific

## Our Approach
```python
trainer = Trainer(model, optimizer, config)
trainer.train(train_loader, val_loader, epochs=100)
```
- Balance flexibility and convenience
- Work with any model
- Extensible via callbacks
- Graph-based workflows supported