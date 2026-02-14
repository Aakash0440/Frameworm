# Registry Pattern Examples

## PyTorch Lightning
```python
class MyModel(LightningModule):
    ...
# Uses class inheritance
```

## Hugging Face Transformers
```python
AutoModel.from_pretrained("bert-base")
# Uses string-based registry
```

## Our Approach
```python
@register_model("my-model")
class MyModel(BaseModel):
    ...

model = get_model("my-model")(config)
# Combines decorator + registry + factory
```

## Key Decisions
- Use decorators (clean, explicit)
- Namespace isolation (avoid collisions)
- Lazy loading (performance)
- Validation on registration (fail early)