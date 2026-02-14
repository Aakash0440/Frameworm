# Error Handling

## Overview

Frameworm provides helpful, actionable error messages that explain:
- What went wrong
- Why it happened  
- How to fix it
- Where to learn more

## Error Types

### Configuration Errors

**ConfigNotFoundError** - Config file not found
```python
try:
    cfg = Config("missing.yaml")
except ConfigNotFoundError as e:
    print(e)  # Helpful message with suggestions
```

**ConfigValidationError** - Invalid config value
```python
# Triggered when required fields missing or values invalid
```

### Model Errors

**DimensionMismatchError** - Tensor shape mismatch
```python
# Provides analysis of dimension mismatch
# Suggests fixes (reshape, unsqueeze, etc.)
```

**ModelNotFoundError** - Model not in registry
```python
try:
    model = get_model("nonexistent")
except ModelNotFoundError as e:
    print(e)  # Shows available models
```

### Plugin Errors

**PluginValidationError** - Plugin missing required methods
```python
# Shows which methods are missing
# Suggests how to implement them
```

## Example Error Messages

### Dimension Mismatch
DimensionMismatchError: Tensor dimension mismatch
Details:
Expected shape: (4, 100, 1, 1)
Received shape: (4, 100)
Layer: generator.main[0]
Likely Causes:

Input has 2 fewer dimension(s) than expected

Suggested Fixes:
→ Add spatial dimensions: x.unsqueeze(-1).unsqueeze(-1)
→ Or reshape: x.view(batch, channels, 1, 1)

### Model Not Found
ModelNotFoundError: Model 'my-model' not found in registry
Details:
Requested: my-model
Available models: dcgan, stylegan2, vae
Suggested Fixes:
→ List all models: frameworm list-models
→ Re-discover plugins: discover_plugins(force=True)

## Best Practices

1. **Don't catch errors silently** - Let them propagate
2. **Read the suggestions** - They're usually right
3. **Check documentation links** - Learn the concepts
4. **Use the debug tools** - They help diagnose issues

## Custom Errors

Create custom errors for your plugins:
```python
from frameworm.core.exceptions import FramewormError

class MyCustomError(FramewormError):
    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)
        
        self.add_cause("Your cause here")
        self.add_suggestion("Your fix here")
        self.set_doc_link("your-docs-url")