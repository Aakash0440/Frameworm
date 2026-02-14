# Plugin System Design

## Goals
1. Allow users to drop Python files in plugins/ directory
2. Auto-discover and register plugins on import
3. Isolate plugins by namespace (models/, trainers/, etc.)
4. Validate plugin interfaces
5. Handle name collisions gracefully

## Architecture

### Registry Pattern
@register_model("my-gan")
class MyGAN(BaseModel):
...
Internally:
ModelRegistry['my-gan'] = MyGAN

### Discovery Pattern
plugins/
├── my_model.py
│   └── @register_model("my-model")
└── my_trainer.py
└── @register_trainer("my-trainer")

Auto-discovered on:
from frameworm.models import get_model

## Components

1. **Registry Class** - Stores registered items
2. **Decorator Functions** - @register_model, @register_trainer, etc.
3. **Getter Functions** - get_model, get_trainer, etc.
4. **Discovery Function** - scan_plugins()
5. **Validation** - Check required methods/attributes

## Implementation Plan

1. Create base Registry class
2. Create namespace-specific registries (ModelRegistry, etc.)
3. Implement decorators
4. Implement getters
5. Add plugin discovery
6. Add validation
7. Test everything