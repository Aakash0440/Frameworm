# Error Explanation System Design

## Goals
1. Clear error messages that explain what went wrong
2. Context about where the error occurred
3. Suggested fixes
4. Links to documentation
5. Beautiful formatting (colors, structure)

## Example Error
DimensionMismatchError: Tensor dimension mismatch in forward pass
Location:
File: frameworm/models/gan/dcgan.py
Line: 142
Function: Generator.forward()
Context:
Expected shape: (batch_size, 100, 1, 1)
Received shape: (batch_size, 100)
Layer: generator.main[0] (ConvTranspose2d)
Likely Causes:

Input tensor missing spatial dimensions (1, 1)
Wrong reshape applied before generator
Using 2D latent instead of 4D

Suggested Fixes:
→ Reshape input: z = z.view(batch_size, 100, 1, 1)
→ Or use: z = z.unsqueeze(-1).unsqueeze(-1)
Documentation:
https://frameworm.readthedocs.io/errors/dimension-mismatch
Debug Command:
python -m frameworm.debug --trace-shapes

## Architecture

### Exception Hierarchy
FramewormError (base)
├── ConfigurationError
│   ├── ConfigNotFoundError
│   ├── ConfigValidationError
│   └── ConfigInheritanceError
├── ModelError
│   ├── DimensionMismatchError
│   ├── ModelNotFoundError
│   └── ArchitectureError
├── TrainingError
│   ├── ConvergenceError
│   └── DataLoaderError
└── PluginError
├── PluginNotFoundError
├── PluginValidationError
└── PluginImportError

### Components
1. Base exception classes
2. Context capture mechanism
3. Suggestion engine (pattern matching)
4. Formatter (colors, structure)
5. Documentation linker