# Common Error Patterns in ML Frameworks

## PyTorch Errors (Not Great)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x100 and 10x256)
- Shows what's wrong
- But no suggestion how to fix
- No context about where in model

## Our Approach (Better)
DimensionMismatchError: Cannot multiply matrices
Expected: (64, 10)
Received: (64, 100)
This likely means:
→ Input dimension doesn't match layer input_dim
→ Check config.model.input_dim = 100 but layer expects 10
Fix:
Set layer = nn.Linear(100, 256) instead of nn.Linear(10, 256)

## Design Principles
1. **Say what happened** - Clear error name
2. **Show the context** - Actual values
3. **Explain why** - Likely causes
4. **Suggest fix** - Actionable steps
5. **Link to docs** - Learn more