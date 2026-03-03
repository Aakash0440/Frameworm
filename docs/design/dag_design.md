# Dependency Graph Engine Design

## Goals
1. Define computation nodes with dependencies
2. Execute in correct order (topological sort)
3. Cache results to avoid recomputation
4. Support parallel execution
5. Visualize graph structure

## Example Use Case
```python
# Define nodes
preprocess = Node("preprocess", fn=preprocess_data)
train_model = Node("train", fn=train, depends_on=["preprocess"])
evaluate = Node("eval", fn=evaluate, depends_on=["train"])
visualize = Node("viz", fn=plot, depends_on=["eval"])

# Create graph
graph = Graph()
graph.add_nodes([preprocess, train_model, evaluate, visualize])

# Execute (automatically figures out order)
results = graph.execute()
```

## Architecture

### Node
- id: unique identifier
- fn: function to execute
- depends_on: list of node IDs this depends on
- result: cached result
- status: pending/running/completed/failed

### Graph
- nodes: dict of node_id → Node
- edges: adjacency list (dependencies)
- execution_order: topologically sorted node IDs

### Algorithms
1. **Topological Sort (Kahn's Algorithm)**
   - Find nodes with no dependencies
   - Execute them, remove from graph
   - Repeat until done or cycle detected

2. **Cycle Detection**
   - DFS with recursion stack
   - Detect back edges

3. **Caching**
   - Hash function inputs
   - Store results by hash
   - Invalidate on dependency change

## Implementation Plan

Hour 1-3: Core DAG
- Node class
- Graph class  
- Topological sort
- Cycle detection
- Basic execution

Hour 4-6: Execution Engine
- Execution context
- Error handling
- Result passing
- Dependency resolution

Hour 7-10: Caching & Optimization
- Result caching
- Incremental execution
- Memoization