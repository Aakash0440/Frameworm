# Dependency Graphs

## Overview

Frameworm's dependency graph system allows you to define complex workflows where tasks have dependencies on other tasks.

## Basic Usage

### Creating a Graph
```python
from frameworm.graph import Graph, Node

# Create graph
graph = Graph()

# Add nodes
graph.add_node(Node("load", fn=load_data))
graph.add_node(Node("process", fn=process, depends_on=["load"]))
graph.add_node(Node("train", fn=train, depends_on=["process"]))

# Execute
results = graph.execute()
```

### Node Definition
```python
node = Node(
    node_id="my_step",           # Unique identifier
    fn=my_function,              # Function to execute
    depends_on=["step1", "step2"],  # Dependencies
    description="My step",       # Optional description
    cache_result=True            # Enable caching
)
```

## Advanced Features

### Conditional Execution
```python
from frameworm.graph import ConditionalNode

# Only execute if condition is met
node = ConditionalNode(
    "conditional_step",
    fn=expensive_operation,
    condition=lambda x: x > threshold,
    depends_on=["input"]
)
```

### Result Caching
```python
from frameworm.graph import CachedGraph

# Cache results to disk
graph = CachedGraph(cache_dir=".cache")
graph.add_node(Node("expensive", fn=slow_function))

# First run: executes function
results1 = graph.execute()

# Second run: uses cached result!
results2 = graph.execute()
```

### Error Handling
```python
# Stop on first error (default)
results = graph.execute()

# Continue even if nodes fail
results = graph.execute(continue_on_error=True)

# Check which nodes failed
summary = graph.get_last_execution_summary()
print(summary['failed'])  # Number of failures
print(summary['errors'])  # Error details
```

## Pipeline Integration
```python
from frameworm.pipelines.base import GraphPipeline

# Create graph-based pipeline
pipeline = GraphPipeline(config)

pipeline.add_step("load", load_data)
pipeline.add_step("train", train_model, depends_on=["load"])
pipeline.add_step("eval", evaluate, depends_on=["train"])

# Execute
results = pipeline.run()
```

## Visualization

### ASCII Visualization
```python
from frameworm.graph.visualization import print_graph_ascii

print_graph_ascii(graph)
```

Output:
Graph Structure:
● load
↓ process
● process
↑ load
↓ train
● train
↑ process

### Graphviz Visualization
```python
from frameworm.graph.visualization import save_graph_image

# Requires: pip install graphviz
save_graph_image(graph, "my_graph.png")
```

## Best Practices

1. **Use descriptive node IDs** - Makes debugging easier
2. **Keep functions pure** - Easier to cache and test
3. **Handle errors explicitly** - Don't rely on continue_on_error
4. **Visualize complex graphs** - Helps understand dependencies
5. **Cache expensive operations** - Use CachedGraph for long-running tasks
6. **Test nodes individually** - Before adding to graph

## Examples

See `examples/graph_pipeline.py` for complete working examples.

## Troubleshooting

### Cycle Detected
CycleDetectedError: Cycle detected in dependency graph
Cycle: a → b → c → a

**Solution:** Remove one dependency to break the cycle.

### Missing Dependency
GraphError: Node 'b' depends on 'a' which doesn't exist

**Solution:** Add the missing node or fix the dependency name.