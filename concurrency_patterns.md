# Python Concurrency Patterns

## ThreadPoolExecutor
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = {executor.submit(fn, arg): name for name, (fn, arg) in tasks.items()}
    
    # Collect results as they complete
    for future in as_completed(futures):
        name = futures[future]
        result = future.result()
```

## ProcessPoolExecutor
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    # Same API as ThreadPoolExecutor
    futures = {executor.submit(fn, arg): name for name, (fn, arg) in tasks.items()}
```

## Dependency Levels
```python
def compute_levels(graph):
    levels = []
    remaining = set(graph.nodes.keys())
    
    while remaining:
        # Find nodes with all dependencies satisfied
        level = [n for n in remaining if all(dep not in remaining for dep in graph.nodes[n].depends_on)]
        levels.append(level)
        remaining -= set(level)
    
    return levels
```