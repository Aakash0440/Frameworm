# Graph Algorithms Reference

## Topological Sort (Kahn's Algorithm)
```python
def topological_sort(graph):
    # 1. Find in-degree of each node
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # 2. Start with nodes that have no dependencies
    queue = [node for node in graph if in_degree[node] == 0]
    result = []
    
    # 3. Process nodes
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        # Remove edge, decrease in-degree
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 4. Check for cycles
    if len(result) != len(graph):
        raise Exception("Cycle detected")
    
    return result
```

## DFS Cycle Detection
```python
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color[neighbor] == GRAY:  # Back edge = cycle
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    return any(dfs(node) for node in graph if color[node] == WHITE)
    ```