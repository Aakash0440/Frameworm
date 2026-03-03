# Execution Engine Design

## Goals
- Execute nodes in topological order
- Pass results between nodes
- Handle errors gracefully
- Provide execution context
- Support conditional execution

## ExecutionContext
- Stores results from completed nodes
- Provides inputs to nodes
- Tracks execution state
- Handles errors

## Execution Modes
1. Sequential: One node at a time
2. Parallel: Independent nodes simultaneously (Day 6)
3. Conditional: Skip nodes based on conditions

## Error Handling
- Node failure stops execution by default
- Option to continue on failure
- Collect all errors for reporting