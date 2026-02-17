"""
Graph visualization utilities.

Provides functions to visualize dependency graphs.
"""

from pathlib import Path
from typing import Optional

from graph import Graph


def graph_to_dot(graph: Graph) -> str:
    """
    Convert graph to DOT format (Graphviz).

    Args:
        graph: Graph to visualize

    Returns:
        DOT format string
    """
    lines = ["digraph G {"]
    lines.append("  rankdir=TB;")  # Top to bottom
    lines.append("  node [shape=box, style=rounded];")

    # Add nodes with status colors
    for node_id, node in graph.nodes.items():
        # Color based on status
        color = {
            "pending": "lightgray",
            "running": "yellow",
            "completed": "lightgreen",
            "failed": "red",
            "skipped": "orange",
        }.get(node.status.value, "white")

        label = node.description or node_id
        lines.append(f'  "{node_id}" [label="{label}", fillcolor={color}, style=filled];')

    # Add edges (dependencies)
    for node_id, node in graph.nodes.items():
        for dep_id in node.depends_on:
            lines.append(f'  "{dep_id}" -> "{node_id}";')

    lines.append("}")
    return "\n".join(lines)


def save_graph_image(graph: Graph, output_path: str, format: str = "png"):
    """
    Save graph as image.

    Requires graphviz to be installed:
      pip install graphviz

    Args:
        graph: Graph to visualize
        output_path: Where to save image
        format: Image format (png, pdf, svg)
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package required for visualization. " "Install with: pip install graphviz"
        )

    dot_str = graph_to_dot(graph)

    # Create graphviz object
    dot = graphviz.Source(dot_str)

    # Render
    output_path = Path(output_path).with_suffix("")  # Remove extension
    dot.render(str(output_path), format=format, cleanup=True)


def print_graph_ascii(graph: Graph):
    """
    Print ASCII representation of graph.

    Args:
        graph: Graph to print
    """
    print("\nGraph Structure:")
    print("=" * 60)

    execution_order = graph.get_execution_order()

    for node_id in execution_order:
        node = graph.nodes[node_id]

        # Status indicator
        status_symbol = {
            "pending": "○",
            "running": "◐",
            "completed": "●",
            "failed": "✗",
            "skipped": "○",
        }.get(node.status.value, "?")

        # Print node
        print(f"{status_symbol} {node_id}")

        # Print dependencies
        if node.depends_on:
            for dep in node.depends_on:
                print(f"  ↑ {dep}")

        # Print dependents
        dependents = graph.get_dependents(node_id)
        if dependents:
            for dep in dependents:
                print(f"  ↓ {dep}")

        print()

    print("=" * 60)
