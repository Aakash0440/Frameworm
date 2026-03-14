"""
frameworm-cost CLI

Commands:
    frameworm-cost report --file costs.json
    frameworm-cost estimate --arch dcgan --hardware t4 --latency 38
    frameworm-cost compare --latency 38 --hardware t4
"""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="frameworm-cost",
    help="FRAMEWORM COST — ML inference cost tracking and optimization.",
    add_completion=False,
)


@app.command()
def report(
    file: Path = typer.Argument(..., help="Path to cost records JSON file"),
):
    """Generate a cost report from a saved records file."""
    from cost.report import CostReport
    from cost.store import CostStore

    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(1)

    store = CostStore(path=file)
    rep = CostReport(store)
    rep.print()


@app.command()
def estimate(
    latency: float = typer.Option(..., "--latency", "-l", help="Inference latency in ms"),
    arch: str = typer.Option("unknown", "--arch", "-a", help="Model architecture"),
    hardware: str = typer.Option("default", "--hardware", "-hw", help="Hardware type"),
    batch: int = typer.Option(1, "--batch", "-b", help="Batch size"),
    params: float = typer.Option(0.0, "--params", "-p", help="Model parameters in millions"),
):
    """Estimate cost for a single inference call."""
    from cost.calculator import CostCalculator

    calc = CostCalculator(
        hardware=hardware,
        architecture=arch,
        parameters_millions=params,
    )
    cost = calc.calculate(latency, batch_size=batch)

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("Metric", style="dim")
        t.add_column("Value", style="bold")
        t.add_row("Architecture", arch)
        t.add_row("Hardware", hardware)
        t.add_row("Latency", f"{latency}ms")
        t.add_row("Batch size", str(batch))
        t.add_row("Cost / request", f"[orange1]${cost.total_cost_usd:.8f}[/orange1]")
        t.add_row("Cost / 1k requests", f"${cost.cost_per_1k_requests:.4f}")
        monthly = cost.monthly_cost_at_rps
        t.add_row("Monthly (1 rps)", f"${monthly['1_rps']:,.2f}")
        t.add_row("Monthly (10 rps)", f"[bold red]${monthly['10_rps']:,.2f}[/bold red]")
        t.add_row("Monthly (100 rps)", f"[bold red]${monthly['100_rps']:,.2f}[/bold red]")
        if cost.optimization_hint:
            t.add_row("💡 Hint", f"[yellow]{cost.optimization_hint}[/yellow]")
        console.print(t)
    except ImportError:
        print(json.dumps(cost.to_dict(), indent=2))


@app.command()
def compare(
    latency: float = typer.Option(..., "--latency", "-l", help="Inference latency in ms"),
    hardware: str = typer.Option("default", "--hardware", help="Hardware type"),
):
    """Compare cost across all supported architectures at a given latency."""
    from cost.calculator import ARCH_COMPLEXITY, CostCalculator

    calc = CostCalculator(hardware=hardware)
    results = calc.compare_architectures(latency)

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t = Table(title="Architecture Cost Comparison", box=box.SIMPLE)
        t.add_column("Architecture", style="bold")
        t.add_column("$/request", style="orange1")
        t.add_column("$/1k requests")
        t.add_column("Monthly (10 rps)")
        t.add_column("Complexity")

        cheapest = results[0].total_cost_usd
        for r in results:
            ratio = r.total_cost_usd / cheapest
            color = "green" if ratio < 1.5 else "yellow" if ratio < 5 else "red"
            monthly = r.total_cost_usd * 10 * 30 * 24 * 3600
            t.add_row(
                r.architecture,
                f"[{color}]${r.total_cost_usd:.8f}[/{color}]",
                f"${r.cost_per_1k_requests:.4f}",
                f"${monthly:,.2f}",
                f"{ratio:.1f}x",
            )
        console.print(t)
    except ImportError:
        for r in results:
            print(f"{r.architecture}: ${r.total_cost_usd:.8f}/request")


if __name__ == "__main__":
    app()
