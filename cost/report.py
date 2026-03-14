"""
CostReport: generates cost analysis reports with savings recommendations.
"""

from __future__ import annotations

from typing import Optional

from cost.calculator import ARCH_COMPLEXITY, CostCalculator
from cost.store import CostStore


class CostReport:
    """
    Generates cost reports and savings recommendations.

    report = CostReport(store)
    report.print()          # rich terminal output
    report.to_dict()        # machine-readable
    report.savings_report() # "here's how to cut your bill"
    """

    def __init__(self, store: CostStore):
        self.store = store

    def to_dict(self) -> dict:
        summary = self.store.summary()
        hints = self.store.hints()
        savings = self._calculate_savings(summary)
        return {
            "summary": summary,
            "hints": hints,
            "savings_opportunities": savings,
        }

    def _calculate_savings(self, summary: dict) -> list[dict]:
        opportunities = []
        if not summary.get("total_requests"):
            return opportunities

        avg_cost = summary.get("avg_cost_usd", 0)
        avg_latency = summary.get("avg_latency_ms", 0)
        monthly = summary.get("projected_monthly_10rps_usd", 0)

        # Batching opportunity
        if avg_latency > 30:
            batch_saving = monthly * 0.65
            opportunities.append(
                {
                    "type": "batching",
                    "title": "Enable request batching",
                    "description": (
                        f"Grouping 8 requests per batch could reduce cost/request by ~65%. "
                        f"Estimated saving: ${batch_saving:,.0f}/month at 10 req/s."
                    ),
                    "estimated_monthly_saving_usd": round(batch_saving, 2),
                    "effort": "low",
                }
            )

        # Quantization opportunity
        if avg_latency > 80:
            quant_saving = monthly * 0.55
            opportunities.append(
                {
                    "type": "quantization",
                    "title": "INT8 quantization",
                    "description": (
                        f"High latency detected ({avg_latency:.0f}ms avg). INT8 quantization "
                        f"typically gives 2-4x speedup with <1% accuracy loss. "
                        f"Estimated saving: ${quant_saving:,.0f}/month."
                    ),
                    "estimated_monthly_saving_usd": round(quant_saving, 2),
                    "effort": "medium",
                }
            )

        # Architecture comparison
        if avg_cost > 0.001:
            opportunities.append(
                {
                    "type": "architecture",
                    "title": "Consider a lighter architecture",
                    "description": (
                        f"Current avg cost is ${avg_cost:.6f}/request. "
                        f"A simpler architecture (e.g. VAE vs DDPM) can be 8-9x cheaper "
                        f"depending on your quality requirements."
                    ),
                    "estimated_monthly_saving_usd": round(monthly * 0.7, 2),
                    "effort": "high",
                }
            )

        return sorted(opportunities, key=lambda x: -x["estimated_monthly_saving_usd"])

    def print(self) -> None:
        """Print a rich terminal report."""
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
        except ImportError:
            print(self._plain_text())
            return

        console = Console()
        data = self.to_dict()
        summary = data["summary"]

        console.print()
        console.print(
            Panel(
                f"[bold orange1]FRAMEWORM COST[/bold orange1]  ·  Cost Analysis Report",
                border_style="orange1",
            )
        )

        # Summary table
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        t.add_column("Metric", style="dim")
        t.add_column("Value", style="bold")
        t.add_row("Total requests", str(summary.get("total_requests", 0)))
        t.add_row("Total cost", f"${summary.get('total_cost_usd', 0):.6f}")
        t.add_row("Avg cost / request", f"${summary.get('avg_cost_usd', 0):.8f}")
        t.add_row("Cost per 1k requests", f"${summary.get('cost_per_1k_usd', 0):.4f}")
        t.add_row(
            "Projected monthly (10 rps)",
            f"[bold red]${summary.get('projected_monthly_10rps_usd', 0):,.2f}[/bold red]",
        )
        t.add_row("Avg latency", f"{summary.get('avg_latency_ms', 0):.1f}ms")
        t.add_row("p95 latency", f"{summary.get('p95_latency_ms', 0):.1f}ms")
        console.print(t)

        # Savings
        savings = data["savings_opportunities"]
        if savings:
            console.print("\n[bold orange1]💡 Savings Opportunities[/bold orange1]")
            for s in savings:
                effort_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
                    s["effort"], "white"
                )
                console.print(
                    f"\n  [bold]{s['title']}[/bold]  [{effort_color}]({s['effort']} effort)[/{effort_color}]"
                )
                console.print(f"  {s['description']}")
                console.print(
                    f"  [green]Est. saving: ${s['estimated_monthly_saving_usd']:,.0f}/month[/green]"
                )

        # Hints
        hints = data["hints"]
        if hints:
            console.print("\n[bold orange1]⚠️  Optimization Hints[/bold orange1]")
            for h in hints:
                console.print(f"  • {h}")

        console.print()

    def _plain_text(self) -> str:
        data = self.to_dict()
        s = data["summary"]
        lines = [
            "=== FRAMEWORM COST REPORT ===",
            f"Total requests:    {s.get('total_requests', 0)}",
            f"Total cost:        ${s.get('total_cost_usd', 0):.6f}",
            f"Avg cost/request:  ${s.get('avg_cost_usd', 0):.8f}",
            f"Monthly (10 rps):  ${s.get('projected_monthly_10rps_usd', 0):,.2f}",
            "",
            "=== SAVINGS OPPORTUNITIES ===",
        ]
        for opp in data["savings_opportunities"]:
            lines.append(
                f"• {opp['title']}: save ${opp['estimated_monthly_saving_usd']:,.0f}/month"
            )
        return "\n".join(lines)
