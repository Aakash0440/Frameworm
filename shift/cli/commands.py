"""
FRAMEWORM SHIFT CLI commands.

Adds to existing frameworm CLI:
    frameworm shift profile   --data train.csv  --name my_model
    frameworm shift check     --name my_model   --current live.csv
    frameworm shift report    --name my_model   --current live.csv --output report.html
    frameworm shift list
    frameworm shift status    --name my_model

Wire into cli/main.py with:
    from shift.cli.commands import register_shift_commands
    register_shift_commands(main_cli_group)
"""

import sys
import json
from pathlib import Path


def register_shift_commands(cli):
    """
    Register shift subcommands onto the root CLI group.
    Works with Click or argparse-style groups.

    If your CLI uses Click:
        @cli.group()
        def shift(): pass
        ... (see below)

    If your CLI uses argparse, call add_shift_subparsers(subparsers) instead.
    """
    try:
        import click
        _register_click(cli)
    except ImportError:
        pass


# ──────────────────────────────────────────────────── Click implementation

def _register_click(cli):
    import click

    @cli.group()
    def shift():
        """FRAMEWORM SHIFT — distribution drift detection."""
        pass

    @shift.command()
    @click.option("--data",     required=True,  help="Path to reference CSV or numpy .npy file")
    @click.option("--name",     required=True,  help="Model/profile name (e.g. fraud_classifier)")
    @click.option("--features", default=None,   help="Comma-separated feature names (optional)")
    @click.option("--store-dir",default=None,   help="Where to save .shift file")
    def profile(data, name, features, store_dir):
        """Save a reference distribution profile from training data."""
        from shift.core.reference_store import ReferenceStore
        import numpy as np

        feature_names = features.split(",") if features else None
        X = _load_data(data)

        store = ReferenceStore(store_dir)
        path = store.save(X, name, feature_names)
        click.echo(f"[SHIFT] Profile saved → {path}")

    @shift.command()
    @click.option("--name",    required=True,  help="Reference profile name")
    @click.option("--current", required=True,  help="Path to current data CSV or .npy")
    @click.option("--features",default=None,   help="Comma-separated feature names")
    @click.option("--json",    "as_json", is_flag=True, help="Output raw JSON")
    def check(name, current, features, as_json):
        """Check current data for drift against saved reference."""
        from shift.sdk.monitor import ShiftMonitor

        feature_names = features.split(",") if features else None
        X = _load_data(current)

        monitor = ShiftMonitor.from_reference(name, auto_alert=False)
        result = monitor.check(X, feature_names)

        if as_json:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            result.print_summary()
            _print_feature_table(result)

        sys.exit(1 if result.overall_drifted else 0)

    @shift.command()
    @click.option("--name",    required=True,  help="Reference profile name")
    @click.option("--current", required=True,  help="Path to current data CSV or .npy")
    @click.option("--output",  default="shift_report.html", help="Output file path")
    @click.option("--features",default=None,   help="Comma-separated feature names")
    def report(name, current, output, features):
        """Generate an HTML drift report."""
        from shift.sdk.monitor import ShiftMonitor
        from shift.report.report_generator import ReportGenerator

        feature_names = features.split(",") if features else None
        X = _load_data(current)

        monitor = ShiftMonitor.from_reference(name, auto_alert=False)
        result  = monitor.check(X, feature_names)

        ref_profile = monitor._reference_profile
        cur_profile = monitor._profiler.profile(X, feature_names)

        gen = ReportGenerator()
        path = gen.generate_html(result, ref_profile, cur_profile, output, model_name=name)
        click.echo(f"[SHIFT] HTML report saved → {path}")

        path_json = output.replace(".html", ".json")
        gen.generate_json(result, path_json)
        click.echo(f"[SHIFT] JSON report saved → {path_json}")

    @shift.command(name="list")
    @click.option("--store-dir", default=None)
    def list_profiles(store_dir):
        """List all saved reference profiles."""
        from shift.core.reference_store import ReferenceStore
        ReferenceStore(store_dir).list_profiles()

    @shift.command()
    @click.option("--name", required=True)
    def status(name):
        """Show drift monitoring status for a model."""
        from shift.sdk.monitor import ShiftMonitor
        monitor = ShiftMonitor.from_reference(name, auto_alert=False)
        monitor.print_status()


# ──────────────────────────────────────────────────── helpers

def _load_data(path: str):
    import numpy as np
    p = Path(path)
    if not p.exists():
        print(f"[SHIFT] File not found: {path}")
        sys.exit(1)

    if p.suffix == ".npy":
        return np.load(path)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        try:
            import pandas as pd
            return pd.read_csv(path, sep=sep)
        except ImportError:
            data = []
            headers = None
            with open(path) as f:
                for i, line in enumerate(f):
                    row = line.strip().split(sep)
                    if i == 0:
                        try:
                            [float(x) for x in row]
                            data.append(row)
                        except ValueError:
                            headers = row
                    else:
                        data.append(row)
            arr = np.array(data, dtype=float)
            return arr
    else:
        print(f"[SHIFT] Unsupported format: {p.suffix}. Use .csv, .tsv, or .npy")
        sys.exit(1)


def _print_feature_table(result):
    """Print a clean terminal table of per-feature drift results."""
    colours = {
        "NONE":   "\033[92m",
        "LOW":    "\033[93m",
        "MEDIUM": "\033[33m",
        "HIGH":   "\033[91m",
    }
    reset = "\033[0m"
    print(f"\n  {'Feature':<28} {'Test':<6} {'Stat':>8} {'p-value':>10} {'Severity'}")
    print(f"  {'─'*28} {'─'*6} {'─'*8} {'─'*10} {'─'*10}")
    for name, r in result.features.items():
        sev = r.severity.value
        c   = colours.get(sev, "")
        marker = "●" if r.drifted else " "
        print(
            f"  {marker} {name:<26} {r.test_used:<6} "
            f"{r.statistic:>8.4f} {r.p_value:>10.4f} "
            f"{c}{sev}{reset}"
        )
    print()