"""
FRAMEWORM CLI — Production-grade ML framework interface.
"""

import os
import sys
import time
from pathlib import Path

import click
from rich import box
from rich.align import Align
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "cli"

console = Console()

# Palette
O = "bold #CC6B2C"
OD = "#8a4618"
A = "#d4a55a"
C = "#f5eedf"
CD = "#8a7a65"
G = "bold #7ab87a"
R = "bold #c85a5a"

BANNER = """[#CC6B2C]
  ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ███╗   ███╗
  ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗████╗ ████║
  █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝██╔████╔██║
  ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██║╚██╔╝██║
  ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝[/#CC6B2C]"""

VERSION = "1.0.0"


def print_banner():
    console.print(BANNER)
    console.print(
        Align.center(
            f"[{CD}]Production-grade generative AI framework  ·  v{VERSION}  ·  github.com/aakash0440/frameworm[/{CD}]"
        )
    )
    console.print()


def print_help_menu():
    print_banner()
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {CD}", padding=(0, 2))
    t.add_column("Command", style=f"{O}", width=14)
    t.add_column("Description", style=f"{C}", width=44)
    t.add_column("Example", style=f"{CD}", width=42)
    commands = [
        ("train", "Train a model from a config file", "frameworm train --config cfg.yaml"),
        ("deploy", "Deploy a checkpoint to production", "frameworm deploy start --model best.pt"),
        ("serve", "Serve model via REST API", "frameworm serve --checkpoint best.pt"),
        ("evaluate", "Evaluate with FID / IS / LPIPS", "frameworm evaluate --config c.yaml"),
        ("export", "Export to ONNX / TorchScript", "frameworm export checkpoint.pt"),
        ("search", "Hyperparameter search", "frameworm search --config c.yaml"),
        ("monitor", "Real-time training monitor", "frameworm monitor ./experiments/run1"),
        ("cost", "Inference cost analysis", "frameworm cost estimate --arch dcgan"),
        ("init", "Scaffold a new project", "frameworm init my-project --template gan"),
        ("plugins", "Manage plugins", "frameworm plugins list"),
        ("config", "Validate / inspect configs", "frameworm config validate cfg.yaml"),
        ("dashboard", "Launch web dashboard", "frameworm dashboard --port 8080"),
        ("shell", "Interactive FRAMEWORM shell", "frameworm shell"),
        ("status", "Show ecosystem status", "frameworm status"),
    ]
    for cmd, desc, ex in commands:
        t.add_row(cmd, desc, ex)
    console.print(Panel(t, title=f"[{O}]Commands[/{O}]", border_style=OD, padding=(1, 2)))
    console.print()


def section(title):
    console.print(Rule(f"[{OD}]{title}[/{OD}]", style=OD))


def success(msg):
    console.print(f"[{G}]✓[/{G}]  [{C}]{msg}[/{C}]")


def error(msg):
    console.print(f"[{R}]✗[/{R}]  [{C}]{msg}[/{C}]")


def info(msg):
    console.print(f"[{A}]·[/{A}]  [{CD}]{msg}[/{CD}]")


# ── CLI ───────────────────────────────────────────────────────────────────────
@click.group(invoke_without_command=True)
@click.version_option(version=VERSION, prog_name="frameworm")
@click.pass_context
def cli(ctx):
    """FRAMEWORM — Production-grade generative AI framework."""
    if ctx.invoked_subcommand is None:
        print_help_menu()


# ── STATUS ────────────────────────────────────────────────────────────────────
@cli.command()
def status():
    """Show FRAMEWORM ecosystem status."""
    print_banner()
    checks = [
        ("Core", "core", "Training framework"),
        ("Training", "training", "Trainer + callbacks"),
        ("Models", "models", "VAE DCGAN DDPM VQ-VAE-2 ViT-GAN CFG-DDPM"),
        ("Deploy", "deploy", "One-command deployment"),
        ("Shift", "shift", "Data drift detection"),
        ("Agent", "agent", "Autonomous training monitor"),
        ("Cost", "cost", "Inference cost tracking"),
        ("Plugins", "plugins", "Plugin loader"),
    ]
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {CD}", padding=(0, 2))
    t.add_column("Module", style=f"{O}", width=12)
    t.add_column("Status", width=12)
    t.add_column("Description", style=f"{CD}", width=40)
    for label, mod, desc in checks:
        try:
            __import__(mod)
            t.add_row(label, f"[{G}]✓  OK[/{G}]", desc)
        except ImportError as e:
            t.add_row(label, f"[{R}]✗  Missing[/{R}]", str(e)[:38])
    console.print(Panel(t, title=f"[{O}]Ecosystem Status[/{O}]", border_style=OD))
    import platform

    info_t = Table.grid(padding=(0, 4))
    info_t.add_row(
        f"[{CD}]Python  [{C}]{platform.python_version()}[/{C}]",
        f"[{CD}]Platform  [{C}]{platform.system()}[/{C}]",
    )
    try:
        import torch

        info_t.add_row(
            f"[{CD}]PyTorch  [{C}]{torch.__version__}[/{C}]",
            f"[{CD}]CUDA  [{C}]{'✓ '+torch.version.cuda if torch.cuda.is_available() else '✗ CPU only'}[/{C}]",
        )
    except ImportError:
        info_t.add_row(f"[{R}]PyTorch not installed[/{R}]", "")
    console.print(Padding(info_t, (0, 2)))
    console.print()


# ── SHELL ─────────────────────────────────────────────────────────────────────
@cli.command()
def shell():
    """Interactive FRAMEWORM shell."""
    print_banner()
    console.print(
        Panel(
            f"[{CD}]Type a command or [bold]help[/bold]. [bold]exit[/bold] to quit.[/{CD}]",
            border_style=OD,
            padding=(0, 2),
        )
    )
    console.print()
    while True:
        try:
            cmd = Prompt.ask(f"[{O}]frameworm[/{O}][{CD}]>[/{CD}]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[{CD}]Goodbye.[/{CD}]")
            break
        if not cmd:
            continue
        if cmd in ("exit", "quit", "q"):
            console.print(f"[{CD}]Goodbye.[/{CD}]")
            break
        if cmd == "help":
            print_help_menu()
            continue
        try:
            cli.main(cmd.split(), standalone_mode=False)
        except Exception as e:
            error(str(e))


# ── TRAIN ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--resume", default=None)
@click.option("--gpus", default="0")
def train(config, resume, gpus):
    """Train a model using a config file."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from core.config import Config
    from core.registry import get_model
    from training.trainer import Trainer

    section("FRAMEWORM TRAIN")
    info(f"Config: {config}")
    cfg = Config(config)
    model_name = cfg.get("model.type", "dcgan")
    info(f"Model: {model_name}")
    try:
        import models.gan.dcgan

        model = get_model(model_name)(cfg)
        success(f"Model ready: {model_name}")
    except Exception as e:
        error(f"Model load failed: {e}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("optimizer.lr", 0.0002))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="checkpoints",
        gradient_accumulation_steps=cfg.get("training.gradient_accumulation_steps", 1),
    )
    dummy = TensorDataset(torch.randn(64, 3, 64, 64))
    loader = DataLoader(dummy, batch_size=cfg.get("training.batch_size", 64))
    trainer.train(train_loader=loader, epochs=cfg.get("training.epochs", 100), resume_from=resume)
    console.print()
    success("Training complete")


# ── SERVE ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--port", default=8080)
@click.option("--host", default="0.0.0.0")
def serve(checkpoint, port, host):
    """Serve a trained model via REST API."""
    section("FRAMEWORM SERVE")
    info(f"Checkpoint: {checkpoint}")
    try:
        import torch
        import uvicorn
        from fastapi import FastAPI

        torch.load(checkpoint, map_location="cpu", weights_only=False)
        success("Checkpoint loaded")
        app = FastAPI(title="FRAMEWORM SERVE", version=VERSION)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/ready")
        def ready():
            return {"ready": True}

        @app.post("/generate")
        def generate(latent_dim: int = 100):
            z = torch.randn(1, latent_dim)
            return {"shape": list(z.shape)}

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_row(f"[{O}]GET[/{O}]", f"[{C}]http://{host}:{port}/health[/{C}]")
        t.add_row(f"[{O}]GET[/{O}]", f"[{C}]http://{host}:{port}/ready[/{C}]")
        t.add_row(f"[{O}]POST[/{O}]", f"[{C}]http://{host}:{port}/generate[/{C}]")
        console.print(Panel(t, title=f"[{O}]Endpoints[/{O}]", border_style=OD))
        success(f"Running at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        error("Missing: pip install fastapi uvicorn[standard]")
    except Exception as e:
        error(f"Failed: {e}")


# ── EVALUATE ──────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--config", required=True, type=str)
@click.option("--checkpoint", required=True, type=str)
@click.option("--metrics", default="fid,is")
@click.option("--num-samples", type=int, default=10000)
def evaluate(config, checkpoint, metrics, num_samples):
    """Evaluate a trained model."""
    from cli.evaluate import run_evaluation

    section("FRAMEWORM EVALUATE")
    info(f"Checkpoint: {checkpoint}  |  Metrics: {metrics}")
    run_evaluation(
        config_path=config,
        checkpoint_path=checkpoint,
        metrics=metrics.split(","),
        num_samples=num_samples,
    )


# ── EXPORT ────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("checkpoint")
@click.option("--format", type=click.Choice(["torchscript", "onnx", "all"]), default="all")
@click.option("--output", default="exported")
@click.option("--quantize", is_flag=True)
@click.option("--benchmark", is_flag=True)
def export(checkpoint, format, output, quantize, benchmark):
    """Export model to ONNX / TorchScript."""
    from cli.export import export_model

    section("FRAMEWORM EXPORT")
    info(f"Checkpoint: {checkpoint}  |  Format: {format}  |  Quantize: {quantize}")
    export_model(
        checkpoint_path=checkpoint,
        export_format=format,
        output_dir=output,
        quantize=quantize,
        benchmark=benchmark,
    )


# ── SEARCH ────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--config", required=True, type=str)
@click.option("--space", required=True, type=str)
@click.option("--method", type=click.Choice(["grid", "random", "bayesian"]), default="random")
@click.option("--trials", type=int, default=50)
@click.option("--parallel", type=int, default=1)
def search(config, space, method, trials, parallel):
    """Run hyperparameter search."""
    from cli.search_cli import run_search

    section("FRAMEWORM SEARCH")
    info(f"Method: {method}  |  Trials: {trials}  |  Parallel: {parallel}")
    run_search(
        config_path=config, space_path=space, method=method, n_trials=trials, n_jobs=parallel
    )


# ── MONITOR ───────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("experiment_dir")
@click.option("--refresh", type=int, default=2)
def monitor(experiment_dir, refresh):
    """Monitor training in real-time."""
    from cli.monitor import TrainingMonitor

    section("FRAMEWORM MONITOR")
    info(f"Watching: {experiment_dir}  (refresh: {refresh}s)")
    TrainingMonitor(experiment_dir).watch(refresh_rate=refresh)


# ── COST ──────────────────────────────────────────────────────────────────────
@cli.group()
def cost():
    """Inference cost tracking and analysis."""
    pass


@cost.command("estimate")
@click.option("--arch", default="unknown")
@click.option("--hardware", default="default")
@click.option("--latency", required=True, type=float)
@click.option("--batch", default=1, type=int)
def cost_estimate(arch, hardware, latency, batch):
    """Estimate cost for a single inference call."""
    from cost.calculator import CostCalculator

    section("FRAMEWORM COST · Estimate")
    c = CostCalculator(hardware=hardware, architecture=arch).calculate(latency, batch_size=batch)
    monthly = c.monthly_cost_at_rps
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_row(f"[{CD}]Architecture[/{CD}]", f"[{C}]{arch}[/{C}]")
    t.add_row(f"[{CD}]Hardware[/{CD}]", f"[{C}]{hardware}[/{C}]")
    t.add_row(f"[{CD}]Latency[/{CD}]", f"[{C}]{latency}ms[/{C}]")
    t.add_row(f"[{CD}]─────────────[/{CD}]", "")
    t.add_row(f"[{O}]Cost / request[/{O}]", f"[bold {C}]${c.total_cost_usd:.8f}[/bold {C}]")
    t.add_row(f"[{CD}]Cost / 1k requests[/{CD}]", f"[{C}]${c.cost_per_1k_requests:.4f}[/{C}]")
    t.add_row(f"[{CD}]Monthly   1 rps[/{CD}]", f"[{C}]${monthly['1_rps']:,.2f}[/{C}]")
    t.add_row(f"[{O}]Monthly  10 rps[/{O}]", f"[bold {C}]${monthly['10_rps']:,.2f}[/bold {C}]")
    t.add_row(f"[{CD}]Monthly 100 rps[/{CD}]", f"[{C}]${monthly['100_rps']:,.2f}[/{C}]")
    console.print(Panel(t, title=f"[{O}]Cost Breakdown[/{O}]", border_style=OD))
    if c.optimization_hint:
        console.print(
            Panel(f"[{A}]{c.optimization_hint}[/{A}]", title=f"[{O}]💡 Hint[/{O}]", border_style=OD)
        )


@cost.command("compare")
@click.option("--latency", required=True, type=float)
@click.option("--hardware", default="default")
def cost_compare(latency, hardware):
    """Compare cost across all architectures."""
    from cost.calculator import CostCalculator

    section("FRAMEWORM COST · Architecture Comparison")
    results = CostCalculator(hardware=hardware).compare_architectures(latency)
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {CD}", padding=(0, 2))
    t.add_column("Architecture", style=f"{O}", width=14)
    t.add_column("$/request", width=16)
    t.add_column("$/1k", width=12)
    t.add_column("Monthly 10rps", width=14)
    t.add_column("vs cheapest", width=10)
    cheapest = results[0].total_cost_usd
    for r in results:
        ratio = r.total_cost_usd / cheapest
        color = G if ratio < 1.5 else (A if ratio < 5 else R)
        monthly = r.total_cost_usd * 10 * 30 * 24 * 3600
        t.add_row(
            r.architecture,
            f"[{color}]${r.total_cost_usd:.8f}[/{color}]",
            f"${r.cost_per_1k_requests:.4f}",
            f"${monthly:,.2f}",
            f"[{color}]{ratio:.1f}x[/{color}]",
        )
    console.print(t)


@cost.command("report")
@click.argument("file", type=click.Path(exists=True))
def cost_report(file):
    """Generate cost report from saved records."""
    from cost.report import CostReport
    from cost.store import CostStore

    section("FRAMEWORM COST · Report")
    CostReport(CostStore(path=file)).print()


# ── INIT ──────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("project_name")
@click.option(
    "--template", type=click.Choice(["basic", "gan", "vae", "diffusion"]), default="basic"
)
@click.option("--path", type=str, default=".")
def init(project_name, template, path):
    """Scaffold a new FRAMEWORM project."""
    import yaml as _yaml

    section("FRAMEWORM INIT")
    project_path = Path(path) / project_name
    dirs = ["configs", "experiments", "checkpoints", "data", "plugins"]
    with console.status(f"[{A}]Creating {project_name}...[/{A}]", spinner="dots"):
        project_path.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (project_path / d).mkdir(exist_ok=True)
        _yaml.dump(
            {
                "model": {
                    "type": template if template != "basic" else "dcgan",
                    "latent_dim": 100,
                    "image_size": 64,
                    "channels": 3,
                    "features_g": 64,
                    "features_d": 64,
                },
                "training": {"epochs": 100, "batch_size": 64, "gradient_accumulation_steps": 1},
                "optimizer": {"lr": 0.0002, "beta1": 0.5, "beta2": 0.999},
            },
            open(project_path / "configs" / "config.yaml", "w"),
            default_flow_style=False,
        )
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_row(f"[{O}]{project_name}/[/{O}]", "")
    for d in dirs:
        t.add_row(f"[{CD}]  ├── {d}/[/{CD}]", "")
    t.add_row(f"[{CD}]  └── configs/config.yaml[/{CD}]", f"[{CD}]({template})[/{CD}]")
    console.print(Panel(t, title=f"[{O}]Project Created[/{O}]", border_style=OD))
    console.print(f"\n  [{O}]cd {project_name}[/{O}]")
    console.print(f"  [{O}]frameworm train --config configs/config.yaml[/{O}]\n")


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--port", type=int, default=8080)
@click.option("--host", type=str, default="0.0.0.0")
def dashboard(port, host):
    """Launch web dashboard."""
    from ui.api import run_dashboard

    section("FRAMEWORM DASHBOARD")
    success(f"Starting at http://{host}:{port}")
    run_dashboard(host=host, port=port)


# ── PIPELINE ──────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("pipeline_file")
@click.option("--dry-run", is_flag=True)
def pipeline(pipeline_file, dry_run):
    """Run automated pipeline."""
    from cli.pipeline import Pipeline

    section("FRAMEWORM PIPELINE")
    pipe = Pipeline(pipeline_file)
    if dry_run:
        info("Dry run:")
        for i, step in enumerate(pipe.steps, 1):
            console.print(f"  [{A}]{i:02d}[/{A}]  [{C}]{step['name']}[/{C}]")
    else:
        pipe.run()


# ── PLUGINS ───────────────────────────────────────────────────────────────────
@cli.group()
def plugins():
    """Manage plugins."""
    pass


@plugins.command("list")
@click.option("--verbose", "-v", is_flag=True)
def list_plugins(verbose):
    """List all available plugins."""
    from plugins.loader import get_plugin_loader

    section("FRAMEWORM PLUGINS")
    loader = get_plugin_loader()
    if verbose:
        loader.print_plugins()
        return
    ps = loader.list_plugins()
    if not ps:
        info("No plugins found.")
        return
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {CD}", padding=(0, 2))
    t.add_column("Status", width=6)
    t.add_column("Name", style=f"{O}", width=20)
    t.add_column("Version", width=8)
    t.add_column("Description", style=f"{CD}")
    for name, meta in ps.items():
        t.add_row(
            f"[{G}]✓[/{G}]" if loader._loaded.get(name) else f"[{CD}]○[/{CD}]",
            name,
            meta.version,
            meta.description,
        )
    console.print(t)


@plugins.command("load")
@click.argument("plugin_name")
def load_plugin_cmd(plugin_name):
    """Load a plugin."""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    with console.status(f"[{A}]Loading {plugin_name}...[/{A}]", spinner="dots"):
        ok = loader.load(plugin_name)
    success(f"Loaded: {plugin_name}") if ok else error(f"Failed: {plugin_name}")


@plugins.command("unload")
@click.argument("plugin_name")
def unload_plugin_cmd(plugin_name):
    """Unload a plugin."""
    from plugins.loader import get_plugin_loader

    get_plugin_loader().unload(plugin_name)
    success(f"Unloaded: {plugin_name}")


@plugins.command("info")
@click.argument("plugin_name")
def plugin_info(plugin_name):
    """Show plugin details."""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    ps = loader.list_plugins()
    if plugin_name not in ps:
        error(f"Plugin '{plugin_name}' not found")
        return
    meta = ps[plugin_name]
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    for k, v in [
        ("Name", meta.name),
        ("Version", meta.version),
        ("Author", meta.author),
        ("Description", meta.description),
        ("Hooks", ", ".join(meta.hooks)),
    ]:
        t.add_row(f"[{CD}]{k}[/{CD}]", f"[{C}]{v}[/{C}]")
    console.print(Panel(t, title=f"[{O}]{plugin_name}[/{O}]", border_style=OD))


@plugins.command("create")
@click.argument("name")
@click.option("--dir", default="frameworm_plugins")
def create_plugin(name, dir):
    """Scaffold a new plugin."""
    import yaml as _yaml

    plugin_dir = Path(dir) / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    _yaml.dump(
        {
            "name": name,
            "version": "0.1.0",
            "author": "Your Name",
            "description": f"{name} plugin",
            "entry_point": "plugin:register",
            "dependencies": ["torch>=2.0.0"],
            "hooks": ["model"],
        },
        open(plugin_dir / "plugin.yaml", "w"),
    )
    open(plugin_dir / "__init__.py", "w").write(
        f'def register():\n    print("✓ {name} registered")\n'
    )
    success(f"Plugin created: {plugin_dir}")
    info(f"frameworm plugins load {name}")


# ── CONFIG ────────────────────────────────────────────────────────────────────
@cli.group()
def config():
    """Manage configurations."""
    pass


@config.command("list")
def config_list():
    configs = [
        c
        for c in list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml"))
        if "venv" not in str(c)
    ]
    if not configs:
        info("No config files found.")
        return
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    for c in configs[:20]:
        t.add_row(f"[{O}]{c}[/{O}]")
    console.print(Panel(t, title=f"[{O}]Config Files[/{O}]", border_style=OD))


@config.command("show")
@click.argument("config_file")
def config_show(config_file):
    from core import Config

    console.print(Syntax(Config(config_file).to_yaml(), "yaml", theme="monokai", line_numbers=True))


@config.command("validate")
@click.argument("config_file")
def config_validate(config_file):
    from core import Config

    try:
        Config(config_file)
        success(f"Valid: {config_file}")
    except Exception as e:
        error(f"Invalid: {e}")
        sys.exit(1)


# ── DEPLOY + SHIFT (forwarded) ────────────────────────────────────────────────
from deploy.cli.commands import register_deploy_commands

register_deploy_commands(cli)
from shift.cli.commands import register_shift_commands

register_shift_commands(cli)


# ── COMPLETION ────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default="bash")
def completion(shell):
    """Generate shell completion script."""
    import click_completion

    click_completion.init()
    click.echo(click_completion.get_code(shell=shell))


if __name__ == "__main__":
    cli()
