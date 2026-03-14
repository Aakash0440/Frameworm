"""
FRAMEWORM Command Line Interface.
"""

import sys
from pathlib import Path

import click
import click_completion

click_completion.init()
# ================= TOP-LEVEL IMPORTS FOR PRE-LAUNCH =================
import core
import training
from core.config import Config
from training.trainer import Trainer

# ====================================================================

# Fix for running directly via python cli/main.py
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "cli"


# ================= CLI GROUP =================
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    FRAMEWORM - Complete Machine Learning Framework

    Train, evaluate, and deploy models with ease.
    """
    pass


@cli.group()
def plugins():
    """Manage plugins"""
    pass


from deploy.cli.commands import register_deploy_commands

register_deploy_commands(cli)
from shift.cli.commands import register_shift_commands

register_shift_commands(cli)


@plugins.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_plugins(verbose):
    """List all available plugins."""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    if verbose:
        loader.print_plugins()
    else:
        plugins = loader.list_plugins()
        click.echo(f"\nAvailable plugins: {len(plugins)}")
        for name, meta in plugins.items():
            status = "✓" if loader._loaded.get(name) else "○"
            click.echo(f"  {status} {name} ({meta.version}) - {meta.description}")


@plugins.command("load")
@click.argument("plugin_name")
def load_plugin_cmd(plugin_name):
    """Load a specific plugin."""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    if loader.load(plugin_name):
        click.echo(f"✓ Loaded: {plugin_name}")
    else:
        click.echo(f"✗ Failed to load: {plugin_name}")
        raise click.Abort()


@plugins.command("unload")
@click.argument("plugin_name")
def unload_plugin_cmd(plugin_name):
    """Unload a plugin"""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    loader.unload(plugin_name)
    click.echo(f"✓ Unloaded: {plugin_name}")


@plugins.command("info")
@click.argument("plugin_name")
def plugin_info(plugin_name):
    """Show detailed plugin information."""
    from plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    plugins = loader.list_plugins()
    if plugin_name not in plugins:
        click.echo(f"Plugin '{plugin_name}' not found")
        raise click.Abort()
    meta = plugins[plugin_name]
    click.echo(f"\nPlugin: {meta.name}")
    click.echo(f"  Version: {meta.version}")
    click.echo(f"  Author: {meta.author}")
    click.echo(f"  Description: {meta.description}")
    click.echo(f"  Entry point: {meta.entry_point}")
    click.echo(f"  Hooks: {', '.join(meta.hooks)}")
    click.echo(f"  Dependencies: {', '.join(meta.dependencies)}")
    click.echo(f"  Enabled: {meta.enabled}")


@plugins.command("create")
@click.argument("name")
@click.option("--dir", default="frameworm_plugins", help="Plugin directory")
def create_plugin(name, dir):
    """Create a new plugin template."""
    from pathlib import Path

    import yaml

    plugin_dir = Path(dir) / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": name,
        "version": "0.1.0",
        "author": "Your Name",
        "description": f"{name} plugin",
        "entry_point": "plugin:register",
        "dependencies": ["torch>=2.0.0"],
        "hooks": ["model"],
    }
    with open(plugin_dir / "plugin.yaml", "w") as f:
        yaml.dump(metadata, f)
    template = '''"""
{name} plugin for FRAMEWORM.
"""

from frameworm.core import register_model
from frameworm.plugins.hooks import HookRegistry
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        return x

    def compute_loss(self, x, y=None):
        return {{'loss': 0.0}}


def register():
    register_model('{name}_model', CustomModel)
    print(f"✓ Registered: {name}_model")

    @HookRegistry.on('on_epoch_end')
    def log_custom(trainer, epoch, metrics):
        print(f"Custom hook: epoch {{epoch}}")
'''
    with open(plugin_dir / "__init__.py", "w") as f:
        f.write(template.format(name=name))
    click.echo(f"✓ Created plugin template: {plugin_dir}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Edit {plugin_dir}/__init__.py")
    click.echo(f"  2. Edit {plugin_dir}/plugin.yaml")
    click.echo(f"  3. Load: frameworm plugins load {name}")


# ================= DASHBOARD =================
@cli.command()
@click.option("--port", type=int, default=8080, help="Port to run on")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
def dashboard(port, host):
    """Launch web dashboard."""
    from ui.api import run_dashboard

    click.echo(f"Starting dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    run_dashboard(host=host, port=port)


# ================= COMPLETION =================
@cli.command()
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default="bash")
def completion(shell):
    """Generate shell completion script."""
    from click_completion import get_code

    click.echo(get_code(shell=shell))


# ================= PIPELINE =================
@cli.command()
@click.argument("pipeline_file")
@click.option("--dry-run", is_flag=True, help="Show steps without executing")
def pipeline(pipeline_file, dry_run):
    """Run automated pipeline."""
    from cli.pipeline import Pipeline

    pipe = Pipeline(pipeline_file)
    if dry_run:
        click.echo("Pipeline steps (dry run):")
        for i, step in enumerate(pipe.steps, 1):
            click.echo(f"{i}. {step['name']}")
    else:
        pipe.run()


# ================= INIT =================
@cli.command()
@click.argument("project_name")
@click.option(
    "--template", type=click.Choice(["basic", "gan", "vae", "diffusion"]), default="basic"
)
@click.option("--path", type=str, default=".")
def init(project_name, template, path):
    """Initialize a new FRAMEWORM project."""
    import os
    from pathlib import Path

    project_path = Path(path) / project_name
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "configs").mkdir(exist_ok=True)
    (project_path / "experiments").mkdir(exist_ok=True)
    (project_path / "checkpoints").mkdir(exist_ok=True)
    config = f"""model:
  type: {template if template != 'basic' else 'dcgan'}
  latent_dim: 100
  image_size: 64
  channels: 3
  features_g: 64
  features_d: 64
training:
  epochs: 100
  batch_size: 64
  gradient_accumulation_steps: 1
optimizer:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
"""
    (project_path / "configs" / "config.yaml").write_text(config)
    click.echo(f"\n✓ Project created successfully at {project_path}!")
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  frameworm train --config configs/config.yaml")


# ================= MONITOR =================
@cli.command()
@click.argument("experiment_dir")
@click.option("--refresh", type=int, default=2, help="Refresh rate in seconds")
def monitor(experiment_dir, refresh):
    """Monitor training in real-time."""
    from cli.monitor import TrainingMonitor

    monitor_obj = TrainingMonitor(experiment_dir)
    monitor_obj.watch(refresh_rate=refresh)


# ================= TRAIN =================
@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Config file path")
@click.option("--resume", default=None, help="Resume from checkpoint")
@click.option("--gpus", default="0", help="GPU IDs to use")
def train(config, resume, gpus):
    """Train a model using a config file"""
    import torch
    from rich.console import Console

    from core.config import Config
    from core.registry import ModelRegistry
    from training.trainer import Trainer

    console = Console()

    console.print(f"[green]Loading config: {config}[/green]")
    cfg = Config(config)

    model_name = cfg.get("model.type", "dcgan")
    console.print(f"[green]Building model: {model_name}[/green]")
    import models.gan.dcgan
    from core.registry import get_model

    model = get_model(model_name)(cfg)

    lr = cfg.get("optimizer.lr", 0.0002)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[green]Using device: {device}[/green]")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="checkpoints",
        gradient_accumulation_steps=cfg.get("training.gradient_accumulation_steps", 1),
    )
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Placeholder dataloader — replace with real dataset later
    dummy = TensorDataset(torch.randn(64, 3, 64, 64))
    loader = DataLoader(dummy, batch_size=cfg.get("training.batch_size", 64))

    epochs = cfg.get("training.epochs", 100)
    trainer.train(train_loader=loader, epochs=epochs, resume_from=resume)


# ================= SERVE =================
@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Model checkpoint")
@click.option("--port", default=8080, help="Port to serve on")
@click.option("--host", default="0.0.0.0", help="Host to serve on")
def serve(checkpoint, port, host):
    """Serve a trained model via REST API"""
    from rich.console import Console

    console = Console()
    console.print(f"[green]Loading model from: {checkpoint}[/green]")
    console.print(f"[green]Serving on {host}:{port}[/green]")

    try:
        import torch
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        from core.config import Config
        from core.registry import get_model

        # Load checkpoint
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", None)

        app = FastAPI(title="FRAMEWORM DEPLOY", version="1.0")

        @app.get("/health")
        def health():
            return {"status": "ok", "checkpoint": checkpoint}

        @app.get("/ready")
        def ready():
            return {"ready": True}

        @app.post("/generate")
        def generate(latent_dim: int = 100):
            z = torch.randn(1, latent_dim)
            return {"status": "generated", "shape": list(z.shape)}

        console.print(f"[bold green]✓ Server running at http://{host}:{port}[/bold green]")
        console.print("[dim]Endpoints: GET /health  GET /ready  POST /generate[/dim]")
        uvicorn.run(app, host=host, port=port)

    except ImportError:
        console.print("[red]✗ Missing dependency: pip install fastapi uvicorn[standard][/red]")
    except Exception as e:
        console.print(f"[red]✗ Failed to start server: {e}[/red]")


# ================= EVALUATE =================
@cli.command()
@click.option("--config", type=str, required=True, help="Config file")
@click.option("--checkpoint", type=str, required=True, help="Model checkpoint")
@click.option("--metrics", type=str, default="fid,is", help="Metrics to compute")
@click.option("--num-samples", type=int, default=10000, help="Number of samples")
def evaluate(config, checkpoint, metrics, num_samples):
    """Evaluate a trained model."""
    from cli.evaluate import run_evaluation

    metric_list = metrics.split(",")
    click.echo(f"Evaluating checkpoint: {checkpoint}")
    click.echo(f"Metrics: {metric_list}")
    run_evaluation(
        config_path=config, checkpoint_path=checkpoint, metrics=metric_list, num_samples=num_samples
    )


# ================= SEARCH =================
@cli.command()
@click.option("--config", type=str, required=True, help="Base config")
@click.option("--space", type=str, required=True, help="Search space YAML")
@click.option("--method", type=click.Choice(["grid", "random", "bayesian"]), default="random")
@click.option("--trials", type=int, default=50)
@click.option("--parallel", type=int, default=1)
def search(config, space, method, trials, parallel):
    """Run hyperparameter search."""
    from cli.search_cli import run_search

    click.echo(f"Starting {method} search with {trials} trials and {parallel} parallel jobs")
    run_search(
        config_path=config, space_path=space, method=method, n_trials=trials, n_jobs=parallel
    )


# ================= EXPORT =================
@cli.command()
@click.argument("checkpoint")
@click.option("--format", type=click.Choice(["torchscript", "onnx", "all"]), default="all")
@click.option("--output", type=str, default="exported")
@click.option("--quantize", is_flag=True)
@click.option("--benchmark", is_flag=True)
def export(checkpoint, format, output, quantize, benchmark):
    """Export trained model."""
    from cli.export import export_model

    click.echo(f"Exporting checkpoint: {checkpoint}, format: {format}")
    export_model(
        checkpoint_path=checkpoint,
        export_format=format,
        output_dir=output,
        quantize=quantize,
        benchmark=benchmark,
    )


# ================= CONFIG GROUP =================
@cli.group()
def config():
    """Manage configurations."""
    pass


@config.command("list")
def config_list():
    """List available configs."""
    click.echo("Available configurations:")


@config.command("show")
@click.argument("config_file")
def config_show(config_file):
    """Show config contents."""
    from core import Config

    cfg = Config(config_file)
    click.echo(cfg.to_yaml())


@config.command("validate")
@click.argument("config_file")
def config_validate(config_file):
    """Validate config file."""
    from core import Config

    try:
        cfg = Config(config_file)
        click.echo("✓ Config is valid")
    except Exception as e:
        click.echo(f"✗ Config error: {e}", err=True)
        sys.exit(1)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    cli()
