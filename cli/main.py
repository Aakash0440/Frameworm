"""
FRAMEWORM Command Line Interface.
"""

import sys
from pathlib import Path

import click
import click_completion
click_completion.init()


# ================= CLI GROUP =================
@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    FRAMEWORM - Complete Machine Learning Framework

    Train, evaluate, and deploy models with ease.
    """
    pass


# ================= DASHBOARD =================
@cli.command()
@click.option('--port', type=int, default=8080, help='Port to run on')
@click.option('--host', type=str, default='0.0.0.0', help='Host to bind to')
def dashboard(port, host):
    """Launch web dashboard."""
    from ui.api import run_dashboard

    click.echo(f"Starting dashboard at http://{host}:{port}")
    click.echo(f"Press Ctrl+C to stop")
    run_dashboard(host=host, port=port)


# ================= COMPLETION =================
@cli.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish']), default='bash')
def completion(shell):
    """Generate shell completion script."""
    from click_completion import get_code
    click.echo(get_code(shell=shell))


# ================= PIPELINE =================
@cli.command()
@click.argument('pipeline_file')
@click.option('--dry-run', is_flag=True, help='Show steps without executing')
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
@click.argument('project_name')
@click.option('--template', type=click.Choice(['basic', 'gan', 'vae', 'diffusion']), default='basic')
@click.option('--path', type=str, default='.')
def init(project_name, template, path):
    """Initialize a new FRAMEWORM project."""
    from cli.init import create_project

    project_path = Path(path) / project_name

    click.echo(f"Creating new project: {project_name}")
    click.echo(f"Template: {template}")
    click.echo(f"Location: {project_path}")

    create_project(project_path, template)

    click.echo(f"\n✓ Project created successfully!")
    click.echo(f"Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo(f"  frameworm train --config config.yaml")


# ================= MONITOR =================
@cli.command()
@click.argument('experiment_dir')
@click.option('--refresh', type=int, default=2, help='Refresh rate in seconds')
def monitor(experiment_dir, refresh):
    """Monitor training in real-time."""
    from cli.monitor import TrainingMonitor

    monitor_obj = TrainingMonitor(experiment_dir)
    monitor_obj.watch(refresh_rate=refresh)


# ================= TRAIN =================
@cli.command()
@click.option('--config', type=str, required=True, help='Config file path')
@click.option('--gpus', type=str, default=None, help='GPU IDs (e.g., 0,1,2,3)')
@click.option('--experiment', type=str, default=None, help='Experiment name')
@click.option('--resume', type=str, default=None, help='Resume from checkpoint')
@click.option('--debug', is_flag=True, help='Debug mode')
def train(config, gpus, experiment, resume, debug):
    """Train a model."""
    from cli.train import run_training

    if debug:
        click.echo("Debug mode enabled")

    click.echo(f"Training with config: {config}")

    gpu_ids = [int(g) for g in gpus.split(',')] if gpus else None
    if gpu_ids:
        click.echo(f"Using GPUs: {gpu_ids}")

    run_training(
        config_path=config,
        gpu_ids=gpu_ids,
        experiment_name=experiment,
        resume_from=resume,
        debug=debug
    )


# ================= EVALUATE =================
@cli.command()
@click.option('--config', type=str, required=True, help='Config file')
@click.option('--checkpoint', type=str, required=True, help='Model checkpoint')
@click.option('--metrics', type=str, default='fid,is', help='Metrics to compute')
@click.option('--num-samples', type=int, default=10000, help='Number of samples')
def evaluate(config, checkpoint, metrics, num_samples):
    """Evaluate a trained model."""
    from cli.evaluate import run_evaluation

    metric_list = metrics.split(',')
    click.echo(f"Evaluating checkpoint: {checkpoint}")
    click.echo(f"Metrics: {metric_list}")

    run_evaluation(
        config_path=config,
        checkpoint_path=checkpoint,
        metrics=metric_list,
        num_samples=num_samples
    )


# ================= SEARCH =================
@cli.command()
@click.option('--config', type=str, required=True, help='Base config')
@click.option('--space', type=str, required=True, help='Search space YAML')
@click.option('--method', type=click.Choice(['grid', 'random', 'bayesian']), default='random')
@click.option('--trials', type=int, default=50)
@click.option('--parallel', type=int, default=1)
def search(config, space, method, trials, parallel):
    """Run hyperparameter search."""
    from cli.search import run_search

    click.echo(f"Starting {method} search with {trials} trials and {parallel} parallel jobs")
    run_search(
        config_path=config,
        space_path=space,
        method=method,
        n_trials=trials,
        n_jobs=parallel
    )


# ================= EXPORT =================
@cli.command()
@click.argument('checkpoint')
@click.option('--format', type=click.Choice(['torchscript', 'onnx', 'all']), default='all')
@click.option('--output', type=str, default='exported')
@click.option('--quantize', is_flag=True)
@click.option('--benchmark', is_flag=True)
def export(checkpoint, format, output, quantize, benchmark):
    """Export trained model."""
    from cli.export import export_model

    click.echo(f"Exporting checkpoint: {checkpoint}, format: {format}")
    export_model(
        checkpoint_path=checkpoint,
        export_format=format,
        output_dir=output,
        quantize=quantize,
        benchmark=benchmark
    )


# ================= SERVE =================
@cli.command()
@click.argument('model_path')
@click.option('--port', type=int, default=8000)
@click.option('--workers', type=int, default=1)
@click.option('--host', type=str, default='0.0.0.0')
def serve(model_path, port, workers, host):
    """Serve model via REST API."""
    from deployment.server import create_server

    click.echo(f"Starting server at http://{host}:{port} with {workers} worker(s)")
    click.echo(f"Model: {model_path}")
    click.echo(f"Docs: http://{host}:{port}/docs")
    create_server(model_path, host, port)


# ================= CONFIG GROUP =================
@cli.group()
def config():
    """Manage configurations."""
    pass


@config.command('list')
def config_list():
    """List available configs."""
    click.echo("Available configurations:")
    # Implement listing logic


@config.command('show')
@click.argument('config_file')
def config_show(config_file):
    """Show config contents."""
    from core import Config
    cfg = Config(config_file)
    click.echo(cfg.to_yaml())


@config.command('validate')
@click.argument('config_file')
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
if __name__ == '__main__':
    cli()
