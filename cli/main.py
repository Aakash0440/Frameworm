"""
FRAMEWORM Command Line Interface.
"""

import click
from pathlib import Path
import sys


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    FRAMEWORM - Complete Machine Learning Framework
    
    Train, evaluate, and deploy models with ease.
    """
    pass


@cli.command()
@click.argument('project_name')
@click.option('--template', type=click.Choice(['basic', 'gan', 'vae', 'diffusion']), 
              default='basic', help='Project template')
@click.option('--path', type=str, default='.', help='Project directory')
def init(project_name, template, path):
    """
    Initialize a new FRAMEWORM project.
    
    Example:
        frameworm init my-project --template vae
    """
    from frameworm.cli.init import create_project
    
    project_path = Path(path) / project_name
    
    click.echo(f"Creating new project: {project_name}")
    click.echo(f"Template: {template}")
    click.echo(f"Location: {project_path}")
    
    create_project(project_path, template)
    
    click.echo(f"\n✓ Project created successfully!")
    click.echo(f"\nNext steps:")
    click.echo(f"  cd {project_name}")
    click.echo(f"  frameworm train --config config.yaml")


@cli.command()
@click.option('--config', type=str, required=True, help='Config file path')
@click.option('--gpus', type=str, default=None, help='GPU IDs (e.g., 0,1,2,3)')
@click.option('--experiment', type=str, default=None, help='Experiment name')
@click.option('--resume', type=str, default=None, help='Resume from checkpoint')
@click.option('--debug', is_flag=True, help='Debug mode')
def train(config, gpus, experiment, resume, debug):
    """
    Train a model.
    
    Example:
        frameworm train --config config.yaml --gpus 0,1,2,3
    """
    from frameworm.cli.train import run_training
    
    if debug:
        click.echo("Debug mode enabled")
    
    click.echo(f"Training with config: {config}")
    
    if gpus:
        gpu_ids = [int(g) for g in gpus.split(',')]
        click.echo(f"Using GPUs: {gpu_ids}")
    else:
        gpu_ids = None
    
    run_training(
        config_path=config,
        gpu_ids=gpu_ids,
        experiment_name=experiment,
        resume_from=resume,
        debug=debug
    )


@cli.command()
@click.option('--config', type=str, required=True, help='Config file')
@click.option('--checkpoint', type=str, required=True, help='Model checkpoint')
@click.option('--metrics', type=str, default='fid,is', help='Metrics to compute')
@click.option('--num-samples', type=int, default=10000, help='Number of samples')
def evaluate(config, checkpoint, metrics, num_samples):
    """
    Evaluate a trained model.
    
    Example:
        frameworm evaluate --checkpoint best.pt --metrics fid,is
    """
    from frameworm.cli.evaluate import run_evaluation
    
    metric_list = metrics.split(',')
    
    click.echo(f"Evaluating checkpoint: {checkpoint}")
    click.echo(f"Metrics: {metric_list}")
    
    run_evaluation(
        config_path=config,
        checkpoint_path=checkpoint,
        metrics=metric_list,
        num_samples=num_samples
    )


@cli.command()
@click.option('--config', type=str, required=True, help='Base config')
@click.option('--space', type=str, required=True, help='Search space YAML')
@click.option('--method', type=click.Choice(['grid', 'random', 'bayesian']),
              default='random', help='Search method')
@click.option('--trials', type=int, default=50, help='Number of trials')
@click.option('--parallel', type=int, default=1, help='Parallel jobs')
def search(config, space, method, trials, parallel):
    """
    Run hyperparameter search.
    
    Example:
        frameworm search --config config.yaml --space search.yaml --method bayesian
    """
    from frameworm.cli.search import run_search
    
    click.echo(f"Starting {method} search")
    click.echo(f"Trials: {trials}")
    click.echo(f"Parallel jobs: {parallel}")
    
    run_search(
        config_path=config,
        space_path=space,
        method=method,
        n_trials=trials,
        n_jobs=parallel
    )


@cli.command()
@click.argument('checkpoint')
@click.option('--format', type=click.Choice(['torchscript', 'onnx', 'all']),
              default='all', help='Export format')
@click.option('--output', type=str, default='exported', help='Output directory')
@click.option('--quantize', is_flag=True, help='Also quantize model')
@click.option('--benchmark', is_flag=True, help='Benchmark exported model')
def export(checkpoint, format, output, quantize, benchmark):
    """
    Export trained model.
    
    Example:
        frameworm export best.pt --format onnx --quantize
    """
    from frameworm.cli.export import export_model
    
    click.echo(f"Exporting checkpoint: {checkpoint}")
    click.echo(f"Format: {format}")
    
    export_model(
        checkpoint_path=checkpoint,
        export_format=format,
        output_dir=output,
        quantize=quantize,
        benchmark=benchmark
    )


@cli.command()
@click.argument('model_path')
@click.option('--port', type=int, default=8000, help='Port to serve on')
@click.option('--workers', type=int, default=1, help='Number of workers')
@click.option('--host', type=str, default='0.0.0.0', help='Host to bind to')
def serve(model_path, port, workers, host):
    """
    Serve model via REST API.
    
    Example:
        frameworm serve model.pt --port 8000 --workers 4
    """
    from frameworm.deployment.server import create_server
    
    click.echo(f"Starting server")
    click.echo(f"Model: {model_path}")
    click.echo(f"URL: http://{host}:{port}")
    click.echo(f"Docs: http://{host}:{port}/docs")
    
    create_server(model_path, host, port)


@cli.group()
def config():
    """Manage configurations"""
    pass


@config.command('list')
def config_list():
    """List available configs"""
    click.echo("Available configurations:")
    # List config files


@config.command('show')
@click.argument('config_file')
def config_show(config_file):
    """Show config contents"""
    from frameworm.core import Config
    cfg = Config(config_file)
    click.echo(cfg.to_yaml())


@config.command('validate')
@click.argument('config_file')
def config_validate(config_file):
    """Validate config file"""
    try:
        from frameworm.core import Config
        cfg = Config(config_file)
        click.echo("✓ Config is valid")
    except Exception as e:
        click.echo(f"✗ Config error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()