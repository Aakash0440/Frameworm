"""
Command-line interface for experiment management.
"""

import click
from pathlib import Path
from experiment.manager import ExperimentManager
from experiment.visualization import (
    plot_metric_comparison,
    plot_multiple_metrics
)
from tabulate import tabulate


@click.group()
def cli():
    """Frameworm Experiment Management CLI"""
    pass


@cli.command()
@click.option('--root-dir', default='experiments', help='Experiments directory')
@click.option('--status', help='Filter by status')
@click.option('--limit', type=int, default=20, help='Maximum results')
def list(root_dir, status, limit):
    """List experiments"""
    manager = ExperimentManager(root_dir)
    df = manager.list_experiments(status=status, limit=limit)
    
    if len(df) == 0:
        click.echo("No experiments found.")
        return
    
    # Select columns to display
    display_cols = ['experiment_id', 'name', 'status', 'created_at']
    display_df = df[display_cols]
    
    click.echo("\n" + tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    click.echo(f"\nTotal: {len(df)} experiments\n")


@cli.command()
@click.argument('experiment_id')
@click.option('--root-dir', default='experiments', help='Experiments directory')
def show(experiment_id, root_dir):
    """Show experiment details"""
    manager = ExperimentManager(root_dir)
    exp = manager.get_experiment(experiment_id)
    
    click.echo("\n" + "="*60)
    click.echo(f"EXPERIMENT: {exp['name']}")
    click.echo("="*60)
    
    click.echo(f"\nID: {exp['experiment_id']}")
    click.echo(f"Status: {exp['status']}")
    click.echo(f"Created: {exp['created_at']}")
    
    if exp.get('description'):
        click.echo(f"Description: {exp['description']}")
    
    if exp.get('git_hash'):
        click.echo(f"Git Hash: {exp['git_hash']}")
        if exp.get('git_dirty'):
            click.echo("  (dirty working directory)")
    
    # Config
    if exp.get('config'):
        click.echo("\nConfiguration:")
        for key, value in sorted(exp['config'].items()):
            click.echo(f"  {key}: {value}")
    
    # Metrics summary
    if exp.get('metrics_summary'):
        click.echo("\nMetrics Summary:")
        metrics_data = []
        for metric_name, summary in exp['metrics_summary'].items():
            metrics_data.append([
                metric_name,
                f"{summary['min_value']:.4f}",
                f"{summary['max_value']:.4f}",
                f"{summary['avg_value']:.4f}",
                summary['count']
            ])
        
        click.echo(tabulate(
            metrics_data,
            headers=['Metric', 'Min', 'Max', 'Avg', 'Count'],
            tablefmt='grid'
        ))
    
    click.echo("="*60 + "\n")


@cli.command()
@click.argument('experiment_ids', nargs=-1, required=True)
@click.option('--root-dir', default='experiments', help='Experiments directory')
@click.option('--metrics', multiple=True, help='Specific metrics to compare')
def compare(experiment_ids, root_dir, metrics):
    """Compare multiple experiments"""
    manager = ExperimentManager(root_dir)
    
    metrics_list = list(metrics) if metrics else None
    df = manager.compare_experiments(list(experiment_ids), metrics=metrics_list)
    
    if len(df) == 0:
        click.echo("No experiments found.")
        return
    
    click.echo("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    click.echo()


@cli.command()
@click.argument('experiment_ids', nargs=-1, required=True)
@click.argument('metric_name')
@click.option('--root-dir', default='experiments', help='Experiments directory')
@click.option('--output', help='Output file path')
def plot(experiment_ids, metric_name, root_dir, output):
    """Plot metric comparison"""
    manager = ExperimentManager(root_dir)
    
    plot_metric_comparison(
        manager,
        list(experiment_ids),
        metric_name,
        save_path=output
    )
    
    if output:
        click.echo(f"Plot saved to {output}")
    else:
        click.echo("Displaying plot...")


@cli.command()
@click.argument('experiment_id')
@click.option('--root-dir', default='experiments', help='Experiments directory')
@click.option('--confirm', is_flag=True, help='Skip confirmation')
def delete(experiment_id, root_dir, confirm):
    """Delete an experiment"""
    manager = ExperimentManager(root_dir)
    
    if not confirm:
        exp = manager.get_experiment(experiment_id)
        click.echo(f"Delete experiment '{exp['name']}' ({experiment_id})?")
        if not click.confirm('Are you sure?'):
            click.echo("Cancelled.")
            return
    
    manager.delete_experiment(experiment_id)
    click.echo(f"Deleted experiment {experiment_id}")


@cli.command()
@click.option('--root-dir', default='experiments', help='Experiments directory')
def summary(root_dir):
    """Show experiments summary"""
    manager = ExperimentManager(root_dir)
    manager.print_summary()


if __name__ == '__main__':
    cli()