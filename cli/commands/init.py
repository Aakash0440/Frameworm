"""
Init Command - Create new FRAMEWORM project
"""

import click


@click.command()
@click.option("--name", default=None, help="Project name")
@click.option("--model", default=None, help="Model type (gan, vae, diffusion)")
@click.option("--dataset", default=None, help="Dataset name")
def init(name, model, dataset):
    """Create a new FRAMEWORM project"""
    click.echo("FRAMEWORM Project Wizard")
