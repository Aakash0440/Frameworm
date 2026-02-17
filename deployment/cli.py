"""
Deployment CLI tools.
"""

import os
import subprocess
from pathlib import Path

import click


@click.group()
def cli():
    """FRAMEWORM Deployment Tools"""
    pass


@cli.command()
@click.option("--model", type=str, required=True, help="Path to model file")
@click.option("--format", type=click.Choice(["torchscript", "onnx", "all"]), default="all")
@click.option("--output-dir", type=str, default="exported_models")
@click.option("--quantize", is_flag=True, help="Also export quantized version")
def export(model, format, output_dir, quantize):
    """Export model to deployment formats"""
    click.echo(f"Exporting model: {model}")

    # This would use ModelExporter in practice
    click.echo(f"Format: {format}")
    click.echo(f"Output: {output_dir}")
    if quantize:
        click.echo("✓ Quantization enabled")


@cli.command()
@click.option("--model", type=str, required=True, help="Path to model file")
@click.option("--port", type=int, default=8000, help="Port to run on")
@click.option("--workers", type=int, default=1, help="Number of workers")
def serve(model, port, workers):
    """Start model server"""
    click.echo(f"Starting server with model: {model}")

    cmd = [
        "python",
        "-m",
        "frameworm.deployment.server",
        "--model",
        model,
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]

    subprocess.run(cmd)


@cli.command()
@click.option("--tag", type=str, default="frameworm-model:latest", help="Docker image tag")
def docker_build(tag):
    """Build Docker image"""
    click.echo(f"Building Docker image: {tag}")

    cmd = ["docker", "build", "-t", tag, "."]
    subprocess.run(cmd)

    click.echo("✓ Docker image built")


@cli.command()
@click.option("--tag", type=str, default="frameworm-model:latest", help="Docker image tag")
@click.option("--port", type=int, default=8000, help="Port to expose")
def docker_run(tag, port):
    """Run Docker container"""
    click.echo(f"Running Docker container: {tag}")

    cmd = ["docker", "run", "-p", f"{port}:8000", "-v", f"{os.getcwd()}/models:/app/models", tag]

    subprocess.run(cmd)


@cli.command()
def k8s_deploy():
    """Deploy to Kubernetes"""
    click.echo("Deploying to Kubernetes...")

    cmd = ["kubectl", "apply", "-f", "k8s/"]
    subprocess.run(cmd)

    click.echo("✓ Deployed to Kubernetes")


if __name__ == "__main__":
    cli()
