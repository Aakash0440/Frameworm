"""
FRAMEWORM DEPLOY CLI commands.

frameworm deploy start    --model path/to/checkpoint.pt --name my_model
frameworm deploy stop     --name my_model
frameworm deploy status   --name my_model
frameworm deploy list
frameworm deploy rollback --name my_model
frameworm deploy promote  --name my_model --version v1.2 --stage production

Wire into cli/main.py with:
    from deploy.cli.commands import register_deploy_commands
    register_deploy_commands(main_cli_group)
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def register_deploy_commands(cli):
    try:
        import click

        _register_click(cli)
    except ImportError:
        print("[DEPLOY] Click not installed — CLI unavailable")


def _register_click(cli):
    import click

    @cli.group()
    def deploy():
        """FRAMEWORM DEPLOY — one command from model to production."""
        pass

    @deploy.command()
    @click.option("--model", required=True, help="Path to model checkpoint (.pt or .onnx)")
    @click.option("--name", required=True, help="Deployment name (e.g. fraud_classifier)")
    @click.option("--version", default="v1.0", help="Version tag (default: v1.0)")
    @click.option(
        "--type",
        "model_type",
        default="generic",
        type=click.Choice(["vae", "dcgan", "ddpm", "vqvae2", "vitgan", "cfg_ddpm", "generic"]),
        help="Model architecture type",
    )
    @click.option("--port", default=8000, help="Port to serve on (default: 8000)")
    @click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
    @click.option(
        "--shift", default=None, help="Reference .shift profile name for drift monitoring"
    )
    @click.option("--build-docker", is_flag=True, help="Build Docker image after generating server")
    @click.option("--quantize", is_flag=True, help="Apply quantization before export")
    def start(model, name, version, model_type, port, device, shift, build_docker, quantize):
        """Deploy a FRAMEWORM model to a production API."""
        click.echo(f"\n[DEPLOY] Starting deployment: {name} v{version}")
        click.echo(f"         Model:  {model}")
        click.echo(f"         Type:   {model_type}")
        click.echo(f"         Port:   {port}\n")

        # 1. Export model
        from deploy.core.model_exporter import ModelExporter

        exporter = ModelExporter()
        export_dir = f"deploy/generated/{name}"
        Path(export_dir).mkdir(parents=True, exist_ok=True)

        click.echo("[DEPLOY] Step 1/4 — Exporting model...")
        try:
            ts_path = exporter.export_torchscript(
                model, f"{export_dir}/{name}_torchscript.pt", quantize=quantize
            )
            click.echo(f"         TorchScript → {ts_path}")
        except Exception as e:
            click.echo(f"         [WARN] TorchScript export failed: {e}")
            ts_path = model  # fall back to original checkpoint

        # 2. Register model
        from deploy.core.registry import ModelRegistry

        registry = ModelRegistry()
        click.echo("[DEPLOY] Step 2/4 — Registering in model registry...")
        registry.register(
            name,
            version,
            model_type,
            checkpoint_path=str(ts_path),
            stage="staging",
        )
        click.echo(f"         Registered {name}:{version} → staging")

        # 3. Generate server
        from deploy.core.server_builder import ServerBuilder

        builder = ServerBuilder()
        click.echo("[DEPLOY] Step 3/4 — Generating server...")
        server_path = builder.build(
            model_type=model_type,
            model_name=name,
            model_version=version,
            model_path=str(ts_path),
            output_dir=export_dir,
            shift_reference=shift,
            port=port,
            device=device,
        )
        click.echo(f"         Server → {server_path}")

        # 4. Docker (optional)
        if build_docker:
            from deploy.core.docker_builder import DockerBuilder

            docker = DockerBuilder()
            click.echo("[DEPLOY] Step 4/4 — Building Docker image...")
            docker.generate_dockerfile(export_dir, name, version, str(ts_path), port)
            docker.generate_compose(name, version, port, f"{export_dir}/docker-compose.yml")
            try:
                tag = docker.build_image(export_dir, name, version)
                click.echo(f"         Image → {tag}")
                registry.promote(name, version, "production")
                click.echo(f"         Promoted {name}:{version} → production")
            except Exception as e:
                click.echo(f"         [WARN] Docker build failed: {e}")
        else:
            click.echo("[DEPLOY] Step 4/4 — Skipping Docker (--build-docker not set)")
            click.echo(f"         Run locally: python {server_path}")

        click.echo(f"\n[DEPLOY] ✓ Done — {name} v{version} deployed")
        if not build_docker:
            click.echo(f"         Start server: python {server_path}")
            click.echo(f"         Health check: curl http://localhost:{port}/health\n")

    @deploy.command()
    @click.option("--name", required=True)
    def stop(name):
        """Stop a running deployment and archive it in the registry."""
        if shutil.which("docker"):
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name=frameworm-{name}", "-q"],
                capture_output=True,
                text=True,
            )
            containers = result.stdout.strip().split("\n")
            for cid in containers:
                if cid:
                    subprocess.run(["docker", "stop", cid])
                    click.echo(f"[DEPLOY] Stopped container: {cid}")
        else:
            click.echo("[DEPLOY] Docker not available — stop manually")

        from deploy.core.registry import ModelRegistry

        registry = ModelRegistry()
        versions = registry.get_by_name(name)
        for v in versions:
            if v["stage"] == "production":
                registry.demote(name, v["version"], "archived", note="Stopped via CLI")
                click.echo(f"[DEPLOY] Archived {name}:{v['version']}")

    @deploy.command()
    @click.option("--name", required=True)
    def status(name):
        """Show deployment status and live metrics."""
        from deploy.core.registry import ModelRegistry

        registry = ModelRegistry()
        versions = registry.get_by_name(name)
        if not versions:
            click.echo(f"[DEPLOY] No deployments found for '{name}'")
            return
        click.echo(f"\n[DEPLOY] Status: {name}")
        click.echo(f"  {'Version':<12} {'Stage':<14} {'Deployed At':<22} {'Type'}")
        click.echo(f"  {'─'*12} {'─'*14} {'─'*22} {'─'*12}")
        for v in versions:
            marker = "▶" if v["stage"] == "production" else " "
            click.echo(
                f"  {marker} {v['version']:<10} {v['stage']:<14} "
                f"{v.get('deployed_at','—'):<22} {v.get('model_type','—')}"
            )
        click.echo()

    @deploy.command(name="list")
    def list_deployments():
        """List all registered model deployments."""
        from deploy.core.registry import ModelRegistry

        ModelRegistry().list_all()

    @deploy.command()
    @click.option("--name", required=True)
    @click.option("--reason", default="Manual rollback via CLI")
    def rollback(name, reason):
        """Manually trigger rollback to previous version."""
        from deploy.core.registry import ModelRegistry

        registry = ModelRegistry()
        current = registry.get_current_production(name)
        if not current:
            click.echo(f"[DEPLOY] No production version found for '{name}'")
            return
        from deploy.rollback.controller import RollbackController

        controller = RollbackController(name, current["version"])
        controller.rollback(reason)
        click.echo(f"[DEPLOY] Rollback initiated for {name}")

    @deploy.command()
    @click.option("--name", required=True)
    @click.option("--version", required=True)
    @click.option(
        "--stage", required=True, type=click.Choice(["dev", "staging", "production", "archived"])
    )
    def promote(name, version, stage):
        """Promote a model version to a lifecycle stage."""
        from deploy.core.registry import ModelRegistry

        ModelRegistry().promote(name, version, stage)
        click.echo(f"[DEPLOY] {name}:{version} → {stage}")
