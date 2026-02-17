"""
Automated pipeline execution.
"""

import yaml
from pathlib import Path
from click import echo, progressbar
import subprocess


class Pipeline:
    """
    Execute automated ML pipelines.

    Runs sequences of commands from YAML definition.
    """

    def __init__(self, pipeline_file: str):
        with open(pipeline_file) as f:
            self.config = yaml.safe_load(f)

        self.name = self.config.get("name", "unnamed")
        self.steps = self.config.get("steps", [])

    def run(self):
        """Execute all pipeline steps"""
        echo(f"Running pipeline: {self.name}")
        echo(f"Steps: {len(self.steps)}")
        echo("=" * 60)

        for i, step in enumerate(self.steps, 1):
            echo(f"\nStep {i}/{len(self.steps)}: {step['name']}")

            command = step.get("command")
            if command:
                self._run_command(command)

            script = step.get("script")
            if script:
                self._run_script(script)

        echo("\n" + "=" * 60)
        echo("✓ Pipeline complete!")

    def _run_command(self, command: str):
        """Run shell command"""
        echo(f"  $ {command}")
        result = subprocess.run(command, shell=True)

        if result.returncode != 0:
            echo(f"  ✗ Command failed with code {result.returncode}")
            raise RuntimeError("Pipeline step failed")

    def _run_script(self, script: str):
        """Run Python script"""
        echo(f"  Running script: {script}")
        exec(open(script).read())
