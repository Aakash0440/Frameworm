"""
FRAMEWORM DEPLOY — Docker builder.
Generates Dockerfile and docker-compose.yml for a deployed model server.
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("frameworm.deploy")


class DockerBuilder:
    """
    Generates production-grade Docker deployment files.

    Features:
        - Non-root user (frameworm)
        - HEALTHCHECK using /health endpoint
        - Model checkpoint baked in
        - Layer-cache optimised

    Usage:
        builder = DockerBuilder()
        builder.generate_dockerfile(server_dir, model_name, model_version,
                                    model_path, port)
        builder.generate_compose(model_name, model_version, port, output_path)
    """

    BASE_IMAGE = "python:3.11-slim"

    def generate_dockerfile(
        self,
        server_dir: str,
        model_name: str,
        model_version: str,
        model_path: str,
        port: int = 8000,
        use_gpu: bool = False,
    ) -> str:
        """Write Dockerfile to server_dir. Returns path as string."""
        os.makedirs(server_dir, exist_ok=True)

        model_filename = Path(model_path).name
        base = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" if use_gpu else self.BASE_IMAGE

        content = f"""\
# FRAMEWORM DEPLOY - Auto-generated Dockerfile
# Model: {model_name} v{model_version}

FROM {base}

# Security: non-root user named 'frameworm'
RUN groupadd -r frameworm && useradd -r -g frameworm frameworm

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt* ./
RUN pip install --no-cache-dir fastapi uvicorn torch --index-url \\
    https://download.pytorch.org/whl/cpu 2>/dev/null || \\
    pip install --no-cache-dir fastapi uvicorn torch

# Copy application code and model
COPY . /app/
COPY {model_path} /app/model/{model_filename}

RUN mkdir -p /app/experiments && chown -R frameworm:frameworm /app
USER frameworm

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
    CMD python -c "import urllib.request; \\
urllib.request.urlopen('http://localhost:{port}/health')"

CMD ["python", "server.py"]
"""
        out_path = os.path.join(server_dir, "Dockerfile")
        # encoding="utf-8" prevents cp1252 errors on Windows
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[DEPLOY] Dockerfile generated -> {out_path}")
        return out_path

    def generate_compose(
        self,
        model_name: str,
        model_version: str,
        port: int,
        output_path: Optional[str] = None,
    ) -> str:
        """Write docker-compose.yml. Returns path as string."""
        if output_path is None:
            output_path = f"deploy/generated/{model_name}/docker-compose.yml"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        safe_name = model_name.replace("_", "-").lower()
        content = f"""\
version: "3.9"

services:
  {safe_name}:
    image: frameworm/{model_name}:{model_version}
    container_name: {safe_name}
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_NAME={model_name}
      - MODEL_VERSION={model_version}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    volumes:
      - ./experiments:/app/experiments
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
"""
        # encoding="utf-8" prevents cp1252 errors on Windows
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[DEPLOY] docker-compose.yml generated -> {output_path}")
        return output_path

    def build_image(
        self, server_dir: str, model_name: str, model_version: str, no_cache: bool = False
    ) -> str:
        """Build Docker image. Returns image tag."""
        if not shutil.which("docker"):
            raise RuntimeError("[DEPLOY] Docker not found.")
        tag = f"frameworm/{model_name}:{model_version}"
        cmd = ["docker", "build", "-t", tag, server_dir]
        if no_cache:
            cmd.append("--no-cache")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")
        return tag
