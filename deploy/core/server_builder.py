"""
FRAMEWORM DEPLOY — Server builder.
Generates a FastAPI inference server tailored to each model architecture.
"""

import os
import textwrap
from pathlib import Path
from typing import Optional

MODEL_SPECS = {
    "vae": {
        "description": "Variational Autoencoder — encode/decode image tensors",
        "input_fields": "x: list  # image tensor [B, C, H, W] normalised to [-1, 1]",
        "inference_code": textwrap.dedent("""
            import torch
            t = torch.tensor(body.x, dtype=torch.float32)
            recon, mu, log_var = self.model(t)
            return {"reconstruction": recon.tolist(), "mu": mu.tolist()}
        """).strip(),
    },
    "dcgan": {
        "description": "DCGAN — generate images from noise",
        "input_fields": "z: list  # latent noise vector [B, latent_dim]",
        "inference_code": textwrap.dedent("""
            import torch
            z = torch.tensor(body.z, dtype=torch.float32)
            images = self.model(z)
            return {"generated_images": images.tolist()}
        """).strip(),
    },
    "ddpm": {
        "description": "DDPM — denoising diffusion probabilistic model",
        "input_fields": ("noisy_image: list  # shape (B,C,H,W)\n" "    timestep: int"),
        "inference_code": textwrap.dedent("""
            import torch
            x = torch.tensor(body.noisy_image, dtype=torch.float32)
            t = torch.tensor([body.timestep])
            out = self.model(x, t)
            return {"denoised": out.tolist()}
        """).strip(),
    },
    "vqvae2": {
        "description": "VQ-VAE-2 — hierarchical discrete representation",
        "input_fields": "x: list  # image tensor [B, C, H, W]",
        "inference_code": textwrap.dedent("""
            import torch
            t = torch.tensor(body.x, dtype=torch.float32)
            recon, loss = self.model(t)
            return {"reconstruction": recon.tolist(), "commitment_loss": float(loss)}
        """).strip(),
    },
    "vitgan": {
        "description": "ViTGAN — Vision Transformer GAN",
        "input_fields": "z: list  # latent noise vector [B, latent_dim]",
        "inference_code": textwrap.dedent("""
            import torch
            z = torch.tensor(body.z, dtype=torch.float32)
            images = self.model(z)
            return {"generated_images": images.tolist()}
        """).strip(),
    },
    "cfg_ddpm": {
        "description": "CFG-DDPM — classifier-free guidance diffusion",
        "input_fields": ("noisy_image: list\n" "    timestep: int\n" "    class_label: int"),
        "inference_code": textwrap.dedent("""
            import torch
            x = torch.tensor(body.noisy_image, dtype=torch.float32)
            t = torch.tensor([body.timestep])
            c = torch.tensor([body.class_label])
            out = self.model(x, t, c)
            return {"denoised": out.tolist()}
        """).strip(),
    },
    "generic": {
        "description": "Generic FRAMEWORM model",
        "input_fields": "inputs: list  # input tensor as nested list",
        "inference_code": textwrap.dedent("""
            import torch
            x = torch.tensor(body.inputs, dtype=torch.float32)
            out = self.model(x)
            return {"outputs": out.tolist()}
        """).strip(),
    },
}


class ServerBuilder:
    """
    Generates a ready-to-run FastAPI server.py for a FRAMEWORM model.

    Usage:
        builder = ServerBuilder()
        path = builder.build(
            model_type="dcgan",
            model_name="face_gen",
            model_version="v1.0",
            model_path="exports/face_gen.pt",
            output_dir="deploy/generated/face_gen",
        )
    """

    GENERATED_DIR = Path("deploy/generated")

    def build(
        self,
        model_type: str,
        model_name: str,
        model_version: str,
        model_path: str,
        output_dir: Optional[str] = None,
        shift_reference: Optional[str] = None,
        port: int = 8000,
        device: str = "cpu",
    ) -> str:
        """Generate server.py in output_dir. Returns path as string."""
        spec = MODEL_SPECS.get(model_type.lower(), MODEL_SPECS["generic"])
        out_dir = Path(output_dir) if output_dir else self.GENERATED_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        code = self._render(
            spec, model_type, model_name, model_version, model_path, shift_reference, port, device
        )

        out_path = out_dir / "server.py"
        # encoding="utf-8" is required on Windows — cp1252 (the default) cannot
        # encode the Unicode arrow/box characters in comments and docstrings.
        out_path.write_text(code, encoding="utf-8")

        # Also write requirements
        req_path = out_dir / "requirements.txt"
        req_path.write_text(
            "fastapi>=0.100.0\nuvicorn[standard]>=0.23.0\n"
            "torch>=2.0.0\npydantic>=2.0.0\nnumpy>=1.24.0\npsutil>=5.9.0\n",
            encoding="utf-8",
        )

        print(f"[DEPLOY] Server generated -> {out_path}")
        return str(out_path)

    def _render(
        self, spec, model_type, model_name, model_version, model_path, shift_reference, port, device
    ) -> str:
        shift_arg = f'"{shift_reference}"' if shift_reference else "None"
        input_field = spec["input_fields"]
        inference = textwrap.indent(spec["inference_code"], "            ")

        return f'''\
"""
Auto-generated by FRAMEWORM DEPLOY
Model  : {model_name}
Type   : {model_type}
Version: {model_version}
"""

import torch
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.serving.server import FramewormModelServer, build_app


class InferenceRequest(BaseModel):
    """Input schema: {spec["description"]}"""
    {input_field}


class {model_type.upper().replace("-","").replace("_","")}Server(FramewormModelServer):

    def __init__(self):
        super().__init__(
            model_path      = "{model_path}",
            model_name      = "{model_name}",
            model_version   = "{model_version}",
            model_type      = "{model_type}",
            shift_reference = {shift_arg},
            device          = "{device}",
        )
        self.load()

    def predict(self, body: InferenceRequest) -> dict:
        self._check_drift(body.dict())
        with torch.no_grad():
{inference}


_server = {model_type.upper().replace("-","").replace("_","")}Server()
app     = build_app(_server)


@app.post("/predict", summary="Run inference")
async def predict(body: InferenceRequest) -> dict:
    return _server.predict(body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
