"""
Exports FRAMEWORM model checkpoints to TorchScript and ONNX.
Applies quantization for size/speed optimisation.
Model-aware: knows FRAMEWORM's 6 architectures and picks correct export strategy.
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("frameworm.deploy")

# FRAMEWORM model architectures and their input signatures
# Maps model class name → (input_shape, dynamic_axes, export_notes)
FRAMEWORM_MODEL_SIGNATURES: Dict[str, dict] = {
    "VAE": {
        "input_shape": (1, 3, 64, 64),
        "input_names": ["image"],
        "output_names": ["reconstruction", "mu", "logvar"],
        "dynamic_axes": {"image": {0: "batch_size"}, "reconstruction": {0: "batch_size"}},
        "export_notes": "Exports encoder+decoder as single graph",
    },
    "DCGAN": {
        "input_shape": (1, 100, 1, 1),
        "input_names": ["noise"],
        "output_names": ["generated_image"],
        "dynamic_axes": {"noise": {0: "batch_size"}, "generated_image": {0: "batch_size"}},
        "export_notes": "Generator only — discriminator not exported",
    },
    "DDPM": {
        "input_shape": (1, 3, 32, 32),
        "input_names": ["noisy_image", "timestep"],
        "output_names": ["denoised"],
        "dynamic_axes": {"noisy_image": {0: "batch_size"}},
        "export_notes": "Denoising network only, not full diffusion loop",
    },
    "VQVAE2": {
        "input_shape": (1, 3, 256, 256),
        "input_names": ["image"],
        "output_names": ["reconstruction", "quantized"],
        "dynamic_axes": {"image": {0: "batch_size"}},
        "export_notes": "Full encoder-quantizer-decoder graph",
    },
    "ViTGAN": {
        "input_shape": (1, 100),
        "input_names": ["latent"],
        "output_names": ["generated_image"],
        "dynamic_axes": {"latent": {0: "batch_size"}},
        "export_notes": "Generator with Vision Transformer backbone",
    },
    "CFG_DDPM": {
        "input_shape": (1, 3, 32, 32),
        "input_names": ["noisy_image", "timestep", "class_label"],
        "output_names": ["denoised"],
        "dynamic_axes": {"noisy_image": {0: "batch_size"}},
        "export_notes": "Classifier-free guidance denoiser",
    },
}


@dataclass
class ExportManifest:
    """Records everything about an exported model."""

    model_name: str
    model_class: str
    checkpoint_path: str
    checkpoint_hash: str
    export_dir: str
    torchscript_path: Optional[str]
    onnx_path: Optional[str]
    quantized_path: Optional[str]
    input_shape: list
    input_names: list
    output_names: list
    export_formats: list
    quantized: bool
    exported_at: str
    frameworm_version: str = "1.0.0"
    metadata: dict = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExportManifest":
        with open(path) as f:
            return cls(**json.load(f))


class ModelExporter:
    """
    Exports FRAMEWORM checkpoints to TorchScript and ONNX.
    Model-aware: auto-detects architecture and uses correct input signature.

    Usage:
        exporter = ModelExporter()
        manifest = exporter.export(
            checkpoint_path="experiments/checkpoints/dcgan_best.pt",
            model_name="dcgan_v1",
            formats=["torchscript", "onnx"],
            quantize=False,
        )
    """

    def __init__(self, exports_dir: Optional[str] = None):
        self.exports_dir = Path(exports_dir or "experiments/deploy_exports")
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────── public

    def export(
        self,
        checkpoint_path: str,
        model_name: str,
        formats: Optional[list] = None,
        quantize: bool = False,
        metadata: Optional[dict] = None,
    ) -> ExportManifest:
        """
        Load checkpoint, detect architecture, export to requested formats.

        Args:
            checkpoint_path: path to .pt checkpoint file
            model_name:      deployment name (e.g. "dcgan_v1")
            formats:         ["torchscript", "onnx"] (default: both)
            quantize:        apply int8 quantization (4x smaller, 2-3x faster)
            metadata:        extra info embedded in manifest

        Returns:
            ExportManifest with paths to all exported files
        """
        import torch

        formats = formats or ["torchscript", "onnx"]
        checkpoint_path = str(checkpoint_path)

        print(f"\n[DEPLOY] Exporting '{model_name}' from {checkpoint_path}")

        # ── load checkpoint ──
        checkpoint = self._load_checkpoint(checkpoint_path)
        model, model_class = self._load_model(checkpoint)
        model.eval()

        print(f"[DEPLOY] Detected architecture: {model_class}")

        # ── get input signature ──
        sig = self._get_signature(model_class)
        dummy_inputs = self._make_dummy_inputs(sig, model_class)

        # ── create export directory ──
        export_dir = self.exports_dir / model_name
        export_dir.mkdir(parents=True, exist_ok=True)

        ts_path = None
        onnx_path = None
        quant_path = None

        # ── TorchScript export ──
        if "torchscript" in formats:
            ts_path = str(export_dir / f"{model_name}.pt")
            self._export_torchscript(model, dummy_inputs, ts_path, model_class)

        # ── ONNX export ──
        if "onnx" in formats:
            onnx_path = str(export_dir / f"{model_name}.onnx")
            self._export_onnx(model, dummy_inputs, onnx_path, sig, model_class)

        # ── Quantization ──
        if quantize and ts_path:
            quant_path = str(export_dir / f"{model_name}_quantized.pt")
            self._quantize(ts_path, quant_path)

        # ── Manifest ──
        manifest = ExportManifest(
            model_name=model_name,
            model_class=model_class,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=self._file_hash(checkpoint_path),
            export_dir=str(export_dir),
            torchscript_path=ts_path,
            onnx_path=onnx_path,
            quantized_path=quant_path,
            input_shape=list(sig["input_shape"]),
            input_names=sig["input_names"],
            output_names=sig["output_names"],
            export_formats=formats,
            quantized=quantize,
            exported_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        manifest_path = str(export_dir / "manifest.json")
        manifest.save(manifest_path)

        print(f"[DEPLOY] Export complete → {export_dir}")
        if ts_path:
            print(f"         TorchScript: {ts_path}")
        if onnx_path:
            print(f"         ONNX:        {onnx_path}")
        if quant_path:
            print(f"         Quantized:   {quant_path}")

        return manifest

    # ──────────────────────────────────────────────── export methods

    def _export_torchscript(self, model, dummy_inputs, out_path, model_class):
        import torch

        print(f"[DEPLOY] Tracing TorchScript...")
        try:
            with torch.no_grad():
                if isinstance(dummy_inputs, tuple):
                    traced = torch.jit.trace(model, dummy_inputs)
                else:
                    traced = torch.jit.trace(model, dummy_inputs)
            traced.save(out_path)
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"[DEPLOY] TorchScript saved ({size_mb:.1f} MB)")
        except Exception as e:
            logger.warning(f"[DEPLOY] TorchScript trace failed, trying script: {e}")
            try:
                scripted = torch.jit.script(model)
                scripted.save(out_path)
            except Exception as e2:
                logger.error(f"[DEPLOY] TorchScript export failed: {e2}")
                raise

    def _export_onnx(self, model, dummy_inputs, out_path, sig, model_class):
        import torch

        print(f"[DEPLOY] Exporting ONNX...")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_inputs,
                    out_path,
                    input_names=sig["input_names"],
                    output_names=sig["output_names"],
                    dynamic_axes=sig.get("dynamic_axes", {}),
                    opset_version=17,
                    do_constant_folding=True,
                )
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"[DEPLOY] ONNX saved ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"[DEPLOY] ONNX export failed: {e}")
            raise

    def _quantize(self, ts_path: str, quant_path: str):
        import torch

        print(f"[DEPLOY] Applying int8 quantization...")
        try:
            model = torch.jit.load(ts_path)
            quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            torch.jit.save(torch.jit.script(quantized), quant_path)
            orig_mb = os.path.getsize(ts_path) / 1024 / 1024
            quant_mb = os.path.getsize(quant_path) / 1024 / 1024
            ratio = orig_mb / quant_mb if quant_mb > 0 else 1
            print(
                f"[DEPLOY] Quantized: {orig_mb:.1f}MB → {quant_mb:.1f}MB "
                f"({ratio:.1f}x reduction)"
            )
        except Exception as e:
            logger.warning(f"[DEPLOY] Quantization failed (non-fatal): {e}")

    # ──────────────────────────────────────────────── helpers

    def _load_checkpoint(self, path: str) -> dict:
        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"[DEPLOY] Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint

    def _load_model(self, checkpoint) -> Tuple[Any, str]:
        """
        Loads model from checkpoint.
        Tries FRAMEWORM's existing model registry first.
        Falls back to direct model_state_dict loading.
        """
        import torch

        # ── try FRAMEWORM plugin registry ──
        model_class = "unknown"
        if isinstance(checkpoint, dict):
            model_class = checkpoint.get(
                "model_class", checkpoint.get("arch", checkpoint.get("model_type", "unknown"))
            )

        try:
            from models import get_model

            model = get_model(model_class)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            return model, model_class
        except (ImportError, KeyError, Exception) as e:
            logger.warning(
                f"[DEPLOY] Could not load via registry ({e}), " f"loading raw state dict"
            )

        # ── fallback: return checkpoint model directly ──
        if hasattr(checkpoint, "eval"):
            return checkpoint, checkpoint.__class__.__name__

        raise ValueError(
            "[DEPLOY] Could not load model from checkpoint. "
            "Ensure checkpoint contains 'model_class' and 'model_state_dict' keys."
        )

    def _get_signature(self, model_class: str) -> dict:
        """Get input/output signature for a FRAMEWORM model class."""
        sig = FRAMEWORM_MODEL_SIGNATURES.get(model_class)
        if sig is None:
            logger.warning(
                f"[DEPLOY] Unknown model class '{model_class}'. "
                f"Using generic signature. Override with custom_signature param."
            )
            sig = {
                "input_shape": (1, 3, 64, 64),
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamic_axes": {"input": {0: "batch_size"}},
                "export_notes": "Generic signature",
            }
        return sig

    def _make_dummy_inputs(self, sig: dict, model_class: str):
        import torch

        shape = sig["input_shape"]
        # Multi-input models (DDPM needs noise + timestep, CFG_DDPM needs class_label)
        if model_class in ("DDPM", "CFG_DDPM"):
            noise = torch.randn(*shape)
            timestep = torch.randint(0, 1000, (shape[0],))
            if model_class == "CFG_DDPM":
                class_label = torch.randint(0, 10, (shape[0],))
                return (noise, timestep, class_label)
            return (noise, timestep)
        return torch.randn(*shape)

    def _file_hash(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
