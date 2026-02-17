"""
Model export utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any
from pathlib import Path
import warnings


class ModelExporter:
    """
    Export PyTorch models to various formats.

    Supports TorchScript, ONNX, and quantization.

    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for tracing/export

    Example:
        >>> exporter = ModelExporter(model, example_inputs)
        >>> exporter.to_torchscript('model.pt')
        >>> exporter.to_onnx('model.onnx')
        >>> exporter.quantize('model_quant.pt')
    """

    def __init__(self, model: nn.Module, example_inputs: Optional[torch.Tensor] = None):
        self.model = model
        self.example_inputs = example_inputs

        # Ensure model is in eval mode
        self.model.eval()

    def to_torchscript(
        self, save_path: str, method: str = "trace", optimize: bool = True, strict: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Export model to TorchScript.

        Args:
            save_path: Path to save .pt file
            method: 'trace' or 'script'
                - trace: Traces execution with example inputs (recommended)
                - script: Direct Python→TorchScript compilation
            optimize: Apply optimization passes
            strict: Strict type checking

        Returns:
            TorchScript module

        Example:
            >>> exporter = ModelExporter(model, example_inputs)
            >>> traced = exporter.to_torchscript('model.pt', method='trace')
        """
        if method == "trace":
            if self.example_inputs is None:
                raise ValueError("example_inputs required for tracing")

            print(f"Tracing model with input shape: {self.example_inputs.shape}")

            # Trace model
            with torch.no_grad():
                traced = torch.jit.trace(
                    self.model, self.example_inputs, strict=strict, check_trace=True
                )

        elif method == "script":
            print("Scripting model (no example inputs needed)")

            # Script model
            traced = torch.jit.script(self.model)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

        # Optimize
        if optimize:
            print("Optimizing TorchScript...")
            traced = torch.jit.optimize_for_inference(traced)

        # Save
        traced.save(save_path)

        # Get file size
        size_mb = Path(save_path).stat().st_size / (1024 * 1024)

        print(f"✓ TorchScript saved: {save_path} ({size_mb:.2f} MB)")

        return traced

    def to_onnx(
        self,
        save_path: str,
        opset_version: int = 14,
        dynamic_axes: bool = True,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        verbose: bool = False,
    ):
        """
        Export model to ONNX format.

        Args:
            save_path: Path to save .onnx file
            opset_version: ONNX opset version (11-17)
            dynamic_axes: Enable dynamic batch size
            input_names: Names for inputs
            output_names: Names for outputs
            verbose: Print detailed export info

        Example:
            >>> exporter.to_onnx('model.onnx', opset_version=14)
        """
        if self.example_inputs is None:
            raise ValueError("example_inputs required for ONNX export")

        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Dynamic axes for variable batch size
        if dynamic_axes:
            dynamic_axes_dict = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            }
        else:
            dynamic_axes_dict = None

        print(f"Exporting to ONNX (opset {opset_version})...")

        # Export
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            torch.onnx.export(
                self.model,
                self.example_inputs,
                save_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                do_constant_folding=True,
                verbose=verbose,
            )

        # Verify
        try:
            import onnx

            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed")
        except ImportError:
            print("⚠️  onnx not installed, skipping validation")
        except Exception as e:
            print(f"⚠️  ONNX validation warning: {e}")

        # Get file size
        size_mb = Path(save_path).stat().st_size / (1024 * 1024)

        print(f"✓ ONNX saved: {save_path} ({size_mb:.2f} MB)")

    def quantize(
        self, save_path: str, method: str = "dynamic", dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Quantize model for faster inference.

        Reduces model size by ~4x and speeds up CPU inference by 2-4x.

        Args:
            save_path: Path to save quantized model
            method: 'dynamic' or 'static'
                - dynamic: Post-training dynamic quantization (no calibration)
                - static: Static quantization (requires calibration data)
            dtype: Quantization dtype (qint8 or float16)

        Returns:
            Quantized model

        Example:
            >>> quantized = exporter.quantize('model_quant.pt', method='dynamic')
        """
        print(f"Quantizing model ({method})...")

        if method == "dynamic":
            # Dynamic quantization (easiest, no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU}, dtype=dtype
            )

        elif method == "static":
            # Static quantization (requires calibration)
            print("⚠️  Static quantization requires calibration data")

            # Prepare for quantization
            quantized_model = self.model
            quantized_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            # Prepare
            torch.quantization.prepare(quantized_model, inplace=True)

            # TODO: Run calibration with representative data
            # with torch.no_grad():
            #     for batch in calibration_loader:
            #         quantized_model(batch)

            # Convert
            torch.quantization.convert(quantized_model, inplace=True)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Save
        torch.save(quantized_model.state_dict(), save_path)

        # Compare sizes
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = Path(save_path).stat().st_size

        reduction = (1 - quantized_size / original_size) * 100

        print(f"✓ Quantized model saved: {save_path}")
        print(f"  Size reduction: {reduction:.1f}%")
        print(f"  Original: {original_size / (1024**2):.2f} MB")
        print(f"  Quantized: {quantized_size / (1024**2):.2f} MB")

        return quantized_model

    def benchmark_inference(self, num_runs: int = 100, warmup_runs: int = 10):
        """
        Benchmark inference speed.

        Args:
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
        """
        if self.example_inputs is None:
            raise ValueError("example_inputs required for benchmarking")

        import time

        print(f"Benchmarking inference ({num_runs} runs)...")

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(self.example_inputs)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(self.example_inputs)
                end = time.perf_counter()
                times.append(end - start)

        # Stats
        import numpy as np

        mean_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        throughput = 1000 / mean_time  # inferences/sec

        print(f"\nInference Benchmark:")
        print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} inferences/sec")
