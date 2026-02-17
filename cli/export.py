"""
Model export command.
"""

from deployment import ModelExporter
import torch
from pathlib import Path
from click import echo


def export_model(
    checkpoint_path: str,
    export_format: str = "all",
    output_dir: str = "exported",
    quantize: bool = False,
    benchmark: bool = False,
):
    """Export model to deployment formats"""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint.get("model")  # Or reconstruct from state_dict

    if model is None:
        echo("✗ Model not found in checkpoint")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Example input (you'd get this from config)
    example_input = torch.randn(1, 3, 64, 64)

    # Create exporter
    exporter = ModelExporter(model, example_input)

    # Export formats
    formats_to_export = []
    if export_format == "all":
        formats_to_export = ["torchscript", "onnx"]
    else:
        formats_to_export = [export_format]

    for fmt in formats_to_export:
        output_file = output_path / f"model.{fmt.replace('torchscript', 'pt')}"

        if fmt == "torchscript":
            exporter.to_torchscript(str(output_file))
        elif fmt == "onnx":
            exporter.to_onnx(str(output_file))

    # Quantize if requested
    if quantize:
        quant_file = output_path / "model_quant.pt"
        exporter.quantize(str(quant_file))

    # Benchmark if requested
    if benchmark:
        echo("\nBenchmarking...")
        exporter.benchmark_inference()

    echo(f"\n✓ Export complete: {output_path}")
