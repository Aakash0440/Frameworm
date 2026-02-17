"""
Complete performance benchmarks for FRAMEWORM.
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from utils.profiler import TrainingProfiler, InferenceProfiler


def benchmark_training_speed():
    """Benchmark training across configurations"""
    print("\n" + "=" * 60)
    print("TRAINING SPEED BENCHMARK")
    print("=" * 60)

    results = {}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

        def compute_loss(self, x, y):
            return {"loss": nn.CrossEntropyLoss()(self.forward(x), y)}

    dataset = TensorDataset(torch.randn(1000, 3, 64, 64), torch.randint(0, 10, (1000,)))
    loader = DataLoader(dataset, batch_size=64, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    configs = [
        {"name": "Baseline", "mixed_precision": False, "grad_accum": 1},
        {"name": "Mixed Precision", "mixed_precision": True, "grad_accum": 1},
        {"name": "Grad Accum (4x)", "mixed_precision": False, "grad_accum": 4},
    ]

    if torch.cuda.is_available():
        configs.append({"name": "MP + Grad Accum", "mixed_precision": True, "grad_accum": 4})

    for cfg in configs:
        from training import Trainer

        model = Model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, device=device)

        if cfg["mixed_precision"] and device == "cuda":
            trainer.enable_mixed_precision()
        if cfg["grad_accum"] > 1:
            trainer.enable_gradient_accumulation(cfg["grad_accum"])

        # Warmup
        trainer.train(loader, epochs=1)

        # Measure
        start = time.perf_counter()
        trainer.train(loader, epochs=3)
        elapsed = time.perf_counter() - start

        samples_per_sec = (1000 * 3) / elapsed

        results[cfg["name"]] = {"time_s": elapsed, "samples_per_sec": samples_per_sec}

        print(f"{cfg['name']:<25} {elapsed:.2f}s  {samples_per_sec:.0f} samples/s")

    return results


def benchmark_inference():
    """Benchmark inference across formats"""
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel()
    example = torch.randn(1, 512)

    from deployment import ModelExporter

    exporter = ModelExporter(model, example)

    # TorchScript
    traced = exporter.to_torchscript("/tmp/bench_model.pt")

    results = {}

    def bench(fn, name, n=500):
        # Warmup
        for _ in range(50):
            fn(example)

        times = []
        for _ in range(n):
            start = time.perf_counter()
            fn(example)
            times.append(time.perf_counter() - start)

        import numpy as np

        mean_ms = np.mean(times) * 1000
        results[name] = mean_ms
        print(f"{name:<30} {mean_ms:.3f} ms")

    with torch.no_grad():
        bench(model, "PyTorch (eager)")
        bench(traced, "TorchScript")

    return results


def run_all_benchmarks():
    """Run all benchmarks and save report"""
    print("FRAMEWORM PERFORMANCE BENCHMARKS")
    print("=" * 60)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "benchmarks": {},
    }

    report["benchmarks"]["training"] = benchmark_training_speed()
    report["benchmarks"]["inference"] = benchmark_inference()

    # Save report
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Benchmark report saved: benchmark_report.json")
    return report


if __name__ == "__main__":
    run_all_benchmarks()
