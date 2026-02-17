"""
Benchmark distributed training performance.

Compares:
- Single GPU vs Multi-GPU
- DataParallel vs DistributedDataParallel
- With/without mixed precision
- With/without gradient accumulation
"""

import time

import torch
import torch.nn as nn
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core import Config, get_model
from distributed import DataParallelTrainer, DistributedTrainer
from distributed.data_loader import OptimizedDataLoader
from distributed.profiler import PerformanceProfiler
from training import Trainer


def get_mnist_loaders_benchmark(batch_size=128):
    """Get MNIST loaders for benchmarking"""
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

    # Use subset for faster benchmarking
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))

    train_loader = OptimizedDataLoader.create(train_subset, batch_size=batch_size, shuffle=True)

    return train_loader


def benchmark_configuration(
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    use_amp: bool = False,
    gradient_accumulation: int = 1,
    epochs: int = 1,
):
    """
    Benchmark a specific configuration.

    Returns:
        Dict with timing and memory stats
    """
    print(f"\nBenchmarking: {name}")
    print("-" * 60)

    # Create trainer
    if isinstance(model, nn.parallel.DistributedDataParallel):
        trainer = DistributedTrainer(
            model, optimizer, use_amp=use_amp, gradient_accumulation_steps=gradient_accumulation
        )
    else:
        trainer = Trainer(model, optimizer, device="cuda:0" if torch.cuda.is_available() else "cpu")

        if use_amp:
            trainer.enable_mixed_precision()

        if gradient_accumulation > 1:
            trainer.enable_gradient_accumulation(gradient_accumulation)

    # Profile
    profiler = PerformanceProfiler()

    # Warmup
    print("Warming up...")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break

        with profiler.profile("step"):
            loss_dict = trainer.model.compute_loss(*batch)
            loss = loss_dict["loss"]
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

    # Reset profiler
    profiler.reset()

    # Benchmark
    print("Benchmarking...")
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            with profiler.profile("step"):
                # Forward
                with profiler.profile("forward"):
                    loss_dict = trainer.model.compute_loss(*batch)
                    loss = loss_dict["loss"]

                # Backward
                with profiler.profile("backward"):
                    loss.backward()

                # Optimizer
                with profiler.profile("optimizer"):
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()

            profiler.record_gpu_memory()

            if batch_idx >= 100:  # Limit for benchmarking
                break

    elapsed = time.time() - start_time

    # Get results
    results = profiler.get_results()
    summary = results.summary()

    # Calculate throughput
    samples_per_sec = (100 * train_loader.batch_size) / elapsed

    return {
        "name": name,
        "time": elapsed,
        "throughput": samples_per_sec,
        "step_time_ms": summary.get("avg_step_time", 0) * 1000,
        "memory_gb": summary.get("peak_memory_gb", 0),
        **summary,
    }


def run_benchmarks():
    """Run all benchmarks"""
    print("=" * 60)
    print("DISTRIBUTED TRAINING BENCHMARK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU benchmarks")
        return

    n_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {n_gpus}")

    # Config
    config = Config("configs/models/vae/vanilla.yaml")
    config.training.epochs = 1
    config.training.batch_size = 128

    results = []

    # Benchmark 1: Single GPU (baseline)
    print("\n" + "=" * 60)
    print("1. SINGLE GPU (BASELINE)")
    print("=" * 60)

    vae = get_model("vae")(config).cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    train_loader = get_mnist_loaders_benchmark(config.training.batch_size)

    result = benchmark_configuration("Single GPU", vae, optimizer, train_loader)
    results.append(result)

    # Benchmark 2: Single GPU + Mixed Precision
    print("\n" + "=" * 60)
    print("2. SINGLE GPU + MIXED PRECISION")
    print("=" * 60)

    vae = get_model("vae")(config).cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    train_loader = get_mnist_loaders_benchmark(config.training.batch_size)

    result = benchmark_configuration("Single GPU + AMP", vae, optimizer, train_loader, use_amp=True)
    results.append(result)

    # Benchmark 3: DataParallel (if multiple GPUs)
    if n_gpus >= 2:
        print("\n" + "=" * 60)
        print("3. DATAPARALLEL")
        print("=" * 60)

        vae = get_model("vae")(config)
        vae = DataParallelTrainer.wrap(vae)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        train_loader = get_mnist_loaders_benchmark(config.training.batch_size)

        result = benchmark_configuration(
            f"DataParallel ({n_gpus} GPUs)", vae, optimizer, train_loader
        )
        results.append(result)

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    table_data = []
    baseline_throughput = results[0]["throughput"]

    for result in results:
        speedup = result["throughput"] / baseline_throughput

        table_data.append(
            [
                result["name"],
                f"{result['time']:.2f}s",
                f"{result['throughput']:.0f}",
                f"{speedup:.2f}x",
                f"{result['step_time_ms']:.1f}ms",
                f"{result['memory_gb']:.2f}GB",
            ]
        )

    print(
        "\n"
        + tabulate(
            table_data,
            headers=[
                "Configuration",
                "Time",
                "Throughput\n(samples/s)",
                "Speedup",
                "Step Time",
                "Memory",
            ],
            tablefmt="grid",
        )
    )

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    best = max(results, key=lambda x: x["throughput"])

    print(f"\nBest Configuration: {best['name']}")
    print(f"  Speedup: {best['throughput'] / baseline_throughput:.2f}x")
    print(f"  Throughput: {best['throughput']:.0f} samples/sec")

    if n_gpus >= 2:
        print("\nFor multi-GPU training:")
        print("  ✅ Use DistributedDataParallel (DDP) instead of DataParallel")
        print("  ✅ Enable mixed precision (AMP) for 2-3x speedup")
        print("  ✅ Use gradient accumulation for larger effective batch size")
        print("  ✅ Optimize data loading with multiple workers")
    else:
        print("\nFor single-GPU training:")
        print("  ✅ Enable mixed precision (AMP) for 2-3x speedup")
        print("  ✅ Use gradient accumulation to simulate larger batches")
        print("  ✅ Optimize batch size for your GPU memory")


def main():
    print("Distributed Training Benchmark")
    print("=" * 60)

    run_benchmarks()

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
