"""
Performance profiling utilities.
"""

import time
import torch
import cProfile
import pstats
import io
from typing import Optional, Dict, Any, Callable, List
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np


class Timer:
    """Simple high-precision timer"""

    def __init__(self, name: str = ""):
        self.name = name
        self.times: List[float] = []

    @contextmanager
    def __call__(self):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.times.append(end - start)

    @property
    def mean_ms(self):
        return np.mean(self.times) * 1000 if self.times else 0.0

    @property
    def std_ms(self):
        return np.std(self.times) * 1000 if self.times else 0.0

    @property
    def total_s(self):
        return sum(self.times)

    def __repr__(self):
        return f"Timer({self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms)"


class TrainingProfiler:
    """
    Profile training loop components.

    Measures time spent in each phase of training:
    data loading, forward pass, loss computation,
    backward pass, optimizer step, and logging.

    Example:
        >>> profiler = TrainingProfiler()
        >>> with profiler.profile_epoch():
        ...     for batch in loader:
        ...         with profiler.data_loading:
        ...             inputs, targets = batch
        ...         with profiler.forward_pass:
        ...             output = model(inputs)
        ...         with profiler.backward_pass:
        ...             loss.backward()
        ...         with profiler.optimizer_step:
        ...             optimizer.step()
        >>> profiler.print_summary()
    """

    def __init__(self):
        self.data_loading = Timer("data_loading")
        self.forward_pass = Timer("forward_pass")
        self.loss_computation = Timer("loss_computation")
        self.backward_pass = Timer("backward_pass")
        self.optimizer_step = Timer("optimizer_step")
        self.logging = Timer("logging")
        self.epoch_timer = Timer("epoch")

        self.epochs: List[Dict] = []

    @contextmanager
    def profile_epoch(self):
        """Context manager for profiling an entire epoch"""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        # Record epoch stats
        epoch_stats = {
            "duration_s": elapsed,
            "data_loading_ms": self.data_loading.mean_ms,
            "forward_pass_ms": self.forward_pass.mean_ms,
            "backward_pass_ms": self.backward_pass.mean_ms,
            "optimizer_step_ms": self.optimizer_step.mean_ms,
        }
        self.epochs.append(epoch_stats)

    def print_summary(self, top_n: int = 5):
        """Print profiling summary"""
        print("\n" + "=" * 70)
        print("TRAINING PROFILER SUMMARY")
        print("=" * 70)

        timers = [
            self.data_loading,
            self.forward_pass,
            self.loss_computation,
            self.backward_pass,
            self.optimizer_step,
            self.logging,
        ]

        total_ms = sum(t.mean_ms for t in timers if t.times)

        print(f"\n{'Phase':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'% Total':<10}")
        print("-" * 70)

        for timer in sorted(timers, key=lambda t: t.mean_ms, reverse=True):
            if not timer.times:
                continue
            pct = (timer.mean_ms / total_ms * 100) if total_ms > 0 else 0
            bar = "█" * int(pct / 5)
            print(
                f"{timer.name:<25} {timer.mean_ms:<12.2f} {timer.std_ms:<12.2f} {pct:<10.1f} {bar}"
            )

        print("-" * 70)
        print(f"{'Total (measured)':<25} {total_ms:<12.2f}")

        # Bottleneck identification
        bottleneck = max(timers, key=lambda t: t.mean_ms if t.times else 0)
        if bottleneck.times:
            pct = bottleneck.mean_ms / total_ms * 100
            print(f"\n⚠️  Bottleneck: {bottleneck.name} ({pct:.1f}% of time)")
            self._print_recommendations(bottleneck.name)

        print("=" * 70)

    def _print_recommendations(self, bottleneck: str):
        """Print optimization recommendations"""
        recommendations = {
            "data_loading": [
                "Increase num_workers in DataLoader",
                "Enable pin_memory=True in DataLoader",
                "Use prefetch_factor parameter",
                "Consider caching dataset in RAM",
            ],
            "forward_pass": [
                "Enable mixed precision (torch.cuda.amp)",
                "Use torch.jit.script for model",
                "Profile model architecture",
                "Check for unnecessary operations",
            ],
            "backward_pass": [
                "Enable mixed precision",
                "Use gradient checkpointing for large models",
                "Consider gradient accumulation",
            ],
            "optimizer_step": [
                "Try faster optimizer (AdamW vs Adam)",
                "Enable fused optimizer (if available)",
                "Reduce number of parameter groups",
            ],
        }

        recs = recommendations.get(bottleneck, [])
        if recs:
            print("\n💡 Recommendations:")
            for rec in recs:
                print(f"   • {rec}")

    def save_report(self, path: str):
        """Save profiling report to JSON"""
        report = {
            "timers": {
                "data_loading": {
                    "mean_ms": self.data_loading.mean_ms,
                    "std_ms": self.data_loading.std_ms,
                    "samples": len(self.data_loading.times),
                },
                "forward_pass": {
                    "mean_ms": self.forward_pass.mean_ms,
                    "std_ms": self.forward_pass.std_ms,
                    "samples": len(self.forward_pass.times),
                },
                "backward_pass": {
                    "mean_ms": self.backward_pass.mean_ms,
                    "std_ms": self.backward_pass.std_ms,
                    "samples": len(self.backward_pass.times),
                },
                "optimizer_step": {
                    "mean_ms": self.optimizer_step.mean_ms,
                    "std_ms": self.optimizer_step.std_ms,
                    "samples": len(self.optimizer_step.times),
                },
            },
            "epochs": self.epochs,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ Profile report saved: {path}")


class InferenceProfiler:
    """
    Profile model inference performance.

    Measures throughput, latency, and GPU utilization.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def benchmark(
        self,
        example_input: torch.Tensor,
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark inference across different batch sizes.

        Args:
            example_input: Single example input
            batch_sizes: Batch sizes to test
            num_runs: Measurement runs per batch size
            warmup_runs: Warmup runs

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        print("\nInference Benchmark:")
        print("=" * 60)
        print(f"{'Batch Size':<12} {'Latency (ms)':<16} {'Throughput':<16} {'GPU Mem (MB)'}")
        print("-" * 60)

        for batch_size in batch_sizes:
            # Create batch
            batch = example_input.repeat(batch_size, *([1] * (example_input.dim() - 1)))
            batch = batch.to(self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(batch)

            # Synchronize
            if self.device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.model(batch)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)

            mean_time_ms = np.mean(times) * 1000
            throughput = batch_size / (np.mean(times))

            # Memory
            if self.device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                gpu_mem = 0

            results[batch_size] = {
                "latency_ms": mean_time_ms,
                "throughput": throughput,
                "gpu_mem_mb": gpu_mem,
            }

            print(f"{batch_size:<12} {mean_time_ms:<16.2f} {throughput:<16.0f} {gpu_mem:<.0f}")

        print("=" * 60)

        # Find optimal batch size
        optimal = max(results.items(), key=lambda x: x[1]["throughput"])
        print(f"\n✓ Optimal batch size: {optimal[0]} ({optimal[1]['throughput']:.0f} samples/s)")

        return results

    def profile_with_pytorch(self, example_input: torch.Tensor):
        """Profile using PyTorch profiler"""
        batch = example_input.to(self.device)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                for _ in range(10):
                    output = self.model(batch)

        # Print top operations
        print("\nTop 10 Operations by CUDA Time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        return prof


class DataLoaderProfiler:
    """Profile DataLoader performance"""

    def benchmark(self, loader, num_batches: int = 50) -> Dict[str, float]:
        """
        Benchmark DataLoader speed.

        Measures:
        - Time to load first batch
        - Average batch loading time
        - Throughput (samples/sec)
        """
        print(f"\nDataLoader Benchmark ({num_batches} batches)...")

        times = []
        first_batch_time = None
        total_samples = 0

        start = time.perf_counter()

        for i, batch in enumerate(loader):
            batch_time = time.perf_counter() - start

            if i == 0:
                first_batch_time = batch_time

            # Get batch size
            if isinstance(batch, (tuple, list)):
                batch_size = len(batch[0])
            else:
                batch_size = len(batch)

            total_samples += batch_size
            times.append(batch_time)

            start = time.perf_counter()

            if i >= num_batches:
                break

        avg_time = np.mean(times) * 1000
        throughput = total_samples / sum(times)

        results = {
            "first_batch_ms": first_batch_time * 1000,
            "avg_batch_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_samples": total_samples,
        }

        print(f"  First batch: {results['first_batch_ms']:.2f} ms")
        print(f"  Avg batch: {results['avg_batch_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")

        return results
