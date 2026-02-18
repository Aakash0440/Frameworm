"""
Final performance benchmarks for v1.0.0 release.
"""

import time
import torch
from core import Config, get_model, Trainer
from torch.utils.data import DataLoader, TensorDataset


def benchmark_training_speed():
    """Benchmark training throughput"""
    
    print("\n📊 TRAINING SPEED BENCHMARK")
    print("="*60)
    
    config = Config.from_dict({
        'model': {'type': 'vae', 'in_channels': 3, 'latent_dim': 64}
    })
    
    model = get_model('vae')(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, device='cuda')
    
    # Synthetic dataset
    data = TensorDataset(torch.randn(10000, 3, 64, 64))
    loader = DataLoader(data, batch_size=128, num_workers=4)
    
    # Warmup
    for batch in list(loader)[:5]:
        trainer._train_step(batch)
    
    # Benchmark
    start = time.time()
    for batch in loader:
        trainer._train_step(batch)
    elapsed = time.time() - start
    
    throughput = len(data) / elapsed
    
    print(f"  Dataset size: {len(data)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print("  ✅ Training speed benchmark complete")
    
    return throughput


def benchmark_inference_latency():
    """Benchmark inference latency"""
    
    print("\n⚡ INFERENCE LATENCY BENCHMARK")
    print("="*60)
    
    config = Config.from_dict({
        'model': {'type': 'vae', 'in_channels': 3, 'latent_dim': 64}
    })
    
    model = get_model('vae')(config).cuda().eval()
    
    batch_sizes = [1, 8, 16, 32, 64]
    results = {}
    
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 64, 64).cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        latency_ms = (elapsed / 100) * 1000
        throughput = (bs * 100) / elapsed
        
        results[bs] = {'latency_ms': latency_ms, 'throughput': throughput}
        
        print(f"  Batch size {bs:2d}: {latency_ms:6.2f}ms/batch, {throughput:7.0f} samples/sec")
    
    print("  ✅ Inference latency benchmark complete")
    
    return results


def benchmark_memory_usage():
    """Benchmark GPU memory usage"""
    
    print("\n💾 MEMORY USAGE BENCHMARK")
    print("="*60)
    
    torch.cuda.reset_peak_memory_stats()
    
    config = Config.from_dict({
        'model': {'type': 'vae', 'in_channels': 3, 'latent_dim': 128}
    })
    
    model = get_model('vae')(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training step
    x = torch.randn(64, 3, 64, 64).cuda()
    output = model.compute_loss(x)
    loss = output['loss']
    loss.backward()
    optimizer.step()
    
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Peak memory: {memory_mb:.1f} MB")
    print(f"  Batch size: 64")
    print("  ✅ Memory benchmark complete")
    
    return memory_mb


def main():
    """Run all benchmarks"""
    
    print("\n" + "="*60)
    print("FRAMEWORM v1.0.0 - FINAL PERFORMANCE BENCHMARKS")
    print("="*60)
    
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True)
    gpu_name = result.stdout.strip()
    print(f"\nGPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    train_throughput = benchmark_training_speed()
    inference_results = benchmark_inference_latency()
    memory_mb = benchmark_memory_usage()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Training: {train_throughput:.0f} samples/sec")
    print(f"✅ Inference (batch=1): {inference_results[1]['latency_ms']:.2f}ms")
    print(f"✅ Inference (batch=64): {inference_results[64]['throughput']:.0f} samples/sec")
    print(f"✅ Peak memory: {memory_mb:.1f} MB")
    print("="*60)
    
    # Save results
    import json
    results = {
        'gpu': gpu_name,
        'pytorch_version': torch.__version__,
        'training_throughput': train_throughput,
        'inference': {k: v for k, v in inference_results.items()},
        'memory_mb': memory_mb
    }
    
    with open('benchmarks/results_v1.0.0.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Benchmark results saved to benchmarks/results_v1.0.0.json")


if __name__ == '__main__':
    main()