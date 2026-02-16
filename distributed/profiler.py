"""
Performance profiling for distributed training.
"""

import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class ProfileResults:
    """Results from profiling"""
    step_times: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    data_loading_times: List[float] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    backward_times: List[float] = field(default_factory=list)
    optimizer_times: List[float] = field(default_factory=list)
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        summary = {}
        
        if self.step_times:
            summary['avg_step_time'] = np.mean(self.step_times)
            summary['throughput'] = 1.0 / np.mean(self.step_times)  # steps/sec
        
        if self.gpu_memory:
            summary['peak_memory_gb'] = np.max(self.gpu_memory)
            summary['avg_memory_gb'] = np.mean(self.gpu_memory)
        
        if self.data_loading_times:
            summary['avg_data_time'] = np.mean(self.data_loading_times)
            summary['data_percent'] = np.mean(self.data_loading_times) / np.mean(self.step_times) * 100
        
        if self.forward_times:
            summary['avg_forward_time'] = np.mean(self.forward_times)
        
        if self.backward_times:
            summary['avg_backward_time'] = np.mean(self.backward_times)
        
        if self.optimizer_times:
            summary['avg_optimizer_time'] = np.mean(self.optimizer_times)
        
        return summary


class PerformanceProfiler:
    """
    Profile distributed training performance.
    
    Tracks:
    - Step time
    - GPU memory
    - Data loading time
    - Forward/backward/optimizer time
    - GPU utilization
    
    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile('step'):
        ...     # Training step
        ...     pass
        >>> results = profiler.get_results()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.results = ProfileResults()
        self._timers = {}
        self._start_times = {}
    
    def start(self, name: str):
        """Start timing a section"""
        if not self.enabled:
            return
        
        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._start_times[name] = time.time()
    
    def end(self, name: str):
        """End timing a section"""
        if not self.enabled:
            return
        
        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name not in self._start_times:
            return
        
        elapsed = time.time() - self._start_times[name]
        
        # Store result
        if name == 'step':
            self.results.step_times.append(elapsed)
        elif name == 'data':
            self.results.data_loading_times.append(elapsed)
        elif name == 'forward':
            self.results.forward_times.append(elapsed)
        elif name == 'backward':
            self.results.backward_times.append(elapsed)
        elif name == 'optimizer':
            self.results.optimizer_times.append(elapsed)
        
        del self._start_times[name]
    
    def record_gpu_memory(self):
        """Record current GPU memory usage"""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        self.results.gpu_memory.append(memory_gb)
    
    def profile(self, name: str):
        """Context manager for profiling a section"""
        class ProfileContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                self.profiler.start(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.end(self.name)
                return False
        
        return ProfileContext(self, name)
    
    def get_results(self) -> ProfileResults:
        """Get profiling results"""
        return self.results
    
    def print_summary(self):
        """Print profiling summary"""
        summary = self.results.summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'avg_step_time' in summary:
            print(f"\nStep Time:")
            print(f"  Average: {summary['avg_step_time']*1000:.1f} ms")
            print(f"  Throughput: {summary['throughput']:.1f} steps/sec")
        
        if 'peak_memory_gb' in summary:
            print(f"\nGPU Memory:")
            print(f"  Peak: {summary['peak_memory_gb']:.2f} GB")
            print(f"  Average: {summary['avg_memory_gb']:.2f} GB")
        
        if 'avg_data_time' in summary:
            print(f"\nData Loading:")
            print(f"  Average: {summary['avg_data_time']*1000:.1f} ms")
            print(f"  Percent of step: {summary['data_percent']:.1f}%")
        
        if 'avg_forward_time' in summary:
            print(f"\nForward Pass: {summary['avg_forward_time']*1000:.1f} ms")
        
        if 'avg_backward_time' in summary:
            print(f"Backward Pass: {summary['avg_backward_time']*1000:.1f} ms")
        
        if 'avg_optimizer_time' in summary:
            print(f"Optimizer Step: {summary['avg_optimizer_time']*1000:.1f} ms")
        
        print("="*60)
    
    def reset(self):
        """Reset profiler"""
        self.results = ProfileResults()
        self._timers.clear()
        self._start_times.clear()