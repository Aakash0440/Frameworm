"""
ONNX Runtime inference wrapper.
"""

import numpy as np
from typing import Union, List, Dict, Any
import warnings

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ONNXInferenceSession:
    """
    Fast inference with ONNX Runtime.
    
    Often 2-5x faster than PyTorch for production inference.
    
    Args:
        model_path: Path to .onnx file
        providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
    Example:
        >>> session = ONNXInferenceSession('model.onnx')
        >>> output = session.run(input_data)
    """
    
    def __init__(
        self,
        model_path: str,
        providers: List[str] = None
    ):
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        
        self.model_path = model_path
        
        # Default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            
            # Try CUDA if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Get input shapes
        self.input_shapes = {
            inp.name: inp.shape for inp in self.session.get_inputs()
        }
        
        print(f"✓ ONNX Runtime session created")
        print(f"  Provider: {self.session.get_providers()[0]}")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def run(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        return_numpy: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference.
        
        Args:
            inputs: Input data (numpy array or dict)
            return_numpy: Return numpy arrays (vs list)
            
        Returns:
            Model outputs
        """
        # Handle dict vs array input
        if isinstance(inputs, dict):
            input_feed = inputs
        else:
            # Assume single input
            input_feed = {self.input_names[0]: inputs}
        
        # Run inference
        outputs = self.session.run(self.output_names, input_feed)
        
        # Return format
        if return_numpy and len(outputs) == 1:
            return outputs[0]
        
        return outputs
    
    def benchmark(
        self,
        example_input: np.ndarray,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Returns:
            Dict with timing statistics
        """
        import time
        
        print(f"Benchmarking ONNX Runtime ({num_runs} runs)...")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.run(example_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.run(example_input)
            end = time.perf_counter()
            times.append(end - start)
        
        # Stats
        mean_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        throughput = 1000 / mean_time
        
        results = {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'throughput': throughput
        }
        
        print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} inferences/sec")
        
        return results