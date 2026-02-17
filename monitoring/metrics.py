
"""
Prometheus metrics for training and inference monitoring.

Exposes:
- Training metrics (loss, lr, epoch progress)
- Inference metrics (latency, throughput, errors)
- System metrics (GPU memory, CPU usage)

Example:
    >>> from frameworm.monitoring import MetricsExporter
    >>> exporter = MetricsExporter(port=9090)
    >>> exporter.start()
    >>> # Metrics now available at http://localhost:9090/metrics
"""

from typing import Dict, Optional, List
import time
import threading
from collections import defaultdict, deque
import json


try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class InMemoryMetrics:
    """
    Lightweight in-memory metrics store when Prometheus is not available.
    
    Stores last N values per metric for local monitoring.
    Always available regardless of prometheus_client installation.
    
    Example:
        >>> metrics = InMemoryMetrics()
        >>> metrics.record('train_loss', 0.45)
        >>> metrics.record('val_loss', 0.50)
        >>> print(metrics.get_latest('train_loss'))
        0.45
        >>> print(metrics.get_history('train_loss'))
        [0.45]
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value"""
        with self._lock:
            self._data[name].append(value)
            self._timestamps[name].append(timestamp or time.time())
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get the most recent value"""
        with self._lock:
            if name in self._data and self._data[name]:
                return self._data[name][-1]
        return None
    
    def get_history(self, name: str, n: int = None) -> List[float]:
        """Get value history"""
        with self._lock:
            if name not in self._data:
                return []
            data = list(self._data[name])
            return data[-n:] if n else data
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get statistical summary"""
        history = self.get_history(name)
        if not history:
            return {}
        
        import statistics
        return {
            'count': len(history),
            'latest': history[-1],
            'mean': statistics.mean(history),
            'min': min(history),
            'max': max(history),
            'stdev': statistics.stdev(history) if len(history) > 1 else 0.0
        }
    
    def export_json(self) -> str:
        """Export all metrics as JSON"""
        result = {}
        with self._lock:
            for name in self._data:
                result[name] = self.get_summary(name)
        return json.dumps(result, indent=2)
    
    def reset(self, name: Optional[str] = None):
        """Clear metrics (all or specific)"""
        with self._lock:
            if name:
                self._data.pop(name, None)
                self._timestamps.pop(name, None)
            else:
                self._data.clear()
                self._timestamps.clear()


class TrainingMetricsCollector:
    """
    Collect and expose training metrics.
    
    Integrates with Trainer via callback or direct calls.
    Falls back to InMemoryMetrics when Prometheus is not installed.
    
    Args:
        namespace: Metric name prefix (e.g., 'frameworm')
        use_prometheus: Try to use prometheus_client (falls back if unavailable)
        
    Example:
        >>> collector = TrainingMetricsCollector()
        >>> collector.record_epoch(
        ...     epoch=1, train_loss=0.45, val_loss=0.50, lr=0.001
        ... )
        >>> print(collector.get_latest('train_loss'))
        0.45
    """
    
    def __init__(
        self,
        namespace: str = 'frameworm',
        use_prometheus: bool = True
    ):
        self.namespace = namespace
        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        
        # Always keep in-memory store
        self.store = InMemoryMetrics()
        
        if self.use_prometheus:
            self._setup_prometheus_metrics(namespace)
        else:
            if use_prometheus:
                print("⚠️  prometheus_client not installed. Using in-memory metrics.")
                print("   Install with: pip install prometheus-client")
    
    def _setup_prometheus_metrics(self, namespace: str):
        """Create Prometheus metric objects"""
        # Training metrics
        self.prom_train_loss = Gauge(f'{namespace}_train_loss', 'Training loss')
        self.prom_val_loss = Gauge(f'{namespace}_val_loss', 'Validation loss')
        self.prom_learning_rate = Gauge(f'{namespace}_learning_rate', 'Current learning rate')
        self.prom_epoch = Gauge(f'{namespace}_epoch', 'Current training epoch')
        self.prom_global_step = Counter(f'{namespace}_global_step_total', 'Total training steps')
        
        # Throughput
        self.prom_samples_per_sec = Gauge(
            f'{namespace}_samples_per_second', 'Training samples per second'
        )
        self.prom_step_duration = Histogram(
            f'{namespace}_step_duration_seconds',
            'Duration of each training step',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
    
    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        **extra_metrics
    ):
        """Record all metrics for one epoch"""
        self.store.record('train_loss', train_loss)
        self.store.record('epoch', float(epoch))
        
        if val_loss is not None:
            self.store.record('val_loss', val_loss)
        if lr is not None:
            self.store.record('learning_rate', lr)
        if samples_per_sec is not None:
            self.store.record('samples_per_second', samples_per_sec)
        
        for name, value in extra_metrics.items():
            self.store.record(name, float(value))
        
        # Update Prometheus if available
        if self.use_prometheus:
            self.prom_train_loss.set(train_loss)
            self.prom_epoch.set(epoch)
            if val_loss is not None:
                self.prom_val_loss.set(val_loss)
            if lr is not None:
                self.prom_learning_rate.set(lr)
            if samples_per_sec is not None:
                self.prom_samples_per_sec.set(samples_per_sec)
    
    def record_step(self, step_duration_s: float, batch_size: int):
        """Record a single training step"""
        self.store.record('step_duration_s', step_duration_s)
        self.store.record('samples_per_step', float(batch_size))
        
        if self.use_prometheus:
            self.prom_step_duration.observe(step_duration_s)
            self.prom_global_step.inc()
    
    def get_latest(self, metric: str) -> Optional[float]:
        return self.store.get_latest(metric)
    
    def get_history(self, metric: str) -> List[float]:
        return self.store.get_history(metric)
    
    def print_summary(self):
        """Print a formatted summary of all collected metrics"""
        print("\n📊 Training Metrics Summary")
        print("─" * 50)
        
        key_metrics = ['train_loss', 'val_loss', 'learning_rate', 'samples_per_second']
        for name in key_metrics:
            summary = self.store.get_summary(name)
            if summary:
                print(f"  {name:<25} latest={summary['latest']:.4f}  "
                      f"min={summary['min']:.4f}  max={summary['max']:.4f}")
        print("─" * 50)


class InferenceMetricsCollector:
    """
    Collect real-time inference metrics.
    
    Tracks latency, throughput, error rates.
    Thread-safe for concurrent inference workers.
    
    Example:
        >>> collector = InferenceMetricsCollector()
        >>> 
        >>> # In your inference code:
        >>> with collector.time_request():
        ...     output = model(input)
        >>> 
        >>> collector.print_summary()
    """
    
    def __init__(self, namespace: str = 'frameworm'):
        self.store = InMemoryMetrics(history_size=10000)
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.Lock()
        
        if PROMETHEUS_AVAILABLE:
            self.prom_request_count = Counter(
                f'{namespace}_inference_requests_total', 'Total inference requests'
            )
            self.prom_latency = Histogram(
                f'{namespace}_inference_latency_ms',
                'Inference latency in milliseconds',
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
            )
            self.prom_error_rate = Gauge(
                f'{namespace}_inference_error_rate', 'Request error rate (0-1)'
            )
            self.prom_throughput = Gauge(
                f'{namespace}_inference_throughput',
                'Inference throughput (requests/sec)'
            )
            self._use_prometheus = True
        else:
            self._use_prometheus = False
    
    class _TimerContext:
        """Context manager for timing inference"""
        def __init__(self, collector):
            self.collector = collector
            self.start = None
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self.start) * 1000
            self.collector._record_request(duration_ms, error=(exc_type is not None))
    
    def time_request(self):
        """Context manager to time a single inference request"""
        return self._TimerContext(self)
    
    def _record_request(self, latency_ms: float, error: bool = False):
        with self._lock:
            self._request_count += 1
            if error:
                self._error_count += 1
            
            self.store.record('latency_ms', latency_ms)
            error_rate = self._error_count / self._request_count
            self.store.record('error_rate', error_rate)
        
        if self._use_prometheus:
            self.prom_request_count.inc()
            self.prom_latency.observe(latency_ms)
            self.prom_error_rate.set(error_rate)
    
    def get_percentile(self, percentile: float) -> float:
        """Get latency percentile (e.g., p95, p99)"""
        import statistics
        history = self.store.get_history('latency_ms')
        if not history:
            return 0.0
        sorted_hist = sorted(history)
        idx = int(len(sorted_hist) * percentile / 100)
        return sorted_hist[min(idx, len(sorted_hist) - 1)]
    
    def print_summary(self):
        print("\n🚀 Inference Metrics Summary")
        print("─" * 50)
        latency_summary = self.store.get_summary('latency_ms')
        if latency_summary:
            print(f"  Requests:   {self._request_count}")
            print(f"  Errors:     {self._error_count}")
            print(f"  Latency p50: {self.get_percentile(50):.1f} ms")
            print(f"  Latency p95: {self.get_percentile(95):.1f} ms")
            print(f"  Latency p99: {self.get_percentile(99):.1f} ms")
            print(f"  Mean latency: {latency_summary['mean']:.1f} ms")
        print("─" * 50)


class MetricsExporter:
    """
    HTTP server exposing Prometheus metrics endpoint.
    
    Args:
        port: HTTP port to listen on (default: 9090)
        
    Example:
        >>> exporter = MetricsExporter(port=9090)
        >>> exporter.start()
        >>> # curl http://localhost:9090/metrics
    """
    
    def __init__(self, port: int = 9090):
        self.port = port
        self._server_thread = None
        
    def start(self):
        """Start the metrics HTTP server in background thread"""
        if not PROMETHEUS_AVAILABLE:
            print("⚠️  prometheus_client not installed. Cannot start metrics server.")
            print("   Install: pip install prometheus-client")
            return
        
        self._server_thread = threading.Thread(
            target=lambda: start_http_server(self.port),
            daemon=True
        )
        self._server_thread.start()
        
        print(f"✓ Metrics server started at http://localhost:{self.port}/metrics")
        print(f"  Add to Prometheus scrape config:")
        print(f"    - targets: ['localhost:{self.port}']")