
"""
Production health check system.

Provides liveness, readiness, and dependency health checks
for production deployments.

Example:
    >>> from frameworm.production import HealthChecker
    >>> 
    >>> health = HealthChecker()
    >>> health.add_check('database', check_database_connection)
    >>> health.add_check('gpu', check_gpu_available)
    >>> 
    >>> status = health.check_all()  # Returns health status
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading


@dataclass
class HealthStatus:
    """Health check result"""
    healthy: bool
    message: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'healthy': self.healthy,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class HealthChecker:
    """
    Centralized health check management.
    
    Supports multiple check types:
    - Liveness: Is the service running? (basic ping)
    - Readiness: Can the service handle requests? (dependencies OK)
    - Startup: Has initialization completed?
    
    Args:
        startup_timeout: Max seconds to wait for startup checks (default: 60)
        
    Example:
        >>> health = HealthChecker()
        >>> 
        >>> # Add checks
        >>> health.add_liveness_check('basic', lambda: HealthStatus(True, 'OK'))
        >>> health.add_readiness_check('database', check_db_connection)
        >>> health.add_readiness_check('model_loaded', check_model_loaded)
        >>> 
        >>> # Check health
        >>> if health.is_ready():
        ...     start_serving_traffic()
    """
    
    def __init__(self, startup_timeout: int = 60):
        self._liveness_checks: Dict[str, Callable] = {}
        self._readiness_checks: Dict[str, Callable] = {}
        self._startup_checks: Dict[str, Callable] = {}
        
        self._startup_complete = False
        self._startup_timeout = startup_timeout
        self._last_check_time: Dict[str, float] = {}
        self._check_cache: Dict[str, HealthStatus] = {}
        self._cache_ttl = 5.0  # Cache results for 5 seconds
        
        self._lock = threading.Lock()
    
    def add_liveness_check(self, name: str, check_fn: Callable[[], HealthStatus]):
        """Add a liveness check (is service running?)"""
        self._liveness_checks[name] = check_fn
    
    def add_readiness_check(self, name: str, check_fn: Callable[[], HealthStatus]):
        """Add a readiness check (can service handle traffic?)"""
        self._readiness_checks[name] = check_fn
    
    def add_startup_check(self, name: str, check_fn: Callable[[], HealthStatus]):
        """Add a startup check (has initialization completed?)"""
        self._startup_checks[name] = check_fn
    
    def is_alive(self) -> bool:
        """Check if service is alive"""
        results = self._run_checks(self._liveness_checks)
        return all(r.healthy for r in results.values())
    
    def is_ready(self) -> bool:
        """Check if service is ready to handle traffic"""
        if not self._startup_complete:
            return False
        
        results = self._run_checks(self._readiness_checks)
        return all(r.healthy for r in results.values())
    
    def check_startup(self) -> bool:
        """Run startup checks and mark complete if all pass"""
        if self._startup_complete:
            return True
        
        results = self._run_checks(self._startup_checks)
        all_healthy = all(r.healthy for r in results.values())
        
        if all_healthy:
            self._startup_complete = True
            print("✓ Startup checks passed - service ready")
        
        return all_healthy
    
    def _run_checks(self, checks: Dict[str, Callable]) -> Dict[str, HealthStatus]:
        """Run a set of health checks with caching"""
        results = {}
        
        for name, check_fn in checks.items():
            # Check cache
            now = time.time()
            if name in self._check_cache:
                last_check = self._last_check_time.get(name, 0)
                if now - last_check < self._cache_ttl:
                    results[name] = self._check_cache[name]
                    continue
            
            # Run check
            try:
                result = check_fn()
                if not isinstance(result, HealthStatus):
                    result = HealthStatus(
                        healthy=bool(result),
                        message=str(result) if result else "Check passed"
                    )
            except Exception as e:
                result = HealthStatus(
                    healthy=False,
                    message=f"Check failed: {str(e)}"
                )
            
            # Cache result
            with self._lock:
                self._check_cache[name] = result
                self._last_check_time[name] = now
            
            results[name] = result
        
        return results
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get complete health report"""
        liveness = self._run_checks(self._liveness_checks)
        readiness = self._run_checks(self._readiness_checks)
        startup = self._run_checks(self._startup_checks)
        
        return {
            'alive': all(r.healthy for r in liveness.values()),
            'ready': self._startup_complete and all(r.healthy for r in readiness.values()),
            'startup_complete': self._startup_complete,
            'checks': {
                'liveness': {k: v.to_dict() for k, v in liveness.items()},
                'readiness': {k: v.to_dict() for k, v in readiness.items()},
                'startup': {k: v.to_dict() for k, v in startup.items()}
            },
            'timestamp': time.time()
        }


# Built-in health checks
def check_gpu_available() -> HealthStatus:
    """Check if GPU is available"""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'N/A'
        return HealthStatus(
            healthy=True,
            message=f"{gpu_count} GPU(s) available",
            details={'gpu_count': gpu_count, 'gpu_name': gpu_name}
        )
    else:
        return HealthStatus(
            healthy=False,
            message="No GPU available"
        )


def check_memory_usage(threshold_percent: float = 90.0) -> HealthStatus:
    """Check system memory usage"""
    import psutil
    
    mem = psutil.virtual_memory()
    healthy = mem.percent < threshold_percent
    
    return HealthStatus(
        healthy=healthy,
        message=f"Memory usage: {mem.percent:.1f}%",
        details={
            'used_gb': mem.used / (1024**3),
            'total_gb': mem.total / (1024**3),
            'percent': mem.percent
        }
    )


def check_disk_space(path: str = '/', threshold_percent: float = 90.0) -> HealthStatus:
    """Check disk space"""
    import psutil
    
    disk = psutil.disk_usage(path)
    healthy = disk.percent < threshold_percent
    
    return HealthStatus(
        healthy=healthy,
        message=f"Disk usage: {disk.percent:.1f}%",
        details={
            'free_gb': disk.free / (1024**3),
            'total_gb': disk.total / (1024**3),
            'percent': disk.percent
        }
    )