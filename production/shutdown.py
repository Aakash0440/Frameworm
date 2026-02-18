"""
Graceful shutdown handling for production services.

Ensures clean shutdown on SIGTERM, completing in-flight requests
and saving checkpoints before exit.

Example:
    >>> from frameworm.production import GracefulShutdown
    >>> 
    >>> shutdown = GracefulShutdown(timeout=30)
    >>> shutdown.register_handler(save_checkpoint)
    >>> shutdown.start()  # Installs signal handlers
"""

import signal
import time
import threading
from typing import Callable, List, Optional
import sys


class GracefulShutdown:
    """
    Manages graceful shutdown on SIGTERM/SIGINT.
    
    Allows in-flight requests to complete and runs cleanup handlers
    before exiting.
    
    Args:
        timeout: Max seconds to wait for shutdown (default: 30)
        
    Example:
        >>> shutdown = GracefulShutdown(timeout=30)
        >>> 
        >>> @shutdown.on_shutdown
        ... def cleanup():
        ...     save_model()
        ...     close_database()
        >>> 
        >>> shutdown.start()
        >>> # Service runs normally
        >>> # On SIGTERM: cleanup() runs, then graceful exit
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._handlers: List[Callable] = []
        self._shutdown_requested = False
        self._in_flight_requests = 0
        self._lock = threading.Lock()
    
    def register_handler(self, handler: Callable):
        """Register a cleanup handler"""
        self._handlers.append(handler)
    
    def on_shutdown(self, handler: Callable) -> Callable:
        """Decorator to register shutdown handler"""
        self.register_handler(handler)
        return handler
    
    def start(self):
        """Install signal handlers"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        print("✓ Graceful shutdown handlers installed")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self._shutdown_requested:
            print("⚠️  Forced shutdown!")
            sys.exit(1)
        
        self._shutdown_requested = True
        signal_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
        print(f"\n🛑 {signal_name} received - starting graceful shutdown...")
        
        # Wait for in-flight requests
        self._wait_for_requests()
        
        # Run cleanup handlers
        self._run_handlers()
        
        print("✓ Graceful shutdown complete")
        sys.exit(0)
    
    def _wait_for_requests(self):
        """Wait for in-flight requests to complete"""
        if self._in_flight_requests == 0:
            return
        
        print(f"Waiting for {self._in_flight_requests} in-flight request(s)...")
        
        start = time.time()
        while self._in_flight_requests > 0:
            if time.time() - start > self.timeout:
                print(f"⚠️  Timeout reached, {self._in_flight_requests} request(s) incomplete")
                break
            time.sleep(0.1)
        
        if self._in_flight_requests == 0:
            print("✓ All requests completed")
    
    def _run_handlers(self):
        """Run all registered cleanup handlers"""
        print(f"Running {len(self._handlers)} cleanup handler(s)...")
        
        for i, handler in enumerate(self._handlers, 1):
            try:
                print(f"  [{i}/{len(self._handlers)}] Running {handler.__name__}...")
                handler()
            except Exception as e:
                print(f"  ⚠️  Handler {handler.__name__} failed: {e}")
        
        print("✓ Cleanup handlers complete")
    
    def track_request(self):
        """Context manager to track in-flight requests"""
        class RequestTracker:
            def __init__(self, parent):
                self.parent = parent
            
            def __enter__(self):
                with self.parent._lock:
                    self.parent._in_flight_requests += 1
            
            def __exit__(self, *args):
                with self.parent._lock:
                    self.parent._in_flight_requests -= 1
        
        return RequestTracker(self)
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested