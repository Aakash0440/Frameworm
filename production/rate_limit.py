"""
Rate limiting for production APIs.

Implements token bucket and sliding window algorithms.

Example:
    >>> from frameworm.production import RateLimiter
    >>> 
    >>> limiter = RateLimiter(max_requests=100, window_seconds=60)
    >>> 
    >>> if limiter.allow(user_id='user123'):
    ...     process_request()
    ... else:
    ...     return_429_error()
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
import threading


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Limits requests per user/IP within a time window.
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        
    Example:
        >>> limiter = RateLimiter(max_requests=10, window_seconds=60)
        >>> 
        >>> for i in range(15):
        ...     if limiter.allow('user1'):
        ...         print(f"Request {i} allowed")
        ...     else:
        ...         print(f"Request {i} rate limited!")
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Sliding window: store timestamps for each key
        self._timestamps: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.Lock()
    
    def allow(self, key: str) -> bool:
        """
        Check if request is allowed for given key.
        
        Args:
            key: Identifier (user_id, IP address, etc.)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            timestamps = self._timestamps[key]
            
            # Remove old timestamps outside window
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            
            # Check if under limit
            if len(timestamps) < self.max_requests:
                timestamps.append(now)
                return True
            
            return False
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            timestamps = self._timestamps[key]
            
            # Remove old
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            
            return max(0, self.max_requests - len(timestamps))
    
    def reset(self, key: str):
        """Reset rate limit for key"""
        with self._lock:
            self._timestamps.pop(key, None)
    
    def cleanup_old_keys(self, max_age_seconds: float = 3600):
        """Remove keys not used recently (memory cleanup)"""
        now = time.time()
        cutoff = now - max_age_seconds
        
        with self._lock:
            keys_to_remove = []
            
            for key, timestamps in self._timestamps.items():
                if not timestamps or timestamps[-1] < cutoff:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._timestamps[key]
            
            if keys_to_remove:
                print(f"✓ Cleaned up {len(keys_to_remove)} rate limit key(s)")


class TokenBucketLimiter:
    """
    Token bucket rate limiter (smoother than sliding window).
    
    Tokens refill continuously, allowing burst traffic up to bucket capacity.
    
    Args:
        capacity: Bucket capacity (max burst)
        refill_rate: Tokens added per second
        
    Example:
        >>> limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        >>> # Can burst 10 requests immediately, then 1/second sustained
    """
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def allow(self, key: str, tokens: float = 1.0) -> bool:
        """Check if request allowed (consumes tokens if yes)"""
        now = time.time()
        
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    'tokens': self.capacity,
                    'last_update': now
                }
            
            bucket = self._buckets[key]
            
            # Refill tokens based on elapsed time
            elapsed = now - bucket['last_update']
            bucket['tokens'] = min(
                self.capacity,
                bucket['tokens'] + elapsed * self.refill_rate
            )
            bucket['last_update'] = now
            
            # Check if enough tokens
            if bucket['tokens'] >= tokens:
                bucket['tokens'] -= tokens
                return True
            
            return False