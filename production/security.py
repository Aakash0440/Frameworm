"""
Security features for production APIs.

Example:
    >>> from frameworm.production import APIKeyAuth
    >>> 
    >>> auth = APIKeyAuth()
    >>> auth.create_key('user123')  # Returns API key
    >>> 
    >>> if auth.verify_key(request_key):
    ...     process_request()
"""

import secrets
import hashlib
import hmac
from typing import Dict, Optional, Set
import time


class APIKeyAuth:
    """
    API key authentication.
    
    Keys are stored hashed (bcrypt-style).
    
    Example:
        >>> auth = APIKeyAuth()
        >>> key = auth.create_key('user1')
        >>> print(f"API Key: {key}")
        >>> 
        >>> # Later, verify
        >>> if auth.verify_key(key, 'user1'):
        ...     allow_access()
    """
    
    def __init__(self):
        self._keys: Dict[str, Set[str]] = {}  # user_id -> set of key hashes
    
    def create_key(self, user_id: str, prefix: str = 'fwk') -> str:
        """
        Create new API key for user.
        
        Returns:
            API key string (only shown once!)
        """
        # Generate random key
        key = f"{prefix}_{secrets.token_urlsafe(32)}"
        
        # Store hash
        key_hash = self._hash_key(key)
        
        if user_id not in self._keys:
            self._keys[user_id] = set()
        
        self._keys[user_id].add(key_hash)
        
        return key
    
    def verify_key(self, key: str, user_id: Optional[str] = None) -> bool:
        """
        Verify API key.
        
        Args:
            key: API key to verify
            user_id: Optional user ID (checks all users if None)
        """
        key_hash = self._hash_key(key)
        
        if user_id:
            return key_hash in self._keys.get(user_id, set())
        else:
            # Check all users
            return any(key_hash in keys for keys in self._keys.values())
    
    def revoke_key(self, key: str, user_id: str):
        """Revoke an API key"""
        key_hash = self._hash_key(key)
        
        if user_id in self._keys:
            self._keys[user_id].discard(key_hash)
    
    def _hash_key(self, key: str) -> str:
        """Hash API key"""
        return hashlib.sha256(key.encode()).hexdigest()


class RequestSigner:
    """
    Request signing for API security.
    
    Clients sign requests with HMAC, server verifies signature.
    
    Example:
        >>> signer = RequestSigner(secret_key='my-secret')
        >>> 
        >>> # Client side
        >>> signature = signer.sign(request_body, timestamp)
        >>> 
        >>> # Server side
        >>> if signer.verify(request_body, signature, timestamp):
        ...     process_request()
    """
    
    def __init__(self, secret_key: str, max_timestamp_age: int = 300):
        self.secret_key = secret_key.encode()
        self.max_timestamp_age = max_timestamp_age
    
    def sign(self, data: bytes, timestamp: Optional[int] = None) -> str:
        """
        Sign request data.
        
        Args:
            data: Request body bytes
            timestamp: Unix timestamp (auto-generated if None)
            
        Returns:
            Signature string (hex)
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        message = f"{timestamp}:{data.hex()}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def verify(self, data: bytes, signature_with_timestamp: str) -> bool:
        """
        Verify request signature.
        
        Args:
            data: Request body bytes
            signature_with_timestamp: "{timestamp}:{signature}"
            
        Returns:
            True if valid and not expired
        """
        try:
            timestamp_str, signature = signature_with_timestamp.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check timestamp age (prevent replay attacks)
            now = int(time.time())
            if abs(now - timestamp) > self.max_timestamp_age:
                return False
            
            # Verify signature
            expected_sig = self.sign(data, timestamp)
            expected_sig_only = expected_sig.split(':', 1)[1]
            
            return hmac.compare_digest(signature, expected_sig_only)
        
        except (ValueError, IndexError):
            return False