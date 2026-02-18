"""
Request validation and sanitization.

Example:
    >>> from frameworm.production import RequestValidator
    >>> 
    >>> validator = RequestValidator()
    >>> validator.add_rule('image', type=bytes, max_size_mb=10)
    >>> 
    >>> if validator.validate(request_data):
    ...     process_request()
"""

from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class ValidationRule:
    """Single validation rule"""
    field_name: str
    required: bool = True
    type_check: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None
    max_size_mb: Optional[float] = None
    allowed_values: Optional[List] = None
    custom_validator: Optional[Callable] = None


class ValidationError(Exception):
    """Validation failed"""
    pass


class RequestValidator:
    """
    Validate and sanitize incoming requests.
    
    Example:
        >>> validator = RequestValidator()
        >>> 
        >>> validator.add_rule(
        ...     'batch_size',
        ...     type=int,
        ...     min_value=1,
        ...     max_value=128
        ... )
        >>> 
        >>> try:
        ...     validator.validate({'batch_size': 64})
        ... except ValidationError as e:
        ...     return_error(str(e))
    """
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
    
    def add_rule(
        self,
        field_name: str,
        required: bool = True,
        type_check: Optional[type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_length: Optional[int] = None,
        max_size_mb: Optional[float] = None,
        allowed_values: Optional[List] = None,
        custom_validator: Optional[Callable] = None
    ):
        """Add validation rule for a field"""
        self.rules[field_name] = ValidationRule(
            field_name=field_name,
            required=required,
            type_check=type_check,
            min_value=min_value,
            max_value=max_value,
            max_length=max_length,
            max_size_mb=max_size_mb,
            allowed_values=allowed_values,
            custom_validator=custom_validator
        )
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data against rules.
        
        Raises:
            ValidationError: If validation fails
        """
        for field_name, rule in self.rules.items():
            # Check required
            if rule.required and field_name not in data:
                raise ValidationError(f"Missing required field: {field_name}")
            
            if field_name not in data:
                continue
            
            value = data[field_name]
            
            # Type check
            if rule.type_check and not isinstance(value, rule.type_check):
                raise ValidationError(
                    f"Field {field_name} must be {rule.type_check.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            # Min/max value
            if rule.min_value is not None and value < rule.min_value:
                raise ValidationError(f"{field_name} must be >= {rule.min_value}")
            
            if rule.max_value is not None and value > rule.max_value:
                raise ValidationError(f"{field_name} must be <= {rule.max_value}")
            
            # Max length (for strings/lists)
            if rule.max_length and len(value) > rule.max_length:
                raise ValidationError(f"{field_name} length must be <= {rule.max_length}")
            
            # Size check (for bytes)
            if rule.max_size_mb and isinstance(value, bytes):
                size_mb = len(value) / (1024 * 1024)
                if size_mb > rule.max_size_mb:
                    raise ValidationError(
                        f"{field_name} size {size_mb:.1f}MB exceeds max {rule.max_size_mb}MB"
                    )
            
            # Allowed values
            if rule.allowed_values and value not in rule.allowed_values:
                raise ValidationError(
                    f"{field_name} must be one of {rule.allowed_values}"
                )
            
            # Custom validator
            if rule.custom_validator:
                try:
                    if not rule.custom_validator(value):
                        raise ValidationError(f"Custom validation failed for {field_name}")
                except Exception as e:
                    raise ValidationError(f"Validation error for {field_name}: {e}")
        
        return True


def sanitize_string(s: str, max_length: int = 1000) -> str:
    """Sanitize string input (remove dangerous characters, limit length)"""
    # Remove null bytes
    s = s.replace('\x00', '')
    
    # Limit length
    if len(s) > max_length:
        s = s[:max_length]
    
    # Strip whitespace
    s = s.strip()
    
    return s