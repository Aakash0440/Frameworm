"""
Error explanation system for Frameworm.

Provides helpful, actionable error messages with context, suggestions,
and documentation links.

Example:
    >>> raise DimensionMismatchError(
    ...     expected=(4, 100, 1, 1),
    ...     received=(4, 100),
    ...     layer_name="generator.main[0]"
    ... )
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ErrorContext:
    """
    Captures context about where an error occurred.

    Automatically captures:
    - File and line number
    - Function name
    - Local variables
    - Stack trace
    """

    def __init__(self):
        """Initialize context from current stack frame"""
        # Get caller's frame (2 frames up: this __init__ -> FramewormError.__init__ -> actual raise)
        frame = sys._getframe(2)

        self.filename = frame.f_code.co_filename
        self.line_number = frame.f_lineno
        self.function_name = frame.f_code.co_name
        self.local_vars = dict(frame.f_locals)

        # Get relative path if in frameworm
        try:
            self.filepath = Path(self.filename)
            if "frameworm" in str(self.filepath):
                parts = self.filepath.parts
                if "frameworm" in parts:
                    idx = parts.index("frameworm")
                    self.relative_path = str(Path(*parts[idx:]))
                else:
                    self.relative_path = self.filepath.name
            else:
                self.relative_path = self.filepath.name
        except:
            self.relative_path = self.filename

    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary"""
        return {
            "file": self.relative_path,
            "line": self.line_number,
            "function": self.function_name,
        }


class ErrorFormatter:
    """
    Formats error messages with colors and structure.

    Uses ANSI color codes for terminal output.
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

    @classmethod
    def format_error(
        cls,
        error_name: str,
        message: str,
        context: Optional[ErrorContext] = None,
        details: Optional[Dict[str, Any]] = None,
        causes: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        doc_link: Optional[str] = None,
    ) -> str:
        """
        Format a complete error message.

        Args:
            error_name: Name of the error (e.g., "DimensionMismatchError")
            message: Main error message
            context: Error context
            details: Additional details to show
            causes: List of likely causes
            suggestions: List of suggested fixes
            doc_link: Link to documentation

        Returns:
            Formatted error string
        """
        lines = []

        # Header
        lines.append(f"\n{cls.BOLD}{cls.RED}{error_name}: {message}{cls.RESET}")

        # Location (if context provided)
        if context:
            lines.append(f"\n{cls.BOLD}Location:{cls.RESET}")
            lines.append(f"  File: {cls.CYAN}{context.relative_path}{cls.RESET}")
            lines.append(f"  Line: {cls.CYAN}{context.line_number}{cls.RESET}")
            lines.append(f"  Function: {cls.CYAN}{context.function_name}(){cls.RESET}")

        # Details
        if details:
            lines.append(f"\n{cls.BOLD}Details:{cls.RESET}")
            for key, value in details.items():
                lines.append(f"  {key}: {cls.YELLOW}{value}{cls.RESET}")

        # Likely causes
        if causes:
            lines.append(f"\n{cls.BOLD}Likely Causes:{cls.RESET}")
            for i, cause in enumerate(causes, 1):
                lines.append(f"  {i}. {cause}")

        # Suggestions
        if suggestions:
            lines.append(f"\n{cls.BOLD}{cls.GREEN}Suggested Fixes:{cls.RESET}")
            for suggestion in suggestions:
                lines.append(f"  {cls.GREEN}→{cls.RESET} {suggestion}")

        # Documentation
        if doc_link:
            lines.append(f"\n{cls.BOLD}Documentation:{cls.RESET}")
            lines.append(f"  {cls.BLUE}{doc_link}{cls.RESET}")

        lines.append("")  # Empty line at end

        return "\n".join(lines)


class FramewormError(Exception):
    """
    Base exception for all Frameworm errors.

    All custom exceptions should inherit from this.
    Provides automatic context capture and helpful formatting.

    Args:
        message: Main error message
        **kwargs: Additional context (stored as attributes)
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.message = message
        self.context = ErrorContext()

        # Store additional context
        for key, value in kwargs.items():
            setattr(self, key, value)

        # These can be overridden by subclasses
        self.causes: List[str] = []
        self.suggestions: List[str] = []
        self.doc_link: Optional[str] = None

    def add_cause(self, cause: str):
        """Add a likely cause"""
        self.causes.append(cause)
        return self

    def add_suggestion(self, suggestion: str):
        """Add a suggested fix"""
        self.suggestions.append(suggestion)
        return self

    def set_doc_link(self, link: str):
        """Set documentation link"""
        self.doc_link = link
        return self

    def get_details(self) -> Dict[str, Any]:
        """
        Get error details.

        Override in subclasses to provide specific details.
        """
        return {}

    def __str__(self) -> str:
        """Format error message"""
        return ErrorFormatter.format_error(
            error_name=self.__class__.__name__,
            message=self.message,
            context=self.context,
            details=self.get_details(),
            causes=self.causes,
            suggestions=self.suggestions,
            doc_link=self.doc_link,
        )


# ==================== Configuration Errors ====================


class ConfigurationError(FramewormError):
    """Base class for configuration-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.set_doc_link("https://frameworm.readthedocs.io/errors/configuration")


class ConfigNotFoundError(ConfigurationError, FileNotFoundError):
    """Raised when a config file is not found"""

    def __init__(self, config_path: str, **kwargs):
        message = f"Configuration file not found: {config_path}"
        super().__init__(message, config_path=config_path, **kwargs)

        self.add_cause(f"File '{config_path}' does not exist")
        self.add_cause("Path may be relative to wrong directory")
        self.add_cause("File extension might be wrong (.yaml vs .yml)")

        self.add_suggestion(f"Check that '{config_path}' exists")
        self.add_suggestion("Try using absolute path")
        self.add_suggestion("List files: ls -la $(dirname {config_path})")

    def get_details(self) -> Dict[str, Any]:
        import os
        from pathlib import Path

        return {
            "Path": self.config_path,
            "Absolute path": Path(self.config_path).resolve(),
            "Current directory": os.getcwd(),
            "Exists": Path(self.config_path).exists(),
        }


class ConfigValidationError(ConfigurationError):
    """Raised when config validation fails"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, field=field, **kwargs)

        if field:
            self.add_cause(f"Field '{field}' has invalid value or type")
            self.add_suggestion(f"Check config.{field} in your YAML file")

        self.add_suggestion("Run: frameworm validate-config <path>")


class ConfigInheritanceError(ConfigurationError):
    """Raised when config inheritance fails"""

    def __init__(self, message: str, base_config: Optional[str] = None, **kwargs):
        super().__init__(message, base_config=base_config, **kwargs)

        if base_config:
            self.add_cause(f"Base config '{base_config}' not found or invalid")
            self.add_cause("Circular inheritance (A inherits B inherits A)")

            self.add_suggestion(f"Check _base_: {base_config} in your config")
            self.add_suggestion("Verify base config path is correct")


# ==================== Model Errors ====================


class ModelError(FramewormError):
    """Base class for model-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.set_doc_link("https://frameworm.readthedocs.io/errors/models")


class DimensionMismatchError(ModelError):
    """Raised when tensor dimensions don't match expectations"""

    def __init__(
        self,
        message: str = "Tensor dimension mismatch",
        expected: Optional[Tuple] = None,
        received: Optional[Tuple] = None,
        layer_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message, expected=expected, received=received, layer_name=layer_name, **kwargs
        )

        # Analyze the mismatch
        if expected and received:
            self._analyze_mismatch(expected, received, layer_name)

    def _analyze_mismatch(self, expected: Tuple, received: Tuple, layer_name: Optional[str]):
        """Analyze dimension mismatch and provide helpful suggestions"""
        exp_len = len(expected)
        rec_len = len(received)

        # Missing dimensions
        if rec_len < exp_len:
            missing = exp_len - rec_len
            self.add_cause(f"Input has {missing} fewer dimension(s) than expected")

            if missing == 2 and exp_len == 4:
                self.add_suggestion("Add spatial dimensions: x.unsqueeze(-1).unsqueeze(-1)")
                self.add_suggestion("Or reshape: x.view(batch, channels, 1, 1)")
            elif missing == 1:
                self.add_suggestion("Add dimension: x.unsqueeze(-1)")

        # Extra dimensions
        elif rec_len > exp_len:
            extra = rec_len - exp_len
            self.add_cause(f"Input has {extra} extra dimension(s)")

            if extra == 1:
                self.add_suggestion("Remove dimension: x.squeeze(-1)")

        # Same length, different sizes
        else:
            for i, (e, r) in enumerate(zip(expected, received)):
                if e != r and e != -1:  # -1 means any size
                    self.add_cause(f"Dimension {i}: expected {e}, got {r}")

                    if i == 0:
                        self.add_cause("Batch size mismatch (usually OK)")
                    elif i == 1:
                        self.add_cause("Channel dimension mismatch")
                        self.add_suggestion(f"Check config: channels should be {e}")

        # Layer-specific suggestions
        if layer_name:
            if "Conv" in layer_name:
                self.add_suggestion(f"Check input to {layer_name}: should be 4D (B, C, H, W)")
            elif "Linear" in layer_name:
                self.add_suggestion(f"Check input to {layer_name}: flatten spatial dims first")

    def get_details(self) -> Dict[str, Any]:
        details = {}
        if self.expected:
            details["Expected shape"] = self.expected
        if self.received:
            details["Received shape"] = self.received
        if self.layer_name:
            details["Layer"] = self.layer_name
        return details


class ModelNotFoundError(ModelError):
    """Raised when a registered model is not found"""

    def __init__(self, model_name: str, available: Optional[List[str]] = None, **kwargs):
        message = f"Model '{model_name}' not found in registry"
        super().__init__(message, model_name=model_name, available=available, **kwargs)

        self.add_cause(f"Model '{model_name}' is not registered")
        self.add_cause("Model plugin not discovered")
        self.add_cause("Typo in model name")

        if available:
            # Find similar names
            similar = [
                name
                for name in available
                if model_name.lower() in name.lower() or name.lower() in model_name.lower()
            ]
            if similar:
                self.add_suggestion(f"Did you mean: {', '.join(similar)}?")

        self.add_suggestion("List all models: frameworm list-models")
        self.add_suggestion("Re-discover plugins: discover_plugins(force=True)")

    def get_details(self) -> Dict[str, Any]:
        details = {"Requested": self.model_name}
        if self.available:
            details["Available models"] = ", ".join(self.available[:5])
            if len(self.available) > 5:
                details["..."] = f"and {len(self.available) - 5} more"
        return details


# ==================== Training Errors ====================


class TrainingError(FramewormError):
    """Base class for training-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.set_doc_link("https://frameworm.readthedocs.io/errors/training")


class ConvergenceError(TrainingError):
    """Raised when training fails to converge"""

    def __init__(
        self,
        message: str = "Training failed to converge",
        loss_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, loss_value=loss_value, **kwargs)

        if loss_value:
            if loss_value > 1e6:
                self.add_cause("Loss exploded (gradient explosion)")
                self.add_suggestion("Reduce learning rate")
                self.add_suggestion("Add gradient clipping")
                self.add_suggestion("Check for NaN in data")
            elif loss_value == float("nan"):
                self.add_cause("Loss is NaN (numerical instability)")
                self.add_suggestion("Check for division by zero")
                self.add_suggestion("Reduce learning rate")
                self.add_suggestion("Use mixed precision training")

    def get_details(self) -> Dict[str, Any]:
        if self.loss_value:
            return {"Loss value": self.loss_value}
        return {}


# ==================== Plugin Errors ====================


class PluginError(FramewormError):
    """Base class for plugin-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.set_doc_link("https://frameworm.readthedocs.io/errors/plugins")


class PluginValidationError(PluginError):
    """Raised when plugin validation fails"""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        missing_methods: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message, plugin_name=plugin_name, missing_methods=missing_methods, **kwargs
        )

        if missing_methods:
            self.add_cause(f"Plugin missing required methods: {', '.join(missing_methods)}")

            for method in missing_methods:
                self.add_suggestion(f"Add method: def {method}(self, ...): ...")

        self.add_suggestion("Check plugin inherits from correct base class")
        self.add_suggestion("See: docs/user_guide/plugins.md")

    def get_details(self) -> Dict[str, Any]:
        details = {}
        if self.plugin_name:
            details["Plugin"] = self.plugin_name
        if self.missing_methods:
            details["Missing methods"] = ", ".join(self.missing_methods)
        return details
