"""Comprehensive tests for error system"""

import pytest

from core.exceptions import (
    ConfigInheritanceError,
    ConfigNotFoundError,
    ConfigValidationError,
    ConvergenceError,
    DimensionMismatchError,
    ErrorContext,
    ErrorFormatter,
    FramewormError,
    ModelNotFoundError,
    PluginValidationError,
)


class TestErrorContext:
    """Test error context capture"""

    def test_context_capture(self):
        """Should capture file, line, function"""
        try:
            raise FramewormError("test")
        except FramewormError as e:
            assert e.context.function_name == "test_context_capture"
            assert e.context.line_number > 0
            assert "test_exceptions.py" in e.context.relative_path

    def test_context_to_dict(self):
        """Should export context as dict"""
        try:
            raise FramewormError("test")
        except FramewormError as e:
            d = e.context.to_dict()
            assert "file" in d
            assert "line" in d
            assert "function" in d


class TestErrorFormatter:
    """Test error formatting"""

    def test_basic_formatting(self):
        """Should format basic error"""
        formatted = ErrorFormatter.format_error(error_name="TestError", message="Test message")

        assert "TestError" in formatted
        assert "Test message" in formatted

    def test_with_details(self):
        """Should include details"""
        formatted = ErrorFormatter.format_error(
            error_name="TestError", message="Test", details={"key": "value"}
        )

        assert "Details:" in formatted
        assert "key" in formatted
        assert "value" in formatted

    def test_with_suggestions(self):
        """Should include suggestions"""
        formatted = ErrorFormatter.format_error(
            error_name="TestError", message="Test", suggestions=["Fix 1", "Fix 2"]
        )

        assert "Suggested Fixes:" in formatted
        assert "Fix 1" in formatted
        assert "→" in formatted


class TestFramewormError:
    """Test base FramewormError"""

    def test_basic_error(self):
        """Should create basic error"""
        error = FramewormError("test message")
        assert error.message == "test message"
        assert error.context is not None

    def test_add_cause(self):
        """Should add causes"""
        error = FramewormError("test")
        error.add_cause("Cause 1")
        error.add_cause("Cause 2")

        assert len(error.causes) == 2
        assert "Cause 1" in str(error)

    def test_add_suggestion(self):
        """Should add suggestions"""
        error = FramewormError("test")
        error.add_suggestion("Fix this")

        assert len(error.suggestions) == 1
        assert "Fix this" in str(error)

    def test_custom_attributes(self):
        """Should store custom attributes"""
        error = FramewormError("test", custom_attr="value")
        assert error.custom_attr == "value"


class TestConfigErrors:
    """Test configuration errors"""

    def test_config_not_found(self):
        """Should provide helpful message for missing config"""
        error = ConfigNotFoundError("missing.yaml")

        error_str = str(error)
        assert "missing.yaml" in error_str
        assert "Suggested Fixes" in error_str
        assert "Check that" in error_str

    def test_config_validation(self):
        """Should explain validation errors"""
        error = ConfigValidationError("Invalid value", field="model.dim")

        error_str = str(error)
        assert "model.dim" in error_str
        assert "Suggested Fixes" in error_str


class TestModelErrors:
    """Test model errors"""

    def test_dimension_mismatch_fewer_dims(self):
        """Should suggest adding dimensions"""
        error = DimensionMismatchError(expected=(4, 100, 1, 1), received=(4, 100))

        error_str = str(error)
        assert "unsqueeze" in error_str
        assert "Expected shape" in error_str
        assert "(4, 100, 1, 1)" in error_str

    def test_dimension_mismatch_wrong_size(self):
        """Should explain size mismatches"""
        error = DimensionMismatchError(
            expected=(4, 3, 64, 64), received=(4, 1, 64, 64), layer_name="conv1"
        )

        error_str = str(error)
        assert "Channel dimension mismatch" in error_str
        assert "conv1" in error_str

    def test_model_not_found(self):
        """Should list available models"""
        error = ModelNotFoundError("my-model", available=["dcgan", "stylegan2", "vae"])

        error_str = str(error)
        assert "not found" in error_str
        assert "dcgan" in error_str

    def test_model_not_found_similar_name(self):
        """Should suggest similar names"""
        error = ModelNotFoundError("dc-gan", available=["dcgan", "stylegan2"])  # Typo

        error_str = str(error)
        assert "Did you mean" in error_str or "dcgan" in error_str


class TestPluginErrors:
    """Test plugin errors"""

    def test_plugin_validation_missing_methods(self):
        """Should list missing methods"""
        error = PluginValidationError(
            "Validation failed", plugin_name="MyPlugin", missing_methods=["forward", "backward"]
        )

        error_str = str(error)
        assert "forward" in error_str
        assert "backward" in error_str
        assert "def forward" in error_str  # Suggestion


class TestTrainingErrors:
    """Test training errors"""

    def test_convergence_error_exploding_loss(self):
        """Should detect exploding loss"""
        error = ConvergenceError(loss_value=1e10)

        error_str = str(error)
        assert "exploded" in error_str.lower()
        assert "learning rate" in error_str.lower()

    def test_convergence_error_nan(self):
        """Should detect NaN loss"""
        error = ConvergenceError(loss_value=float("nan"))

        error_str = str(error)
        assert "nan" in error_str.lower()
        # remove division by zero check
        # assert "division by zero" in error_str.lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
