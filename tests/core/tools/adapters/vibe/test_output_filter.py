"""
Unit tests for output filter module.
"""

from xagent.core.tools.adapters.vibe.output_filter import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    OutputValueFilter,
    create_output_filter,
)


def test_string_within_limit():
    """Test that strings within limit are not modified."""
    filter = OutputValueFilter(max_chars=100)
    result = filter.filter("a" * 50, "test_tool")
    assert len(result) == 50
    assert result == "a" * 50


def test_string_exceeds_limit():
    """Test that strings exceeding limit are truncated."""
    filter = OutputValueFilter(max_chars=100, truncation_message=" [TRUNCATED]")
    result = filter.filter("a" * 200, "test_tool")
    assert len(result) == 100 + len(" [TRUNCATED]")
    assert result.endswith(" [TRUNCATED]")
    assert result.startswith("a" * 100)


def test_dict_preservation():
    """Test that dict structure is preserved."""
    filter = OutputValueFilter(max_chars=50, truncation_message=" [TRUNCATED]")
    data = {"short": "ok", "long": "a" * 100, "nested": {"value": "b" * 100}}
    result = filter.filter(data, "test_tool")
    assert result["short"] == "ok"
    # After truncation, length is max_chars + truncation_message length
    assert len(result["long"]) == 50 + len(" [TRUNCATED]")
    assert len(result["nested"]["value"]) == 50 + len(" [TRUNCATED]")


def test_dict_with_non_string_values():
    """Test that dict with non-string values is handled correctly."""
    filter = OutputValueFilter(max_chars=50, truncation_message=" [TRUNCATED]")
    data = {
        "number": 42,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "long_string": "a" * 100,
    }
    result = filter.filter(data, "test_tool")
    # Primitive types are preserved (better design)
    assert result["number"] == 42
    assert result["boolean"] is True
    assert result["none"] is None
    # List elements are also preserved
    assert result["list"] == [1, 2, 3]
    assert len(result["long_string"]) == 50 + len(" [TRUNCATED]")


def test_list_filtering():
    """Test that list items are filtered."""
    filter = OutputValueFilter(max_chars=50, truncation_message=" [TRUNCATED]")
    data = ["short", "a" * 100, {"key": "b" * 100}]
    result = filter.filter(data, "test_tool")
    assert result[0] == "short"
    # After truncation, length is max_chars + truncation_message length
    assert len(result[1]) == 50 + len(" [TRUNCATED]")
    assert len(result[2]["key"]) == 50 + len(" [TRUNCATED]")


def test_none_passthrough():
    """Test that None values pass through."""
    filter = OutputValueFilter()
    assert filter.filter(None, "test_tool") is None


def test_empty_string():
    """Test that empty strings pass through."""
    filter = OutputValueFilter(max_chars=100)
    result = filter.filter("", "test_tool")
    assert result == ""


def test_exact_limit():
    """Test that strings at exact limit are not modified."""
    filter = OutputValueFilter(max_chars=100)
    result = filter.filter("a" * 100, "test_tool")
    assert len(result) == 100
    assert not result.endswith("[TRUNCATED]")


def test_default_limit():
    """Test default limit is 50K characters."""
    filter = create_output_filter()
    assert filter.max_chars == DEFAULT_MAX_OUTPUT_LENGTH
    assert DEFAULT_MAX_OUTPUT_LENGTH == 50 * 1024


def test_custom_truncation_message():
    """Test custom truncation message."""
    filter = OutputValueFilter(max_chars=10, truncation_message=" ... [CUT]")
    result = filter.filter("a" * 100, "test_tool")
    assert result.endswith(" ... [CUT]")
    assert len(result) == 10 + len(" ... [CUT]")


def test_unicode_string():
    """Test that unicode strings are handled correctly."""
    filter = OutputValueFilter(max_chars=20)
    result = filter.filter("你好世界" * 10, "test_tool")
    # Each Chinese character is counted as 1 character
    assert len(result) <= 20 + len(filter.truncation_message)


def test_nested_structures():
    """Test deeply nested structures."""
    filter = OutputValueFilter(max_chars=10, truncation_message=" [CUT]")
    data = {"level1": {"level2": {"level3": {"value": "a" * 100}}}}
    result = filter.filter(data, "test_tool")
    # After truncation, length is max_chars + truncation_message length
    assert len(result["level1"]["level2"]["level3"]["value"]) == 10 + len(" [CUT]")


def test_list_of_dicts():
    """Test list of dictionaries."""
    filter = OutputValueFilter(max_chars=20, truncation_message=" [CUT]")
    data = [
        {"name": "short", "value": "ok"},
        {"name": "long", "value": "a" * 100},
    ]
    result = filter.filter(data, "test_tool")
    assert result[0]["name"] == "short"
    assert result[0]["value"] == "ok"
    # After truncation, length is max_chars + truncation_message length
    assert len(result[1]["value"]) == 20 + len(" [CUT]")


def test_number_conversion():
    """Test that numbers are preserved (not converted to strings)."""
    filter = OutputValueFilter(max_chars=5, truncation_message=" [CUT]")
    result = filter.filter(1234567890, "test_tool")
    # Numbers are preserved as-is (no conversion)
    assert result == 1234567890
    assert isinstance(result, int)


def test_boolean_conversion():
    """Test that booleans are preserved (not converted to strings)."""
    filter = OutputValueFilter(max_chars=10)
    result = filter.filter(True, "test_tool")
    # Booleans are preserved as-is
    assert result is True


def test_zero_max_chars():
    """Test edge case of zero max_chars."""
    filter = OutputValueFilter(max_chars=0)
    result = filter.filter("a" * 100, "test_tool")
    assert result == filter.truncation_message


def test_small_limit():
    """Test very small limit."""
    filter = OutputValueFilter(max_chars=5, truncation_message=" [CUT]")
    result = filter.filter("a" * 100, "test_tool")
    assert result.startswith("a" * 5)
    assert result.endswith(" [CUT]")


def test_env_variable_default():
    """Test that environment variable is used as default."""
    import os

    from xagent.core.tools.adapters.vibe.output_filter import (
        _ENV_MAX_OUTPUT_LENGTH,
        _get_default_max_output_length,
    )

    # Save original value
    original_value = os.getenv(_ENV_MAX_OUTPUT_LENGTH)

    try:
        # Test with valid env var
        os.environ[_ENV_MAX_OUTPUT_LENGTH] = "100000"
        assert _get_default_max_output_length() == 100000

        # Test with invalid env var (should fallback to default)
        os.environ[_ENV_MAX_OUTPUT_LENGTH] = "invalid"
        result = _get_default_max_output_length()
        assert result == 50 * 1024  # Fallback to default

        # Test without env var (should use default)
        os.environ.pop(_ENV_MAX_OUTPUT_LENGTH, None)
        assert _get_default_max_output_length() == 50 * 1024
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop(_ENV_MAX_OUTPUT_LENGTH, None)
        else:
            os.environ[_ENV_MAX_OUTPUT_LENGTH] = original_value


def test_filter_uses_env_var_when_none():
    """Test that filter uses env var when max_chars is None."""
    import os

    from xagent.core.tools.adapters.vibe.output_filter import (
        _ENV_MAX_OUTPUT_LENGTH,
        OutputValueFilter,
    )

    # Save original value
    original_value = os.getenv(_ENV_MAX_OUTPUT_LENGTH)

    try:
        # Set env var
        os.environ[_ENV_MAX_OUTPUT_LENGTH] = "25"
        filter = OutputValueFilter()  # max_chars=None, should use env var
        assert filter.max_chars == 25
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop(_ENV_MAX_OUTPUT_LENGTH, None)
        else:
            os.environ[_ENV_MAX_OUTPUT_LENGTH] = original_value
