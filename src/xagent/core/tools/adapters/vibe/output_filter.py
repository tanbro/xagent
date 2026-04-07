"""
Tool Output Value Filtering Module

Provides unified output length limiting for all tools.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default max output length (50K chars, ~12K tokens, suitable for most LLMs)
DEFAULT_MAX_OUTPUT_LENGTH = 50 * 1024

# Environment variable name for overriding default max output length
_ENV_MAX_OUTPUT_LENGTH = "XAGENT_TOOL_MAX_OUTPUT_LENGTH"


def _get_default_max_output_length() -> int:
    """Get default max output length from environment variable or fallback to DEFAULT."""
    env_value = os.getenv(_ENV_MAX_OUTPUT_LENGTH)
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logger.warning(
                f"Invalid value for {_ENV_MAX_OUTPUT_LENGTH}: '{env_value}'. "
                f"Using default: {DEFAULT_MAX_OUTPUT_LENGTH}"
            )
    return DEFAULT_MAX_OUTPUT_LENGTH


# Default truncation message
DEFAULT_TRUNCATION_MESSAGE = "\n\n[OUTPUT TRUNCATED: exceeded maximum length]"


class OutputValueFilter:
    """Filter and truncate tool return values based on character limit."""

    def __init__(
        self,
        max_chars: int | None = None,
        truncation_message: str = DEFAULT_TRUNCATION_MESSAGE,
    ):
        """
        Initialize output filter.

        Args:
            max_chars: Maximum output length in characters. If None, reads from
                      XAGENT_TOOL_MAX_OUTPUT_LENGTH env var or uses DEFAULT_MAX_OUTPUT_LENGTH.
            truncation_message: Message to append when output is truncated
        """
        if max_chars is None:
            max_chars = _get_default_max_output_length()
        self.max_chars = max_chars
        self.truncation_message = truncation_message
        """
        Initialize output filter.

        Args:
            max_chars: Maximum output length in characters
            truncation_message: Message to append when output is truncated
        """
        self.max_chars = max_chars
        self.truncation_message = truncation_message

    def filter(self, value: Any, tool_name: str = "unknown") -> Any:
        """
        Filter return value based on character limit.

        Args:
            value: Return value to filter
            tool_name: Name of the tool (for logging)

        Returns:
            Filtered value (may be truncated)
        """
        if value is None:
            return None

        # Handle strings
        if isinstance(value, str):
            return self._filter_string(value, tool_name)

        # Handle dicts - recursively filter each string value
        elif isinstance(value, dict):
            return {k: self.filter(v, tool_name) for k, v in value.items()}

        # Handle lists - recursively filter each element
        elif isinstance(value, list):
            return [self.filter(item, tool_name) for item in value]

        # Handle Pydantic models
        elif hasattr(value, "model_dump"):
            filtered_dict = self.filter(value.model_dump(), tool_name)
            try:
                return value.__class__(**filtered_dict)
            except Exception:
                return filtered_dict

        # Handle primitives (bool, int, float, etc.) - return as-is
        elif isinstance(value, (bool, int, float)):
            return value

        # Handle other types by converting to string (as last resort)
        else:
            str_value = str(value)
            return self._filter_string(str_value, tool_name)

    def _filter_string(self, value: str, tool_name: str) -> str:
        """
        Filter a string value.

        Args:
            value: String value to filter
            tool_name: Name of the tool (for logging)

        Returns:
            Filtered string value
        """
        if len(value) <= self.max_chars:
            return value

        truncated = value[: self.max_chars]
        logger.warning(
            f"Tool '{tool_name}' output truncated: "
            f"{len(value)} -> {len(truncated)} characters"
        )
        return truncated + self.truncation_message


def create_output_filter(
    max_chars: int | None = None,
    truncation_message: str = DEFAULT_TRUNCATION_MESSAGE,
) -> OutputValueFilter:
    """
    Create an output value filter.

    Args:
        max_chars: Maximum output length in characters. If None, reads from
                  XAGENT_TOOL_MAX_OUTPUT_LENGTH env var or uses DEFAULT_MAX_OUTPUT_LENGTH.
        truncation_message: Message to append when output is truncated

    Returns:
        Configured output filter
    """
    return OutputValueFilter(max_chars, truncation_message)
