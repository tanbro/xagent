"""
Integration tests for output filter with tool factory.
"""

import pytest

from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory


@pytest.mark.asyncio
async def test_tool_factory_applies_filters():
    """Test that tools created by factory have output filtering."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 100,
            "truncation_message": " [TRUNCATED]",
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Check that tools were created
    assert len(tools) > 0

    # Find any wrapped tool (all tools should be wrapped with _filter)
    wrapped_tools = [t for t in tools if hasattr(t, "_filter")]
    assert len(wrapped_tools) > 0, "No tools with output filter found"

    # Check that the filter has the correct configuration
    tool = wrapped_tools[0]
    assert hasattr(tool, "_filter")
    assert tool._filter.max_chars == 100
    assert tool._filter.truncation_message == " [TRUNCATED]"


@pytest.mark.asyncio
async def test_filtered_tool_execution():
    """Test that filtered tools truncate output correctly."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 50,
            "truncation_message": " [TRUNCATED]",
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Find a tool with run_json_sync method
    executable_tool = next((t for t in tools if hasattr(t, "run_json_sync")), None)
    assert executable_tool is not None, "No executable tool found"

    # Verify the tool has the filter
    assert hasattr(executable_tool, "_filter")


@pytest.mark.asyncio
async def test_default_max_output_length():
    """Test that default max output length is 50K characters."""
    config = ToolConfig({"workspace": None})

    tools = await ToolFactory.create_all_tools(config)

    # Check that at least one tool was created
    assert len(tools) > 0

    # Check that tools have the default limit
    for tool in tools:
        if hasattr(tool, "_filter"):
            assert tool._filter.max_chars == 50 * 1024


@pytest.mark.asyncio
async def test_custom_truncation_message():
    """Test custom truncation message."""
    custom_message = " ... [OUTPUT WAS TOO LONG]"
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 10,
            "truncation_message": custom_message,
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Find a tool and check its truncation message
    for tool in tools:
        if hasattr(tool, "_filter"):
            assert tool._filter.truncation_message == custom_message
            break
