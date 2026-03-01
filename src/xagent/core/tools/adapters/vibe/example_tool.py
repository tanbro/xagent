"""
Example: How to create and register custom tools using @register_tool decorator

This demonstrates the new auto-discovery mechanism for tools.
"""

# mypy: ignore-errors
# ruff: noqa

import logging
from typing import TYPE_CHECKING, List

from ..base import ToolCategory
from ..config import BaseToolConfig
from ..factory import ToolFactory, register_tool
from ..function import FunctionTool

logger = logging.getLogger(__name__)


@register_tool
async def create_my_custom_tools(config: BaseToolConfig) -> List[FunctionTool]:
    """
    Create custom tools for my specific use case.

    This function is automatically discovered and called during tool creation
    thanks to the @register_tool decorator.

    No need to modify factory.py!

    Args:
        config: Tool configuration with workspace, user settings, etc.

    Returns:
        List of tool instances
    """
    # Get workspace from config
    _workspace = ToolFactory._create_workspace(config.get_workspace_config())

    # Create your custom tools
    tools = []

    # Example: A simple calculator tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    tools.append(
        FunctionTool(
            calculate,
            name="calculate",
            description="Safely evaluate mathematical expressions",
            category=ToolCategory.BASIC,
        )
    )

    logger.info(f"Created custom tools: {[t.name for t in tools]}")
    return tools


# For demonstration, you could also create tools that depend on configuration
@register_tool
async def create_weather_tools(config: BaseToolConfig) -> List[FunctionTool]:
    """
    Example: Tools that use configuration from database/API keys.

    The config parameter gives you access to:
    - config.get_db(): Database session
    - config.get_user_id(): Current user ID
    - config.get_workspace_config(): Workspace settings
    - etc.
    """
    # Example: Only create tools if certain conditions are met
    _user_id = config.get_user_id()

    # You could check database for user-specific settings
    # db = config.get_db()
    # enabled = db.query(UserSettings).filter_by(user_id=user_id).first()

    # For this example, just return empty list
    return []
