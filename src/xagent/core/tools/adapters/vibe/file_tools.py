"""File tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_file_tools(config: "BaseToolConfig") -> List[Any]:
    """Create workspace-bound file tools."""
    if not config.get_file_tools_enabled():
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        from .workspace_file_tool import create_workspace_file_tools

        return create_workspace_file_tools(workspace)
    except Exception as e:
        logger.warning(f"Failed to create file tools: {e}")
        return []
