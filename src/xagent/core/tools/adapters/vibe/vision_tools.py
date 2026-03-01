"""Vision tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_vision_tools(config: "BaseToolConfig") -> List[Any]:
    """Create vision understanding tools."""
    vision_model = config.get_vision_model()
    if not vision_model:
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())

    try:
        from .vision_tool import get_vision_tool

        return get_vision_tool(vision_model=vision_model, workspace=workspace)
    except Exception as e:
        logger.warning(f"Failed to create vision tools: {e}")
        return []
