"""Image generation tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_image_tools_from_config(config: "BaseToolConfig") -> List[Any]:
    """Create image generation tools from configuration."""
    image_models = config.get_image_models()
    if not image_models:
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        from .image_tool import create_image_tool

        default_generate_model = config.get_image_generate_model()
        default_edit_model = config.get_image_edit_model()

        return create_image_tool(
            image_models,
            workspace=workspace,
            default_generate_model=default_generate_model,
            default_edit_model=default_edit_model,
        )
    except Exception as e:
        logger.warning(f"Failed to create image tools: {e}")
        return []
