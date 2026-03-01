"""Agent tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import register_tool

if TYPE_CHECKING:
    from xagent.web.tools.config import WebToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_agent_tools(config: "WebToolConfig") -> List[Any]:
    """Create tools from published agents."""
    if not config.get_enable_agent_tools():
        return []

    try:
        from .factory import ToolFactory

        db = config.get_db()
        user_id = config.get_user_id()
        if not user_id:
            return []

        return ToolFactory._create_agent_tools(
            db=db,
            user_id=user_id,
            task_id=config.get_task_id(),
            config=config,
        )
    except Exception as e:
        logger.warning(f"Failed to create agent tools: {e}")
        return []
