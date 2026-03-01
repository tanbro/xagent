# Import tool modules to trigger @register_tool decorator registration
# This enables auto-discovery of tools without manual factory.py modifications
from .agent_tools import create_agent_tools  # noqa: F401
from .base import Tool as Tool
from .base import ToolMetadata as ToolMetadata
from .base import ToolVisibility as ToolVisibility
from .basic_tools import create_basic_tools  # noqa: F401
from .browser_tools import create_browser_tools  # noqa: F401
from .file_tools import create_file_tools  # noqa: F401
from .image_tools import create_image_tools_from_config  # noqa: F401
from .knowledge_tools import create_knowledge_tools  # noqa: F401
from .mcp_tools import create_mcp_tools  # noqa: F401
from .pptx_tool import create_pptx_tool  # noqa: F401
from .special_image_tools import create_special_image_tools  # noqa: F401
from .vision_tools import create_vision_tools  # noqa: F401

# # Export basic file tools (unsafe, only for backward compatibility)
# from .file_tool import BASIC_FILE_TOOLS

# # Export workspace file tool creation function
# from .workspace_file_tool import (
#     create_workspace_file_tools as create_workspace_file_tools,
# )

# # FILE_TOOLS now points to basic tools (not recommended)
# FILE_TOOLS = BASIC_FILE_TOOLS
