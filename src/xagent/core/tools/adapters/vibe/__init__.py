from .base import Tool as Tool
from .base import ToolMetadata as ToolMetadata
from .base import ToolVisibility as ToolVisibility

# Import tool modules to trigger @register_tool decorator registration
# This enables auto-discovery of tools without manual factory.py modifications
from .image_tool import create_image_tool  # noqa: F401
from .pptx_tool import create_pptx_tool  # noqa: F401

# # Export basic file tools (unsafe, only for backward compatibility)
# from .file_tool import BASIC_FILE_TOOLS

# # Export workspace file tool creation function
# from .workspace_file_tool import (
#     create_workspace_file_tools as create_workspace_file_tools,
# )

# # FILE_TOOLS now points to basic tools (not recommended)
# FILE_TOOLS = BASIC_FILE_TOOLS
