"""
Workspace-bound file tools for xagent

This module provides file tools that are bound to specific workspace instances.
Each tool instance operates within its designated workspace only.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from xagent.core.workspace import TaskWorkspace

from ...core.workspace_file_tool import FileInfo, WorkspaceFileOperations
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class FileTool(FunctionTool):
    """Base class for file tools with FILE category."""

    category = ToolCategory.FILE


class WorkspaceFileTools(WorkspaceFileOperations):
    """
    Workspace-bound file tools.

    Each instance is bound to a specific workspace and provides
    file operations restricted to that workspace.
    """

    def __init__(self, workspace: TaskWorkspace, skills_root: str = ".xagent/skills"):
        """
        Initialize with workspace binding.

        Args:
            workspace: The workspace to bind to
            skills_root: Root directory for skills (default: .xagent/skills)
        """
        self.inner = WorkspaceFileOperations(workspace)
        self.workspace = workspace
        self.skills_root = skills_root

    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file content in workspace"""
        return self.inner.read_file(file_path, encoding)

    def write_file(
        self,
        file_path: str | None = None,
        content: str | None = None,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        filename: str | None = None,
    ) -> bool:
        """Write file content in workspace"""
        if file_path is None:
            file_path = filename
        if not file_path:
            raise ValueError("file_path is required")
        if content is None:
            raise ValueError("content is required")
        return self.inner.write_file(file_path, content, encoding, create_dirs)

    def append_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> bool:
        """Append content to file in workspace"""
        return self.inner.append_file(file_path, content, encoding, create_dirs)

    def delete_file(self, file_path: str) -> bool:
        """Delete file in workspace"""
        return self.inner.delete_file(file_path)

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in workspace"""
        return self.inner.file_exists(file_path)

    def list_files(
        self,
        directory_path: str = ".",
        show_hidden: bool = False,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """List files in workspace directory (defaults to all directories)"""
        return self.inner.list_files(directory_path, show_hidden, recursive)

    def create_directory(self, directory_path: str, parents: bool = True) -> bool:
        """Create directory in workspace"""
        return self.inner.create_directory(directory_path, parents)

    def get_file_info(self, file_path: str) -> FileInfo:
        """Get detailed file information in workspace"""
        return self.inner.get_file_info(file_path)

    def read_json_file(self, file_path: str, encoding: str = "utf-8") -> Any:
        """Read JSON file in workspace"""
        return self.inner.read_json_file(file_path, encoding)

    def write_json_file(
        self,
        file_path: str,
        data: Dict[str, Any],
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> bool:
        """Write JSON file in workspace"""
        return self.inner.write_json_file(file_path, data, encoding, indent)

    def read_csv_file(
        self, file_path: str, encoding: str = "utf-8", delimiter: str = ","
    ) -> List[Dict[str, str]]:
        """Read CSV file in workspace"""
        return self.inner.read_csv_file(file_path, encoding, delimiter)

    def write_csv_file(
        self,
        file_path: str,
        data: List[Dict[str, str]],
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> bool:
        """Write CSV file in workspace"""
        return self.inner.write_csv_file(file_path, data, encoding, delimiter)

    def get_workspace_output_files(self) -> Dict[str, Any]:
        """Get output file list from current workspace"""
        return self.inner.get_workspace_output_files()

    def read_skill_file(self, skill_name: str, file_path: str) -> str:
        """
        Read reference/resource files from a skill directory.

        This allows accessing skill documentation (schema files, references, etc.)
        that are stored in the skill directory outside of the workspace.

        Args:
            skill_name: Name of the skill
            file_path: Relative path within the skill

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If the skill file doesn't exist
        """

        skill_dir = Path(self.skills_root) / skill_name
        full_path = skill_dir / file_path

        if not full_path.exists():
            # List available files for better error message
            available_files = []
            if skill_dir.exists():
                for f in skill_dir.rglob("*"):
                    if f.is_file():
                        available_files.append(str(f.relative_to(skill_dir)))

            raise FileNotFoundError(
                f"File not found: '{file_path}' in skill '{skill_name}'\n"
                f"Use list_skill_files('{skill_name}') to see available files."
            )

        return full_path.read_text(encoding="utf-8")

    def list_skill_files(
        self,
        skill_name: str,
        directory_path: str = ".",
        show_hidden: bool = False,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        List files in a skill directory.

        This allows discovering what reference/resource files are available in a skill.
        Behavior matches list_files() for consistency.

        Args:
            skill_name: Name of the skill
            directory_path: Optional subdirectory path relative to the skill directory (default: '.' for all files)
            show_hidden: Whether to show hidden files (default: False)
            recursive: Whether to list recursively (default: True)

        Returns:
            Dict with keys:
            - files: List of FileInfo objects (as dicts), with 'path' being relative to skill directory
            - total_count: Number of files
            - current_path: Full path to the searched directory
            - directory: The skill name

        Raises:
            FileNotFoundError: If the skill directory doesn't exist
        """
        skill_dir = Path(self.skills_root) / skill_name

        if not skill_dir.exists():
            raise FileNotFoundError(
                f"Skill not found: '{skill_name}'\n"
                f"Please check the skill name is correct."
            )

        # Determine search path
        if directory_path == ".":
            search_path = skill_dir
        else:
            search_path = skill_dir / directory_path
            if not search_path.exists():
                raise FileNotFoundError(
                    f"Directory not found: '{directory_path}' in skill '{skill_name}'"
                )

        files = []

        def scan_directory(current_path: Path) -> None:
            try:
                for item in current_path.iterdir():
                    if not show_hidden and item.name.startswith("."):
                        continue

                    stat = item.stat()
                    # Use relative path as the "path" field for compatibility
                    rel_path = item.relative_to(skill_dir)
                    file_info = FileInfo(
                        name=item.name,
                        path=str(rel_path),
                        size=stat.st_size,
                        is_file=item.is_file(),
                        is_dir=item.is_dir(),
                        modified_time=stat.st_mtime,
                    )
                    files.append(file_info)

                    if recursive and item.is_dir():
                        scan_directory(item)

            except PermissionError:
                pass

        scan_directory(search_path)

        # Return relative path as current_path to avoid exposing system paths
        relative_search_path = (
            search_path.relative_to(skill_dir) if search_path != skill_dir else "."
        )

        return {
            "files": [file.dict() for file in files],
            "total_count": len(files),
            "current_path": str(relative_search_path),
            "directory": skill_name,
        }

    def get_tools(self) -> List[FunctionTool]:
        """Get all tool instances"""
        return [
            FileTool(
                self.read_file,
                name="read_file",
                description="Read file content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.write_file,
                name="write_file",
                description="Write file content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.\n\nImportant: For HTML files, when referencing resources in the same directory (CSS, JS, images), only use filenames (e.g., '1.png'), not absolute paths (e.g., '/uploads/xxx/1.png'). All files are in the workspace, and browsers will automatically resolve relative paths.",
            ),
            FileTool(
                self.append_file,
                name="append_file",
                description="Append content to file in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.delete_file,
                name="delete_file",
                description="Delete file in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.list_files,
                name="list_files",
                description="List files in workspace directory (defaults to all directories including input, output, temp. Can also specify specific directory like list_files('input'))",
            ),
            FileTool(
                self.create_directory,
                name="create_directory",
                description="Create directory in workspace",
            ),
            FileTool(
                self.file_exists,
                name="file_exists",
                description="Check if file exists in workspace",
            ),
            FileTool(
                self.get_file_info,
                name="get_file_info",
                description="Get detailed file information in workspace",
            ),
            FileTool(
                self.read_json_file,
                name="read_json_file",
                description="Read JSON file in workspace",
            ),
            FileTool(
                self.write_json_file,
                name="write_json_file",
                description="Write JSON file in workspace",
            ),
            FileTool(
                self.read_csv_file,
                name="read_csv_file",
                description="Read CSV file in workspace",
            ),
            FileTool(
                self.write_csv_file,
                name="write_csv_file",
                description="Write CSV file in workspace",
            ),
            FileTool(
                self.get_workspace_output_files,
                name="get_workspace_output_files",
                description="Get output file list from current workspace",
            ),
            FileTool(
                self.edit_file,
                name="edit_file",
                description="Precisely edit file content in workspace, supporting multiple edit operations based on line numbers and pattern matching. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.find_and_replace,
                name="find_and_replace",
                description="Convenience function to find and replace text content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FunctionTool(
                self.read_skill_file,
                name="read_skill_file",
                tags=["skill"],
                description="""
    Read reference/resource files from a skill directory.

    Use this to access skill documentation like schema files, reference guides, and resources stored in the skill directory.

    Args:
        skill_name: Name of the skill
        file_path: Relative path within the skill directory

    Note: If you don't know the exact file path, use list_skill_files() first to see available files.
""",
            ),
            FunctionTool(
                self.list_skill_files,
                name="list_skill_files",
                tags=["skill"],
                description="""
    List files in a skill directory.

    Use this to discover what reference/resource files are available in a skill before reading them.
    Returns FileInfo objects with name, path, size, is_file, is_dir, and modified_time.

    Args:
        skill_name: Name of the skill
        directory_path: Optional subdirectory path relative to skill directory (default: '.' for all files)
        show_hidden: Whether to show hidden files (default: False)
        recursive: Whether to list recursively (default: True)

    Returns:
        Dict with keys:
        - files: List of file info objects (path is relative to skill directory)
        - total_count: Number of files
        - current_path: Relative path to the searched directory (within the skill)
        - directory: The skill name

    Note: All paths are relative to the skill directory.
""",
            ),
        ]


def create_workspace_file_tools(
    workspace: TaskWorkspace, skills_root: str = ".xagent/skills"
) -> List[FunctionTool]:
    """
    Create list of file tools bound to specified workspace

    Args:
        workspace: Workspace to bind to
        skills_root: Root directory for skills (default: .xagent/skills)

    Returns:
        List of tool instances
    """
    tools_instance = WorkspaceFileTools(workspace, skills_root=skills_root)
    return tools_instance.get_tools()


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool
async def create_file_tools(config: "BaseToolConfig") -> List[Any]:
    """Create workspace-bound file tools."""
    if not config.get_file_tools_enabled():
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_workspace_file_tools(workspace)
    except Exception as e:
        logger.warning(f"Failed to create file tools: {e}")
        return []
