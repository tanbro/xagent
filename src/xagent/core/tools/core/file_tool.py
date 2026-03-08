"""
Basic file tool module for xagent

This module provides basic file operations, excluding workspace-related functionality.
For workspace-related file operations, use the workspace_file_tool.py module.
"""

import csv
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

# Constants for file safety
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB default limit
PREVIEW_LINES = 50  # Number of lines to show in preview for large files
BINARY_CHECK_BYTES = 1024  # Number of bytes to check for binary content


class FileInfo(BaseModel):
    """File information model"""

    name: str
    path: str
    size: int
    is_file: bool
    is_dir: bool
    modified_time: float
    encoding: Optional[str] = None


class ListFilesResult(BaseModel):
    """List files result model"""

    files: List[FileInfo]
    total_count: int
    current_path: str


class EditOperation(BaseModel):
    """Edit operation model"""

    operation_type: str  # 'replace', 'insert', 'delete'
    line_number: Optional[int] = None  # Line number (1-based)
    pattern: Optional[str] = None  # Pattern matching
    content: Optional[str] = None  # New content


class EditResult(BaseModel):
    """Edit result model"""

    success: bool
    message: str
    lines_changed: int = 0
    preview: Optional[str] = None


def _is_binary_by_mime(file_path: str) -> bool:
    """Check if file is binary using MIME type detection.

    Args:
        file_path: Path to the file

    Returns:
        True if MIME type is not text/, False otherwise
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Unknown MIME type - don't treat as binary yet, let content check decide
        return False
    if mime_type.startswith("text/"):
        return False
    if mime_type in ["application/json", "application/javascript"]:
        # JSON and JS are text files
        return False
    return True


def _is_binary_by_content(chunk: bytes) -> bool:
    """Check if bytes chunk appears to be binary.

    Args:
        chunk: Bytes chunk to examine

    Returns:
        True if chunk appears to be binary, False otherwise
    """
    # Check 1: Null bytes (strong indicator of binary files)
    if b"\x00" in chunk:
        return True

    # Check 2: High byte ratio
    # If >30% of bytes are >0x7F, likely binary
    if len(chunk) > 0:
        high_bytes = sum(1 for b in chunk if b > 0x7F)
        if high_bytes / len(chunk) > 0.3:
            return True

    return False


def _is_binary_file(file_path: str, chunk: bytes) -> bool:
    """Comprehensive binary file detection using multiple checks.

    Args:
        file_path: Path to the file (for MIME type check)
        chunk: First bytes chunk of the file

    Returns:
        True if file is binary, False if text
    """
    # Layer 1: MIME type check (fast)
    if _is_binary_by_mime(file_path):
        return True

    # Layer 2: Content check (accurate)
    if _is_binary_by_content(chunk):
        return True

    return False


def _get_file_preview_from_content(
    content: str, file_size: int, max_lines: int = PREVIEW_LINES
) -> str:
    """Generate a preview from decoded content string.

    Args:
        content: Decoded file content
        file_size: File size in bytes
        max_lines: Number of lines to show from head and tail

    Returns:
        Preview string with file metadata and content preview
    """
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    # Extract head and tail lines
    head_lines = lines[:max_lines]
    head_lines_str = "".join(head_lines).rstrip("\n")

    tail_lines = []
    if total_lines > max_lines * 2:
        tail_lines = lines[-max_lines:]
        tail_lines_str = "".join(tail_lines).rstrip("\n")
    else:
        tail_lines_str = ""

    # Build preview
    size_mb = file_size / 1024 / 1024
    preview_parts = [
        "# File too large for complete reading",
        f"# Size: {file_size:,} bytes ({size_mb:.2f} MB)",
        f"# Lines: {total_lines:,}",
        f"# Showing preview (first {len(head_lines)} and last {len(tail_lines)} lines):",
        "",
        head_lines_str,
    ]

    if tail_lines:
        omitted_lines = total_lines - len(head_lines) - len(tail_lines)
        preview_parts.append(f"\n# ... ({omitted_lines:,} lines omitted) ...\n")
        preview_parts.append(tail_lines_str)

    return "\n".join(preview_parts)


def read_file(
    file_path: str, encoding: str = "utf-8", max_size: int = DEFAULT_MAX_FILE_SIZE
) -> str:
    """
    Read the content of a text file.

    This function can only read text files. Binary files (like images, PDFs, executables) will be rejected.
    For files larger than 1MB, a preview with the first and last 50 lines will be returned instead.

    Args:
        file_path: Path to the text file you want to read
        encoding: Text encoding format (default: utf-8). Most text files use utf-8.
        max_size: Maximum file size in bytes (default: 1,048,576 = 1MB).

    Returns:
        - For small files (≤1MB): The complete file content as text
        - For large files (>1MB): A preview showing the first and last 50 lines with file metadata
    """
    # Step 1: Quick MIME check (no file I/O)
    if _is_binary_by_mime(file_path):
        raise ValueError(
            f"Cannot read binary file: {file_path}. This tool only supports text files."
        )

    # Step 2: Open file ONCE in binary mode for all checks
    with open(file_path, "rb") as f:
        # Read first chunk for safety checks
        chunk = f.read(8192)

        # Binary file detection (MIME + content checks)
        if _is_binary_file(file_path, chunk):
            raise ValueError(
                f"Cannot read binary file: {file_path}. This tool only supports text files."
            )

        # Get file size
        file_size = os.path.getsize(file_path)

        # Check if file exceeds size limit
        if file_size > max_size:
            # File is too large - read all content and generate preview
            f.seek(0)  # Seek back to beginning
            content = f.read().decode(encoding, errors="ignore")
            return _get_file_preview_from_content(content, file_size, PREVIEW_LINES)

        # File is within size limit - read and decode full content
        f.seek(0)  # Seek back to beginning
        return f.read().decode(encoding)


def write_file(
    file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> bool:
    """
    Write string to text file

    Args:
        file_path: File path
        content: Content to write
        encoding: File encoding, defaults to utf-8
        create_dirs: Whether to auto-create directories, defaults to True

    Returns:
        True if write succeeded

    Raises:
        OSError: File operation error
    """
    if not file_path:
        raise ValueError("file_path is required")
    if not content:
        raise ValueError("content is required")
    if create_dirs:
        dir_path = os.path.dirname(file_path)
        if dir_path:  # Only create directory if directory path is not empty
            os.makedirs(dir_path, exist_ok=True)
        # If dir_path is empty (e.g., 'hello_world.html'), file is in current directory, no need to create directory

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)
    return True


def append_file(
    file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> bool:
    """
    Append string to text file

    Args:
        file_path: File path
        content: Content to append
        encoding: File encoding, defaults to utf-8
        create_dirs: Whether to auto-create directories, defaults to True

    Returns:
        True if append succeeded

    Raises:
        OSError: File operation error
    """
    # Handle relative paths, use current directory if file path has no directory part
    if create_dirs:
        dir_path = os.path.dirname(file_path)
        if dir_path:  # Only create directory if directory path is not empty
            os.makedirs(dir_path, exist_ok=True)
        # If dir_path is empty (e.g., 'hello_world.html'), file is in current directory, no need to create directory

    with open(file_path, "a", encoding=encoding) as f:
        f.write(content)
    return True


def delete_file(file_path: str) -> bool:
    """
    Delete file

    Args:
        file_path: File path

    Returns:
        True if deletion succeeded

    Raises:
        FileNotFoundError: File doesn't exist
        OSError: Deletion failed
    """
    os.remove(file_path)
    return True


def list_files(
    directory_path: str = ".", show_hidden: bool = False, recursive: bool = False
) -> ListFilesResult:
    """
    List files in directory

    Args:
        directory_path: Directory path, defaults to current directory
        show_hidden: Whether to show hidden files, defaults to False
        recursive: Whether to recursively list subdirectories, defaults to False

    Returns:
        ListFilesResult object containing file information
    """
    files = []
    path = Path(directory_path)

    def scan_directory(current_path: Path, is_root: bool = True) -> None:
        try:
            for item in current_path.iterdir():
                # Skip hidden files
                if not show_hidden and item.name.startswith("."):
                    continue

                stat = item.stat()
                file_info = FileInfo(
                    name=item.name,
                    path=str(item.absolute()),
                    size=stat.st_size,
                    is_file=item.is_file(),
                    is_dir=item.is_dir(),
                    modified_time=stat.st_mtime,
                )
                files.append(file_info)

                # Recursively scan subdirectories
                if recursive and item.is_dir():
                    scan_directory(item, False)

        except PermissionError:
            # Skip directories without permission to access
            pass

    scan_directory(path)

    return ListFilesResult(
        files=files, total_count=len(files), current_path=str(path.absolute())
    )


def create_directory(directory_path: str, parents: bool = True) -> bool:
    """
    Create directory

    Args:
        directory_path: Directory path
        parents: Whether to create parent directories, defaults to True

    Returns:
        True if creation succeeded

    Raises:
        OSError: Creation failed
    """
    os.makedirs(directory_path, exist_ok=parents)
    return True


def file_exists(file_path: str) -> bool:
    """
    Check if file exists

    Args:
        file_path: File path

    Returns:
        True if file exists, otherwise False
    """
    return os.path.exists(file_path)


def get_file_info(file_path: str) -> FileInfo:
    """
    Get detailed file information

    Args:
        file_path: File path

    Returns:
        File information object

    Raises:
        FileNotFoundError: File doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    path = Path(file_path)
    stat = path.stat()

    return FileInfo(
        name=path.name,
        path=str(path.absolute()),
        size=stat.st_size,
        is_file=path.is_file(),
        is_dir=path.is_dir(),
        modified_time=stat.st_mtime,
    )


def read_json_file(file_path: str, encoding: str = "utf-8") -> Any:
    """
    Read JSON file

    Args:
        file_path: JSON file path
        encoding: File encoding, defaults to utf-8

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: File doesn't exist
        json.JSONDecodeError: JSON parsing error
    """
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def write_json_file(
    file_path: str, data: Dict[str, Any], encoding: str = "utf-8", indent: int = 2
) -> bool:
    """
    Write JSON file

    Args:
        file_path: JSON file path
        data: JSON data to write
        encoding: File encoding, defaults to utf-8
        indent: Number of spaces for JSON indentation, defaults to 2

    Returns:
        True if write succeeded

    Raises:
        OSError: File operation error
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return True


def read_csv_file(
    file_path: str, encoding: str = "utf-8", delimiter: str = ","
) -> List[Dict[str, str]]:
    """
    Read CSV file

    Args:
        file_path: CSV file path
        encoding: File encoding, defaults to utf-8
        delimiter: Delimiter, defaults to comma

    Returns:
        List of dictionaries containing CSV data

    Raises:
        FileNotFoundError: File doesn't exist
        csv.Error: CSV parsing error
    """
    with open(file_path, "r", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def write_csv_file(
    file_path: str,
    data: List[Dict[str, str]],
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> bool:
    """
    Write CSV file

    Args:
        file_path: CSV file path
        data: CSV data to write
        encoding: File encoding, defaults to utf-8
        delimiter: Delimiter, defaults to comma

    Returns:
        True if write succeeded

    Raises:
        OSError: File operation error
        csv.Error: CSV write error
    """
    if not data:
        return True

    with open(file_path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)
    return True


def edit_file(
    file_path: str,
    operations: List[Union[Dict[str, Any], EditOperation]],
    encoding: str = "utf-8",
    backup: bool = False,
) -> EditResult:
    """
    Precisely edit file content, supporting multiple edit operations based on line numbers and pattern matching

    Args:
        file_path: File path
        operations: List of edit operations, each operation can be a dictionary or EditOperation object
        encoding: File encoding, defaults to utf-8
        backup: Whether to create backup file, defaults to False

    Returns:
        Edit result object

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Invalid operation parameters
        OSError: File operation error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Convert operations to EditOperation objects
    edit_ops = []
    for op in operations:
        if isinstance(op, dict):
            # Handle field name mapping
            mapped_op = {}
            for key, value in op.items():
                if key == "type":
                    mapped_op["operation_type"] = value
                elif key == "target":
                    mapped_op["pattern"] = value
                elif key == "replacement":
                    mapped_op["content"] = value
                else:
                    mapped_op[key] = value
            edit_ops.append(EditOperation(**mapped_op))
        else:
            edit_ops.append(op)

    # Read original file content
    try:
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Try other encodings
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()

    # Create backup
    if backup:
        backup_path = f"{file_path}.backup"
        with open(backup_path, "w", encoding=encoding) as f:
            f.writelines(lines)

    lines_changed = 0
    results = []

    # Execute edit operations
    for operation in edit_ops:
        try:
            if operation.operation_type == "replace":
                result = _replace_lines(lines, operation)
            elif operation.operation_type == "insert":
                result = _insert_lines(lines, operation)
            elif operation.operation_type == "delete":
                result = _delete_lines(lines, operation)
            else:
                result = EditResult(
                    success=False,
                    message=f"Unknown operation type: {operation.operation_type}",
                )

            results.append(result)
            if result.success:
                lines_changed += result.lines_changed

        except Exception as e:
            results.append(
                EditResult(success=False, message=f"Operation failed: {str(e)}")
            )

    # Write edited content
    try:
        with open(file_path, "w", encoding=encoding) as f:
            f.writelines(lines)
    except Exception as e:
        # If write fails, try to restore backup
        if backup and os.path.exists(f"{file_path}.backup"):
            try:
                with open(f"{file_path}.backup", "r", encoding=encoding) as f:
                    backup_lines = f.readlines()
                with open(file_path, "w", encoding=encoding) as f:
                    f.writelines(backup_lines)
            except Exception:
                pass
        raise OSError(f"Failed to write file: {str(e)}")

    # Generate preview
    preview_lines = lines[:10] + lines[-10:] if len(lines) > 20 else lines
    preview = "".join(preview_lines)

    # Summarize results
    failed_ops = [r for r in results if not r.success]
    if failed_ops:
        return EditResult(
            success=False,
            message=f"Some operations failed: {'; '.join(r.message for r in failed_ops)}",
            lines_changed=lines_changed,
            preview=preview,
        )

    return EditResult(
        success=True,
        message=f"Successfully edited file with {len(edit_ops)} operations",
        lines_changed=lines_changed,
        preview=preview,
    )


def _replace_lines(lines: List[str], operation: EditOperation) -> EditResult:
    """Replace line content"""
    if operation.line_number is not None:
        # Replace based on line number
        line_idx = operation.line_number - 1
        if 0 <= line_idx < len(lines):
            if operation.content is None:
                return EditResult(
                    success=False,
                    message="Content cannot be None for replace operation",
                )
            lines[line_idx] = (
                operation.content + "\n"
                if not operation.content.endswith("\n")
                else operation.content
            )
            return EditResult(
                success=True,
                message=f"Replaced line {operation.line_number}",
                lines_changed=1,
            )
        else:
            return EditResult(
                success=False,
                message=f"Line number {operation.line_number} out of range (file has {len(lines)} lines)",
            )
    elif operation.pattern is not None:
        # Replace based on pattern matching
        pattern = re.compile(operation.pattern, re.MULTILINE)
        lines_changed = 0
        for i, line in enumerate(lines):
            if pattern.search(line):
                if operation.content is None:
                    return EditResult(
                        success=False,
                        message="Content cannot be None for replace operation",
                    )
                lines[i] = (
                    operation.content + "\n"
                    if not operation.content.endswith("\n")
                    else operation.content
                )
                lines_changed += 1

        if lines_changed > 0:
            return EditResult(
                success=True,
                message=f"Replaced {lines_changed} lines matching pattern",
                lines_changed=lines_changed,
            )
        else:
            return EditResult(
                success=False,
                message=f"No lines found matching pattern: {operation.pattern}",
            )
    else:
        return EditResult(
            success=False,
            message="Either line_number or pattern must be specified for replace operation",
        )


def _insert_lines(lines: List[str], operation: EditOperation) -> EditResult:
    """Insert line content"""
    if operation.line_number is not None:
        # Insert based on line number
        line_idx = operation.line_number - 1
        if 0 <= line_idx <= len(lines):
            if operation.content is None:
                return EditResult(
                    success=False,
                    message="Content cannot be None for insert operation",
                )
            content = (
                operation.content + "\n"
                if not operation.content.endswith("\n")
                else operation.content
            )
            lines.insert(line_idx, content)
            return EditResult(
                success=True,
                message=f"Inserted content at line {operation.line_number}",
                lines_changed=1,
            )
        else:
            return EditResult(
                success=False,
                message=f"Line number {operation.line_number} out of range for insertion (file has {len(lines)} lines)",
            )
    elif operation.pattern is not None:
        # Insert based on pattern matching (insert after matching lines)
        pattern = re.compile(operation.pattern, re.MULTILINE)
        lines_changed = 0
        for i in range(
            len(lines) - 1, -1, -1
        ):  # Insert from back to front to avoid line number offset
            if pattern.search(lines[i]):
                if operation.content is None:
                    return EditResult(
                        success=False,
                        message="Content cannot be None for insert operation",
                    )
                content = (
                    operation.content + "\n"
                    if not operation.content.endswith("\n")
                    else operation.content
                )
                lines.insert(i + 1, content)
                lines_changed += 1

        if lines_changed > 0:
            return EditResult(
                success=True,
                message=f"Inserted content after {lines_changed} lines matching pattern",
                lines_changed=lines_changed,
            )
        else:
            return EditResult(
                success=False,
                message=f"No lines found matching pattern: {operation.pattern}",
            )
    else:
        return EditResult(
            success=False,
            message="Either line_number or pattern must be specified for insert operation",
        )


def _delete_lines(lines: List[str], operation: EditOperation) -> EditResult:
    """Delete line content"""
    if operation.line_number is not None:
        # Delete based on line number
        line_idx = operation.line_number - 1
        if 0 <= line_idx < len(lines):
            del lines[line_idx]
            return EditResult(
                success=True,
                message=f"Deleted line {operation.line_number}",
                lines_changed=1,
            )
        else:
            return EditResult(
                success=False,
                message=f"Line number {operation.line_number} out of range (file has {len(lines)} lines)",
            )
    elif operation.pattern:
        # Delete based on pattern matching
        pattern = re.compile(operation.pattern, re.MULTILINE)
        original_length = len(lines)
        lines[:] = [line for line in lines if not pattern.search(line)]
        lines_changed = original_length - len(lines)

        if lines_changed > 0:
            return EditResult(
                success=True,
                message=f"Deleted {lines_changed} lines matching pattern",
                lines_changed=lines_changed,
            )
        else:
            return EditResult(
                success=False,
                message=f"No lines found matching pattern: {operation.pattern}",
            )
    else:
        return EditResult(
            success=False,
            message="Either line_number or pattern must be specified for delete operation",
        )


def find_and_replace(
    file_path: str,
    search_pattern: str,
    replacement: str,
    encoding: str = "utf-8",
    use_regex: bool = True,
    case_sensitive: bool = True,
    backup: bool = False,
) -> EditResult:
    """
    Convenience function to find and replace text content

    Args:
        file_path: File path
        search_pattern: Pattern to search for
        replacement: Replacement content
        encoding: File encoding, defaults to utf-8
        use_regex: Whether to use regular expression, defaults to True
        case_sensitive: Whether to be case-sensitive, defaults to True
        backup: Whether to create backup file, defaults to False

    Returns:
        Edit result object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file content
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

    # Create backup
    if backup:
        backup_path = f"{file_path}.backup"
        with open(backup_path, "w", encoding=encoding) as f:
            f.write(content)

    # Prepare regular expression
    if not use_regex:
        # Escape special characters for literal matching
        search_pattern = re.escape(search_pattern)

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(search_pattern, flags)

    # Execute replacement
    matches = list(pattern.finditer(content))
    if not matches:
        return EditResult(
            success=False,
            message=f"No matches found for pattern: {search_pattern}",
            lines_changed=0,
        )

    # Execute replacement
    new_content = pattern.sub(replacement, content)
    lines_changed = len(matches)

    # Write back to file
    try:
        with open(file_path, "w", encoding=encoding) as f:
            f.write(new_content)
    except Exception as e:
        # If write fails, try to restore backup
        if backup and os.path.exists(f"{file_path}.backup"):
            try:
                with open(f"{file_path}.backup", "r", encoding=encoding) as f:
                    backup_content = f.read()
                with open(file_path, "w", encoding=encoding) as f:
                    f.write(backup_content)
            except Exception:
                pass
        raise OSError(f"Failed to write file: {str(e)}")

    # Generate preview
    preview_lines = new_content.split("\n")
    if len(preview_lines) > 20:
        preview = "\n".join(preview_lines[:10] + ["..."] + preview_lines[-10:])
    else:
        preview = new_content

    return EditResult(
        success=True,
        message=f"Successfully replaced {lines_changed} occurrences",
        lines_changed=lines_changed,
        preview=preview,
    )
