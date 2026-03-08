"""
Unit tests for file_tool safety features.

Tests for binary file detection, file size limits, and preview generation.
"""

import os
import tempfile

import pytest

from xagent.core.tools.core.file_tool import (
    DEFAULT_MAX_FILE_SIZE,
    _get_file_preview_from_file,
    _is_binary_by_content,
    _is_binary_by_mime,
    _is_binary_file,
    read_file,
)


class TestBinaryDetection:
    """Test binary file detection using multiple methods."""

    def test_is_binary_by_mime_with_image(self):
        """Test MIME type detection rejects image files."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is True
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_mime_with_png(self):
        """Test MIME type detection rejects PNG files."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is True
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_mime_with_pdf(self):
        """Test MIME type detection rejects PDF files."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is True
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_mime_with_text(self):
        """Test MIME type detection accepts text files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is False
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_mime_with_py(self):
        """Test MIME type detection accepts Python files."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is False
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_mime_with_json(self):
        """Test MIME type detection accepts JSON files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name
        try:
            assert _is_binary_by_mime(temp_file) is False
        finally:
            os.unlink(temp_file)

    def test_is_binary_by_content_with_null_bytes(self):
        """Test content detection rejects files with null bytes."""
        chunk = b"Hello\x00World"
        assert _is_binary_by_content(chunk) is True

    def test_is_binary_by_content_with_text(self):
        """Test content detection accepts plain text."""
        chunk = b"Hello, World!\nThis is plain text."
        assert _is_binary_by_content(chunk) is False

    def test_is_binary_file_with_text_file(self):
        """Test comprehensive binary detection accepts text files."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            temp_file = f.name
            f.write("This is a plain text file.\nMultiple lines.\nSafe for LLMs.")
        try:
            chunk = b"This is a plain text file.\nMultiple lines.\nSafe for LLMs."
            assert _is_binary_file(temp_file, chunk) is False
        finally:
            os.unlink(temp_file)

    def test_is_binary_file_with_binary_file(self):
        """Test comprehensive binary detection rejects binary files."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
            temp_file = f.name
            f.write(b"\x00\x01\x02\x03\x04\x05")
        try:
            chunk = b"\x00\x01\x02\x03\x04\x05"
            assert _is_binary_file(temp_file, chunk) is True
        finally:
            os.unlink(temp_file)


class TestFileSizeLimit:
    """Test file size limiting functionality."""

    def test_read_small_file(self):
        """Test reading small files (under limit)."""
        content = "Hello, World!\n" * 10  # Small file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8", newline="\n"
        ) as f:
            temp_file = f.name
            f.write(content)
        try:
            result = read_file(temp_file, max_size=DEFAULT_MAX_FILE_SIZE)
            # Normalize line endings for comparison
            expected = content.replace("\r\n", "\n")
            actual = result.replace("\r\n", "\n")
            assert actual == expected
        finally:
            os.unlink(temp_file)

    def test_read_file_exceeds_limit(self):
        """Test reading files that exceed the limit returns preview."""
        # Create a file larger than DEFAULT_MAX_FILE_SIZE
        lines = ["Line number {}\n".format(i) for i in range(2000)]
        content = "".join(lines)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            temp_file = f.name
            f.write(content)
        try:
            result = read_file(temp_file, max_size=1024)  # Very small limit
            assert "# File too large for complete reading" in result
            assert "Showing preview" in result
            # Check that preview has line numbers
            assert "\tLine number 0\r\n" in result or "\tLine number 0\n" in result
            # Check that only first lines are shown (not tail)
            assert "Line number 1999" not in result
        finally:
            os.unlink(temp_file)

    def test_read_file_with_custom_limit(self):
        """Test reading files with custom size limit."""
        content = "x" * 2000  # 2000 bytes
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            temp_file = f.name
            f.write(content)
        try:
            # Set limit to 1000 bytes
            result = read_file(temp_file, max_size=1000)
            assert "# File too large for complete reading" in result
        finally:
            os.unlink(temp_file)


class TestGetFilePreview:
    """Test preview generation for large files."""

    def test_get_file_preview_with_large_file(self):
        """Test preview generation includes metadata and head only (no tail)."""
        lines = ["Line {}\n".format(i) for i in range(200)]
        content = "".join(lines)
        file_size = len(content.encode("utf-8"))

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_file = f.name
            f.write(content.encode("utf-8"))

        try:
            with open(temp_file, "rb") as f:
                preview = _get_file_preview_from_file(
                    f, file_size, "utf-8", max_lines=50
                )
                assert "# File too large for complete reading" in preview
                assert "Size:" in preview
                assert "Showing preview" in preview
                # Check that preview has line numbers (cat -n format)
                assert "\tLine 0\n" in preview
                # Check that only first lines are shown, not tail
                assert "Line 199" not in preview  # Tail should NOT be present
        finally:
            os.unlink(temp_file)

    def test_get_file_preview_with_small_file(self):
        """Test preview with file smaller than preview lines."""
        lines = ["Line {}\n".format(i) for i in range(10)]
        content = "".join(lines)
        file_size = len(content.encode("utf-8"))

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_file = f.name
            f.write(content.encode("utf-8"))

        try:
            with open(temp_file, "rb") as f:
                preview = _get_file_preview_from_file(
                    f, file_size, "utf-8", max_lines=50
                )
                assert "Showing preview" in preview
                # All 10 lines should be present
                assert "\tLine 0\n" in preview
                assert "\tLine 9\n" in preview
        finally:
            os.unlink(temp_file)

    def test_preview_bytes_limit(self):
        """Test preview respects bytes limit to prevent reading huge lines."""
        # Create a file with one very long line (no newlines)
        long_line = "x" * 100000  # 100KB single line
        file_size = len(long_line.encode("utf-8"))

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_file = f.name
            f.write(long_line.encode("utf-8"))

        try:
            with open(temp_file, "rb") as f:
                preview = _get_file_preview_from_file(
                    f, file_size, "utf-8", max_lines=100
                )
                # Should truncate after 64KB
                assert "preview truncated after 64KB" in preview
                assert "# File too large for complete reading" in preview
        finally:
            os.unlink(temp_file)


class TestReadFileErrors:
    """Test error handling in read_file function."""

    def test_read_binary_file_raises_error(self):
        """Test reading binary files raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_file = f.name
            f.write(b"\xff\xd8\xff\xe0")  # JPEG header
        try:
            with pytest.raises(ValueError) as exc_info:
                read_file(temp_file)
            assert "Cannot read binary file" in str(exc_info.value)
            assert "only supports text files" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_read_file_with_invalid_encoding_raises_error(self):
        """Test reading file with invalid encoding raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
            temp_file = f.name
            f.write(b"Hello\x00World\xff\xfe")
        try:
            with pytest.raises(ValueError) as exc_info:
                read_file(temp_file, encoding="utf-8")
            assert "Cannot read binary file" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_read_nonexistent_file_raises_error(self):
        """Test reading nonexistent file raises FileNotFoundError from runtime."""
        # The runtime will raise FileNotFoundError naturally when trying to open the file
        with pytest.raises(FileNotFoundError):
            read_file("/non/existent/file.txt")
