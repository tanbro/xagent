"""
SQL Tool adapter for xagent framework.

Wraps the core SQL tool for framework integration.
"""

import logging
from textwrap import dedent, indent
from typing import Any, Optional

from ....workspace import TaskWorkspace
from ...core.sql_tool import execute_sql_query
from .function import FunctionTool

logger = logging.getLogger(__name__)


class SqlQueryTool:
    """
    SQL query tool that executes SQL queries on configured databases.
    """

    def __init__(self, workspace: Optional[TaskWorkspace] = None):
        """
        Initialize SQL query tool.

        Args:
            workspace: Optional workspace for file-based operations
        """
        self._workspace = workspace

    async def execute_sql_query(
        self, connection_name: str, query: str, output_file: Optional[str] = None
    ) -> dict[str, Any]:
        return await execute_sql_query(
            connection_name, query, output_file, self._workspace
        )

    def get_tools(self) -> list:
        """Get all tool instances."""
        tools = [
            FunctionTool(
                self.execute_sql_query,
                name="execute_sql_query",
                description=indent(
                    dedent("""
                    Execute SQL queries on databases and return structured results.

                        Args:
                            connection_name: Database connection name to use
                            query: SQL statement to execute
                            output_file: Optional file path to export query results.
                                Supported formats: .csv, .parquet, .json, .jsonl, .ndjson (relative to workspace).
                                Use this for large datasets or complex analysis with Python.
                                Formats: .csv (streaming), .parquet (streaming + compression),
                                          .json/.jsonl/.ndjson (JSON Lines streaming).
                                Example: output_file="results.parquet" for optimized pandas analysis.

                        Returns:
                            dict with keys:
                            - success: true if query worked, false if it failed
                            - rows: query results as list of dicts (SELECT only, empty when exported)
                            - row_count: number of rows returned or affected
                            - columns: column names in the result
                            - message: what happened (includes export info when applicable)
                            - error: error details if success is false

                """),
                    "" * 4,
                ),
                tags=[
                    "sql",
                    "database",
                    "query",
                    "postgresql",
                    "mysql",
                    "sqlite",
                    "async",
                ],
            ),
        ]

        return tools


def get_sql_tool(info: Optional[dict[str, Any]] = None) -> list[FunctionTool]:
    workspace: TaskWorkspace | None = None
    if info and "workspace" in info:
        workspace = (
            info["workspace"] if isinstance(info["workspace"], TaskWorkspace) else None
        )

    tool_instance = SqlQueryTool(workspace=workspace)
    return tool_instance.get_tools()
