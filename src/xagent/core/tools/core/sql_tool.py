"""
SQL Tool for xagent - Async SQL execution using SQLAlchemy

Database connections are configured via environment variables, not raw URLs.
Connection format: XAGENT_DB_<NAME>=<connection_url>

Example:
    XAGENT_DB_ANALYTICS=postgresql+asyncpg://user:pass@localhost:5432/analytics
    XAGENT_DB_PROD=mysql+aiomysql://user:pass@localhost:3306/production
    XAGENT_DB_LOCAL=sqlite+aiosqlite:///path/to/database.db
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import URL, text
from sqlalchemy.engine import Row, make_url
from sqlalchemy.ext.asyncio import AsyncResult, create_async_engine

if TYPE_CHECKING:
    from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)


class SQLQueryArgs(BaseModel):
    """Arguments for SQL query execution."""

    connection_name: str = Field(description="Database connection name to use")
    query: str = Field(description="SQL query to execute")


class SQLQueryResult(BaseModel):
    """Result from SQL query execution in LLM-friendly format"""

    success: bool = Field(description="Whether the query executed successfully")
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Query result rows as list of dictionaries"
    )
    row_count: int = Field(default=0, description="Number of rows affected/returned")
    columns: list[str] = Field(
        default_factory=list, description="Column names in result set"
    )
    message: str = Field(default="", description="Summary of what happened")
    error: str = Field(default="", description="Error message if execution failed")


def _get_connection_url(connection_name: str) -> URL:
    """Get database connection URL from environment variable.

    Environment variable format: XAGENT_DB_<NAME>=<connection_url>

    Args:
        connection_name: Name of the connection (case-insensitive)

    Returns:
        Connection URL if found, None otherwise
    """
    env_key = f"XAGENT_DB_{connection_name.upper()}"
    url = os.getenv(env_key)

    if not url:
        # List available DB connections (filter only XAGENT_DB_* and strip prefix)
        available = [
            key.removeprefix("XAGENT_DB_")
            for key in os.environ.keys()
            if key.startswith("XAGENT_DB_")
        ]
        if available:
            raise ValueError(
                f"Database connection '{connection_name}' not found. "
                f"Available databases: {', '.join(available)}"
            )
        else:
            raise ValueError(
                f"Database connection '{connection_name}' not found. No databases configured."
            )

    # Validate URL format using SQLAlchemy
    parsed_url = make_url(url)
    drivername = parsed_url.drivername

    # Ensure async driver is being used
    if drivername == "postgresql" and "+" not in drivername:
        raise ValueError(
            f"Connection '{connection_name}' must use async driver (postgresql+asyncpg://), got: {drivername}"
        )
    elif drivername == "mysql" and "+" not in drivername:
        raise ValueError(
            f"Connection '{connection_name}' must use async driver (mysql+aiomysql://), got: {drivername}"
        )
    elif drivername == "sqlite" and "+" not in drivername:
        raise ValueError(
            f"Connection '{connection_name}' must use async driver (sqlite+aiosqlite:///), got: {drivername}"
        )

    return parsed_url


def _row_to_dict(row: Row) -> dict[str, Any]:
    """Convert SQLAlchemy Row to dictionary"""
    return dict(row._mapping)


async def execute_sql_query(
    connection_name: str,
    query: str,
    output_file: Optional[str] = None,
    workspace: Optional["TaskWorkspace"] = None,
) -> dict[str, Any]:
    """Execute SQL queries on databases and return structured results.

    Args:
        connection_name: Database connection name to use
        query: SQL statement to execute
        output_file: Optional file path to export query results.
            Supported formats: .csv, .parquet, .json, .jsonl, .ndjson (relative to workspace output directory).
            When provided, query results are exported to file instead of being returned.
        workspace: Optional TaskWorkspace instance for file exports.

    Returns:
        dict:
            with keys:
            - success: true if query worked, false if it failed
            - rows: query results as list of dicts (SELECT only, empty when exported)
            - row_count: number of rows returned or affected
            - columns: column names in the result
            - message: what happened
            - error: error details if success is false
    """
    # Get connection URL from environment
    try:
        url = _get_connection_url(connection_name)
        stmt = text(query)
        engine = create_async_engine(url)

        try:
            async with engine.connect() as conn:
                # Check if export to file is requested first
                if output_file and workspace:
                    file_ext = Path(output_file).suffix.lower()
                    if file_ext == ".csv":
                        # Async streaming export for large datasets
                        stream_result = await conn.stream(stmt)
                        _, exported_count, columns = await _async_stream_export_to_csv(
                            workspace, output_file, stream_result
                        )
                        return SQLQueryResult(
                            success=True,
                            rows=[],
                            row_count=exported_count,
                            columns=columns,
                            message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                        ).model_dump()
                    elif file_ext == ".parquet":
                        # Async streaming export with Parquet (better compression & type preservation)
                        stream_result = await conn.stream(stmt)
                        (
                            _,
                            exported_count,
                            columns,
                        ) = await _async_stream_export_to_parquet(
                            workspace, output_file, stream_result
                        )
                        return SQLQueryResult(
                            success=True,
                            rows=[],
                            row_count=exported_count,
                            columns=columns,
                            message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                        ).model_dump()
                    elif file_ext in (".json", ".jsonl", ".ndjson"):
                        # Async streaming JSON Lines (NDJSON) export
                        stream_result = await conn.stream(stmt)
                        (
                            _,
                            exported_count,
                            columns,
                        ) = await _async_stream_export_to_jsonlines(
                            workspace, output_file, stream_result
                        )
                        return SQLQueryResult(
                            success=True,
                            rows=[],
                            row_count=exported_count,
                            columns=columns,
                            message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                        ).model_dump()
                    else:
                        raise ValueError(
                            f"Unsupported file format: {file_ext}. "
                            f"Supported: .csv (async streaming), .parquet (async streaming), .json/.jsonl/.ndjson (async streaming JSON Lines)"
                        )

                # Original behavior: return data in response
                result = await conn.execute(stmt)

                # Get column names from result
                if result.returns_rows:
                    rows = result.all()
                    row_list = [_row_to_dict(row) for row in rows]

                    # Extract column names from first row
                    columns = list(row_list[0].keys()) if row_list else []

                    return SQLQueryResult(
                        success=True,
                        rows=row_list,
                        row_count=len(row_list),
                        columns=columns,
                        message=f"Query executed successfully on '{connection_name}', returned {len(row_list)} row(s)",
                    ).model_dump()
                else:
                    # For INSERT, UPDATE, DELETE operations
                    rowcount = result.rowcount if hasattr(result, "rowcount") else 0

                    # Commit the transaction for non-SELECT queries
                    await conn.commit()

                    return SQLQueryResult(
                        success=True,
                        rows=[],
                        row_count=rowcount,
                        columns=[],
                        message=f"Query executed successfully on '{connection_name}', affected {rowcount} row(s)",
                    ).model_dump()

        finally:
            await engine.dispose()

    except Exception as e:
        logger.error(
            f"SQL execution error on connection '{connection_name}': {e}", exc_info=True
        )

        # Return error in LLM-friendly format
        return SQLQueryResult(
            success=False,
            rows=[],
            row_count=0,
            columns=[],
            message=f"Query execution failed on connection '{connection_name}'",
            error=str(e),
        ).model_dump()


async def _async_stream_export_to_csv(
    workspace: "TaskWorkspace",
    file_path: str,
    result: AsyncResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Async streaming export to CSV using conn.stream() with partitions().

    Uses partitions() to fetch pre-buffered rows in batches, reducing iteration
    overhead while maintaining non-blocking async behavior.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration (result may be closed after)
    columns = list(result.keys())

    row_count = 0
    writer: csv.DictWriter | None = None

    with open(resolved_path, "w", encoding="utf-8", newline="") as f:
        # partitions() yields batches of pre-buffered rows (async, non-blocking)
        async for partition in result.partitions(batch_size):
            # Convert batch to dict format
            batch_dicts = [_row_to_dict(row) for row in partition]

            # Initialize writer on first batch
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

            # Write batch to file
            if writer is not None:
                writer.writerows(batch_dicts)
            row_count += len(batch_dicts)

    return str(resolved_path), row_count, columns


async def _async_stream_export_to_jsonlines(
    workspace: "TaskWorkspace",
    file_path: str,
    result: AsyncResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Async streaming export to JSON Lines (NDJSON) using conn.stream() with partitions().

    Uses partitions() to fetch pre-buffered rows in batches for efficient async streaming.
    Each row is written as a separate JSON object on its own line.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration (result may be closed after)
    columns = list(result.keys())

    row_count = 0

    with open(resolved_path, "w", encoding="utf-8") as f:
        # partitions() yields batches of pre-buffered rows (async, non-blocking)
        async for partition in result.partitions(batch_size):
            # Convert batch to JSON lines and write
            for row in partition:
                row_dict = _row_to_dict(row)
                print(json.dumps(row_dict, ensure_ascii=False), file=f)
                row_count += 1

    return str(resolved_path), row_count, columns


async def _async_stream_export_to_parquet(
    workspace: "TaskWorkspace",
    file_path: str,
    result: AsyncResult,
    batch_size: int = 5000,
) -> tuple[str, int, list[str]]:
    """Async streaming export to Parquet using conn.stream() with partitions().

    Uses partitions() to fetch pre-buffered rows in batches for efficient async streaming.
    Parquet provides excellent compression and preserves data types.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet export. "
            "Install it with: pip install pyarrow"
        )

    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration (result may be closed after)
    columns = list(result.keys())

    row_count = 0
    writer = None

    # partitions() yields batches of pre-buffered rows (async, non-blocking)
    async for partition in result.partitions(batch_size):
        # Convert batch to dict format
        batch_dicts = [_row_to_dict(row) for row in partition]

        # Create Arrow Table from batch
        table = pa.Table.from_pylist(batch_dicts)

        # Initialize writer with schema from first batch
        if writer is None:
            writer = pq.ParquetWriter(resolved_path, table.schema)

        # Write batch to file
        writer.write_table(table)
        row_count += len(batch_dicts)

    # Close writer to finalize file
    if writer:
        writer.close()

    return str(resolved_path), row_count, columns
