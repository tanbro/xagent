"""
Output Filter Tool Wrapper

Wraps any tool with output length filtering capabilities.
"""

import inspect
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel

from .base import AbstractBaseTool
from .output_filter import (
    DEFAULT_TRUNCATION_MESSAGE,
    OutputValueFilter,
)

if TYPE_CHECKING:
    from .base import ToolCategory

logger = logging.getLogger(__name__)


class OutputFilteredToolWrapper(AbstractBaseTool):
    """
    Wrapper that applies output filtering to any tool.

    This wrapper intercepts the return value from run_json_sync/async
    and applies length limiting before returning to the caller.
    """

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        max_chars: int | None = None,
        truncation_message: str = DEFAULT_TRUNCATION_MESSAGE,
    ):
        """
        Initialize output filter wrapper.

        Args:
            target_tool: Tool to wrap
            max_chars: Maximum output length in characters. If None, reads from
                      XAGENT_TOOL_MAX_OUTPUT_LENGTH env var or uses default.
            truncation_message: Message to append when truncated
        """
        self._target = target_tool
        self._visibility = getattr(target_tool, "_visibility", None)
        self._allow_users = getattr(target_tool, "_allow_users", None)

        # Create output filter
        self._filter = OutputValueFilter(max_chars, truncation_message)

    @property
    def name(self) -> str:
        return self._target.name

    @property
    def description(self) -> str:
        return self._target.description

    @property
    def tags(self) -> list[str]:
        return self._target.tags

    @property
    def category(self) -> "ToolCategory":
        """Get tool category (delegates to target tool)."""
        return getattr(self._target, "category", None)  # type: ignore[return-value]

    @property
    def metadata(self) -> Any:  # ToolMetadata (avoid circular import)
        return self._target.metadata

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def state_type(self) -> Optional[Type[BaseModel]]:
        return self._target.state_type()

    def is_async(self) -> bool:
        return self._target.is_async()

    def return_value_as_string(self, value: Any) -> str:
        """Convert return value to string (delegates to target tool)."""
        return self._target.return_value_as_string(value)

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Execute tool synchronously with output filtering."""
        result = self._target.run_json_sync(args)
        return self._filter.filter(result, self._target.name)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute tool asynchronously with output filtering."""
        result = await self._target.run_json_async(args)
        return self._filter.filter(result, self._target.name)

    async def save_state_json(self) -> Mapping[str, Any]:
        """Save state (delegates to target tool)."""
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """Load state (delegates to target tool)."""
        await self._target.load_state_json(state)

    async def setup(self, task_id: Optional[str] = None) -> None:
        """Setup tool (delegates to target tool)."""
        if hasattr(self._target, "setup"):
            await self._target.setup(task_id)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        """Teardown tool (delegates to target tool)."""
        if hasattr(self._target, "teardown"):
            await self._target.teardown(task_id)

    @property
    def func(self) -> Any:
        """Get the underlying function, wrapped with output filtering."""
        if (func_obj := getattr(self._target, "func", None)) is None:
            raise AttributeError(f"Tool '{self._target.name}' has no 'func' attribute")

        # Return a wrapped function that applies output filtering
        original_func = func_obj

        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            result = original_func(*args, **kwargs)
            return self._filter.filter(result, self._target.name)

        async def wrapped_func_async(*args: Any, **kwargs: Any) -> Any:
            result = await original_func(*args, **kwargs)
            return self._filter.filter(result, self._target.name)

        # Return async wrapper if original function is async
        if inspect.iscoroutinefunction(original_func):
            return wrapped_func_async
        return wrapped_func
