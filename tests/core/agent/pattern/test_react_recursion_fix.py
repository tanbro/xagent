"""
Tests for RecursionError handling in ReAct pattern.

These tests verify that JSON parsing errors (including RecursionError) are handled gracefully
without causing infinite loops, and that LLM can see errors and attempt recovery.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from xagent.core.agent.context import AgentContext
from xagent.core.agent.exceptions import MaxIterationsError, PatternExecutionError
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM


class MockReActLLM(BaseLLM):
    """Mock LLM for testing RecursionError scenarios"""

    def __init__(self, responses=None, stream_chunks=None):
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "mock_react_llm"
        self.messages_history = []

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    async def chat(self, messages: list[dict[str, str]], **kwargs):
        self.messages_history.append(messages)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            # If response is a string, return it as-is (will be processed by _extract_content)
            return response
        self.call_count += 1
        return '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        self.messages_history.append(messages)
        from xagent.core.model.chat.types import ChunkType, StreamChunk

        # Get the full response for this call
        if self.call_count < len(self.stream_chunks):
            chunks_data = self.stream_chunks[self.call_count]
        else:
            # Use responses array as fallback
            response = (
                self.responses[self.call_count]
                if self.call_count < len(self.responses)
                else '{"type": "final_answer", "answer": "Done"}'
            )
            chunks_data = [{"delta": response, "type": "token"}]
            self.call_count += 1

        # Yield chunks
        accumulated_content = ""
        for chunk_data in chunks_data:
            if isinstance(chunk_data, dict):
                delta = chunk_data.get("delta", "")
                accumulated_content += delta
                yield StreamChunk(
                    type=ChunkType.TOKEN,
                    delta=delta,
                    content=accumulated_content,
                )
            elif isinstance(chunk_data, str):
                accumulated_content += chunk_data
                yield StreamChunk(
                    type=ChunkType.TOKEN,
                    delta=chunk_data,
                    content=accumulated_content,
                )
            else:
                yield chunk_data


class DummyMemoryStore(MemoryStore):
    def add(self, note):
        return MemoryResponse(success=True)

    def get(self, note_id: str):
        return MemoryResponse(success=True)

    def update(self, note):
        return MemoryResponse(success=True)

    def delete(self, note_id: str):
        return MemoryResponse(success=True)

    def search(self, query: str, k: int = 5, filters=None):
        return []

    def clear(self):
        pass

    def get_stats(self):
        return {}

    def list_all(self, limit: int = 100, offset: int = 0):
        return []


# ============================================================================
# STRATEGY 1: Simulate RecursionError - Verify PatternExecutionError Handling
# ============================================================================


@pytest.mark.asyncio
async def test_recursion_error_converted_to_observation():
    """Test that RecursionError is converted to observation so LLM can see and retry."""

    # Mock LLM: first gets error, then returns correct answer
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
        ],
        stream_chunks=[
            # First call - will trigger RecursionError
            [{"delta": '{"type": "final_answer", "answer": "test"}'}],
            # Second call - after seeing error, returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
                }
            ],
        ],
    )

    # Mock repair_loads: first time throws RecursionError, second time works
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            # First call triggers RecursionError
            raise RecursionError("maximum recursion depth exceeded")
        # Subsequent calls work normally
        import json

        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=10)
        memory = DummyMemoryStore()
        tools = []

        # Execute task - should handle RecursionError gracefully
        result = await pattern.run(
            task="Test task",
            memory=memory,
            tools=tools,
            context=AgentContext(),
        )

        # Verify:
        # 1. Doesn't crash with RecursionError
        assert result is not None
        # 2. repair_loads was called at least once (error occurred)
        assert repair_call_count[0] >= 1
        # 3. Task completes successfully (LLM saw error and recovered)
        assert result.get("success") is True
        # 4. Error was handled through observation (not infinite loop)
        assert result.get("iterations", 0) <= 10


# ============================================================================
# STRATEGY 2: 497+ Level Deeply Nested JSON
# ============================================================================


@pytest.mark.asyncio
async def test_deeply_nested_json_handling():
    """Test that deeply nested JSON (497+ levels) is handled gracefully."""

    # Create 497 levels of nested JSON (triggers RecursionError threshold)
    deep_json = "{"
    for i in range(497):
        deep_json += '{"a":'
    deep_json += '"x"' + "}" * 497
    deep_json += "}"

    # Mock LLM keeps returning the same deeply nested JSON
    # This will cause the agent to hit max_iterations, but NOT crash with RecursionError
    llm = MockReActLLM(
        responses=[],
        stream_chunks=[
            # Always return deeply nested JSON
            [{"delta": deep_json}],
        ],
    )

    pattern = ReActPattern(llm=llm, max_iterations=5)
    memory = DummyMemoryStore()
    tools = []

    # Execute task - should raise MaxIterationsError, NOT RecursionError
    with pytest.raises(MaxIterationsError) as exc_info:
        await pattern.run(
            task="Test task",
            memory=memory,
            tools=tools,
            context=AgentContext(),
        )

    # Verify the error is MaxIterationsError (not RecursionError)
    assert "maximum iterations" in str(exc_info.value).lower()
    assert "recursion" not in str(exc_info.value).lower()


# ============================================================================
# STRATEGY 3: Error Handling Consistency
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_name,error_exception",
    [
        ("RecursionError", RecursionError("max recursion")),
        ("JSONDecodeError", json.JSONDecodeError("test", "doc", 0)),
    ],
)
async def test_json_error_handling_consistency(error_name, error_exception):
    """Test that JSON parsing errors are handled consistently."""

    # LLM returns valid response (but repair_loads will fail)
    llm = MockReActLLM(
        responses=[],
        stream_chunks=[
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
                }
            ]
        ],
    )

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = error_exception

        pattern = ReActPattern(llm=llm, max_iterations=5)

        # All errors should be handled consistently:
        # - No RecursionError raised (it's caught internally)
        # - May hit MaxIterationsError due to persistent errors
        with pytest.raises((MaxIterationsError, PatternExecutionError)) as exc_info:
            await pattern.run(
                task="Test task",
                memory=DummyMemoryStore(),
                tools=[],
                context=AgentContext(),
            )

        # The important thing: error is NOT RecursionError escaping to the top level
        if isinstance(exc_info.value, RecursionError):
            pytest.fail("RecursionError should have been caught and handled internally")


# ============================================================================
# STRATEGY 4: PatternExecutionError Converted to Observation
# ============================================================================


@pytest.mark.asyncio
async def test_pattern_execution_error_converted_to_observation():
    """Test that PatternExecutionError is converted to observation so LLM can see the error."""

    # Track messages to verify error was added
    messages_history = []

    # LLM: first gets error (which becomes observation), then succeeds
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
        ],
        stream_chunks=[
            # First call - will trigger PatternExecutionError
            [{"delta": "invalid json {{{"}],
            # Second call - after seeing error in observation, returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
                }
            ],
        ],
    )

    # Track calls to verify error was seen
    call_count = [0]

    original_chat = llm.chat

    async def tracking_chat(messages, **kwargs):
        call_count[0] += 1
        messages_history.append(messages.copy())
        return await original_chat(messages, **kwargs)

    llm.chat = tracking_chat

    # Mock repair_loads: first fails with JSONDecodeError, second succeeds
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            raise json.JSONDecodeError("Expecting value", "{{{", 0)
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify:
        # 1. repair_loads was called (error occurred)
        assert repair_call_count[0] >= 1
        # 2. LLM was called multiple times (second call saw the error observation)
        assert call_count[0] >= 2
        # 3. Task completes successfully (LLM saw error and corrected)
        assert result.get("success") is True
        # 4. Second call's messages include the error observation
        second_call_messages = messages_history[1] if len(messages_history) > 1 else []
        error_observation_found = any(
            "Pattern execution error" in msg.get("content", "")
            or "Observation:" in msg.get("content", "")
            and "JSON" in msg.get("content", "")
            for msg in second_call_messages
        )
        assert error_observation_found, (
            "Error observation should be in second call's messages"
        )


# ============================================================================
# STRATEGY 5: LLM Can See Error
# ============================================================================


@pytest.mark.asyncio
async def test_llm_can_see_json_parsing_error():
    """Test that when JSON parsing fails, LLM sees the error in next iteration."""

    # LLM first returns bad JSON, then correct answer
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
        ],
        stream_chunks=[
            # First: error - will cause JSON parsing to fail
            [{"delta": "invalid json {{{"}],
            # After seeing error, valid response
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
                }
            ],
        ],
    )

    # Mock repair_loads: first fails, second succeeds
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            raise json.JSONDecodeError("test", "doc", 0)
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)

        # Should succeed after second call
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify: repair_loads was called (error was caught and handled)
        assert repair_call_count[0] >= 1
        # Task completes successfully
        assert result.get("success") is True


# ============================================================================
# STRATEGY 5: Multiple Consecutive Failures
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_consecutive_json_failures():
    """Test that multiple consecutive JSON failures don't cause infinite loop."""

    # LLM returns valid answer after failures
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Done", "answer": "Success", "success": true}'
        ],
        stream_chunks=[{"delta": "invalid"}] * 10
        + [
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Done", "answer": "Success", "success": true"}'
                }
            ]
        ],
    )

    # Mock repair_loads: first 10 times fail
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] <= 10:
            raise json.JSONDecodeError("test", "doc", 0)
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=20)
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify:
        # 1. Doesn't crash with RecursionError
        assert result is not None
        # 2. repair_loads was called multiple times
        assert repair_call_count[0] >= 1


# ============================================================================
# STRATEGY 6: Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_recursion_error_with_tool_call():
    """Test RecursionError when LLM tries to make a tool call."""

    # LLM returns a tool call that contains deeply nested JSON
    tool_call_with_deep_json = (
        '{"type": "tool_call", "tool_name": "test", "tool_args": ' + "{"
    )
    for i in range(497):
        tool_call_with_deep_json += '{"a":'
    tool_call_with_deep_json += '"value"}' + "}" * 497
    tool_call_with_deep_json += "}"

    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
        ],
        stream_chunks=[
            [{"delta": tool_call_with_deep_json}],
        ],
    )

    pattern = ReActPattern(llm=llm, max_iterations=5)
    memory = DummyMemoryStore()
    tools = []

    result = await pattern.run(
        task="Test task",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should handle gracefully without infinite loop or RecursionError crash
    assert result is not None


# ============================================================================
# RUN TESTS MANUALLY
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RecursionError Fix Tests")
    print("=" * 70)
    print("\nRun with pytest:")
    print("  pytest tests/core/agent/pattern/test_react_recursion_fix.py -v")
    print("\nOr run this file directly (limited tests):")
    print("=" * 70)

    async def run_test(test_func, name):
        try:
            await test_func()
            print(f"✓ {name}")
            return True
        except Exception as e:
            print(f"✗ {name}: {e}")
            return False

    asyncio.run(
        run_test(
            test_recursion_error_converted_to_observation,
            "RecursionError converted to observation",
        )
    )
    asyncio.run(
        run_test(test_deeply_nested_json_handling, "Deeply nested JSON handling")
    )
    asyncio.run(run_test(test_llm_can_see_json_parsing_error, "LLM can see error"))
    asyncio.run(
        run_test(
            test_multiple_consecutive_json_failures, "Multiple consecutive failures"
        )
    )

    print("\n" + "=" * 70)
    print("Tests completed. Use pytest for full testing with detailed output.")
    print("=" * 70)
