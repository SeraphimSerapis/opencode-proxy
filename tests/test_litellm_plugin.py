"""LiteLLM plugin adapter tests.

The behavior tests need the optional ``litellm`` dependency and are skipped
without it (``uv run --with litellm pytest tests/test_litellm_plugin.py``).
The factory-guard test runs everywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    import litellm  # noqa: F401
    from litellm.types.utils import ModelResponseStream

    _HAVE_LITELLM = True
except ImportError:
    _HAVE_LITELLM = False


def test_factory_requires_litellm() -> None:
    if _HAVE_LITELLM:
        pytest.skip("litellm is installed; the guard only applies without it")
    from opencode_proxy.litellm_plugin import create_repair_handler

    with pytest.raises(RuntimeError, match="litellm"):
        create_repair_handler()


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        model="test-model",
        choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    )


async def _collect(handler: Any, chunks: list[Any]) -> list[ModelResponseStream]:
    async def iterator() -> Any:
        for chunk in chunks:
            yield chunk

    out: list[ModelResponseStream] = []
    async for item in handler.async_post_call_streaming_iterator_hook(None, iterator(), {}):
        out.append(item)
    return out


@pytest.mark.skipif(not _HAVE_LITELLM, reason="litellm not installed")
class TestStreamingIteratorHook:
    async def test_plain_text_passes_through(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        out = await _collect(handler, [_chunk({"content": "hello"}, "stop")])

        assert len(out) == 2  # text chunk + terminator
        assert out[0].choices[0].delta.content == "hello"
        assert out[-1].choices[0].finish_reason == "stop"

    async def test_raw_tool_block_becomes_tool_call_chunks(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        raw = (
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="city">Paris</parameter></invoke></tool_calls>'
        )
        # The guard holds back the tail until the stream ends; both chunks must go in.
        out = await _collect(
            handler,
            [_chunk({"content": raw[:40]}), _chunk({"content": raw[40:]}, "stop")],
        )

        tool_chunks = [c for c in out if c.choices and c.choices[0].delta.tool_calls]
        assert tool_chunks, f"no tool-call chunks in {[c.model_dump() for c in out]}"
        first = tool_chunks[0].choices[0].delta.tool_calls[0]
        assert first.function.name == "get_weather"
        arguments = "".join(
            c.choices[0].delta.tool_calls[0].function.arguments or ""
            for c in tool_chunks
            if c.choices[0].delta.tool_calls
        )
        assert arguments == '{"city":"Paris"}'
        assert out[-1].choices[0].finish_reason == "tool_calls"

    async def test_truncated_arguments_are_completed(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        out = await _collect(
            handler,
            [
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"a": "va'},
                            }
                        ]
                    },
                    "tool_calls",
                ),
            ],
        )

        repairs = [c for c in out if c.choices[0].delta.tool_calls]
        joined = "".join(c.choices[0].delta.tool_calls[0].function.arguments or "" for c in repairs)
        assert joined == '{"a": "va"}'

    async def test_empty_turn_gets_synthesized_terminator(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler
        from opencode_proxy.stream_repair import StreamRepairConfig

        handler = create_repair_handler(config=StreamRepairConfig(empty_turn_notice="[dead turn]"))
        out = await _collect(handler, [_chunk({}, "stop")])

        contents = [c.choices[0].delta.content for c in out if c.choices[0].delta.content]
        assert contents == ["[dead turn]"]
        assert out[-1].choices[0].finish_reason == "stop"


@pytest.mark.skipif(not _HAVE_LITELLM, reason="litellm not installed")
class TestPostCallSuccessHook:
    async def test_buffered_raw_tool_block_is_repaired(self) -> None:
        from litellm.types.utils import ModelResponse

        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        response = ModelResponse(
            id="y",
            object="chat.completion",
            created=1,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            '<tool_calls><invoke name="get_weather">'
                            '<parameter name="city">Paris</parameter>'
                            "</invoke></tool_calls>"
                        ),
                    },
                }
            ],
        )

        result = await handler.async_post_call_success_hook({}, None, response)

        message = result.choices[0].message
        assert message.tool_calls is not None
        assert message.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].finish_reason == "tool_calls"

    async def test_clean_response_is_returned_untouched(self) -> None:
        from litellm.types.utils import ModelResponse

        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        response = ModelResponse(
            id="z",
            object="chat.completion",
            created=1,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "plain answer"},
                }
            ],
        )

        result = await handler.async_post_call_success_hook({}, None, response)
        assert result is response


@pytest.mark.skipif(not _HAVE_LITELLM, reason="litellm not installed")
class TestPreCallHook:
    async def test_null_assistant_content_is_normalized(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        data = {"model": "m", "messages": [{"role": "assistant", "content": None}]}

        result = await handler.async_pre_call_hook(None, None, data, "completion")

        assert result["messages"][0]["content"] == ""

    async def test_non_completion_calls_pass_through(self) -> None:
        from opencode_proxy.litellm_plugin import create_repair_handler

        handler = create_repair_handler()
        data = {"messages": [{"role": "assistant", "content": None}]}

        result = await handler.async_pre_call_hook(None, None, data, "embeddings")

        assert result["messages"][0]["content"] is None
