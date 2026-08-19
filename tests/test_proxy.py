from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from conftest import collect_content

from opencode_proxy.app import create_app
from opencode_proxy.proxy import classify_upstream_error
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BAR = "\uff5c"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_calls"


def test_classify_upstream_error_types() -> None:
    assert classify_upstream_error(httpx.ConnectTimeout("connect timed out")) == (
        "upstream connect timeout",
        "connect_timeout",
    )
    assert classify_upstream_error(httpx.ReadTimeout("read timed out")) == (
        "upstream read timeout",
        "read_timeout",
    )
    assert classify_upstream_error(httpx.WriteTimeout("write timed out")) == (
        "upstream write timeout",
        "write_timeout",
    )
    assert classify_upstream_error(httpx.PoolTimeout("pool timed out")) == (
        "upstream pool timeout",
        "pool_timeout",
    )
    assert classify_upstream_error(
        httpx.ConnectError("connection refused to secret-host:4000")
    ) == ("upstream connection refused", "connection_refused")
    assert classify_upstream_error(httpx.ConnectError("[Errno -2] Name or service not known")) == (
        "upstream DNS resolution failed",
        "dns_error",
    )
    assert classify_upstream_error(httpx.ConnectError("network unreachable")) == (
        "upstream connection failed",
        "connect_error",
    )
    assert classify_upstream_error(httpx.RemoteProtocolError("peer closed")) == (
        "upstream protocol error",
        "protocol_error",
    )
    message, error_type = classify_upstream_error(
        httpx.ConnectError("connection refused to secret-host:4000")
    )
    assert "secret-host" not in message
    assert error_type == "connection_refused"


async def _client(settings: Settings | None = None) -> httpx.AsyncClient:
    app = create_app(settings or Settings(upstream_url="http://upstream.test"))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://proxy.test")


def _no_retry_settings() -> Settings:
    return Settings(upstream_url="http://upstream.test", upstream_max_retries=0)


def _stream_payloads(response: httpx.Response) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@respx.mock
async def test_non_streaming_chat_completion_is_converted() -> None:
    upstream = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call><name>read</name>"
                                '<parameters>{"path":"README.md"}</parameters></tool_call>'
                            ),
                        },
                        "finish_reason": "stop",
                    },
                ],
            },
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "read"}]},
        )

    assert upstream.called
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": '{"path":"README.md"}',
    }


@respx.mock
async def test_streaming_chat_completion_is_converted() -> None:
    first_chunk = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "model": "deepseek",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    raw_tool_chunk = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "model": "deepseek",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": (
                        f"<{BAR}DSML{BAR}tool_calls><name>bash</name>"
                        f'<parameters>{{"cmd":"pwd"}}</parameters></{BAR}DSML{BAR}tool_calls>'
                    ),
                    "tool_calls": [],
                },
                "finish_reason": None,
            },
        ],
    }
    sse = (
        f"data: {json.dumps(first_chunk)}\n\n"
        f"data: {json.dumps(raw_tool_chunk, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek",
                "stream": True,
                "messages": [{"role": "user", "content": "where"}],
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [line.removeprefix("data: ") for line in lines if line != "data: [DONE]"]
    chunks = [json.loads(payload) for payload in payloads]
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
        or chunk["choices"][0]["finish_reason"] == "tool_calls"
    ]
    assert tool_chunks
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "bash"
    streamed_arguments = "".join(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for chunk in tool_chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    )
    assert json.loads(streamed_arguments) == {"cmd": "pwd"}
    assert lines[-1] == "data: [DONE]"


@respx.mock
async def test_streaming_sse_comments_multiline_data_and_duplicate_done() -> None:
    chunk = {
        "id": "chatcmpl-sse",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ],
    }
    payload = json.dumps(chunk)
    first_part, second_part = payload.split('"object"', 1)
    sse = (
        ": keepalive\n\n"
        f"data: {first_part}\n"
        f'data: "object"{second_part}\n\n'
        "data: [DONE]\n\n"
        "data: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert ": keepalive" in response.text
    assert response.text.count("data: [DONE]") == 1
    assert _stream_payloads(response)[0]["choices"][0]["delta"]["content"] == "hello"


@respx.mock
async def test_streaming_duplicate_terminal_choice_is_ignored() -> None:
    frames = [
        {
            "id": "chatcmpl-duplicate-finish",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {"content": "before"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-duplicate-finish",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-duplicate-finish",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-duplicate-finish",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {"content": "after"}, "finish_reason": None}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    assert collect_content(_stream_payloads(response)) == "before"
    assert _finish_reasons(response) == ["stop"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_terminal_state_is_independent_per_choice() -> None:
    frames = [
        {
            "id": "chatcmpl-choice-terminals",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [
                {"index": 0, "delta": {"content": "zero"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "one"}, "finish_reason": None},
            ],
        },
        {
            "id": "chatcmpl-choice-terminals",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-choice-terminals",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
                {"index": 1, "delta": {"content": "-continued"}, "finish_reason": None},
            ],
        },
        {
            "id": "chatcmpl-choice-terminals",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    content_by_index: dict[int, str] = {}
    finish_indexes: list[int] = []
    for payload in _stream_payloads(response):
        for choice in payload.get("choices") or []:
            index = choice["index"]
            delta = choice.get("delta") or {}
            content_by_index[index] = content_by_index.get(index, "") + delta.get("content", "")
            if choice.get("finish_reason"):
                finish_indexes.append(index)
    assert content_by_index == {0: "zero", 1: "one-continued"}
    assert finish_indexes == [0, 1]


@respx.mock
async def test_streaming_non_sse_response_passes_through() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b'{"message":"not an sse stream"}',
            headers={"content-type": "application/json"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"message": "not an sse stream"}


@respx.mock
async def test_chat_completion_request_drops_non_function_tools() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "parameters": {"type": "object"},
                },
            },
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "ls"}],
                "tools": [
                    {"type": "custom", "name": "bad"},
                    {
                        "type": "function",
                        "function": {
                            "name": "ls",
                            "parameters": {"type": "object"},
                        },
                    },
                ],
            },
        )

    assert response.status_code == 200


@respx.mock
async def test_chat_completion_request_keeps_non_function_tools_when_disabled() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"] == [{"type": "custom", "name": "kept"}]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)
    settings = Settings(upstream_url="http://upstream.test", sanitize_tools=False)

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "custom", "name": "kept"}],
            },
        )

    assert response.status_code == 200


@respx.mock
async def test_chat_completion_request_drops_configured_fields() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "parallel_tool_calls" not in payload
        assert payload["model"] == "test"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)
    settings = Settings(
        upstream_url="http://upstream.test",
        request_drop_fields="parallel_tool_calls",
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "parallel_tool_calls": False,
                "messages": [{"role": "user", "content": "x"}],
            },
        )

    assert response.status_code == 200


@respx.mock
async def test_non_chat_route_passes_through() -> None:
    route = respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "model-a"}]},
        ),
    )

    async with await _client() as client:
        response = await client.get("/v1/models")

    assert route.called
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "model-a"


@respx.mock
async def test_root_models_route_forwards_to_v1_models() -> None:
    route = respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "model-a"}]},
        ),
    )

    async with await _client() as client:
        response = await client.get("/models")

    assert route.called
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "model-a"


@respx.mock
async def test_custom_headers_are_forwarded_to_upstream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer upstream-token"
        assert request.headers["x-skip-auth"] == "true"
        return httpx.Response(200, json={"object": "list", "data": []})

    respx.get("http://upstream.test/v1/models").mock(side_effect=handler)
    settings = Settings(
        upstream_url="http://upstream.test",
        custom_headers='{"Authorization":"Bearer upstream-token","X-Skip-Auth":"true"}',
    )

    async with await _client(settings) as client:
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer client-token"},
        )

    assert response.status_code == 200


@respx.mock
async def test_model_alias_rewrites_chat_completion_request() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "DeepSeek-V4-Flash"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)
    settings = Settings(
        upstream_url="http://upstream.test",
        model_aliases='{"dsv4-flash":"DeepSeek-V4-Flash"}',
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "dsv4-flash", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200


@respx.mock
async def test_model_aliases_are_added_to_v1_models() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "DeepSeek-V4-Flash",
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            },
        ),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_aliases=(
            '{"dsv4-flash":"DeepSeek-V4-Flash",'
            '"deepseek-ai/DeepSeek-V4-Flash-DSpark":"DeepSeek-V4-Flash"}'
        ),
    )

    async with await _client(settings) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {model["id"] for model in response.json()["data"]}
    assert model_ids == {
        "DeepSeek-V4-Flash",
        "dsv4-flash",
        "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    }


@respx.mock
async def test_model_alias_conflict_error_policy_returns_409() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "alias", "object": "model", "owned_by": "upstream"},
                    {"id": "target", "object": "model", "owned_by": "upstream"},
                ],
            },
        ),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_aliases='{"alias":"target"}',
        alias_conflict_policy="error",
    )

    async with await _client(settings) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "alias_conflict"


@respx.mock
async def test_model_alias_conflict_shadow_policy_replaces_discovery_entry() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "alias", "object": "model", "owned_by": "upstream-original"},
                    {"id": "target", "object": "model", "owned_by": "target-owner"},
                ],
            },
        ),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_aliases='{"alias":"target"}',
        alias_conflict_policy="shadow",
    )

    async with await _client(settings) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    aliases = [entry for entry in response.json()["data"] if entry["id"] == "alias"]
    assert aliases == [{"id": "alias", "object": "model", "owned_by": "target-owner"}]


@respx.mock
async def test_custom_hop_by_hop_headers_are_not_forwarded() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-test"] == "yes"
        assert request.headers.get("connection") != "close"
        assert request.headers.get("content-length") != "999"
        return httpx.Response(200, json={"object": "list", "data": []})

    respx.get("http://upstream.test/v1/models").mock(side_effect=handler)
    settings = Settings(
        upstream_url="http://upstream.test",
        custom_headers='{"Connection":"close","Content-Length":"999","X-Test":"yes"}',
    )

    async with await _client(settings) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200


@respx.mock
async def test_streaming_preserves_reasoning_content_separate_from_content() -> None:
    reasoning_chunk = {
        "id": "chatcmpl-r1",
        "object": "chat.completion.chunk",
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello", "reasoning_content": "Thinking about it"},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(reasoning_chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-r1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    reasoning_text = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "") or "" for p in payloads
    )
    content_text = "".join(p["choices"][0]["delta"].get("content", "") or "" for p in payloads)

    assert "Thinking about it" in reasoning_text
    assert "Thinking about it" not in content_text
    assert content_text == "Hello"


@respx.mock
async def test_streaming_mixed_reasoning_and_content_documents_field_ordering() -> None:
    mixed_chunk = {
        "id": "chatcmpl-r1",
        "object": "chat.completion.chunk",
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "Short answer",
                    "reasoning_content": "Thinking first",
                },
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(mixed_chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-r1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert payloads[0]["choices"][0]["delta"] == {"reasoning_content": "Thinking first"}
    assert payloads[1]["choices"][0]["delta"] == {"content": "Short answer"}


@respx.mock
async def test_streaming_reasoning_tail_does_not_trail_content() -> None:
    """stream_guard must not leave reasoning tails after the answer body."""

    reasoning = "THINK-" + ("R" * 300)
    content = "ANSWER-" + ("C" * 400)
    sse_lines: list[str] = []
    for start in range(0, len(reasoning), 40):
        chunk = {
            "id": "chatcmpl-r-order",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": reasoning[start : start + 40]},
                    "finish_reason": None,
                }
            ],
        }
        sse_lines.append(f"data: {json.dumps(chunk)}\n\n")
    for start in range(0, len(content), 40):
        chunk = {
            "id": "chatcmpl-r-order",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content[start : start + 40]},
                    "finish_reason": None,
                }
            ],
        }
        sse_lines.append(f"data: {json.dumps(chunk)}\n\n")
    finish = {
        "id": "chatcmpl-r-order",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    sse = "".join(sse_lines) + f"data: {json.dumps(finish)}\n\ndata: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    order: list[str] = []
    reasoning_out = []
    content_out = []
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line.removeprefix("data: "))
        delta = payload["choices"][0]["delta"]
        if delta.get("reasoning_content"):
            order.append("R")
            reasoning_out.append(delta["reasoning_content"])
        if delta.get("content"):
            order.append("C")
            content_out.append(delta["content"])

    assert "".join(reasoning_out) == reasoning
    assert "".join(content_out) == content
    assert "R" not in "".join(order).split("C", 1)[1]


def _reasoning_chunk(field: str, text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mix",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}],
    }


class _BoomStream(httpx.AsyncByteStream):
    """Yield one frame then raise, mimicking an upstream that dies mid-stream."""

    def __init__(self, first: bytes, error: Exception | None = None) -> None:
        self.first = first
        self.sent = False
        self.error = error or httpx.ReadError("connection reset by peer")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if not self.sent:
            self.sent = True
            yield self.first
            raise self.error
        return


@respx.mock
async def test_streaming_upstream_error_ends_as_truncated() -> None:
    """A transport failure ends the partial SSE turn without claiming success."""

    first = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-boom",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "thinking..."},
                        "finish_reason": None,
                    }
                ],
            }
        )
        + "\n\n"
    ).encode()

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=_BoomStream(first),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.text.strip().endswith("data: [DONE]")
    payloads = _stream_payloads(response)
    finished = payloads[-1]
    assert finished["choices"][0]["finish_reason"] == "length"
    assert "thinking..." in "".join(
        p["choices"][0]["delta"].get("reasoning_content", "")
        for p in payloads
        if isinstance(p, dict) and p["choices"] and isinstance(p["choices"][0].get("delta"), dict)
    )


@respx.mock
async def test_streaming_upstream_error_before_first_choice_ends_as_truncated() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=_BoomStream(b""),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.text.strip().endswith("data: [DONE]")
    payloads = _stream_payloads(response)
    assert len(payloads) == 1
    assert payloads[0]["choices"][0]["finish_reason"] == "length"


def _tool_arg_frames(fragments: list[str], finish_with_last: bool = False) -> bytes:
    """Build an upstream tool-call stream whose arguments arrive in fragments."""
    frames = [
        {
            "id": "chatcmpl-trunc",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "grep", "arguments": ""},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
        },
    ]
    for position, fragment in enumerate(fragments):
        last = position == len(fragments) - 1
        frames.append(
            {
                "id": "chatcmpl-trunc",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": fragment}},
                            ],
                        },
                        "finish_reason": "tool_calls" if (last and finish_with_last) else None,
                    },
                ],
            },
        )
    if not finish_with_last:
        frames.append(
            {
                "id": "chatcmpl-trunc",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        )
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    return (body + "data: [DONE]\n\n").encode()


def _accumulated_tool_arguments(response: httpx.Response) -> str:
    arguments = ""
    for payload in _stream_payloads(response):
        for choice in payload.get("choices") or []:
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                arguments += (tool_call.get("function") or {}).get("arguments") or ""
    return arguments


def _accumulated_tool_arguments_by_index(response: httpx.Response) -> dict[int, str]:
    arguments: dict[int, str] = {}
    for payload in _stream_payloads(response):
        for choice in payload.get("choices") or []:
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                index = tool_call.get("index", 0)
                fragment = (tool_call.get("function") or {}).get("arguments") or ""
                arguments[index] = arguments.get(index, "") + fragment
    return arguments


@respx.mock
@pytest.mark.parametrize("finish_with_last", [False, True])
async def test_streaming_truncated_tool_arguments_are_completed(finish_with_last: bool) -> None:
    """A turn that closes as ``tool_calls`` must carry parseable arguments.

    Observed against vLLM: the upstream reports success while the streamed JSON
    is cut off mid-string, so the client receives a call it cannot execute and
    that fails validation when the history is replayed.
    """
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_tool_arg_frames(
                ['{"pattern": "def ', "main|uvicorn|FastAPI"],
                finish_with_last=finish_with_last,
            ),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _finish_reasons(response) == ["tool_calls"]
    assert response.text.count("data: [DONE]") == 1
    arguments = _accumulated_tool_arguments(response)
    assert json.loads(arguments) == {"pattern": "def main|uvicorn|FastAPI"}


@respx.mock
async def test_streaming_valid_tool_arguments_are_left_alone() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_tool_arg_frames(['{"pattern": "def ', 'main"}']),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _accumulated_tool_arguments(response) == '{"pattern": "def main"}'


@respx.mock
async def test_streaming_tool_argument_repair_precedes_finish_without_delta() -> None:
    frames = [
        {
            "id": "chatcmpl-null-delta",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "grep", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-null-delta",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a": '}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-null-delta",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "finish_reason": "tool_calls"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    payloads = _stream_payloads(response)
    repair_position = next(
        position
        for position, payload in enumerate(payloads)
        if (payload["choices"][0]["delta"].get("tool_calls"))
        and payload["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments") == "null}"
    )
    finish_position = next(
        position
        for position, payload in enumerate(payloads)
        if payload["choices"][0].get("finish_reason") == "tool_calls"
    )
    assert repair_position < finish_position
    assert json.loads(_accumulated_tool_arguments(response)) == {"a": None}


@respx.mock
async def test_streaming_tool_argument_repair_ignores_invalid_tool_index() -> None:
    frames = [
        {
            "id": "chatcmpl-invalid-index",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "grep", "arguments": '{"a": '},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-invalid-index",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _accumulated_tool_arguments(response) == '{"a": '


@respx.mock
async def test_streaming_multiple_tool_argument_repairs_keep_indexes_separate() -> None:
    frames = [
        {
            "id": "chatcmpl-multi-args",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "read", "arguments": ""},
                            },
                            {
                                "id": "call_b",
                                "index": 1,
                                "type": "function",
                                "function": {"name": "write", "arguments": ""},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-multi-args",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"path": '}},
                            {"index": 1, "function": {"arguments": '{"content": "hi'}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-multi-args",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    arguments = _accumulated_tool_arguments_by_index(response)
    assert json.loads(arguments[0]) == {"path": None}
    assert json.loads(arguments[1]) == {"content": "hi"}
    assert _finish_reasons(response) == ["tool_calls"]


@respx.mock
async def test_streaming_standard_tool_call_without_finish_uses_tool_calls_reason() -> None:
    frames = [
        {
            "id": "chatcmpl-standard-no-finish",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    assert _finish_reasons(response) == ["tool_calls"]


@respx.mock
async def test_streaming_unrepairable_tool_arguments_are_not_guessed() -> None:
    """Refuse to invent a completion; a wrong repair is worse than none."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_tool_arg_frames(['{"a": 1, ']),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _accumulated_tool_arguments(response) == '{"a": 1, '
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_tool_argument_repair_buffer_respects_argument_limit() -> None:
    original = '{"a": 1, '
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_tool_arg_frames([original]),
            headers={"content-type": "text/event-stream"},
        ),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        max_tool_argument_chars=4,
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    # The original delta is still transparent, but the repair accumulator must
    # stop retaining it once the configured bound is exceeded.
    assert _accumulated_tool_arguments(response) == original
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_tool_argument_repair_buffer_respects_tool_call_limit() -> None:
    frames = [
        {
            "id": "chatcmpl-call-limit",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ""}},
                            {"index": 1, "function": {"arguments": ""}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-call-limit",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"a": '}},
                            {"index": 1, "function": {"arguments": '{"b": '}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-call-limit",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )
    settings = Settings(upstream_url="http://upstream.test", max_tool_calls=1)

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    arguments = _accumulated_tool_arguments_by_index(response)
    assert json.loads(arguments[0]) == {"a": None}
    assert arguments[1] == '{"b": '
    assert _finish_reasons(response) == ["tool_calls"]


@respx.mock
async def test_streaming_empty_reasoning_only_turn_gets_a_notice() -> None:
    """A turn that spends its whole budget reasoning leaves the agent nothing.

    The stream is well formed, but with no content and no tool call the client
    has nothing to render and nothing to run, which looks exactly like a hang.
    """
    body = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [
                    {"index": 0, "delta": {"reasoning": "thinking hard..."}, "finish_reason": None},
                ],
            }
        )
        + "\n\ndata: "
        + json.dumps(
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
        )
        + "\n\ndata: [DONE]\n\n"
    ).encode()

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    content = collect_content(_stream_payloads(response))
    assert "proxy:" in content
    assert _finish_reasons(response) == ["length"]
    assert response.text.count("data: [DONE]") == 1
    payloads = _stream_payloads(response)
    notice_position = next(
        position
        for position, payload in enumerate(payloads)
        if "proxy:" in (payload["choices"][0]["delta"].get("content") or "")
    )
    finish_position = next(
        position
        for position, payload in enumerate(payloads)
        if payload["choices"][0].get("finish_reason")
    )
    assert notice_position < finish_position


@respx.mock
async def test_streaming_empty_done_without_a_choice_gets_a_notice() -> None:
    body = (
        b'data: {"id":"chatcmpl-empty","object":"chat.completion.chunk",'
        b'"model":"deepseek-v4","choices":[]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert "proxy:" in collect_content(_stream_payloads(response))
    assert _finish_reasons(response) == ["stop"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_empty_turn_notice_can_be_disabled() -> None:
    body = (
        b'data: {"id":"chatcmpl-empty","object":"chat.completion.chunk",'
        b'"model":"deepseek-v4","choices":[]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(upstream_url="http://upstream.test", empty_turn_notice="")
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert "proxy:" not in response.text
    assert _finish_reasons(response) == []
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_turn_with_content_gets_no_notice() -> None:
    body = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-ok",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [{"index": 0, "delta": {"content": "here you go"}}],
            }
        )
        + "\n\ndata: "
        + json.dumps(
            {
                "id": "chatcmpl-ok",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
        )
        + "\n\ndata: [DONE]\n\n"
    ).encode()

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert collect_content(_stream_payloads(response)) == "here you go"


@respx.mock
async def test_streaming_transport_failure_gets_no_misleading_notice() -> None:
    """The notice blames the token budget, which is wrong for a broken stream."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=_BoomStream(b""),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert "proxy:" not in response.text
    assert _finish_reasons(response) == ["length"]


@respx.mock
async def test_streaming_non_transport_error_still_terminates_the_turn() -> None:
    """A bug in the rewrite path must not strand the client on an open turn.

    The status is already committed to 200/SSE, so re-raising cannot produce an
    HTTP error - it only truncates the body, which is what leaves agent clients
    spinning. The turn is closed and the failure is logged instead.
    """
    first = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-bug",
                "object": "chat.completion.chunk",
                "model": "deepseek-v4",
                "choices": [
                    {"index": 0, "delta": {"content": "partial"}, "finish_reason": None},
                ],
            }
        )
        + "\n\n"
    ).encode()

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=_BoomStream(first, RuntimeError("bug in the rewrite path")),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.text.count("data: [DONE]") == 1
    assert _finish_reasons(response) == ["length"]
    assert "partial" in response.text


@respx.mock
async def test_streaming_reasoning_tool_call_survives_interleaved_content() -> None:
    """Pre-flushing reasoning must not break a raw tool block still being parsed."""

    chunks = [
        _reasoning_chunk("reasoning_content", "I should call a tool. <tool_call>"),
        _reasoning_chunk("reasoning_content", '{"name": "get_weather", "arg'),
        _reasoning_chunk("content", "Meanwhile here is text. "),
        _reasoning_chunk("reasoning_content", 'uments": {"city": "Berlin"}}</tool_call>'),
        {
            "id": "chatcmpl-mix",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response)
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "")
        for p in payloads
        if p["choices"] and isinstance(p["choices"][0].get("delta"), dict)
    )
    names = [
        call["function"]["name"]
        for p in payloads
        if p["choices"] and isinstance(p["choices"][0].get("delta"), dict)
        for call in p["choices"][0]["delta"].get("tool_calls", [])
        if "function" in call and "name" in call["function"]
    ]

    assert names == ["get_weather"]
    assert "<tool_call>" not in reasoning
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


@respx.mock
async def test_streaming_reasoning_precedes_unscanned_content() -> None:
    """Reasoning tails flush first even when content is not a scanned field."""

    reasoning = "R" * 300
    content = "ANSWER " * 40
    chunks = [
        _reasoning_chunk("reasoning_content", reasoning),
        _reasoning_chunk("content", content),
        {
            "id": "chatcmpl-mix",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        tool_call_scan_fields="reasoning,reasoning_content",
    )
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    order: list[str] = []
    reasoning_out: list[str] = []
    content_out: list[str] = []
    for payload in _stream_payloads(response):
        if not payload["choices"]:
            continue
        delta = payload["choices"][0].get("delta")
        if not isinstance(delta, dict):
            continue
        if delta.get("reasoning_content"):
            order.append("R")
            reasoning_out.append(delta["reasoning_content"])
        if delta.get("content"):
            order.append("C")
            content_out.append(delta["content"])

    assert "".join(reasoning_out) == reasoning
    assert "".join(content_out) == content
    assert "R" not in "".join(order).split("C", 1)[1]


@respx.mock
async def test_streaming_false_tool_prefix_does_not_starve() -> None:
    filler = "x" * 200
    pieces = ["I need a tool to do ", filler, ". The end."]

    sse_lines = []
    for piece in pieces:
        chunk = {
            "id": "chatcmpl-fp",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None,
                }
            ],
        }
        sse_lines.append(f"data: {json.dumps(chunk)}\n\n")
    sse = "".join(sse_lines) + "data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "tell me"}],
            },
        )

    assert response.status_code == 200
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content_text = "".join(p["choices"][0]["delta"].get("content", "") or "" for p in payloads)
    expected = "I need a tool to do " + filler + ". The end."
    assert content_text == expected
    # Monitoring triggered by "tool" must not hold the entire stream back until
    # [DONE]; with 220+ buffer chars between the prefix match and stream end,
    # the proxy must flush at least one intermediate content chunk.
    assert len(payloads) >= 2


@respx.mock
async def test_upstream_connection_error_returns_typed_502() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused to secret-host:4000")
    )

    async with await _client(_no_retry_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == {
        "message": "upstream connection refused",
        "type": "connection_refused",
    }
    assert "secret-host" not in json.dumps(body)


@respx.mock
async def test_non_json_upstream_response_passes_through() -> None:
    """Non-JSON content-type from upstream should be passed through unchanged."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b"<html>Error page</html>",
            headers={"content-type": "text/html"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.status_code == 200
    assert response.text == "<html>Error page</html>"


@respx.mock
async def test_malformed_json_upstream_response_passes_through() -> None:
    """Invalid JSON from upstream should be forwarded as-is."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b"{invalid json",
            headers={"content-type": "application/json"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.status_code == 200
    assert response.text == "{invalid json"


@respx.mock
async def test_streaming_upstream_4xx_returns_error_body() -> None:
    """When the upstream returns a 4xx during streaming, the error body is returned."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
            headers={"content-type": "application/json"},
        ),
    )

    async with await _client(_no_retry_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["message"] == "Rate limit exceeded"


@respx.mock
async def test_passthrough_connection_error_returns_502() -> None:
    """Connection errors on passthrough routes should return a typed 502."""
    respx.get("http://upstream.test/v1/models").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    async with await _client() as client:
        response = await client.get("/v1/models")

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "connection_refused"


@respx.mock
async def test_catch_all_streams_request_and_preserves_encoded_response() -> None:
    request_body = b"input" * 1024
    response_body = b"output" * 1024

    async def handler(request: httpx.Request) -> httpx.Response:
        assert await request.aread() == request_body
        return httpx.Response(
            200,
            stream=httpx.ByteStream(gzip.compress(response_body)),
            headers={"content-type": "application/octet-stream", "content-encoding": "gzip"},
        )

    respx.post("http://upstream.test/v1/embeddings").mock(side_effect=handler)

    async with await _client() as client:
        response = await client.post("/v1/embeddings", content=request_body)

    assert response.content == response_body
    assert response.headers["content-encoding"] == "gzip"


@respx.mock
async def test_streaming_connection_error_returns_502() -> None:
    """Connection errors during streaming setup should return a typed 502."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    async with await _client(_no_retry_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "connection_refused"


@respx.mock
async def test_sanitize_tools_strips_all_non_function_tools() -> None:
    """When all tools are non-function, the tools key should be removed entirely."""

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "tools" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {"type": "code_interpreter"},
                    {"type": "retrieval"},
                ],
            },
        )

    assert response.status_code == 200


@respx.mock
async def test_query_string_forwarded_to_upstream() -> None:
    """Query parameters should be preserved when forwarding to upstream."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert "api-version=2024-06-01" in str(request.url)
        return httpx.Response(200, json={"object": "list", "data": []})

    respx.get("http://upstream.test/v1/models").mock(side_effect=handler)

    async with await _client() as client:
        response = await client.get("/v1/models?api-version=2024-06-01")

    assert response.status_code == 200


@respx.mock
async def test_root_models_route_preserves_query_string() -> None:
    """Root-level model discovery should preserve query parameters."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert "api-version=2024-06-01" in str(request.url)
        return httpx.Response(200, json={"object": "list", "data": []})

    respx.get("http://upstream.test/v1/models").mock(side_effect=handler)

    async with await _client() as client:
        response = await client.get("/models?api-version=2024-06-01")

    assert response.status_code == 200


@respx.mock
async def test_streaming_multiple_tool_calls_converted() -> None:
    """Multiple tool calls in a single streaming message should all be converted."""
    content = (
        f"<{BAR}DSML{BAR}tool_calls>"
        f'<name>read</name><parameters>{{"path":"a.py"}}</parameters>'
        f'<name>write</name><parameters>{{"path":"b.py","content":"hi"}}</parameters>'
        f"</{BAR}DSML{BAR}tool_calls>"
    )
    chunk = {
        "id": "chatcmpl-multi",
        "object": "chat.completion.chunk",
        "model": "deepseek",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek",
                "stream": True,
                "messages": [{"role": "user", "content": "do it"}],
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [line.removeprefix("data: ") for line in lines if line != "data: [DONE]"]
    chunks = [json.loads(p) for p in payloads]
    tool_names = [
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
        and "name" in chunk["choices"][0]["delta"]["tool_calls"][0].get("function", {})
    ]
    assert "read" in tool_names
    assert "write" in tool_names
    assert lines[-1] == "data: [DONE]"


@respx.mock
@pytest.mark.parametrize(
    ("opener", "closer"),
    [
        ("<DSML:tool_calls>", "</DSML:tool_calls>"),
        ("<DSML tool_calls>", "</DSML tool_calls>"),
        ("<tool_calls >", "</tool_calls >"),
        ('<tool_call name="ignored">', "</tool_call >"),
    ],
)
async def test_streaming_guard_reassembles_accepted_marker_variants(
    opener: str,
    closer: str,
) -> None:
    block = opener + '<name>read</name><parameters>{"path":"a"}</parameters>' + closer
    split = max(1, len(opener) - 2)
    chunks = [
        {
            "id": "chatcmpl-marker-variant",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {"content": block[:split]}}],
        },
        {
            "id": "chatcmpl-marker-variant",
            "object": "chat.completion.chunk",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {"content": block[split:]}}],
        },
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(sse + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    assert opener not in response.text
    assert closer not in response.text
    tool_names = [
        tool_call["function"]["name"]
        for payload in _stream_payloads(response)
        for tool_call in payload["choices"][0]["delta"].get("tool_calls", [])
        if "name" in tool_call.get("function", {})
    ]
    assert tool_names == ["read"]


@respx.mock
async def test_streaming_trailing_text_after_tool_call_is_preserved() -> None:
    content = (
        "Before "
        '<tool_call><name>read</name><parameters>{"path":"README.md"}</parameters></tool_call>'
        " after."
    )
    chunk = {
        "id": "chatcmpl-trailing",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    finish_chunk = {
        "id": "chatcmpl-trailing",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(finish_chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "read"}],
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response)
    content_text = "".join(
        payload["choices"][0]["delta"].get("content", "") or "" for payload in payloads
    )
    assert content_text == "Before  after."
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert [line for line in response.text.splitlines() if line][-1] == "data: [DONE]"


@respx.mock
async def test_streaming_long_raw_tool_block_does_not_leak_markup() -> None:
    long_value = "x" * 500
    first = {
        "id": "chatcmpl-long",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": (
                        '<tool_call><name>write</name><parameters>{"content":"' + long_value[:250]
                    )
                },
                "finish_reason": None,
            }
        ],
    }
    second = {
        "id": "chatcmpl-long",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": long_value[250:] + '"}</parameters></tool_call>'},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(first)}\n\ndata: {json.dumps(second)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "write"}],
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response)
    assert "<tool_call>" not in response.text
    streamed_arguments = "".join(
        payload["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    )
    assert json.loads(streamed_arguments) == {"content": long_value}


@respx.mock
async def test_streaming_oversized_raw_tool_block_passes_through_as_text() -> None:
    content = (
        '<tool_call><name>write</name><parameters>{"content":"abcdef"}</parameters></tool_call>'
    )
    chunk = {
        "id": "chatcmpl-limit",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        max_raw_tool_block_chars=10,
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "write"}],
            },
        )

    assert response.status_code == 200
    content_text = "".join(
        payload["choices"][0]["delta"].get("content", "") or ""
        for payload in _stream_payloads(response)
    )
    assert content_text == content
    assert '"tool_calls"' not in response.text


@respx.mock
async def test_streaming_over_tool_call_count_limit_passes_through_as_text() -> None:
    content = (
        "<tool_calls>"
        "<name>read</name><parameters>{}</parameters>"
        "<name>write</name><parameters>{}</parameters>"
        "</tool_calls>"
    )
    chunk = {
        "id": "chatcmpl-count-limit",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )
    settings = Settings(upstream_url="http://upstream.test", max_tool_calls=1)

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "write"}],
            },
        )

    assert response.status_code == 200
    assert content in response.text
    assert '"tool_calls"' not in response.text


@respx.mock
async def test_streaming_upstream_disconnect_mid_tool_block_flushes_text_and_done() -> None:
    content = '<tool_call><name>write</name><parameters>{"content":"unfinished"'
    chunk = {
        "id": "chatcmpl-disconnect",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(chunk)}\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "write"}],
            },
        )

    assert response.status_code == 200
    content_text = "".join(
        payload["choices"][0]["delta"].get("content", "") or ""
        for payload in _stream_payloads(response)
    )
    assert content_text == content
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_repairs_multiple_choice_indexes() -> None:
    chunk = {
        "id": "chatcmpl-choices",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": (
                        "<tool_call><name>read</name>"
                        '<parameters>{"path":"a"}</parameters></tool_call>'
                    )
                },
                "finish_reason": None,
            },
            {
                "index": 1,
                "delta": {
                    "content": (
                        "<tool_call><name>write</name>"
                        '<parameters>{"path":"b"}</parameters></tool_call>'
                    )
                },
                "finish_reason": None,
            },
        ],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "do both"}],
            },
        )

    assert response.status_code == 200
    names_by_choice = {
        payload["choices"][0]["index"]: payload["choices"][0]["delta"]["tool_calls"][0]["function"][
            "name"
        ]
        for payload in _stream_payloads(response)
        if payload["choices"][0]["delta"].get("tool_calls")
        and "name" in payload["choices"][0]["delta"]["tool_calls"][0].get("function", {})
    }
    assert names_by_choice == {0: "read", 1: "write"}


@respx.mock
async def test_streaming_reasoning_content_tool_call_is_converted() -> None:
    chunk = {
        "id": "chatcmpl-reasoning-tool",
        "object": "chat.completion.chunk",
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": (
                        "<tool_call><name>read</name>"
                        '<parameters>{"path":"README.md"}</parameters></tool_call>'
                    )
                },
                "finish_reason": None,
            }
        ],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-r1",
                "stream": True,
                "messages": [{"role": "user", "content": "read"}],
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response)
    assert "reasoning_content" not in response.text
    tool_names = [
        payload["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
        and "name" in payload["choices"][0]["delta"]["tool_calls"][0].get("function", {})
    ]
    assert tool_names == ["read"]


@respx.mock
async def test_e2e_opencode_like_stream_with_alias_and_fixture() -> None:
    fixture_content = (FIXTURE_DIR / "qwen_json.txt").read_text()

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "DeepSeek-V4-Flash"
        chunk = {
            "id": "chatcmpl-e2e",
            "object": "chat.completion.chunk",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": fixture_content},
                    "finish_reason": None,
                }
            ],
        }
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)
    settings = Settings(
        upstream_url="http://upstream.test",
        model_aliases='{"dsv4-flash":"DeepSeek-V4-Flash"}',
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "dsv4-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "search"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response)
    tool_names = [
        payload["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
        and "name" in payload["choices"][0]["delta"]["tool_calls"][0].get("function", {})
    ]
    assert tool_names == ["search"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_non_streaming_request_without_body() -> None:
    """A request with an empty body should still be forwarded."""
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "Missing model field"}},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"",
        )

    assert response.status_code == 400


@respx.mock
async def test_non_streaming_laguna_tool_call_is_converted() -> None:
    upstream = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-laguna-1",
                "object": "chat.completion",
                "model": "laguna",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call>terminal"
                                "<arg_key>cmd</arg_key><arg_value>uname -a</arg_value>"
                                "</tool_call>"
                            ),
                        },
                        "finish_reason": "stop",
                    },
                ],
            },
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "laguna", "messages": [{"role": "user", "content": "check OS"}]},
        )

    assert upstream.called
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "terminal",
        "arguments": '{"cmd":"uname -a"}',
    }


@respx.mock
async def test_streaming_laguna_tool_call_is_converted() -> None:
    first_chunk = {
        "id": "chatcmpl-laguna-stream",
        "object": "chat.completion.chunk",
        "model": "laguna",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    raw_tool_chunk = {
        "id": "chatcmpl-laguna-stream",
        "object": "chat.completion.chunk",
        "model": "laguna",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": (
                        "<tool_call>terminal"
                        "<arg_key>cmd</arg_key><arg_value>pwd</arg_value>"
                        "</tool_call>"
                    ),
                },
                "finish_reason": None,
            },
        ],
    }
    sse = (
        f"data: {json.dumps(first_chunk)}\n\ndata: {json.dumps(raw_tool_chunk)}\n\ndata: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "laguna",
                "stream": True,
                "messages": [{"role": "user", "content": "where"}],
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [line.removeprefix("data: ") for line in lines if line != "data: [DONE]"]
    chunks = [json.loads(payload) for payload in payloads]
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
        or chunk["choices"][0]["finish_reason"] == "tool_calls"
    ]
    assert tool_chunks
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "terminal"
    streamed_arguments = "".join(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for chunk in tool_chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    )
    assert json.loads(streamed_arguments) == {"cmd": "pwd"}
    assert lines[-1] == "data: [DONE]"


@respx.mock
async def test_streaming_adjacent_raw_blocks_use_distinct_tool_indexes() -> None:
    content = (
        '<tool_call><name>read</name><parameters>{"path":"a"}</parameters></tool_call>'
        '<tool_call><name>write</name><parameters>{"path":"b"}</parameters></tool_call>'
    )
    chunk = {
        "id": "chatcmpl-adjacent",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    named_calls = [
        tool_call
        for payload in _stream_payloads(response)
        for tool_call in payload["choices"][0]["delta"].get("tool_calls", [])
        if "name" in tool_call.get("function", {})
    ]
    assert [(call["function"]["name"], call["index"]) for call in named_calls] == [
        ("read", 0),
        ("write", 1),
    ]


@respx.mock
async def test_streaming_close_tag_split_mid_buffer_is_converted() -> None:
    """A close tag whose `</` lands mid-frame must still complete the block.

    Observed against real vLLM: the close tag streams as `</|DSML|tool_c` then
    `alls>`, so `</` is several characters before the frame boundary. The guard
    must re-parse whenever any close-tag start is held, not only when `</`
    straddles the boundary.
    """
    block = (
        "<|DSML|tool_calls><name>read</name>"
        '<parameters>{"path":"a"}</parameters></|DSML|tool_calls>'
    )
    # First frame ends with `</|DSML|tool_c`; the completing `alls>` arrives alone.
    split = len(block) - 5
    frames = [
        {
            "id": "chatcmpl-split-close",
            "object": "chat.completion.chunk",
            "model": "deepseek",
            "choices": [{"index": 0, "delta": {"content": block[:split]}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-split-close",
            "object": "chat.completion.chunk",
            "model": "deepseek",
            "choices": [{"index": 0, "delta": {"content": block[split:]}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-split-close",
            "object": "chat.completion.chunk",
            "model": "deepseek",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek", "stream": True, "messages": []},
        )

    payloads = _stream_payloads(response)
    leaked = "".join(
        payload["choices"][0]["delta"].get("content", "") or "" for payload in payloads
    )
    assert leaked == ""
    named = [
        tool_call
        for payload in payloads
        for tool_call in payload["choices"][0]["delta"].get("tool_calls", [])
        if "name" in tool_call.get("function", {})
    ]
    assert [(call["function"]["name"], call["index"]) for call in named] == [("read", 0)]
    assert _finish_reasons(response) == ["tool_calls"]


@respx.mock
async def test_compressed_chat_response_drops_stale_content_headers() -> None:
    payload = json.dumps({"id": "chatcmpl-gzip", "choices": []}).encode()
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=gzip.compress(payload),
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "etag": '"compressed-body"',
            },
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )

    assert response.json()["id"] == "chatcmpl-gzip"
    assert "content-encoding" not in response.headers
    assert "etag" not in response.headers


@respx.mock
async def test_streaming_scanned_content_preserves_choice_and_event_metadata() -> None:
    chunk = {
        "id": "chatcmpl-metadata",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "qwen",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "logprobs": {"content": [{"token": "hello", "logprob": -0.1}]},
                "finish_reason": None,
            }
        ],
    }
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": []},
        )

    payloads = _stream_payloads(response)
    assert any(payload.get("created") == 123 for payload in payloads)
    assert any(payload.get("system_fingerprint") == "fp_test" for payload in payloads)
    assert any(payload["choices"][0].get("logprobs") for payload in payloads)


@respx.mock
async def test_streaming_null_choice_extras_do_not_emit_empty_deltas() -> None:
    """vLLM sends logprobs/stop_reason as null on every choice; they carry nothing."""

    chunks = [
        {
            "id": "chatcmpl-vllm",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text, "reasoning_content": None},
                    "logprobs": None,
                    "finish_reason": None,
                    "stop_reason": None,
                }
            ],
        }
        for text in ("hello ", "world")
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    payloads = _stream_payloads(response)
    deltas = [payload["choices"][0]["delta"] for payload in payloads if payload["choices"]]
    assert "".join(delta.get("content", "") for delta in deltas) == "hello world"
    # The terminal finish chunk carries an empty delta by design; nothing before it should.
    content_deltas = [
        payload["choices"][0]["delta"]
        for payload in payloads
        if payload["choices"] and not payload["choices"][0].get("finish_reason")
    ]
    assert all(delta for delta in content_deltas)
    assert _finish_reasons(response) == ["stop"]


def _finish_reasons(response: httpx.Response) -> list[str]:
    return [
        choice["finish_reason"]
        for payload in _stream_payloads(response)
        for choice in payload.get("choices", [])
        if choice.get("finish_reason")
    ]


@respx.mock
async def test_streaming_without_upstream_finish_reason_still_closes_the_turn() -> None:
    chunk = {
        "id": "chatcmpl-nofinish",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "done thinking"}, "finish_reason": None}],
    }
    sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _finish_reasons(response) == ["stop"]
    assert [line for line in response.text.splitlines() if line][-1] == "data: [DONE]"


@respx.mock
async def test_streaming_truncated_upstream_stream_still_closes_the_turn() -> None:
    chunk = {
        "id": "chatcmpl-truncated",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "partial answer"}, "finish_reason": None}],
    }
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _finish_reasons(response) == ["length"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_empty_upstream_stream_ends_as_truncated() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b"",
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert _finish_reasons(response) == ["length"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_streaming_idle_upstream_terminates_instead_of_hanging() -> None:
    chunk = {
        "id": "chatcmpl-idle",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "answer text"}, "finish_reason": "stop"}],
    }

    async def stalling_stream() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        # Upstream stops sending without closing the connection: without an idle
        # guard the proxy would hold the client stream open forever.
        await asyncio.sleep(30)
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=stalling_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        upstream_stream_idle_timeout=0.2,
    )
    async with await _client(settings) as client:
        response = await asyncio.wait_for(
            client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4", "stream": True, "messages": []},
            ),
            timeout=10,
        )

    assert _finish_reasons(response) == ["stop"]
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_slow_prefill_is_not_killed_by_the_between_frame_timeout() -> None:
    """A long silence before the first frame is prefill, not a stall."""
    chunk = {
        "id": "chatcmpl-prefill",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "answer text"}, "finish_reason": "stop"}],
    }

    async def slow_prefill_stream() -> AsyncIterator[bytes]:
        # Longer than the between-frame budget, well inside the first-frame one.
        await asyncio.sleep(0.5)
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=slow_prefill_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        upstream_stream_idle_timeout=0.2,
        upstream_stream_first_frame_timeout=10,
    )
    async with await _client(settings) as client:
        response = await asyncio.wait_for(
            client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4", "stream": True, "messages": []},
            ),
            timeout=10,
        )

    assert _finish_reasons(response) == ["stop"]
    assert "answer text" in response.text


@respx.mock
async def test_prefill_that_never_produces_a_frame_hits_the_first_frame_timeout() -> None:
    async def silent_stream() -> AsyncIterator[bytes]:
        await asyncio.sleep(30)
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=silent_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        upstream_stream_idle_timeout=10,
        upstream_stream_first_frame_timeout=0.2,
    )
    async with await _client(settings) as client:
        response = await asyncio.wait_for(
            client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4", "stream": True, "messages": []},
            ),
            timeout=10,
        )

    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_idle_termination_records_the_phase_it_stalled_in() -> None:
    chunk = {
        "id": "chatcmpl-phase",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "partial"}}],
    }

    async def stalling_stream() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(30)
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=stalling_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        upstream_stream_idle_timeout=0.2,
        upstream_stream_first_frame_timeout=10,
    )
    async with await _client(settings) as client:
        await asyncio.wait_for(
            client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4", "stream": True, "messages": []},
            ),
            timeout=10,
        )
        metrics = (await client.get("/metrics")).text

    assert 'opencode_proxy_stream_idle_terminations_total{phase="mid_stream"} 1.0' in metrics


@respx.mock
async def test_streaming_response_disables_intermediary_buffering() -> None:
    chunk = {
        "id": "chatcmpl-headers",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


@respx.mock
async def test_quiet_upstream_gets_keepalive_comments() -> None:
    chunk = {
        "id": "chatcmpl-keepalive",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "thinking done"}, "finish_reason": "stop"}],
    }

    async def slow_stream() -> AsyncIterator[bytes]:
        # A long reasoning pause before the first token.
        await asyncio.sleep(0.35)
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=slow_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        sse_keepalive_interval=0.1,
        upstream_stream_idle_timeout=10,
    )
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert ": keepalive" in response.text
    # The pause must not cost any content: the frame still arrives intact.
    assert collect_content(_stream_payloads(response)) == "thinking done"
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_keepalive_header_promotes_comments_to_chunks() -> None:
    """Opt-in keepalives must be frames an SSE parser actually surfaces.

    Comments are dropped by SSE parsers, so a client watching for forward
    progress sees nothing through a long prefill. With the header set the same
    quiet period must produce real chunks instead, sharing the stream's id and
    carrying the requested model, and adding no content of their own.
    """
    chunk = {
        "id": "chatcmpl-keepalive",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "thinking done"}, "finish_reason": "stop"}],
    }

    async def slow_stream() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.35)
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=slow_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        sse_keepalive_interval=0.1,
        upstream_stream_idle_timeout=10,
    )
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
            headers={"x-opencode-proxy-keepalive": "chunk"},
        )

    assert ": keepalive" not in response.text

    payloads = _stream_payloads(response)
    keepalives = [
        p
        for p in payloads
        if p.get("choices")
        and p["choices"][0].get("delta") == {}
        and p["choices"][0].get("finish_reason") is None
    ]
    assert keepalives, "expected at least one empty-delta keepalive chunk"
    # These keepalives all precede the first upstream frame, so they carry the
    # proxy's synthesized id: the upstream id genuinely is not known yet.
    # `chunk_id` adopts the upstream id as soon as a data frame arrives, so
    # keepalives emitted mid-stream do match the chunks around them. Either way
    # the id is stable within the keepalive run, and the model is the one the
    # caller asked for rather than the "unknown" placeholder.
    assert len({p["id"] for p in keepalives}) == 1
    assert {p["model"] for p in keepalives} == {"deepseek-v4"}

    # Keepalives are additive only: content and termination are untouched.
    assert collect_content(payloads) == "thinking done"
    assert response.text.count("data: [DONE]") == 1


@respx.mock
async def test_keepalive_control_header_is_not_forwarded_upstream() -> None:
    """The control header is for this proxy, not part of the upstream contract."""
    seen: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(
            200,
            content=b'data: {"id":"c","object":"chat.completion.chunk","model":"m",'
            b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=_capture)

    settings = Settings(upstream_url="http://upstream.test")
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
            headers={"x-opencode-proxy-keepalive": "chunk"},
        )

    assert response.status_code == 200
    assert "x-opencode-proxy-keepalive" not in seen


@respx.mock
async def test_upstream_503_is_retried_before_the_stream_starts() -> None:
    chunk = {
        "id": "chatcmpl-retry",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "second try"}, "finish_reason": "stop"}],
    }
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": "overloaded"}),
            httpx.Response(
                200,
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
                headers={"content-type": "text/event-stream"},
            ),
        ],
    )

    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=1)
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    assert response.status_code == 200
    assert route.call_count == 2
    assert collect_content(_stream_payloads(response)) == "second try"


@respx.mock
async def test_transport_error_is_retried_for_buffered_requests() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json={"choices": [], "model": "deepseek-v4"}),
        ],
    )

    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=1)
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )

    assert response.status_code == 200
    assert route.call_count == 2


@respx.mock
async def test_failure_after_the_stream_started_is_not_retried() -> None:
    chunk = {
        "id": "chatcmpl-midfail",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
    }

    async def failing_stream() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        raise httpx.RemoteProtocolError("peer closed connection")

    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=failing_stream(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=2)
    async with await _client(settings) as client:
        with contextlib.suppress(httpx.RemoteProtocolError):
            await client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4", "stream": True, "messages": []},
            )

    # Re-sending would replay a prompt whose answer the client already saw part of.
    assert route.call_count == 1


@respx.mock
async def test_retries_give_up_and_forward_the_upstream_error() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "still overloaded"}}),
    )

    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=1)
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )

    assert route.call_count == 2
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "still overloaded"


def _deepseek_settings(**overrides: Any) -> Settings:
    return Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps({"deepseek-v4": {"compatibility": "deepseek_v4"}}),
        **overrides,
    )


@respx.mock
async def test_request_normalization_repairs_messages_before_forwarding() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-normalize",
                "model": "deepseek-v4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    },
                ],
            },
        ),
    )

    async with await _client(_deepseek_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "reasoning_effort": "off",
                "messages": [
                    {"role": "user", "content": "list the files"},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "needed for the passback",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "ls", "arguments": "{}"},
                            },
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": ""},
                    {"role": "assistant", "content": "done", "reasoning_content": "stale"},
                ],
            },
        )
        metrics = (await client.get("/metrics")).text

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assistant_tool_turn, tool_result, plain_turn = forwarded["messages"][1:]
    assert assistant_tool_turn["content"] == ""
    assert assistant_tool_turn["reasoning_content"] == "needed for the passback"
    assert tool_result["content"] == "(no output)"
    assert "reasoning_content" not in plain_turn
    assert "reasoning_effort" not in forwarded
    assert forwarded["thinking"] == {"type": "disabled"}
    assert 'opencode_proxy_request_normalizations_total{kind="null_assistant_content"} 1.0' in (
        metrics
    )


@respx.mock
async def test_request_normalization_can_be_disabled() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [], "model": "deepseek-v4"}),
    )

    settings = _deepseek_settings(normalize_requests=False)
    async with await _client(settings) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4",
                "reasoning_effort": "off",
                "messages": [{"role": "assistant", "content": None}],
            },
        )

    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["messages"][0]["content"] is None
    assert forwarded["reasoning_effort"] == "off"


@respx.mock
async def test_request_normalization_is_scoped_to_compatibility_profiles() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [], "model": "qwen"}),
    )

    async with await _client() as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "max_completion_tokens": 123,
                "messages": [
                    {"role": "developer", "content": "provider-owned instructions"},
                    {
                        "role": "assistant",
                        "content": "provider-owned history",
                        "reasoning_content": "keep this field for qwen",
                    },
                ],
            },
        )

    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["max_completion_tokens"] == 123
    assert forwarded["messages"][0]["role"] == "developer"
    assert forwarded["messages"][1]["reasoning_content"] == "keep this field for qwen"


def _buffered_completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "model": "deepseek-v4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            },
        ],
    }


@respx.mock
async def test_empty_buffered_completion_is_retried() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_buffered_completion("")),
            httpx.Response(200, json=_buffered_completion("second try")),
        ],
    )

    async with await _client(_deepseek_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    assert route.call_count == 2
    assert response.json()["choices"][0]["message"]["content"] == "second try"
    assert 'opencode_proxy_upstream_retries_total{reason="empty_response"} 1.0' in metrics


@respx.mock
async def test_exhausted_empty_completion_retries_annotate_the_turn() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("")),
    )

    async with await _client(_deepseek_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    assert route.call_count == 2
    assert "[proxy:" in response.json()["choices"][0]["message"]["content"]
    assert "opencode_proxy_empty_turns_total 1.0" in metrics


@respx.mock
async def test_empty_completion_retry_can_be_disabled() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("")),
    )

    settings = _deepseek_settings(empty_response_retries=0)
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )

    assert route.call_count == 1
    assert "[proxy:" in response.json()["choices"][0]["message"]["content"]


@respx.mock
async def test_empty_completion_retry_is_scoped_to_deepseek_profiles() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("")),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": []},
        )

    assert route.call_count == 1
    assert "[proxy:" in response.json()["choices"][0]["message"]["content"]


@respx.mock
async def test_length_truncated_completion_is_not_retried() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("", finish_reason="length")),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )

    assert route.call_count == 1
    assert response.json()["choices"][0]["message"]["content"] == ""


@respx.mock
async def test_abnormal_finish_reason_replaces_the_budget_notice() -> None:
    chunk = {
        "id": "chatcmpl-abnormal",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "insufficient_system_resource"},
        ],
    }
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    content = collect_content(_stream_payloads(response))
    assert "finish_reason=insufficient_system_resource" in content
    assert "token budget" not in content
    assert 'opencode_proxy_finish_reasons_total{reason="other",transport="streaming"} 1.0' in (
        metrics
    )


@respx.mock
async def test_empty_reasoning_delta_opens_no_reasoning_block() -> None:
    chunks = [
        {
            "id": "chatcmpl-think",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {"reasoning_content": ""}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-think",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "thinking"}, "finish_reason": None},
            ],
        },
        {
            "id": "chatcmpl-think",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}],
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )

    payloads = _stream_payloads(response)
    reasoning_deltas = [
        choice["delta"]["reasoning_content"]
        for payload in payloads
        for choice in payload.get("choices", [])
        if isinstance(choice.get("delta"), dict) and "reasoning_content" in choice["delta"]
    ]
    # The first thinking chunk of a DeepSeek turn is an empty string; forwarding
    # it opens an empty reasoning block in the client.
    assert reasoning_deltas == ["thinking"]
    assert collect_content(payloads) == "answer"


@respx.mock
async def test_retry_after_header_paces_the_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("opencode_proxy.proxy.asyncio.sleep", record_sleep)
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": "slow down"}, headers={"Retry-After": "7"}),
            httpx.Response(200, json=_buffered_completion("after the wait")),
        ],
    )

    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=1)
    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    assert route.call_count == 2
    # The header wins over the proxy's own backoff curve.
    assert delays == [7.0]
    assert response.json()["choices"][0]["message"]["content"] == "after the wait"
    assert 'opencode_proxy_upstream_retries_total{reason="http_429"} 1.0' in metrics


@respx.mock
async def test_upstream_error_status_is_classified() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "This model's maximum context length is 1000000 tokens"}},
        ),
    )

    async with await _client(_no_retry_settings()) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    assert response.status_code == 400
    assert 'opencode_proxy_upstream_errors_total{type="context_window_exceeded"} 1.0' in metrics


@respx.mock
async def test_streamed_usage_is_counted_disjointly() -> None:
    content_chunk = {
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    usage_chunk = {
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4",
        "choices": [],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 40,
            "prompt_cache_hit_tokens": 900,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
    }
    body = (
        f"data: {json.dumps(content_chunk)}\n\ndata: {json.dumps(usage_chunk)}\n\ndata: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        ),
    )

    async with await _client() as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4", "stream": True, "messages": []},
        )
        metrics = (await client.get("/metrics")).text

    # prompt_tokens includes the cache hits, so input is the miss share only.
    assert 'opencode_proxy_usage_tokens_total{kind="input"} 100.0' in metrics
    assert 'opencode_proxy_usage_tokens_total{kind="cache_read"} 900.0' in metrics
    assert 'opencode_proxy_usage_tokens_total{kind="output"} 40.0' in metrics
    assert 'opencode_proxy_usage_tokens_total{kind="reasoning"} 25.0' in metrics


@respx.mock
async def test_chat_template_transport_forwards_the_vllm_thinking_argument() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("ok")),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps(
            {
                "deepseek-v4-flash": {
                    "compatibility": "deepseek_v4",
                    "thinking_transport": "chat_template_kwargs",
                },
            },
        ),
    )
    async with await _client(settings) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "reasoning_effort": "off",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        metrics = (await client.get("/metrics")).text

    forwarded = json.loads(route.calls[0].request.content)
    # vLLM ignores the DeepSeek API's top-level field, so the profile targets
    # the chat-template argument instead.
    assert forwarded["chat_template_kwargs"] == {"thinking": False}
    assert "thinking" not in forwarded
    assert "reasoning_effort" not in forwarded
    assert 'opencode_proxy_request_normalizations_total{kind="thinking_disabled"} 1.0' in metrics


@respx.mock
async def test_chat_template_transport_converts_official_thinking_field() -> None:
    route = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_buffered_completion("ok")),
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps(
            {
                "deepseek-v4-flash": {
                    "compatibility": "deepseek_v4",
                    "thinking_transport": "chat_template_kwargs",
                },
            },
        ),
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["chat_template_kwargs"] == {"thinking": False}
    assert "thinking" not in forwarded
