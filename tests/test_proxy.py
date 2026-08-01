from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
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

    assert _finish_reasons(response) == ["stop"]
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
