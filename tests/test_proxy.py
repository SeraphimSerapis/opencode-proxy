from __future__ import annotations

import json

import httpx
import respx

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings

BAR = "\uff5c"


async def _client(settings: Settings | None = None) -> httpx.AsyncClient:
    app = create_app(settings or Settings(upstream_url="http://upstream.test"))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://proxy.test")


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
async def test_upstream_connection_error_returns_generic_502() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused to secret-host:4000")
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["message"] == "upstream request failed"
    assert "secret-host" not in json.dumps(body)
    assert "connection refused" not in json.dumps(body)
