from __future__ import annotations

import json

import httpx
import respx

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings

BAR = "\uff5c"


async def _client() -> httpx.AsyncClient:
    app = create_app(Settings(upstream_url="http://upstream.test"))
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
