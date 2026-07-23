from __future__ import annotations

import json

import httpx
import respx

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings


async def _client(settings: Settings | None = None) -> httpx.AsyncClient:
    app = create_app(settings or Settings(upstream_url="http://upstream.test"))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


@respx.mock
async def test_ollama_chat_non_stream_repairs_raw_tool_call() -> None:
    upstream = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call><name>read</name><parameters>"
                                '{"path":"README.md"}</parameters></tool_call>'
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    async with await _client() as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "read"}],
                "stream": False,
            },
        )

    assert upstream.called
    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["message"]["content"] == ""
    assert body["message"]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "README.md"},
    }


@respx.mock
async def test_ollama_chat_stream_converts_sse_to_ndjson_and_keeps_usage_until_done() -> None:
    chunks = [
        {
            "id": "chat-1",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat-1",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat-1",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        },
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )

    async with await _client() as client:
        response = await client.post(
            "/api/chat",
            json={"model": "qwen", "messages": [{"role": "user", "content": "read"}]},
        )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines[0]["message"]["content"] == "hello"
    assert lines[-1]["done"] is True
    assert lines[-1]["prompt_eval_count"] == 2
    tool_lines = [line for line in lines if line.get("message", {}).get("tool_calls")]
    assert tool_lines[0]["message"]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "README.md"},
    }


@respx.mock
async def test_ollama_generate_and_embed_translate_upstream_requests() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "world"}, "finish_reason": "stop"}]},
        )

    chat = respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=chat_handler)
    embed = respx.post("http://upstream.test/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
    )

    async with await _client() as client:
        generate = await client.post(
            "/api/generate",
            json={"model": "qwen", "prompt": "hello", "stream": False},
        )
        embeddings = await client.post("/api/embed", json={"model": "embed", "input": "hello"})

    assert chat.called
    assert embed.called
    assert generate.json()["response"] == "world"
    assert embeddings.json()["embeddings"] == [[0.1, 0.2]]


@respx.mock
async def test_ollama_tags_applies_model_aliases_and_forwards_client_auth() -> None:
    upstream = respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "target", "created": 1}]})
    )
    settings = Settings(upstream_url="http://upstream.test", model_aliases="alias=target")

    async with await _client(settings) as client:
        response = await client.get("/api/tags", headers={"Authorization": "Bearer client-key"})

    assert response.status_code == 200
    assert {model["name"] for model in response.json()["models"]} == {"target", "alias"}
    assert upstream.calls[0].request.headers["authorization"] == "Bearer client-key"


async def test_ollama_noop_endpoints_and_health() -> None:
    async with await _client() as client:
        assert (await client.get("/")).text == "Ollama is running"
        assert (await client.get("/api/version")).json()["version"] == "0.5.1"
        assert (await client.post("/api/pull", json={"name": "qwen"})).json() == {
            "status": "success"
        }
        assert (await client.post("/api/chat", json={"model": "qwen", "messages": []})).json()[
            "done_reason"
        ] == "load"
