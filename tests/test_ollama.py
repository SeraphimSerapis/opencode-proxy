from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from opencode_proxy.app import create_app
from opencode_proxy.compat import (
    DSML_INVOKE_CLOSE,
    DSML_INVOKE_OPEN,
    DSML_PARAMETER_CLOSE,
    DSML_PARAMETER_OPEN,
)
from opencode_proxy.settings import Settings


async def _client(settings: Settings | None = None) -> httpx.AsyncClient:
    app = create_app(settings or Settings(upstream_url="http://upstream.test"))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


def _orphan_tool_call() -> str:
    return (
        f'{DSML_INVOKE_OPEN} name="read">'
        f'{DSML_PARAMETER_OPEN} name="path">README.md{DSML_PARAMETER_CLOSE}'
        f"{DSML_INVOKE_CLOSE}"
    )


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
        },
        {
            "id": "chat-1",
            "model": "qwen",
            "choices": [],
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
    assert lines[-1]["message"]["content"] == ""
    assert lines[-1]["prompt_eval_count"] == 2
    tool_lines = [line for line in lines if line.get("message", {}).get("tool_calls")]
    assert tool_lines[0]["message"]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "README.md"},
    }


@respx.mock
async def test_ollama_generate_stream_defers_done_until_usage_chunk() -> None:
    chunks = [
        {
            "id": "generate-1",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
        },
        {
            "id": "generate-1",
            "model": "qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "generate-1",
            "model": "qwen",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
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
            "/api/generate",
            json={"model": "qwen", "prompt": "question"},
        )

    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines[0]["response"] == "answer"
    assert lines[-1]["done"] is True
    assert lines[-1]["prompt_eval_count"] == 4
    assert lines[-1]["eval_count"] == 6


@respx.mock
async def test_ollama_generate_and_embed_translate_upstream_requests() -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        assert len(request.headers.get_list("content-type")) == 1
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
async def test_ollama_deepseek_profile_normalizes_thinking_history_and_retries_empty_turn() -> None:
    forwarded: list[dict[str, Any]] = []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        if len(forwarded) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        },
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    upstream = respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=chat_handler,
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
            "/api/chat",
            json={
                "model": "deepseek-v4-flash",
                "think": False,
                "stream": False,
                "messages": [
                    {"role": "user", "content": "call the tool"},
                    {
                        "role": "assistant",
                        "content": None,
                        "thinking": "I should call it",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "echo",
                                    "arguments": {"value": "x"},
                                },
                            },
                        ],
                    },
                    {"role": "tool", "tool_name": "echo", "content": ""},
                ],
            },
        )
        metrics = (await client.get("/metrics")).text

    assert upstream.call_count == 2
    assert response.json()["message"]["content"] == "ok"
    first = forwarded[0]
    assert first["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in first
    assistant, tool = first["messages"][1:]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I should call it"
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]
    assert tool["content"] == "(no output)"
    assert 'opencode_proxy_request_normalizations_total{kind="thinking_disabled"} 1.0' in metrics
    assert 'opencode_proxy_upstream_retries_total{reason="empty_response"} 1.0' in metrics


@respx.mock
async def test_ollama_streaming_upstream_error_is_forwarded_and_counted() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "invalid request"}},
            headers={"content-type": "application/json"},
        )
    )
    settings = Settings(upstream_url="http://upstream.test", upstream_max_retries=0)

    async with await _client(settings) as client:
        response = await client.post(
            "/api/chat",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        metrics = (await client.get("/metrics")).text

    assert response.status_code == 400
    assert response.json() == {"error": {"message": "invalid request"}}
    assert 'opencode_proxy_upstream_errors_total{type="invalid_request"} 1.0' in metrics


@respx.mock
async def test_ollama_think_translation_survives_message_normalization_opt_out() -> None:
    upstream = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        normalize_requests=False,
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
            "/api/chat",
            json={
                "model": "deepseek-v4-flash",
                "think": False,
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    forwarded = json.loads(upstream.calls[0].request.content)
    assert forwarded["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in forwarded


@respx.mock
async def test_ollama_buffered_deepseek_profile_recovers_declared_orphan_call() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": _orphan_tool_call()},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps(
            {
                "deepseek-v4-flash": {
                    "compatibility": "deepseek_v4",
                    "recover_orphan_invokes": True,
                },
            },
        ),
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        metrics = (await client.get("/metrics")).text

    assert response.json()["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert 'opencode_proxy_orphan_recovery_total{outcome="accepted",reason="valid"} 1.0' in metrics


@respx.mock
async def test_ollama_streaming_deepseek_profile_recovers_declared_orphan_call() -> None:
    frame = {
        "id": "chatcmpl-orphan",
        "object": "chat.completion.chunk",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "delta": {"content": _orphan_tool_call()},
                "finish_reason": "stop",
            }
        ],
    }
    sse = f"data: {json.dumps(frame, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )
    settings = Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps(
            {
                "deepseek-v4-flash": {
                    "compatibility": "deepseek_v4",
                    "recover_orphan_invokes": True,
                },
            },
        ),
    )

    async with await _client(settings) as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

    lines = [json.loads(line) for line in response.text.splitlines() if line]
    calls = [call for line in lines for call in line.get("message", {}).get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "read"


@respx.mock
async def test_ollama_tags_applies_model_aliases_and_forwards_client_auth() -> None:
    upstream = respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "target", "created": 1}]})
    )
    settings = Settings(upstream_url="http://upstream.test", model_aliases="alias=target")

    async with await _client(settings) as client:
        response = await client.get("/api/tags", headers={"Authorization": "Bearer client-key"})

    assert response.status_code == 200
    assert {model["name"] for model in response.json()["models"]} == {
        "target",
        "alias",
        "primary",
    }
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


async def test_ollama_model_management_accepts_current_model_field() -> None:
    async with await _client() as client:
        show = await client.post("/api/show", json={"model": "qwen3.5-35b"})
        pull = await client.post("/api/pull", json={"model": "qwen3.5-35b"})
        delete = await client.request("DELETE", "/api/delete", json={"model": "qwen3.5-35b"})

    assert show.status_code == 200
    assert show.json()["details"]["family"] == "qwen"
    assert pull.status_code == 200
    assert delete.status_code == 200
