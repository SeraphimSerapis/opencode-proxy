from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx

from opencode_proxy.app import create_app
from opencode_proxy.compat import (
    DSML_INVOKE_CLOSE,
    DSML_INVOKE_OPEN,
    DSML_PARAMETER_CLOSE,
    DSML_PARAMETER_OPEN,
    RepairStats,
    convert_chat_completion_response,
    extract_orphan_dsml_invokes,
)
from opencode_proxy.config_file import load_config_file
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


def _orphan(name: str = "read", value: str = "README.md") -> str:
    return (
        f'{DSML_INVOKE_OPEN} name="{name}">'
        f'{DSML_PARAMETER_OPEN} name="path">{value}{DSML_PARAMETER_CLOSE}'
        f"{DSML_INVOKE_CLOSE}"
    )


def _settings() -> Settings:
    return Settings(
        upstream_url="http://upstream.test",
        model_compatibility=json.dumps(
            {
                "deepseek-v4": {
                    "compatibility": "deepseek_v4",
                    "recover_orphan_invokes": True,
                }
            }
        ),
    )


async def _client(settings: Settings | None = None) -> httpx.AsyncClient:
    app = create_app(settings or _settings())
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    )


def _request(*, stream: bool = False, tool_choice: object = "auto") -> dict[str, object]:
    return {
        "model": "deepseek-v4",
        "stream": stream,
        "messages": [{"role": "user", "content": "read it"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": tool_choice,
    }


def _sse_payloads(response: httpx.Response) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_yaml_model_entry_loads_deepseek_v4_compatibility(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text(
        "models:\n"
        "  deepseek-v4:\n"
        "    aliases: [dsv4]\n"
        "    compatibility: deepseek_v4\n"
        "    recover_orphan_invokes: true\n",
        encoding="utf-8",
    )

    values = load_config_file(config)
    settings = Settings(config_file=str(config))

    assert values["model_compatibility"]["deepseek-v4"]["recover_orphan_invokes"] is True
    assert settings.parsed_model_compatibility["deepseek-v4"].profile == "deepseek_v4"
    assert settings.parsed_model_aliases == {"dsv4": "deepseek-v4"}


def test_yaml_model_entry_selects_the_thinking_transport(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text(
        "models:\n"
        "  deepseek-v4-flash:\n"
        "    compatibility: deepseek_v4\n"
        "    thinking_transport: chat_template_kwargs\n",
        encoding="utf-8",
    )

    settings = Settings(config_file=str(config))
    profile = settings.parsed_model_compatibility["deepseek-v4-flash"]

    assert profile.thinking_transport == "chat_template_kwargs"


def test_thinking_transport_defaults_to_the_api_form(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text(
        "models:\n  deepseek-v4:\n    compatibility: deepseek_v4\n",
        encoding="utf-8",
    )

    settings = Settings(config_file=str(config))

    assert settings.parsed_model_compatibility["deepseek-v4"].thinking_transport == "api"


@pytest.mark.parametrize(
    "config",
    [
        "models:\n  model:\n    compatibility: other\n",
        (
            "models:\n  model:\n    compatibility: deepseek_v4\n"
            "    thinking_transport: enable_thinking\n"
        ),
        ('models:\n  model:\n    compatibility: deepseek_v4\n    recover_orphan_invokes: "yes"\n'),
    ],
)
def test_yaml_rejects_invalid_compatibility(tmp_path: Path, config: str) -> None:
    path = tmp_path / "proxy.yaml"
    path.write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match=r"compatibility|boolean|thinking_transport"):
        load_config_file(path)


def test_orphan_parser_accepts_multiple_unicode_calls_and_trailing_prose() -> None:
    content = _orphan("réad", "日本.txt") + "\n" + _orphan("read", "資料.txt") + "\nDone."
    stats = RepairStats()

    calls, remaining, changed = extract_orphan_dsml_invokes(
        content,
        declared_tool_names=frozenset({"réad", "read"}),
        stats=stats,
    )

    assert changed
    assert [call["function"]["name"] for call in calls] == ["réad", "read"]
    assert remaining == "\n\nDone."
    assert stats.orphan_accepted == 2


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (_orphan("write"), "undeclared_tool"),
        (
            f'{DSML_INVOKE_OPEN} name="read">'
            f'{DSML_PARAMETER_OPEN} name="path" string="false">nope'
            f"{DSML_PARAMETER_CLOSE}{DSML_INVOKE_CLOSE}",
            "malformed",
        ),
    ],
)
def test_rejected_orphans_are_preserved_byte_for_byte(content: str, reason: str) -> None:
    stats = RepairStats()
    calls, remaining, changed = extract_orphan_dsml_invokes(
        content,
        declared_tool_names=frozenset({"read"}),
        stats=stats,
    )

    assert not changed
    assert calls == []
    assert remaining == content
    assert stats.orphan_rejected == {reason: 1}


def test_oversized_orphan_is_preserved_and_counted_as_rejected() -> None:
    content = _orphan(value="x" * 100)
    stats = RepairStats()
    calls, remaining, changed = extract_orphan_dsml_invokes(
        content,
        declared_tool_names=frozenset({"read"}),
        max_raw_tool_block_chars=32,
        stats=stats,
    )

    assert (calls, remaining, changed) == ([], content, False)
    assert stats.orphan_rejected == {"oversized_block": 1}


def test_orphan_recovery_never_scans_reasoning_fields() -> None:
    content = _orphan()
    body = {
        "choices": [
            {
                "message": {"content": None, "reasoning_content": content},
                "finish_reason": "stop",
            }
        ]
    }

    converted, changed = convert_chat_completion_response(
        body,
        recover_orphan_invokes=True,
        declared_tool_names=frozenset({"read"}),
    )

    assert not changed
    assert converted["choices"][0]["message"]["reasoning_content"] == content


@pytest.mark.parametrize(
    "content",
    [
        f'Quoted "{_orphan()}"',
        "Here is an example:\n" + _orphan(),
        _orphan().replace(chr(0xFF5C), "|"),
        _orphan().removesuffix(DSML_INVOKE_CLOSE),
    ],
)
def test_prose_foreign_and_partial_markers_are_not_candidates(content: str) -> None:
    stats = RepairStats()
    calls, remaining, changed = extract_orphan_dsml_invokes(
        content,
        declared_tool_names=frozenset({"read"}),
        stats=stats,
    )

    assert (calls, remaining, changed) == ([], content, False)
    assert not stats.orphan_rejected


@respx.mock
async def test_buffered_orphan_recovery_is_request_aware() -> None:
    raw = _orphan() + "\nAfter."
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    async with await _client() as client:
        response = await client.post("/v1/chat/completions", json=_request())

    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "\nAfter."
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert choice["finish_reason"] == "tool_calls"


@respx.mock
@pytest.mark.parametrize(
    "tool_choice",
    ["none", {"type": "function", "function": {"name": "read"}}],
)
async def test_tool_choice_controls_orphan_recovery(tool_choice: object) -> None:
    raw = _orphan()
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]},
        )
    )

    async with await _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_request(tool_choice=tool_choice),
        )

    message = response.json()["choices"][0]["message"]
    if tool_choice == "none":
        assert message == {"content": raw}
    else:
        assert message["tool_calls"][0]["function"]["name"] == "read"


@respx.mock
@pytest.mark.parametrize("split", range(1, len(DSML_INVOKE_OPEN)))
async def test_streaming_orphan_recovery_handles_every_opener_split(split: int) -> None:
    block = _orphan() + "\nTrailing."
    frames = [
        {
            "id": "chatcmpl-orphan",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [{"index": 0, "delta": {"content": block[:split]}}],
        },
        {
            "id": "chatcmpl-orphan",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": block[split:]},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    body = "".join(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(body + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with await _client() as client:
        response = await client.post("/v1/chat/completions", json=_request(stream=True))

    payloads = _sse_payloads(response)
    tool_calls = [
        call
        for payload in payloads
        for choice in payload["choices"]
        for call in choice["delta"].get("tool_calls", [])
    ]
    content = "".join(
        choice["delta"].get("content", "") for payload in payloads for choice in payload["choices"]
    )
    assert tool_calls[0]["function"]["name"] == "read"
    assert tool_calls[0]["id"].startswith("call_")
    assert content == "\nTrailing."


def test_buffered_native_tool_call_ids_are_synthesized_and_preserved() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                        {
                            "id": "call_upstream",
                            "type": "function",
                            "function": {"name": "write", "arguments": "{}"},
                        },
                    ]
                }
            }
        ]
    }
    converted, changed = convert_chat_completion_response(body)
    calls = converted["choices"][0]["message"]["tool_calls"]

    assert changed
    assert calls[0]["id"].startswith("call_")
    assert calls[1]["id"] == "call_upstream"
    assert calls[0]["id"] != calls[1]["id"]


@respx.mock
async def test_streaming_native_missing_ids_are_stable_per_tool_index() -> None:
    frames = [
        {
            "id": "chatcmpl-native",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "type": "function",
                                "function": {"name": "read", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_upstream",
                                "type": "function",
                                "function": {"name": "write", "arguments": ""},
                            },
                        ]
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-native",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": "{}"}},
                            {"index": 0, "function": {"arguments": "{}"}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=(body + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with await _client() as client:
        response = await client.post("/v1/chat/completions", json=_request(stream=True))

    ids_by_index: dict[int, set[str]] = {}
    for payload in _sse_payloads(response):
        for call in payload["choices"][0]["delta"].get("tool_calls", []):
            ids_by_index.setdefault(call["index"], set()).add(call["id"])
    assert len(ids_by_index[0]) == 1
    assert next(iter(ids_by_index[0])).startswith("call_")
    assert ids_by_index[1] == {"call_upstream"}


@respx.mock
async def test_metrics_expose_bounded_repair_labels() -> None:
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": _orphan()},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    async with await _client() as client:
        await client.post("/v1/chat/completions", json=_request())
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert (
        'opencode_proxy_orphan_recovery_total{outcome="accepted",reason="valid"} 1.0'
        in response.text
    )
    assert (
        'opencode_proxy_raw_tool_repair_total{field="content",format="deepseek_v4_orphan"}'
        " 1.0" in response.text
    )
