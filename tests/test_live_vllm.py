"""Environment-gated DeepSeek V4 capability probe.

This is intentionally excluded from normal local runs. See the DeepSeek V4
runbook for the required environment variables and removal gate.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

BAR = chr(0xFF5C)
ORPHAN_MARKER = f"<{BAR}DSML{BAR}invoke"

DIRECT_URL = os.environ.get("VLLM_PROBE_DIRECT_URL")
PROXY_URL = os.environ.get("VLLM_PROBE_PROXY_URL")
MODEL = os.environ.get("VLLM_PROBE_MODEL")

pytestmark = pytest.mark.skipif(
    not (DIRECT_URL and PROXY_URL and MODEL),
    reason="set VLLM_PROBE_DIRECT_URL, VLLM_PROBE_PROXY_URL, and VLLM_PROBE_MODEL",
)


def _request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call echo_probe exactly once with value 'stable-prefix'. "
                    "Do not answer in prose."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo_probe",
                    "description": "Return the supplied value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
    }


def _first_message(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices")
    assert isinstance(choices, list)
    assert choices
    message = choices[0].get("message")
    assert isinstance(message, dict)
    return message


@pytest.mark.asyncio
async def test_live_deepseek_v4_serving_contract() -> None:
    assert DIRECT_URL is not None
    assert PROXY_URL is not None
    timeout = httpx.Timeout(180)

    async with (
        httpx.AsyncClient(base_url=DIRECT_URL, timeout=timeout) as direct,
        httpx.AsyncClient(base_url=PROXY_URL, timeout=timeout) as proxy,
    ):
        direct_message = _first_message(await direct.post("/v1/chat/completions", json=_request()))
        proxy_message = _first_message(await proxy.post("/v1/chat/completions", json=_request()))

        for message in (direct_message, proxy_message):
            content = message.get("content")
            assert not isinstance(content, str) or ORPHAN_MARKER not in content
            calls = message.get("tool_calls")
            assert isinstance(calls, list)
            assert calls
            assert all(isinstance(call.get("id"), str) and call["id"] for call in calls)

        # Replay the exact provider reasoning field, when present, through a
        # tool-result turn. This detects request-contract failures without the
        # proxy inventing or mutating reasoning history.
        replay_message = dict(direct_message)
        call = replay_message["tool_calls"][0]
        replay = _request()
        replay["tool_choice"] = "auto"
        replay["messages"] = [
            *_request()["messages"],
            replay_message,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": '{"value":"stable-prefix"}',
            },
        ]
        replay_response = await direct.post("/v1/chat/completions", json=replay)
        replay_response.raise_for_status()

        # Interrupt one client-side stream, then prove the server still accepts
        # a fresh request. The proxy's synthetic interruption handling remains
        # covered by deterministic unit tests.
        stream_request = {**_request(), "stream": True}
        async with direct.stream(
            "POST",
            "/v1/chat/completions",
            json=stream_request,
        ) as stream_response:
            stream_response.raise_for_status()
            async for line in stream_response.aiter_lines():
                if line.startswith("data: "):
                    break
        follow_up = await direct.post("/v1/chat/completions", json=_request())
        follow_up.raise_for_status()

        # Repeat the same long prefix, then verify vLLM's own cache counters are
        # exposed directly. APC accelerates repeated-prefix prefill only.
        await direct.post("/v1/chat/completions", json=_request())
        await direct.post("/v1/chat/completions", json=_request())
        metrics = await direct.get("/metrics")
        metrics.raise_for_status()
        assert "vllm:prefix_cache_queries" in metrics.text
        assert "vllm:prefix_cache_hits" in metrics.text
