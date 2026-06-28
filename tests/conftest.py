"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest


@pytest.fixture
def default_settings() -> Settings:
    return Settings(upstream_url="http://upstream.test")


@pytest.fixture
async def proxy_client(default_settings: Settings) -> httpx.AsyncClient:
    app = create_app(default_settings)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://proxy.test")


def build_sse(*chunks: dict[str, object], done: bool = True) -> str:
    """Build an SSE text/event-stream body from a sequence of JSON chunks."""
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines)


def make_content_chunk(
    chunk_id: str,
    model: str,
    content: str,
    *,
    finish_reason: str | None = None,
    extra_delta: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a single SSE chat-completion chunk with content."""
    delta: dict[str, object] = {"content": content}
    if extra_delta:
        delta.update(extra_delta)
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def parse_sse_payloads(response_text: str) -> list[dict[str, object]]:
    """Parse SSE response text into a list of JSON payloads (excluding [DONE])."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def collect_content(payloads: Sequence[dict[str, object]]) -> str:
    """Concatenate all content deltas from parsed SSE payloads."""
    parts: list[str] = []
    for p in payloads:
        choices = p.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if isinstance(delta, dict):
            c = delta.get("content", "")
            if isinstance(c, str):
                parts.append(c)
    return "".join(parts)
