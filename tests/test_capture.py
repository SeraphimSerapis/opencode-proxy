"""Capture writes both sides of a turn, and the analyzer attributes a fault."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import respx
from conftest import build_sse, make_content_chunk

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from analyze_capture import (  # type: ignore[import-not-found]
    collect_events,
    find_duplicates,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

UPSTREAM = "http://upstream.test"

# The fault this infrastructure exists to catch, as observed twice in pi: a
# segment is emitted, cut mid-word, then repeated in full.
TRUNCATED_COPY = "Usage: codex-usage in any new shell. Single-quoted so tokens are rea"
FULL_COPY = "Usage: codex-usage in any new shell. Single-quoted so tokens are read at call time."


def capture_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        upstream_url=UPSTREAM,
        capture_stream_dir=str(tmp_path / "captures"),
        **overrides,  # type: ignore[arg-type]
    )


async def run_stream(
    settings: Settings, body: str, *, request_body: dict[str, object] | None = None
) -> str:
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    payload = request_body or {"model": "deepseek-v4-flash", "stream": True, "messages": []}
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)
        return response.text


def capture_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "captures").glob("*.jsonl"))


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@respx.mock
@pytest.mark.anyio
async def test_capture_disabled_by_default_writes_nothing(tmp_path: Path) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=build_sse(make_content_chunk("c", "m", "hi", finish_reason="stop")),
            headers={"content-type": "text/event-stream"},
        )
    )
    settings = Settings(upstream_url=UPSTREAM)
    await run_stream(settings, "")
    assert not (tmp_path / "captures").exists()


@respx.mock
@pytest.mark.anyio
async def test_capture_records_both_sides(tmp_path: Path) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=build_sse(
                make_content_chunk("c", "m", "hello "),
                make_content_chunk("c", "m", "world", finish_reason="stop"),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    await run_stream(capture_settings(tmp_path), "")

    files = capture_files(tmp_path)
    assert len(files) == 1
    records = read_records(files[0])
    kinds = [record["kind"] for record in records]
    assert kinds[0] == "start"
    assert records[-1]["kind"] == "end"
    assert "upstream" in kinds
    assert "client" in kinds

    upstream, client, meta = collect_events(files[0])
    assert upstream.joined("content") == "hello world"
    assert client.joined("content") == "hello world"
    assert meta["model"] == "deepseek-v4-flash"


@respx.mock
@pytest.mark.anyio
async def test_capture_omits_request_body_unless_enabled(tmp_path: Path) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=build_sse(make_content_chunk("c", "m", "hi", finish_reason="stop")),
            headers={"content-type": "text/event-stream"},
        )
    )
    secret = {
        "model": "deepseek-v4-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "my private prompt"}],
    }
    await run_stream(capture_settings(tmp_path), "", request_body=secret)
    body = capture_files(tmp_path)[0].read_text(encoding="utf-8")
    assert "my private prompt" not in body

    other = tmp_path / "with-request"
    await run_stream(
        capture_settings(other, capture_stream_include_request=True),
        "",
        request_body=secret,
    )
    assert "my private prompt" in capture_files(other)[0].read_text(encoding="utf-8")


@respx.mock
@pytest.mark.anyio
async def test_capture_survives_a_stream_that_never_terminates(tmp_path: Path) -> None:
    """A turn cut off upstream still leaves a readable capture."""

    async def truncated(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=build_sse(make_content_chunk("c", "m", "partial"), done=False),
            headers={"content-type": "text/event-stream"},
        )

    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(side_effect=truncated)
    await run_stream(capture_settings(tmp_path), "")

    records = read_records(capture_files(tmp_path)[0])
    assert records[-1]["kind"] == "end"
    assert any(record["kind"] == "upstream_eof" for record in records)


@respx.mock
@pytest.mark.anyio
async def test_capture_bounded_by_max_bytes(tmp_path: Path) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=build_sse(
                *[make_content_chunk("c", "m", "x" * 500) for _ in range(50)],
                make_content_chunk("c", "m", "done", finish_reason="stop"),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    await run_stream(capture_settings(tmp_path, capture_stream_max_bytes=2000), "")
    path = capture_files(tmp_path)[0]
    assert path.stat().st_size < 6000
    assert any(record["kind"] == "truncated" for record in read_records(path))


@respx.mock
@pytest.mark.anyio
async def test_capture_attributes_an_upstream_duplicate(tmp_path: Path) -> None:
    """The observed fault, injected upstream, is attributed to upstream."""
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=build_sse(
                make_content_chunk("c", "m", TRUNCATED_COPY),
                make_content_chunk("c", "m", "\n" + FULL_COPY, finish_reason="stop"),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    await run_stream(capture_settings(tmp_path), "")

    upstream, client, _ = collect_events(capture_files(tmp_path)[0])
    upstream_hits = find_duplicates(upstream.joined("content"), "content", 24)
    client_hits = find_duplicates(client.joined("content"), "content", 24)

    assert len(upstream_hits) == 1
    assert upstream_hits[0].unit == TRUNCATED_COPY
    # Present on both sides: the proxy relayed a duplicate it did not create.
    assert len(client_hits) == 1


def test_analyzer_ignores_clean_text() -> None:
    clean = "First sentence here. Second sentence here. Third one is different.\n"
    assert find_duplicates(clean, "content", 24) == []


def test_analyzer_finds_exact_repeated_line() -> None:
    text = f"intro\n{FULL_COPY}\n{FULL_COPY}\ntail"
    hits = find_duplicates(text, "content", 24)
    assert [hit.unit for hit in hits] == [FULL_COPY]


def test_analyzer_finds_truncated_then_completed_repeat() -> None:
    text = f"{TRUNCATED_COPY}\n{FULL_COPY}"
    hits = find_duplicates(text, "content", 24)
    assert len(hits) == 1
    assert hits[0].unit == TRUNCATED_COPY
    assert hits[0].separator == "\n"


def test_analyzer_respects_min_length() -> None:
    text = "short\nshort\n"
    assert find_duplicates(text, "content", 24) == []
    assert len(find_duplicates(text, "content", 4)) >= 1


def test_analyzer_reconstructs_reasoning_separately() -> None:
    side_lines: Sequence[str] = [
        'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
        'data: {"choices":[{"delta":{"content":"answer"}}]}',
    ]
    from analyze_capture import SideText, _absorb_sse_line

    side = SideText()
    for line in side_lines:
        _absorb_sse_line(line, side)
    assert side.joined("reasoning_content") == "thinking"
    assert side.joined("content") == "answer"
