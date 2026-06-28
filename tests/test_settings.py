from __future__ import annotations

import pytest
from pydantic import ValidationError

from opencode_proxy.settings import Settings, parse_custom_headers


def test_parse_custom_headers_from_json_object() -> None:
    headers = parse_custom_headers('{"Authorization":"Bearer token","X-Skip-Auth":"true"}')

    assert headers == {"Authorization": "Bearer token", "X-Skip-Auth": "true"}


def test_parse_custom_headers_from_lines() -> None:
    headers = parse_custom_headers(
        """
        Authorization: Bearer token
        X-Skip-Auth: true
        """,
    )

    assert headers == {"Authorization": "Bearer token", "X-Skip-Auth": "true"}


def test_parse_custom_headers_rejects_invalid_line() -> None:
    with pytest.raises(ValueError, match="Invalid CUSTOM_HEADERS line"):
        parse_custom_headers("Authorization")


def test_settings_reads_custom_headers_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_HEADERS", '{"X-Test":"yes"}')

    settings = Settings()

    assert settings.parsed_custom_headers == {"X-Test": "yes"}


@pytest.mark.parametrize("value", [0, -1])
def test_settings_rejects_stream_guard_chars_below_one(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(stream_guard_chars=value)


@pytest.mark.parametrize("value", [0, -10])
def test_settings_rejects_tool_argument_chunk_size_below_one(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(tool_argument_chunk_size=value)
