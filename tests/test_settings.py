from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from opencode_proxy.settings import (
    Settings,
    parse_custom_headers,
    parse_model_aliases,
    parse_request_drop_fields,
    parse_tool_call_scan_fields,
)


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


def test_parse_model_aliases_from_json_object() -> None:
    aliases = parse_model_aliases(
        '{"dsv4-flash":"DeepSeek-V4-Flash",'
        '"deepseek-ai/DeepSeek-V4-Flash-DSpark":"DeepSeek-V4-Flash"}',
    )

    assert aliases == {
        "dsv4-flash": "DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-DSpark": "DeepSeek-V4-Flash",
    }


def test_parse_model_aliases_from_lines() -> None:
    aliases = parse_model_aliases(
        """
        dsv4-flash=DeepSeek-V4-Flash
        short: canonical
        """,
    )

    assert aliases == {
        "dsv4-flash": "DeepSeek-V4-Flash",
        "short": "canonical",
    }


def test_parse_model_aliases_from_comma_separated_pairs() -> None:
    aliases = parse_model_aliases(
        "dsv4-flash=DeepSeek-V4-Flash,deepseek-ai/DeepSeek-V4-Flash-DSpark=DeepSeek-V4-Flash",
    )

    assert aliases == {
        "dsv4-flash": "DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-DSpark": "DeepSeek-V4-Flash",
    }


def test_parse_model_aliases_from_comma_separated_colon_pairs() -> None:
    aliases = parse_model_aliases("short:canonical, other: target")

    assert aliases == {
        "short": "canonical",
        "other": "target",
    }


def test_settings_reads_model_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ALIASES", '{"alias":"target"}')

    settings = Settings()

    assert settings.parsed_model_aliases == {"alias": "target"}


def test_settings_reads_tool_call_scan_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_CALL_SCAN_FIELDS", "content,reasoning_content")

    settings = Settings()

    assert settings.parsed_tool_call_scan_fields == ("content", "reasoning_content")


def test_parse_tool_call_scan_fields_all_shortcut() -> None:
    assert parse_tool_call_scan_fields("all") == ("content", "reasoning", "reasoning_content")


def test_parse_request_drop_fields_from_string() -> None:
    assert parse_request_drop_fields("parallel_tool_calls, unsupported-field") == (
        "parallel_tool_calls",
        "unsupported-field",
    )


def test_settings_safe_config_omits_custom_header_values() -> None:
    settings = Settings(
        upstream_url="http://user:pass@upstream.test:4000/v1",
        custom_headers='{"Authorization":"Bearer secret","X-Skip-Auth":"true"}',
        model_aliases='{"alias":"target"}',
        request_drop_fields="parallel_tool_calls",
    )

    safe_config = cast("dict[str, Any]", settings.safe_config)

    assert safe_config["upstream"]["origin"] == "http://upstream.test:4000"
    assert safe_config["custom_headers"] == {"names": ["Authorization", "X-Skip-Auth"]}
    assert safe_config["model_aliases"]["aliases"] == ["alias"]
    assert safe_config["request_transforms"]["drop_fields"] == ["parallel_tool_calls"]


def test_settings_rejects_invalid_tool_call_scan_field() -> None:
    with pytest.raises(ValidationError):
        Settings(tool_call_scan_fields="content,bad_field")


def test_settings_rejects_invalid_alias_conflict_policy() -> None:
    with pytest.raises(ValidationError):
        Settings(alias_conflict_policy="panic")


def test_settings_rejects_invalid_request_drop_field() -> None:
    with pytest.raises(ValidationError):
        Settings(request_drop_fields="bad.field")


@pytest.mark.parametrize("value", [0, -1])
def test_settings_rejects_stream_guard_chars_below_one(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(stream_guard_chars=value)


@pytest.mark.parametrize("value", [0, -10])
def test_settings_rejects_tool_argument_chunk_size_below_one(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(tool_argument_chunk_size=value)


@pytest.mark.parametrize(
    "field_name",
    ["max_raw_tool_block_chars", "max_tool_calls", "max_tool_argument_chars"],
)
def test_settings_rejects_parser_limits_below_one(field_name: str) -> None:
    kwargs: dict[str, Any] = {field_name: 0}
    with pytest.raises(ValidationError):
        Settings(**kwargs)
