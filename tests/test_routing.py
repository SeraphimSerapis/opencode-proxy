"""Modality detection, YAML config loading, and modality-routed requests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from opencode_proxy.app import create_app
from opencode_proxy.config_file import load_config_file
from opencode_proxy.ollama_models import OllamaChatRequest, OllamaMessage
from opencode_proxy.ollama_translate import ollama_chat_to_openai
from opencode_proxy.routing import (
    detect_modalities,
    parse_modality_routes,
    resolve_upstream_target,
)
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

VISION_ROUTES = json.dumps(
    {"vision": {"upstream": "http://vision.test", "model": "gemma-4-e4b"}},
)


def _text_body() -> dict[str, object]:
    return {"model": "deepseek-v4", "messages": [{"role": "user", "content": "hello"}]}


def _image_body() -> dict[str, object]:
    return {
        "model": "deepseek-v4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    }


def _audio_body() -> dict[str, object]:
    return {
        "model": "deepseek-v4",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}],
            }
        ],
    }


def test_detect_modalities_ignores_plain_text() -> None:
    assert detect_modalities(_text_body()) == frozenset()
    assert detect_modalities({"messages": "not a list"}) == frozenset()
    assert detect_modalities({}) == frozenset()


def test_detect_modalities_finds_images_and_audio() -> None:
    assert detect_modalities(_image_body()) == frozenset({"vision"})
    assert detect_modalities(_audio_body()) == frozenset({"audio"})


def test_detect_modalities_covers_translated_ollama_images() -> None:
    payload = ollama_chat_to_openai(
        OllamaChatRequest(
            model="llava",
            messages=[
                OllamaMessage(role="user", content="what is this?", images=["AAAA"]),
            ],
        )
    )
    assert detect_modalities(payload) == frozenset({"vision"})


def test_detect_modalities_covers_raw_ollama_images() -> None:
    body = {"messages": [{"role": "user", "content": "hi", "images": ["AAAA"]}]}
    assert detect_modalities(body) == frozenset({"vision"})


def test_audio_wins_when_a_request_carries_both_modalities() -> None:
    settings = Settings(
        upstream_url="http://upstream.test",
        modality_routes=json.dumps(
            {
                "vision": {"upstream": "http://vision.test"},
                "audio": {"upstream": "http://audio.test"},
            }
        ),
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "input_audio", "input_audio": {"data": "y"}},
                ],
            }
        ]
    }
    assert resolve_upstream_target(settings, body).base_url == "http://audio.test"


def test_unrouted_modality_falls_back_to_the_primary_upstream() -> None:
    settings = Settings(upstream_url="http://upstream.test")
    target = resolve_upstream_target(settings, _image_body())
    assert target.base_url == "http://upstream.test"
    assert target.model is None
    assert target.modality == ""


def test_parse_modality_routes_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="Unsupported MODALITY_ROUTES modality"):
        parse_modality_routes({"video": {"upstream": "http://x.test"}})
    with pytest.raises(ValueError, match="requires an upstream"):
        parse_modality_routes({"vision": {"model": "gemma"}})
    with pytest.raises(ValueError, match="Unsupported MODALITY_ROUTES keys"):
        parse_modality_routes({"vision": {"upstream": "http://x.test", "temperature": 1}})


def test_parse_modality_routes_accepts_a_bare_upstream() -> None:
    routes = parse_modality_routes({"vision": "http://vision.test/"})
    assert routes["vision"].upstream == "http://vision.test"
    assert routes["vision"].model is None


def test_config_file_flattens_aliases_and_routes(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text(
        "models:\n"
        "  deepseek-v4-flash:\n"
        "    aliases: [dsv4-flash, DeepSeek-V4-Flash]\n"
        "  gemma-4-e4b: [gemma]\n"
        "routes:\n"
        "  vision:\n"
        "    upstream: http://vision.test/v1\n"
        "    model: gemma-4-e4b\n",
        encoding="utf-8",
    )
    values = load_config_file(config)
    assert values["model_aliases"] == {
        "dsv4-flash": "deepseek-v4-flash",
        "DeepSeek-V4-Flash": "deepseek-v4-flash",
        "gemma": "gemma-4-e4b",
    }

    settings = Settings(config_file=str(config), upstream_url="http://upstream.test")
    assert settings.parsed_model_aliases["dsv4-flash"] == "deepseek-v4-flash"
    # The trailing /v1 is stripped for route upstreams too.
    assert settings.parsed_modality_routes["vision"].upstream == "http://vision.test"


def test_environment_wins_over_the_config_file(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text("models:\n  from-file: [alias]\n", encoding="utf-8")
    settings = Settings(config_file=str(config), model_aliases="alias=from-env")
    assert settings.parsed_model_aliases == {"alias": "from-env"}


def test_config_file_rejects_unknown_sections(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text("upstreams:\n  - http://x.test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported section"):
        load_config_file(config)


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_config_file(tmp_path / "nope.yaml")


def test_no_config_file_configured_is_fine() -> None:
    assert load_config_file("") == {}


def test_config_file_rejects_conflicting_aliases(tmp_path: Path) -> None:
    config = tmp_path / "proxy.yaml"
    config.write_text("models:\n  a: [shared]\n  b: [shared]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claimed by both"):
        load_config_file(config)


async def _client(settings: Settings) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://proxy.test")


@respx.mock
async def test_vision_request_is_routed_to_the_vision_upstream() -> None:
    primary = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []}),
    )
    vision = respx.post("http://vision.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []}),
    )

    settings = Settings(
        upstream_url="http://upstream.test",
        upstream_api_key="primary-key",
        modality_routes=json.dumps(
            {
                "vision": {
                    "upstream": "http://vision.test",
                    "model": "gemma-4-e4b",
                    "api_key": "vision-key",
                    "headers": {"X-Vision": "1"},
                }
            }
        ),
    )
    async with await _client(settings) as client:
        response = await client.post("/v1/chat/completions", json=_image_body())

    assert response.status_code == 200
    assert not primary.called
    assert vision.called
    sent = vision.calls[0].request
    assert json.loads(sent.content)["model"] == "gemma-4-e4b"
    assert sent.headers["authorization"] == "Bearer vision-key"
    assert sent.headers["x-vision"] == "1"


@respx.mock
async def test_text_request_still_goes_to_the_primary_upstream() -> None:
    primary = respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []}),
    )
    settings = Settings(upstream_url="http://upstream.test", modality_routes=VISION_ROUTES)

    async with await _client(settings) as client:
        response = await client.post("/v1/chat/completions", json=_text_body())

    assert response.status_code == 200
    assert primary.called
    assert json.loads(primary.calls[0].request.content)["model"] == "deepseek-v4"


@respx.mock
async def test_ollama_chat_with_images_is_routed() -> None:
    vision = respx.post("http://vision.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "a cat"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ),
    )
    settings = Settings(upstream_url="http://upstream.test", modality_routes=VISION_ROUTES)

    async with await _client(settings) as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "llava",
                "stream": False,
                "messages": [{"role": "user", "content": "what is this?", "images": ["AAAA"]}],
            },
        )

    assert response.status_code == 200
    assert vision.called
    assert json.loads(vision.calls[0].request.content)["model"] == "gemma-4-e4b"
    assert response.json()["message"]["content"] == "a cat"
