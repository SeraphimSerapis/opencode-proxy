"""Pure translations between Ollama and OpenAI-compatible payloads."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from opencode_proxy.ollama_models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    OllamaFunction,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModelDetails,
    OllamaModelInfo,
    OllamaModelList,
    OllamaRunningModelInfo,
    OllamaRunningModels,
    OllamaShowResponse,
    OllamaToolCall,
)

THINK_KEYWORD = "[think]"
_QWEN3_THINKING_PATTERNS = ("qwen3.5", "qwen-3.5", "qwen_3.5")


def is_qwen3_thinking_model(model: str) -> bool:
    name = model.lower()
    return any(pattern in name for pattern in _QWEN3_THINKING_PATTERNS)


def rewrite_openclaw_messages(
    messages: list[OllamaMessage], model: str
) -> tuple[list[OllamaMessage], bool]:
    """Normalize OpenClaw roles and Qwen3.5 system/thinking conventions."""
    copied = [message.model_copy(deep=True) for message in messages]
    for message in copied:
        if message.role == "developer":
            message.role = "system"
        elif message.role == "toolResult":
            message.role = "tool"

    thinking_enabled = False
    if is_qwen3_thinking_model(model):
        first_system = next((message for message in copied if message.role == "system"), None)
        normalized: list[OllamaMessage] = []
        for message in copied:
            if (
                message.role == "system"
                and first_system is not None
                and message is not first_system
            ):
                first_system.content = "\n\n".join(
                    value for value in (first_system.content, message.content) if value
                ).strip()
                continue
            normalized.append(message)
        copied = normalized

        for message in reversed(copied):
            if message.role != "user":
                continue
            content = message.content or ""
            if content.lstrip().lower().startswith(THINK_KEYWORD):
                message.content = content.lstrip()[len(THINK_KEYWORD) :].lstrip()
                thinking_enabled = True
            break

    return copied, thinking_enabled


def ollama_chat_to_openai(request: OllamaChatRequest) -> dict[str, Any]:
    messages, thinking_enabled = rewrite_openclaw_messages(request.messages, request.model)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_to_openai(message) for message in messages],
        "stream": request.stream,
    }

    if is_qwen3_thinking_model(request.model):
        payload["chat_template_kwargs"] = {
            "enable_thinking": request.think if request.think is not None else thinking_enabled
        }

    if request.tools:
        payload["tools"] = [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    **(
                        {"description": tool.function.description}
                        if tool.function.description
                        else {}
                    ),
                    **(
                        {"parameters": tool.function.parameters}
                        if tool.function.parameters is not None
                        else {}
                    ),
                },
            }
            for tool in request.tools
        ]

    _apply_format(request.format, payload)
    _apply_options(request.options, payload)
    if request.stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _message_to_openai(message: OllamaMessage) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role}
    if message.images:
        parts: list[dict[str, Any]] = []
        if message.content is not None:
            parts.append({"type": "text", "text": message.content})
        for image in message.images:
            url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        result["content"] = parts
    elif message.content is not None:
        result["content"] = message.content

    if message.role == "assistant" and message.tool_calls:
        result["tool_calls"] = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(
                        tool_call.function.arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            for index, tool_call in enumerate(message.tool_calls)
        ]
        if not message.content:
            result["content"] = None

    if message.role == "tool":
        result["tool_call_id"] = f"call_{message.tool_name or 'unknown'}"
    return result


def openai_chat_to_ollama(response: dict[str, Any], model: str) -> OllamaChatResponse:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else "stop"
    return OllamaChatResponse(
        model=model,
        created_at=_now_iso(),
        message=_message_to_ollama(message),
        done=True,
        done_reason=_finish_reason(finish_reason),
        prompt_eval_count=_int_or_none(usage.get("prompt_tokens")),
        eval_count=_int_or_none(usage.get("completion_tokens")),
    )


def _message_to_ollama(message: dict[str, Any]) -> OllamaMessage:
    raw_calls = message.get("tool_calls")
    tool_calls: list[OllamaToolCall] | None = None
    if isinstance(raw_calls, list):
        parsed: list[OllamaToolCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            parsed.append(
                OllamaToolCall(
                    function=OllamaFunction(
                        name=function["name"], arguments=_json_object(function.get("arguments"))
                    )
                )
            )
        tool_calls = parsed or None

    content = message.get("content")
    return OllamaMessage(
        role=str(message.get("role") or "assistant"),
        content=content if isinstance(content, str) else "",
        thinking=_string_or_none(message.get("reasoning_content") or message.get("reasoning")),
        tool_calls=tool_calls,
    )


def openai_chat_to_ollama_generate(response: dict[str, Any], model: str) -> OllamaGenerateResponse:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else "stop"
    return OllamaGenerateResponse(
        model=model,
        created_at=_now_iso(),
        response=_string_or_empty(message.get("content")),
        thinking=_string_or_none(message.get("reasoning_content")),
        done=True,
        done_reason=_finish_reason(finish_reason),
        prompt_eval_count=_int_or_none(usage.get("prompt_tokens")),
        eval_count=_int_or_none(usage.get("completion_tokens")),
    )


def ollama_generate_to_openai(request: OllamaGenerateRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    user: dict[str, Any] = {"role": "user"}
    if request.images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for image in request.images:
            url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        user["content"] = parts
    else:
        user["content"] = request.prompt
    messages.append(user)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    }
    if is_qwen3_thinking_model(request.model):
        payload["chat_template_kwargs"] = {"enable_thinking": bool(request.think)}
    if request.suffix:
        payload.setdefault("extra_body", {})["suffix"] = request.suffix
    _apply_format(request.format, payload)
    _apply_options(request.options, payload)
    if request.stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def ollama_embed_to_openai(request: OllamaEmbedRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": request.model, "input": request.input}
    if request.dimensions is not None:
        payload["dimensions"] = request.dimensions
    return payload


def openai_embeddings_to_ollama(response: dict[str, Any], model: str) -> OllamaEmbedResponse:
    raw_data = response.get("data")
    embeddings = (
        [
            item.get("embedding", [])
            for item in raw_data
            if isinstance(item, dict) and isinstance(item.get("embedding", []), list)
        ]
        if isinstance(raw_data, list)
        else []
    )
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return OllamaEmbedResponse(
        model=model,
        embeddings=embeddings,
        prompt_eval_count=_int_or_none(usage.get("prompt_tokens")),
    )


def openai_models_to_ollama(response: dict[str, Any]) -> OllamaModelList:
    raw_models = response.get("data")
    models: list[OllamaModelInfo] = []
    if not isinstance(raw_models, list):
        return OllamaModelList()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict) or not isinstance(raw_model.get("id"), str):
            continue
        model_id = raw_model["id"]
        created = raw_model.get("created")
        modified = (
            datetime.fromtimestamp(created, tz=UTC).isoformat()
            if isinstance(created, (int, float)) and created > 0
            else _now_iso()
        )
        models.append(
            OllamaModelInfo(
                name=model_id,
                model=model_id,
                modified_at=modified,
                details=_infer_model_details(model_id),
            )
        )
    return OllamaModelList(models=models)


def openai_models_to_running(response: dict[str, Any]) -> OllamaRunningModels:
    raw_models = response.get("data")
    if not isinstance(raw_models, list):
        return OllamaRunningModels()
    models = [
        OllamaRunningModelInfo(
            name=raw["id"], model=raw["id"], details=_infer_model_details(raw["id"])
        )
        for raw in raw_models
        if isinstance(raw, dict) and isinstance(raw.get("id"), str)
    ]
    return OllamaRunningModels(models=models)


def create_show_response(model_name: str) -> OllamaShowResponse:
    details = _infer_model_details(model_name)
    return OllamaShowResponse(
        modelfile=f"FROM {model_name}",
        template="{{ .Prompt }}",
        details=details,
        model_info={"general.architecture": details.family, "general.name": model_name},
    )


_OPTIONS_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "num_predict": "max_tokens",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}
_EXTRA_BODY_MAP = {"top_k": "top_k", "repeat_penalty": "repetition_penalty"}


def _apply_options(options: dict[str, Any] | None, payload: dict[str, Any]) -> None:
    if not options:
        return
    for source, target in _OPTIONS_MAP.items():
        if source in options:
            payload[target] = options[source]
    extra = {
        target: options[source] for source, target in _EXTRA_BODY_MAP.items() if source in options
    }
    if extra:
        existing = payload.setdefault("extra_body", {})
        if isinstance(existing, dict):
            existing.update(extra)


def _apply_format(fmt: str | dict[str, Any] | None, payload: dict[str, Any]) -> None:
    if fmt == "json":
        payload["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt},
        }


def _infer_model_details(model_name: str) -> OllamaModelDetails:
    lowered = model_name.lower()
    known_families = (
        "llama",
        "qwen",
        "gemma",
        "phi",
        "mistral",
        "mixtral",
        "deepseek",
        "codellama",
        "vicuna",
        "starcoder",
        "falcon",
        "yi",
        "command-r",
        "dbrx",
    )
    family = next((candidate for candidate in known_families if candidate in lowered), "unknown")
    parameter_match = re.search(r"(\d+x?\d*b)", lowered)
    quantization_match = re.search(r"(q\d+[_kms]*\d*|fp\d+|int\d+)", lowered)
    return OllamaModelDetails(
        family=family,
        families=[family] if family != "unknown" else None,
        parameter_size=parameter_match.group(1).upper() if parameter_match else "",
        quantization_level=quantization_match.group(1).upper() if quantization_match else "",
    )


def _finish_reason(reason: object) -> str:
    if reason == "length":
        return "length"
    return "stop"


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
