from __future__ import annotations

from opencode_proxy.ollama_models import (
    OllamaChatRequest,
    OllamaMessage,
    OllamaTool,
    OllamaToolFunction,
)
from opencode_proxy.ollama_translate import (
    ollama_chat_to_openai,
    openai_chat_to_ollama,
    openai_models_to_ollama,
)


def test_chat_translation_handles_openclaw_roles_images_tools_and_options() -> None:
    request = OllamaChatRequest(
        model="qwen3.5-35b",
        messages=[
            OllamaMessage(role="developer", content="Be concise"),
            OllamaMessage(role="system", content="Use JSON"),
            OllamaMessage(role="user", content="[think] inspect this", images=["abc"]),
            OllamaMessage(role="toolResult", content="done", tool_name="inspect"),
        ],
        tools=[
            OllamaTool(
                function=OllamaToolFunction(
                    name="inspect", description="Inspect", parameters={"type": "object"}
                )
            )
        ],
        options={"temperature": 0.2, "num_predict": 100, "top_k": 40, "repeat_penalty": 1.1},
        stream=True,
    )

    translated = ollama_chat_to_openai(request)

    assert translated["model"] == "qwen3.5-35b"
    assert translated["messages"][0] == {"role": "system", "content": "Be concise\n\nUse JSON"}
    assert translated["messages"][1]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,abc"
    )
    assert translated["messages"][2] == {
        "role": "tool",
        "content": "done",
        "tool_call_id": "call_inspect",
    }
    assert translated["chat_template_kwargs"] == {"enable_thinking": True}
    assert translated["max_tokens"] == 100
    assert translated["extra_body"] == {"top_k": 40, "repetition_penalty": 1.1}
    assert translated["stream_options"] == {"include_usage": True}


def test_openai_response_translation_preserves_thinking_and_tool_arguments() -> None:
    response = openai_chat_to_ollama(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should inspect",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "inspect",
                                    "arguments": '{"path":"README.md"}',
                                }
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
        "qwen",
    )

    assert response.done is True
    assert response.done_reason == "stop"
    assert response.message.content == ""
    assert response.message.thinking == "I should inspect"
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].function.arguments == {"path": "README.md"}
    assert response.prompt_eval_count == 3
    assert response.eval_count == 4


def test_models_translation_uses_created_timestamp_and_safe_defaults() -> None:
    result = openai_models_to_ollama({"data": [{"id": "llama-3-8b-q4", "created": 1_700_000_000}]})

    assert result.models[0].name == "llama-3-8b-q4"
    assert result.models[0].modified_at.startswith("2023-")
    assert result.models[0].details.family == "llama"
    assert result.models[0].details.parameter_size == "8B"
