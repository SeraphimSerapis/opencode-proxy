"""Optional LiteLLM proxy plugin exposing the repair core as custom callbacks.

Runs the same repair logic as the standalone FastAPI proxy *inside* a LiteLLM
proxy process, via LiteLLM's ``CustomLogger`` hooks:

* ``async_pre_call_hook`` repairs outgoing message shapes (DeepSeek wire rules).
* ``async_post_call_success_hook`` repairs buffered chat completions.
* ``async_post_call_streaming_iterator_hook`` drives the streaming repair state
  machine over the whole ``/chat/completions`` chunk stream.

LiteLLM is an optional dependency: importing this module never requires it, only
calling :func:`create_repair_handler` does. Wire it up with a two-line shim next
to the LiteLLM config::

    # litellm_opencode_repair.py
    from opencode_proxy.litellm_plugin import create_repair_handler

    repair_handler = create_repair_handler()

and reference it in ``litellm_config.yaml``::

    litellm_settings:
      callbacks: litellm_opencode_repair.repair_handler

Not covered here, by design: model aliases and modality routing (LiteLLM's
router owns both), Ollama-native API endpoints (LiteLLM serves none), SSE
keepalive comments (hooks can only emit ``data:`` frames), and empty-response
re-sends (LiteLLM retries/fallbacks apply).
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from opencode_proxy.compat import (
    JsonObject,
    ToolRepairContext,
    annotate_empty_completion,
    convert_chat_completion_response,
    is_empty_completion,
)
from opencode_proxy.request_compat import normalize_request
from opencode_proxy.stream_repair import (
    StreamChoiceState,
    StreamRepairConfig,
    iter_finish_payloads,
    rewrite_stream_choice,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from opencode_proxy.metrics import ProxyMetrics

LOG = logging.getLogger(__name__)

# Top-level chunk fields LiteLLM's ModelResponseStream understands. Anything
# else carried on a synthesized payload is dropped rather than failing the
# whole stream; standard fields always survive.
_KNOWN_CHUNK_FIELDS = frozenset(
    {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"},
)


def create_repair_handler(
    *,
    config: StreamRepairConfig | None = None,
    normalize_requests: bool = True,
    metrics: ProxyMetrics | None = None,
) -> Any:
    """Build a LiteLLM ``CustomLogger`` driving the shared repair core.

    Raises ``RuntimeError`` when the ``litellm`` package is not installed in the
    running interpreter.
    """
    try:
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.types.utils import ModelResponseStream
        from pydantic import ValidationError
    except ImportError as exc:
        raise RuntimeError(
            "opencode_proxy.litellm_plugin requires the 'litellm' package; "
            "install it into the LiteLLM proxy environment"
        ) from exc

    repair_config = config or StreamRepairConfig()

    def _to_chunk(payload: JsonObject) -> Any:
        """Convert a repaired dict payload into a typed LiteLLM chunk."""
        try:
            return ModelResponseStream(**payload)
        except ValidationError:
            LOG.warning(
                "repair payload did not fit LiteLLM's chunk schema; dropping non-standard fields",
                extra={"keys": sorted(payload)},
            )
            trimmed = {key: value for key, value in payload.items() if key in _KNOWN_CHUNK_FIELDS}
            return ModelResponseStream(**trimmed)

    class RepairHandler(CustomLogger):  # type: ignore[misc]
        """Repairs raw-text tool calls for one LiteLLM proxy process."""

        def __init__(self) -> None:
            super().__init__()
            self.config = repair_config
            self.normalize_requests = normalize_requests
            self.metrics = metrics

        async def async_pre_call_hook(
            self,
            user_api_key_dict: Any,
            cache: Any,
            data: dict[str, Any],
            call_type: str,
        ) -> dict[str, Any]:
            if call_type != "completion" or not self.normalize_requests:
                return data
            messages = data.get("messages")
            if isinstance(messages, list) and messages:
                # DeepSeek-specific wire repairs; other providers pass through
                # untouched because normalize_request only rewrites known shapes.
                normalize_request(data, thinking_transport=None)
            return data

        async def async_post_call_success_hook(
            self,
            data: dict[str, Any],
            user_api_key_dict: Any,
            response: Any,
        ) -> Any:
            if not hasattr(response, "model_dump"):
                return response
            payload = response.model_dump()
            if not isinstance(payload.get("choices"), list) or not payload["choices"]:
                return response

            converted, changed = convert_chat_completion_response(
                payload,
                tool_call_scan_fields=self.config.scan_fields,
                max_raw_tool_block_chars=self.config.max_raw_tool_block_chars,
                max_tool_calls=self.config.max_tool_calls,
                max_tool_argument_chars=self.config.max_tool_argument_chars,
            )
            if changed:
                LOG.info("converted raw tool call(s) in buffered completion")
            if is_empty_completion(converted) and self.config.empty_turn_notice:
                annotate_empty_completion(converted, self.config.empty_turn_notice)
                changed = True
            if not changed:
                return response
            return type(response)(**converted)

        async def async_post_call_streaming_iterator_hook(
            self,
            user_api_key_dict: Any,
            response: Any,
            request_data: dict[str, Any],
        ) -> AsyncIterator[Any]:
            choice_states: dict[int, StreamChoiceState] = {}
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            model = "unknown"

            async for chunk in response:
                if isinstance(chunk, dict):
                    event: JsonObject = chunk
                elif hasattr(chunk, "model_dump"):
                    # exclude_none matches the SSE wire shape: LiteLLM's typed
                    # chunks carry None placeholders for every unset Delta
                    # field, which the repair logic would read as payload.
                    event = chunk.model_dump(exclude_none=True)
                else:
                    yield chunk
                    continue

                chunk_id = str(event.get("id") or chunk_id)
                model = str(event.get("model") or model)
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    yield _to_chunk(event)
                    continue

                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    index = choice.get("index")
                    state_key = index if type(index) is int else 0
                    state = choice_states.setdefault(state_key, StreamChoiceState())
                    for payload in rewrite_stream_choice(
                        event,
                        choice,
                        state,
                        chunk_id=chunk_id,
                        model=model,
                        config=self.config,
                        repair_context=ToolRepairContext(),
                        metrics=self.metrics,
                    ):
                        yield _to_chunk(payload)

            for payload in iter_finish_payloads(
                choice_states,
                chunk_id=chunk_id,
                model=model,
                fallback_finish_reason="length",
                upstream_completed=True,
                empty_turn_notice=self.config.empty_turn_notice,
                metrics=self.metrics,
            ):
                yield _to_chunk(payload)

    return RepairHandler()


def handler_from_env() -> Any:
    """Convenience constructor reading knobs from opencode-proxy env vars.

    Useful for the shim-file wiring LiteLLM expects; ignores the network-facing
    settings (upstream URL, ports, retries) that only the standalone proxy uses.
    """
    from opencode_proxy.settings import Settings

    settings = Settings()
    return create_repair_handler(config=StreamRepairConfig.from_settings(settings))
