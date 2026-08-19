---
type: architecture
title: Request pipeline
description: How an OpenAI or Ollama request travels through the proxy, what is rewritten at each stage, and where the proxy actually sits in the deployed chain.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [proxy, fastapi, streaming, ollama, litellm, vllm, topology]
status: active
generated:
  by: openai-codex/gpt-5.6-sol
  at: 2026-08-19T13:29:49Z
---

# Request pipeline

## Deployed topology

The proxy sits **behind** LiteLLM, not in front of it. This is the inverse of
what the README's provider examples imply, and it is the single most misleading
thing about the deployment:

```text
pi / OpenCode / Home Assistant
        |
        v
LiteLLM  (container 172.18.0.x)
        |
        v
opencode-proxy  (:9526, also published on :11434)
        |
        v
vLLM  http://192.168.10.221:8080
```

`UPSTREAM_URL` is the vLLM server. The Compose fragment used to declare
`depends_on: litellm`, which read as if LiteLLM were upstream; it is not, and
the dependency has been dropped — it gated proxy startup on a service that sits
in front of the proxy. When a streamed turn misbehaves, LiteLLM is a
suspect hop between the client and the proxy, and its logs are worth checking
before concluding the proxy is at fault.

Compose fragment: [`/home/tim/docker/compose/ai/opencode-proxy.yml`](/home/tim/docker/compose/ai/opencode-proxy.yml).
Host ports `9526` and `11434` both map to container `9526`, so Ollama clients
reach the same service on the port they expect.

## Routing table

Routes are registered in this order, and the order matters — the OpenAI router
ends in a catch-all:

1. Health endpoints (`/healthz`, `/readyz`, `/healthz/config`) — [`app.py`](/home/tim/projects/opencode-proxy/src/opencode_proxy/app.py)
2. Ollama routes (`/api/*`) — [`ollama.py`](/home/tim/projects/opencode-proxy/src/opencode_proxy/ollama.py)
3. OpenAI routes (`/v1/chat/completions`, `/v1/models`, `/models`) — [`proxy.py`](/home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py)
4. `/{path:path}` transparent passthrough

The Ollama router is included first so `/api/chat` is not swallowed by the
catch-all. Adding routes after the OpenAI router has no effect.

## Chat completion stages

`proxy_chat_completions` applies these in order. Everything except the upstream
call is pure function work on the parsed body, so it is testable without network
fixtures:

| Stage | What it does | Skipped when |
| --- | --- | --- |
| Parse | JSON body to a dict | Body is absent or not a JSON object |
| Sanitize tools | Drop non-`function` tools; drop `tools` entirely if none remain | `SANITIZE_TOOLS=false` |
| Drop fields | Remove `REQUEST_DROP_FIELDS` entries | No fields configured |
| Model alias | Rewrite a configured alias. For `primary`, query upstream discovery and use its first model when the upstream does not define `primary` itself | No alias matches |
| Modality routing | Pick an alternate upstream and model for image/audio requests | No matching route |
| Normalize messages | For `deepseek_v4` profiles only, repair `null` assistant content, replay reasoning on tool turns, empty tool results, unsupported `developer` roles, `max_completion_tokens`, and map `reasoning_effort`/`thinking` to the configured transport | `NORMALIZE_REQUESTS=false` |
| Concurrency slot | Acquire one of `MAX_CONCURRENT_UPSTREAM`; `429` if full | Limit disabled |
| Upstream send | Retried before the first response byte | `UPSTREAM_MAX_RETRIES=0` |

A body the proxy cannot parse is forwarded verbatim. None of the rewriting
stages run in that case, by design: an unparseable body is not something to
guess at.

The built-in `primary` alias follows the model currently returned first by the
upstream `/v1/models` endpoint. The proxy leaves a real upstream model named
`primary` alone. If discovery fails or returns no model ids, it returns `502`
with `primary_model_discovery_failed` and does not send the completion.

Message normalization follows the rules in DeepSeek's own client; see
[conform to DeepSeek's own client](../decisions/deepseek-wire-contract.md).

Streaming and non-streaming diverge after the upstream call. Non-streaming
responses are repaired in one pass by `convert_chat_completion_response`, then
checked for emptiness: for `deepseek_v4` profiles, a completed turn carrying no
content and no tool call is retried up to `EMPTY_RESPONSE_RETRIES` times before
it is annotated.
Streaming responses go through the SSE rewriter — see
[tool-call repair](tool-call-repair.md) and the
[streaming contract](streaming-contract.md).

## The Ollama path is a translation layer, not a second proxy

`/api/chat` and `/api/generate` translate the Ollama request into an OpenAI chat
completion, send it through the same upstream client, and translate the response
back to Ollama NDJSON. They reuse the same alias map, the same modality routing,
the same tool-call repair, and the same concurrency limiter.

Ollama tool results carry a function name but no OpenAI call ID. Translation
assigns IDs to assistant tool calls in history and consumes them by name and
order as matching results arrive. The request-aware DeepSeek orphan-repair
context is also passed through both buffered and streaming Ollama responses.
Ollama `think` is always translated as protocol behavior; `NORMALIZE_REQUESTS`
controls the additional DeepSeek message and token cleanup.

Notably, `_message_to_openai` converts Ollama `images` into OpenAI `image_url`
parts *before* routing runs, so modality detection covers Ollama clients without
any Ollama-specific code.

Translation lives in `ollama_translate.py`, streaming translation in
`ollama_streaming.py`, and route wiring in `ollama.py`. Keep them separate: the
translators have no FastAPI imports and are unit tested directly.

## Concurrency

`MAX_CONCURRENT_UPSTREAM` (default 8) bounds concurrent chat and generate calls
so a shared GPU is not stampeded by several clients. It is non-blocking: over the
limit, callers get `429` with `Retry-After: 1` rather than queueing.

The slot is released by a Starlette `BackgroundTask` for streaming responses and
in a `finally` for buffered ones. A client that disconnects mid-stream releases
its slot promptly — verified by aborting a stream under uvicorn and watching the
limiter return to zero within half a second.
