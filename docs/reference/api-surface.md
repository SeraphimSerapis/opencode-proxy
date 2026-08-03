---
type: reference
title: API surface
description: Endpoints the proxy serves, which are rewritten, which are translated from Ollama, and which are passed through untouched.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/app.py
tags: [api, endpoints, openai, ollama, health]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# API surface

## Health

| Endpoint | Behaviour |
| --- | --- |
| `GET /healthz` | Liveness only. Never touches the upstream. Used by the container healthcheck. |
| `GET /readyz` | Probes upstream `GET /v1/models`. Any non-5xx counts as ready, so an auth rejection still means "reachable". Connection failure or upstream 5xx returns `503`. |
| `GET /healthz/config` | Effective configuration with credentials and header values omitted — header and alias *names* only. |
| `GET /metrics` | Prometheus counters for proxy-owned repair and transport behavior. vLLM must be scraped separately for model-serving metrics. |

`/readyz` treating `401` as ready is deliberate: readiness asks whether the
backend is up, not whether this proxy holds a valid key.

## OpenAI-compatible

| Endpoint | Behaviour |
| --- | --- |
| `/v1/chat/completions` | Request sanitising, aliasing, modality routing; response tool-call repair for both streaming and buffered replies. |
| `/v1/models`, `/models` | Upstream discovery with alias entries added per `ALIAS_CONFLICT_POLICY`. |
| `/{path:path}` | Transparent passthrough: body streamed both ways, no rewriting, no retries. |

All methods are accepted on the chat and models routes and forwarded as-is; the
proxy does not enforce `POST`.

Models reachable only through a modality route are **not** added to `/v1/models`.
They are reachable by routing, not by name, and advertising them would hand
clients a model id that fails if selected directly.

## Ollama-compatible

Translated to OpenAI chat completions and back:

| Endpoint | Notes |
| --- | --- |
| `POST /api/chat` | Full translation including `images` to `image_url` parts. Streams NDJSON. |
| `POST /api/generate` | Prompt-shaped variant of the same path. |
| `POST /api/embed`, `POST /api/embeddings` | Translated to `/v1/embeddings`. |
| `GET /api/tags`, `GET /api/ps` | Upstream model discovery reshaped to Ollama form, with aliases. |

Served locally without touching the upstream:

| Endpoint | Notes |
| --- | --- |
| `GET /`, `HEAD /` | Returns `Ollama is running`. |
| `GET /api/version` | Returns `OLLAMA_VERSION`. |
| `POST /api/show` | Synthesised metadata. |
| `POST /api/pull`, `/api/push`, `/api/copy`, `/api/create`, `DELETE /api/delete` | Accepted and ignored — models are managed upstream. Logged. |
| `HEAD`/`POST /api/blobs/{digest}` | Always `200`, so clients that probe before uploading proceed. |

Model-management endpoints return success rather than `501` because Ollama
clients treat a failure there as "the server is broken" and stop, rather than
"this model is already available".

An empty `messages` (chat) or `prompt` (generate) is answered locally with
`done_reason` `load` or `unload`, matching how Ollama handles model load and
unload pings, instead of being forwarded as an empty completion.
