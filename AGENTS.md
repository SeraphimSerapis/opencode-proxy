# AGENTS.md

## Project Mission

Build a small, production-ready FastAPI proxy between OpenCode/Ollama clients and an OpenAI-compatible LiteLLM router. The proxy should preserve normal OpenAI-compatible traffic while repairing model responses that emit tool calls as non-standard text formats such as DeepSeek DSML or Qwen XML. OpenAI and Ollama traffic must share one upstream client and one deployed service.

## Engineering Standards

- Keep the proxy transparent by default. Only mutate request or response payloads when needed for OpenCode compatibility.
- Keep parsing and transformation logic isolated from FastAPI route code so it can be unit tested without network fixtures.
- Preserve streaming behavior. Do not buffer an entire SSE response unless a possible raw tool-call block is being detected.
- Treat upstream headers carefully. Strip hop-by-hop headers and avoid compressed SSE from upstreams that would prevent line-by-line parsing.
- Prefer typed, explicit code over broad `Any` use. When dynamic JSON is unavoidable, narrow types near the boundary.
- Add focused tests for every supported tool-call format before changing parser behavior.
- Keep commits atomic: scaffold, parser behavior, proxy routing/streaming, tests/docs, and validation fixes should be separate commits.

The Ollama adapter is separated into `ollama_models.py`, `ollama_translate.py`,
`ollama_streaming.py`, and `ollama.py`; keep translation logic independent from
FastAPI route wiring. `tools/mock_openai.py` is the dependency-free upstream for
Docker smoke tests and must not make real model calls.

## Local Commands

Use `uv` for development:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Run the app locally:

```bash
UPSTREAM_URL=http://127.0.0.1:4000 uv run uvicorn opencode_proxy.app:create_app --factory --host 0.0.0.0 --port 9526
```

## Docker

The container listens on port `9526` by default:

```bash
docker build -t opencode-proxy:local .
docker run --rm -p 9526:9526 -e UPSTREAM_URL=http://host.docker.internal:4000 opencode-proxy:local
```

The Prometheus deployment maps both host ports `9526` and `11434` to container
port `9526`; do not bring up the retired standalone `ollama-proxy` service.

## Environment

- `UPSTREAM_URL`: LiteLLM/OpenAI-compatible base URL, for example `http://127.0.0.1:4000`.
- `UPSTREAM_API_KEY`: optional fallback bearer token. Forwarded caller `Authorization` takes precedence; leave empty to require LiteLLM virtual keys.
- `OLLAMA_VERSION`: version returned by `/api/version`. Default: `0.5.1`.
- `PROXY_HOST`: bind host for direct `python -m opencode_proxy` runs. Default: `0.0.0.0`.
- `PROXY_PORT`: bind port. Default: `9526`.
- `LOG_LEVEL`: Python logging level. Default: `INFO`.
- `STREAM_GUARD_CHARS`: amount of non-tool text to hold while checking for split tool tags. Default: `192`.
- `TOOL_ARGUMENT_CHUNK_SIZE`: streamed function argument chunk size. Default: `64`.
- `CUSTOM_HEADERS`: extra headers applied to upstream requests. Prefer JSON object syntax. `UPSTREAM_HEADERS` is accepted as an alias.
- `PROXY_UPSTREAM_FALLBACK_KEY`: Prometheus compose alias for the optional `UPSTREAM_API_KEY` fallback; do not use the LiteLLM master key as a general client credential.

## Release Expectations

- Do not push unless linting, formatting check, typing, and tests pass locally.
- If Docker is unavailable locally, state that clearly and include the exact build command to run.
- Keep README usage examples current with the supported environment variables and OpenCode provider configuration.

## Cursor Cloud specific instructions

- This is a headless API service (no GUI/frontend). Validate it with `curl`/HTTP, not a browser.
- `uv` is installed to `~/.local/bin` (on `PATH` for login shells via `~/.bashrc`). The startup update script runs `uv sync --dev`, so `.venv` is ready; use the `uv run ...` commands from `## Local Commands` for lint/format/type/test.
- To run and test the proxy end-to-end without a real model, start the dependency-free mock upstream and point the proxy at it. Invoke it with `python3` (the VM has no `python` alias, unlike the README example): `python3 tools/mock_openai.py` (listens on `:4000`), then run the app per `## Local Commands` with `UPSTREAM_URL=http://127.0.0.1:4000`.
- The app binds `:9526`. Quick hello-world checks once both are running: `curl http://127.0.0.1:9526/healthz`, OpenAI path `POST /v1/chat/completions`, and Ollama path `POST /api/chat` (both return `mock response` against the mock upstream).
