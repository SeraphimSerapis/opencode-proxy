# AGENTS.md

## Project Mission

Build a small, production-ready FastAPI proxy between OpenCode and an OpenAI-compatible LiteLLM router. The proxy should preserve normal OpenAI-compatible traffic while repairing model responses that emit tool calls as non-standard text formats such as DeepSeek DSML or Qwen XML.

## Engineering Standards

- Keep the proxy transparent by default. Only mutate request or response payloads when needed for OpenCode compatibility.
- Keep parsing and transformation logic isolated from FastAPI route code so it can be unit tested without network fixtures.
- Preserve streaming behavior. Do not buffer an entire SSE response unless a possible raw tool-call block is being detected.
- Treat upstream headers carefully. Strip hop-by-hop headers and avoid compressed SSE from upstreams that would prevent line-by-line parsing.
- Prefer typed, explicit code over broad `Any` use. When dynamic JSON is unavoidable, narrow types near the boundary.
- Add focused tests for every supported tool-call format before changing parser behavior.
- Keep commits atomic: scaffold, parser behavior, proxy routing/streaming, tests/docs, and validation fixes should be separate commits.

## Local Commands

Use `uv` for development:

```bash
uv sync --all-extras --dev
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

## Environment

- `UPSTREAM_URL`: LiteLLM/OpenAI-compatible base URL, for example `http://127.0.0.1:4000`.
- `PROXY_HOST`: bind host for direct `python -m opencode_proxy` runs. Default: `0.0.0.0`.
- `PROXY_PORT`: bind port. Default: `9526`.
- `LOG_LEVEL`: Python logging level. Default: `INFO`.
- `STREAM_GUARD_CHARS`: amount of non-tool text to hold while checking for split tool tags. Default: `192`.
- `TOOL_ARGUMENT_CHUNK_SIZE`: streamed function argument chunk size. Default: `64`.

## Release Expectations

- Do not push unless linting, formatting check, typing, and tests pass locally.
- If Docker is unavailable locally, state that clearly and include the exact build command to run.
- Keep README usage examples current with the supported environment variables and OpenCode provider configuration.
