---
type: runbook
title: Build, deploy, and verify
description: Validation gates, the homelab build and deploy path for the locally-built proxy image, smoke tests, and rollback.
resource: /home/tim/docker/compose/ai/opencode-proxy.yml
tags: [deploy, docker, compose, build, rollback, verification]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Build, deploy, and verify

## Scope

`opencode-proxy` is a locally-built image: the Compose fragment declares
`build.context: ${PROJECTSDIR}/opencode-proxy` with `pull_policy: build`, so
`./utility/pull.sh` never refreshes it. Deploying means rebuilding from the
checkout. See the stack's
[locally-built image runbook](/home/tim/docker/documentation/runbooks/local-image-builds.md)
for the general procedure.

## 1. Validation gates

All four must pass before deploying. Nothing is pushed or deployed on a failure:

```bash
cd /home/tim/projects/opencode-proxy && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest
```

## 2. Build and deploy

From the stack root, which owns the Compose context:

```bash
cd /home/tim/docker && docker compose up -d --build opencode-proxy
```

The container publishes host `9526` and `11434`, both to container `9526`.

## 3. Verify

Health, which is also the container healthcheck:

```bash
docker compose ps opencode-proxy && curl -sS http://127.0.0.1:9526/healthz
```

Readiness, which probes the backend:

```bash
curl -sS http://127.0.0.1:9526/readyz
```

Effective configuration — confirms the deployed build picked up the intended
settings, with no credentials in the output:

```bash
curl -sS http://127.0.0.1:9526/healthz/config | python3 -m json.tool
```

End to end through a real generation, including stream termination:

```bash
curl -sS -N http://127.0.0.1:9526/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv4-flash","stream":true,"max_tokens":40,"messages":[{"role":"user","content":"Say hi in 3 words."}]}' | tail -4
```

Expect a chunk with `"finish_reason"`, then `data: [DONE]`, and a prompt exit.

The Ollama surface shares the same upstream client and deserves one check:

```bash
curl -sS http://127.0.0.1:11434/api/version && curl -sS http://127.0.0.1:11434/api/tags | head -c 200
```

## 4. Rollback

The previous image is untagged by a rebuild but not deleted. Tag it before
building if a rollback path is wanted:

```bash
docker tag opencode-proxy:latest opencode-proxy:rollback-$(date +%Y%m%d)
```

To roll back, retag and recreate:

```bash
cd /home/tim/docker && docker tag opencode-proxy:rollback-<date> opencode-proxy:latest && docker compose up -d --force-recreate opencode-proxy
```

Reverting source instead means checking out the previous commit and repeating
step 2, since the image is built from the working tree.

## Notes

* Clients hold connections to the proxy. Recreating the container drops
  in-flight streamed turns; expect one failed turn in any active session.
* `depends_on: litellm` orders startup only. The proxy's upstream is vLLM — see
  [request pipeline](/architecture/request-pipeline.md).
* Configuration lives in the Compose fragment's `environment:` block, not in the
  project's `.env`, which is for local development only.
