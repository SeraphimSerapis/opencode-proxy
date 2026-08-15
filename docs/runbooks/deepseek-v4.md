---
type: runbook
title: Serve DeepSeek V4 safely
description: Pin vLLM, launch DeepSeek V4 with the matching parsers, observe cache behavior, and retire the orphan-invoke fallback.
resource: /home/tim/projects/opencode-proxy/tests/test_live_vllm.py
tags: [deepseek-v4, vllm, dsml, prefix-cache, prometheus, deployment]
status: active
generated:
  by: codex/gpt-5
  at: 2026-08-15T16:21:21+02:00
sources:
  - id: llm-deepseek
    title: "@deepseek-ai/dsh-llm-deepseek — the vendor's reference client and its model catalog"
    url: https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/llm/llm-deepseek
---

# Serve DeepSeek V4 safely

## Deployment record and launch profile

Do not deploy a mutable vLLM tag. The deployment record must contain all four
values below before rollout:

| Field | Required value |
| --- | --- |
| Image | Registry/repository plus immutable `@sha256:...` digest |
| Source | A build containing [vLLM PR #49117](https://github.com/vllm-project/vllm/pull/49117), or the first release verified to contain it |
| Model | Exact DeepSeek V4 repository and revision |
| Date/evidence | Probe command, result, and operator |

This repository does not own the vLLM deployment manifest and therefore cannot
safely invent its registry digest. Resolve and record the actual deployment
artifact, for example with `docker image inspect`, before enabling traffic.

The V4 server command must include:

```text
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4
--enable-auto-tool-choice
--enable-prefix-caching
```

Keep vLLM authoritative for tokenization, chat templates, reasoning/tool
parsing, and GPU execution. The proxy does not author prompt text, replay
sessions, execute tools, compact history, or retry after response bytes reach a
client. It does repair message *shapes* the DeepSeek wire format rejects, which
is a different thing and is bounded by
[conform to DeepSeek's own client](../decisions/deepseek-wire-contract.md).

## Wire-format facts from the vendor's own client

`deepseek-harness` ships `packages/llm/llm-deepseek`, DeepSeek's reference
adapter for this wire format. Useful facts to serve against, none of which the
proxy can enforce on its own:

| Fact | Value |
| --- | --- |
| Context window advertised for V4 Flash and V4 Pro | 1,000,000 tokens |
| Default per-request output cap | 256,000 tokens |
| Default stream idle budget | 300s per outstanding read |
| Accepted `reasoning_effort` values | `low`, `high`, `max` (default `high`); `medium`/`xhigh` map to `high` |
| Disabling thinking (vendor API) | `thinking: {"type": "disabled"}`, never `reasoning_effort: "off"` |
| Disabling thinking (our vLLM) | `chat_template_kwargs: {"thinking": false}`; thinking is on by default in the generation config, and the vendor field is ignored |
| Tool selection | `tool_choice` is preserved for the deployed vLLM path; direct vendor API deployments that reject it should drop it with `REQUEST_DROP_FIELDS=tool_choice` |
| Cache accounting | `prompt_tokens` *includes* `prompt_cache_hit_tokens`; no cache-write metric exists |

Two consequences for this deployment. The proxy's
`UPSTREAM_STREAM_FIRST_FRAME_TIMEOUT` (480s) is intentionally above the observed
histogram-estimated prefill p99 tail on a single local GPU, while the separate
mid-stream guard stays tight. And `DSML` appears nowhere in that repository -- the vendor
client assumes native `tool_calls` -- so every raw-text repair in this proxy is
compensating for the self-hosted chat template and should disappear once vLLM's
parser is correct.

## Temporary proxy fallback

Enable the fallback only on the affected canonical upstream model:

```yaml
models:
  deepseek-v4:
    compatibility: deepseek_v4
    recover_orphan_invokes: true
```

It accepts only canonical V4 orphan invokes in `content`, and only when the
request declares the exact function name and permits tools. Set the flag false
after the removal gate passes. This is the rollback switch; it does not require
a proxy image rollback.

## Prometheus targets

Scrape both services:

* Proxy `/metrics` reports transport/compatibility behavior:
  `opencode_proxy_orphan_recovery_total`,
  `opencode_proxy_raw_tool_repair_total`,
  `opencode_proxy_synthesized_tool_call_ids_total`,
  `opencode_proxy_tool_argument_repair_total`,
  `opencode_proxy_upstream_retries_total`,
  `opencode_proxy_stream_idle_terminations_total`, and
  `opencode_proxy_empty_turns_total`. Admission pressure is exposed through
  `opencode_proxy_upstream_overloads_total` and
  `opencode_proxy_upstream_active`.
* vLLM `/metrics` reports model-serving behavior. Track
  `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` (both token
  counters), KV-cache utilization, waiting requests/queue time, time to first
  token, and end-to-end request latency. Metric names beyond the prefix-cache
  counters can change between vLLM releases, so verify them against the pinned
  image's `/metrics` output when writing scrape rules.

Prefix-cache hit rate over an interval is
`rate(vllm:prefix_cache_hits[...]) / rate(vllm:prefix_cache_queries[...])`.
Hits measure cached prefix tokens. Automatic prefix caching improves repeated
prefix prefill; it does not accelerate generation tokens.

References: [automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/),
[production metrics](https://docs.vllm.ai/en/stable/usage/metrics/), and the
[official DeepSeek V4 recipe](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md).

## Live capability probe

Run the normal gates first, then point the gated test at vLLM directly and at
this proxy:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest

VLLM_PROBE_DIRECT_URL=http://vllm-host:8000 \
VLLM_PROBE_PROXY_URL=http://proxy-host:9526 \
VLLM_PROBE_MODEL=deepseek-v4 \
uv run pytest -q tests/test_live_vllm.py
```

The probe requires native structured calls with non-empty IDs directly and
through the proxy, replays the provider reasoning/tool-result shape, interrupts
a client stream and verifies recovery, repeats a cacheable prefix, and checks
vLLM's direct prefix-cache counters. Deterministic unit tests separately inject
wrapped and orphan DSML at every marker split point.

## Fallback removal gate

Set `recover_orphan_invokes: false` only when all are true:

1. The running vLLM image is recorded by immutable digest and contains #49117.
2. The launch profile above is visible in the running process.
3. Ruff, formatting, mypy, and all unit tests pass.
4. The live capability probe passes against the exact deployed digest.
5. Repeated production-like tool prompts show no orphan invokes in direct vLLM
   responses and the proxy accepted-orphan counter remains flat.
6. Direct vLLM metrics show prefix-cache queries/hits and healthy KV, queue, and
   latency behavior.

If direct responses regress, re-enable the flag only for the affected model and
retain the captured response shape privately; never put prompts, arguments,
tool names, or model output into metric labels.

## Resolved: duplicated sentence fragments were a client rendering artifact

DeepSeek-V4-Flash occasionally emits a sentence or fragment twice in a row.
Observed 2026-08-03 in a completed task review (first copy cut mid-word, second
copy completed):

> One flag: TRAEFIK_LEARNINGS.md and TRAEFIK_REVIEW.md were delet
> One flag: TRAEFIK_LEARNINGS.md and TRAEFIK_REVIEW.md were deleted from the repo root, ...

Second occurrence 2026-08-10, again in pi, again the closing sentence of the
turn, again a prefix cut mid-word followed by the complete sentence:

> Usage: codex-usage in any new shell. Single-quoted so tokens are rea
> Usage: codex-usage in any new shell. Single-quoted so tokens are read at call time, ...

Not a model fault. Confirmed on 2026-08-10 by capturing a third sighting with
`CAPTURE_STREAM_DIR` enabled: both sides of the turn are byte-identical and
contain the text exactly once, and the repeated unit was a terminal soft-wrap
line that has no newline in the bytes — a unit only the renderer can produce.
pi drew part of a wrapped line, then repainted it complete without clearing the
partial draw. The earlier working hypothesis of model-side token repetition
under vLLM sampling was wrong, and no sampling change is needed.

Two details made this easy to misread. The cut lengths differ between
occurrences (63 and 68 characters), which rules out any fixed-size buffer in
the chain — `STREAM_GUARD_CHARS` is 192 and would not cut at either point. And
searching the transcripts for the duplicated phrase matches a `role: user`
record too, because pasting the screen text into pi to report the problem
stores it verbatim; only the `role: assistant` record is evidence.

Status: closed. The general procedure, including the proxy-side stream capture
built while chasing this, is in
[diagnose duplicated output](diagnose-duplicated-output.md).
