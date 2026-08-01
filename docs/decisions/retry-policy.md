---
type: decision
title: Retry only before the first byte
description: Which upstream failures are retried, why a started stream never is, and why passthrough routes are excluded.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [retries, reliability, streaming, idempotency]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
sources:
  - id: visionbridge
    title: VisionBridge backend client — retry rule adapted from its `started` flag
    url: https://github.com/thomasunise/visionbridge
---

# Retry only before the first byte

## Context

The proxy had no retries. A vLLM restart, a LiteLLM `502`, or a transient
connection failure surfaced as a failed turn that the user had to retry by hand,
even though the request had never reached a model.

## Decision

Retry chat and generate requests up to `UPSTREAM_MAX_RETRIES` (default 2) on
transport errors and upstream `429`, `500`, `502`, `503`, `504`, with exponential
backoff (`min(0.5 · 2ⁿ, 8)` seconds) plus up to 250 ms of jitter.

**Retries happen only before any response byte has reached the client.**

## Why that boundary

Re-sending after streaming has started would replay a prompt whose partial answer
the client has already rendered — the user would see the beginning of one
generation followed by the whole of another. Worse, for an agent, a replayed turn
can duplicate tool calls the client already dispatched.

The boundary is enforced structurally rather than with a runtime flag: all retry
logic lives inside `send_upstream_with_retries`, which returns the response
before the rewriting generator is ever iterated. There is no code path from a
started stream back into the retry loop, so the invariant cannot be broken by a
later edit that forgets a flag. `test_failure_after_the_stream_started_is_not_retried`
pins it by asserting the upstream route was called exactly once when a stream
dies mid-flight.

## Why these statuses

`429` and `503` are explicit overload signals; `502` and `504` are gateway
failures where the request usually never reached a model. `500` is ambiguous —
the model may have already run — but on a local backend the only cost of a
duplicate is GPU time, and the alternative is surfacing a transient failure to
the user. Transport errors are retried because a connection that never
established cannot have been processed.

Requests are rebuilt on each attempt because httpx consumes the outgoing body.

## Exclusions

**Transparent passthrough** (`/{path:path}`) is not retried. It forwards
`content=request.stream()`, which cannot be replayed — a second attempt would
send an empty body, which is worse than the original failure.

**The proxy's own `429`** from `MAX_CONCURRENT_UPSTREAM` is generated before any
upstream call and is unaffected.

## Consequences

* With the backend genuinely down, a connection refusal now takes roughly 2.7 s
  to surface as a typed `502` instead of failing immediately. Set
  `UPSTREAM_MAX_RETRIES=0` to restore fail-fast behaviour.
* Tests that assert error surfacing must disable retries or they spend real
  seconds in backoff. Three existing tests were updated for this.
* A retried request occupies its concurrency slot for the whole sequence,
  including backoff.
