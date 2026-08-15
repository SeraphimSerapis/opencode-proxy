---
type: decision
title: Retry only before the first byte
description: Which upstream failures are retried, why a started stream never is, and why passthrough routes are excluded.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [retries, reliability, streaming, idempotency]
status: active
generated:
  by: codex/gpt-5
  at: 2026-08-15T16:21:21+02:00
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

A `Retry-After` header on a retryable status replaces that curve, clamped to 30
seconds so a broken or hostile upstream cannot park a caller's request. Both
header forms are accepted; a malformed, negative, or absurd value is ignored and
the normal backoff applies.

**Retries happen only before any response byte has reached the client.**

One narrow exception is judged after the response, not before it: a *buffered*
turn for a configured `deepseek_v4` profile that completes with no content and
no tool call is re-sent up to `EMPTY_RESPONSE_RETRIES` times (default 1). That
is a failed generation rather than an answer, and nothing has been shown to the client yet, so the reasoning
below does not apply to it. A turn cut short by `length` is excluded -- it is
truthfully reported, and a replay burns the same budget again. Streamed turns
are never re-sent for this: their emptiness is only knowable once the bytes have
already reached the client. See
[conform to DeepSeek's own client](deepseek-wire-contract.md).

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
* An empty buffered DeepSeek turn costs one extra generation by default. It is counted as
  `opencode_proxy_upstream_retries{reason="empty_response"}`; set
  `EMPTY_RESPONSE_RETRIES=0` to disable.
