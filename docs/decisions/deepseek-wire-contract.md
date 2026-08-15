---
type: decision
title: Conform to DeepSeek's own client on the way upstream
description: Why the proxy now repairs outgoing message shapes, maps thinking effort, and treats an empty or abnormal turn as a failure, following the wire rules published in deepseek-harness.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/request_compat.py
tags: [deepseek, requests, wire-format, reasoning, reliability]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-15T16:40:00+02:00
sources:
  - id: deepseek-harness
    title: DeepSeek Harness — the vendor's own agent client for V4
    url: https://github.com/deepseek-ai/deepseek-harness
  - id: llm-deepseek
    title: "@deepseek-ai/dsh-llm-deepseek — chat-completions adapter (serialize.ts, translate.ts, sse.ts)"
    url: https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/llm/llm-deepseek
---

# Conform to DeepSeek's own client on the way upstream

## Context

DeepSeek published `deepseek-harness`, an agent harness whose
`packages/llm/llm-deepseek` is the vendor's reference client for the V4 wire
format. It is the authoritative statement of what a DeepSeek request and
response are supposed to look like, written by the people who serve the model.

Two things came out of reading it.

**`DSML` appears nowhere in that repository.** DeepSeek's own client assumes the
API emits native `tool_calls` JSON. Everything in
[tool-call repair](../architecture/tool-call-repair.md) is compensating for the
self-hosted chat template, not for the model contract. That does not make the
repair layer wrong -- it is what makes our vLLM deployment usable -- but it does
mean the target shape is not a matter of taste. The adapter defines it.

**The rules the adapter treats as hard-won are all request-side, and the proxy
had no request-side message handling at all.** It sanitized `tools`, dropped
configured fields, and applied model aliases; `messages` went upstream verbatim.

## Decision

The proxy repairs outgoing messages before forwarding
(`NORMALIZE_REQUESTS`, default on), and judges a completed turn the way the
reference client does.

### Assistant `content` is never `null`

`serialize.ts` carries an unusually blunt comment about this. A reasoning-only
assistant turn -- which V4 Flash produces for short prompts, answering entirely
in the reasoning channel -- serializes as `content: null` with no `tool_calls`,
and the API rejects it with "content or tool_calls must be set". Because that
message sits durably in the caller's session log, it is replayed on every
subsequent request: one bad turn bricks the rest of the session. The proxy
coerces `null` to `""`.

### Reasoning is replayed on tool-call turns and dropped everywhere else

Per DeepSeek's thinking-mode guide, `reasoning_content` **must** be sent back on
assistant turns that carried tool calls, and is ignored on turns that did not.
The proxy keeps it on tool-call turns -- moving a caller's `reasoning` into
`reasoning_content` when that is the only copy present -- and drops it
otherwise, which also stops the caller paying for those tokens again.

Getting this wrong degrades tool round trips without producing an error, which
is the worst failure shape available: it looks like the model is bad at its job.

### Empty tool results carry a placeholder

The reference client sends the literal `(no output)` for an empty tool result,
because empty content is another rejected shape. A command that legitimately
prints nothing is common in an agent loop.

### `off` is `thinking`, not an effort

`reasoning_effort` accepts only `high` and `max`. Disabling thinking is
`thinking: {"type": "disabled"}` with no effort field, and enabled is the
provider default, so an accepted effort is forwarded alone. This mapping is
DeepSeek-specific -- `thinking` is not an OpenAI field -- so it runs only for a
model configured with the `deepseek_v4` compatibility profile. Message hygiene
above is valid for any OpenAI-compatible upstream and is not gated.

### An empty completed turn is a failure, not an answer

`translate.ts` maps a `stop` finish that opened no blocks to `EMPTY_RESPONSE`
and retries it under the default policy. The proxy already
[detected and annotated](turn-usability.md) this shape but still handed the
caller a successful empty turn. Buffered requests now retry it
(`EMPTY_RESPONSE_RETRIES`, default 1) before annotating. A turn cut short by
`length` is excluded: that one is truthfully reported, and retrying it unchanged
burns the same budget again.

Streamed turns are not retried, for the reason in
[retry only before the first byte](retry-policy.md): emptiness is only knowable
once the bytes are already with the client. They keep the annotation.

### An unknown `finish_reason` ended the turn abnormally

`content_filter`, vLLM's `insufficient_system_resource`, and anything added
later are mapped by the reference client to an error finish rather than a stop.
The proxy forwards the value unchanged -- rewriting a terminator would be worse
than reporting an unfamiliar one -- but logs it, counts it under
`opencode_proxy_finish_reasons{reason="other"}`, and, when the turn was also
empty, replaces the token-budget notice with one naming the actual terminator.
The budget notice was misattributing every one of these.

## Where we deliberately differ

**SSE framing.** The adapter is spec-strict: an event dispatches only on its
blank-line terminator, and an unterminated tail at EOF is truncation
(`STREAM_CLOSED`), not something to flush. The proxy flushes pending text and
synthesizes a terminator instead, because a client that hangs is worse than a
client that sees a truthfully-truncated turn. See
[always terminate a streamed turn](stream-termination.md).

**Reasoning-only turns.** The adapter counts a reasoning block as output, so a
reasoning-only turn is not `EMPTY_RESPONSE` to it. The proxy treats content and
tool calls as the only actionable output, because that is what an agent client
can render or execute. This is the shape behind the original empty-turn notice.

## Consequences

* The proxy now rewrites request `messages`. Every repair is counted under
  `opencode_proxy_request_normalizations{kind}` and logged at `info`. Set
  `NORMALIZE_REQUESTS=false` to forward bodies untouched.
* A buffered empty turn costs one extra generation by default. Set
  `EMPTY_RESPONSE_RETRIES=0` to restore the previous behaviour.
* Dropping stale `reasoning_content` from history changes the prompt prefix
  relative to what a caller sent, which can cost an upstream cache hit on the
  first turn after upgrading. It saves the tokens on every turn after that.

## Regression coverage

`tests/test_request_compat.py` covers the normalizer, the emptiness rule, and
the failure classifiers directly. In `tests/test_proxy.py`:
`test_request_normalization_repairs_messages_before_forwarding`,
`test_request_normalization_can_be_disabled`,
`test_empty_buffered_completion_is_retried`,
`test_exhausted_empty_completion_retries_annotate_the_turn`,
`test_empty_completion_retry_can_be_disabled`,
`test_length_truncated_completion_is_not_retried`,
`test_abnormal_finish_reason_replaces_the_budget_notice`,
`test_empty_reasoning_delta_opens_no_reasoning_block`,
`test_retry_after_header_paces_the_retry`,
`test_upstream_error_status_is_classified`,
`test_streamed_usage_is_counted_disjointly`.
