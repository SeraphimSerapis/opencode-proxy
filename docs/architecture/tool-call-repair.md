---
type: architecture
title: Tool-call repair
description: How raw tool-call markup emitted as model text is detected mid-stream and converted into standard OpenAI tool_calls deltas.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/compat.py
tags: [tool-calls, dsml, qwen, deepseek, streaming, parsing]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-03T09:40:00+02:00
verified: 2026-08-03
---

# Tool-call repair

## The problem

Some models emit tool calls as text in their own markup instead of populating
the OpenAI `tool_calls` field. The client then renders a wall of pseudo-XML
instead of executing a tool. DeepSeek emits DSML; Qwen emits `<tool_call>` XML
or bare JSON. The proxy's core job is turning that text back into the structured
field the client expects.

## Recognised formats

Complete opener/closer detection lives in `RAW_TOOL_BLOCK_PATTERNS`.
`RAW_TOOL_START_MARKERS` enumerates common complete openers for the standalone
prefix helper:

| Marker | Origin |
| --- | --- |
| `<｜DSML｜tool_calls>` (fullwidth bar, U+FF5C) | DeepSeek, canonical form |
| `<\|DSML\|tool_calls>` (ASCII pipe) | DeepSeek, degraded tokenisation |
| `<DSML>tool_calls>`, `<DSML: tool_calls>`, `<DSML tool_calls>`, `<DSML:tool_calls>` | DeepSeek, further degraded |
| `<tool_calls>`, `<tool_call ...>` | Qwen XML |

The streaming state machine does not call the prefix helper. It retains the last
`STREAM_GUARD_CHARS` of ordinary text, then applies the complete block patterns
after more text arrives. `test_block_grammar_and_common_marker_table_agree` in
[`tests/test_compat.py`](/home/tim/projects/opencode-proxy/tests/test_compat.py)
keeps the common marker table valid, while
`test_streaming_guard_reassembles_accepted_marker_variants` in
[`tests/test_proxy.py`](/home/tim/projects/opencode-proxy/tests/test_proxy.py)
exercises the production buffering path.

`normalize_raw_tool_markup` folds every variant into the canonical DSML shape
before parsing, so the parser has one grammar to handle rather than five. The
ASCII and spaced variants exist because the fullwidth bar does not always
survive tokenisation intact. Detection and normalisation share the same degraded
grammar, including a close tag that drops the backslashes
(`</DSML>tool_calls>`) and `name=`/`string=` with whitespace around the equals
sign (`name = "bash"`). DeepSeek-V4's reference encoding
(`encoding/encoding_dsv4.py`) uses exactly the U+FF5C delimiter and `string="true|false"`
parameter attribute, both already handled.

Qwen openers may carry trailing whitespace or attributes (`<tool_calls >`,
`<tool_call name="…">`); the same forms are normalized and parsed rather than
being accepted by block detection and then passed through as raw text.

Fixtures for each format: [`/home/tim/projects/opencode-proxy/tests/fixtures/tool_calls`](/home/tim/projects/opencode-proxy/tests/fixtures/tool_calls).
Add a fixture and a test before changing parser behaviour.

## Streaming detection: the guard buffer

A raw tool block arrives split across SSE frames — the opening marker may land
in one delta and the closing marker several frames later. The proxy cannot
forward text as it arrives without risking emitting half a tool block as visible
content, and it cannot buffer the whole response without destroying streaming.

The compromise is `STREAM_GUARD_CHARS` (default 192): text is released to the
client only once that many characters sit behind it, so a marker split across a
frame boundary is still detectable. Per scanned field, the state machine is:

1. **Complete block found** — emit any text before it, convert the block to
   `tool_calls` deltas, continue with the remainder.
2. **Block opener found, no closer yet** — mark the field pending and hold
   everything from the opener onward.
3. **Neither** — release all but the last `STREAM_GUARD_CHARS` characters.

A pending field is re-parsed as soon as the held buffer contains any
close-tag start (`</`), not only when it straddles the incoming frame
boundary. vLLM streams the DSML close tag as `</|DSML|tool_c` then `alls>`,
so the completing text arrives without a `</` of its own; checking the whole
held buffer keeps the block repairable across that boundary.

A block that never closes, or exceeds `MAX_RAW_TOOL_BLOCK_CHARS`, or fails to
parse, or breaches `MAX_TOOL_CALLS` / `MAX_TOOL_ARGUMENT_CHARS`, is emitted as
plain text. Standard streamed tool-call arguments are also accumulated only for
valid indexes, up to those same call and argument limits; an over-limit or
malformed call is passed through without a guessed repair. The proxy degrades
to a passthrough rather than dropping content.

## Scanned fields and reasoning order

`TOOL_CALL_SCAN_FIELDS` defaults to `content,reasoning,reasoning_content`.
Reasoning fields are scanned because these models sometimes emit tool markup
inside their thinking block.

That creates an ordering hazard. Held reasoning text and held content text are
separate buffers, so reasoning from an early frame could otherwise be flushed
*after* content from a later one — the client would render thinking below the
answer. Before any `content` delta is emitted, `_flush_reasoning_before_content`
releases held reasoning tails first.

The exception matters: a reasoning field that is mid raw-tool-block keeps its
buffer. Flushing there would leak a half-parsed block as visible text, which is
exactly the failure the guard buffer exists to prevent.

## Emission shape

`build_tool_call_chunks` emits arguments in `TOOL_ARGUMENT_CHUNK_SIZE` slices
(default 64) so clients that render arguments incrementally behave normally.
`state.next_tool_call_index` keeps indices monotonic across several converted
blocks in one response, and `raw_tool_calls_emitted` forces the terminal
`finish_reason` to `tool_calls` even when the upstream said `stop` — a client
that sees `stop` will not execute the tools it was just handed.

Responses that already contain standard `tool_calls` are passed through
unchanged when their IDs are valid. An otherwise valid buffered call without an
ID receives a distinct synthetic ID. Streaming keeps one ID per choice/tool
index and reuses it on every fragment; valid upstream IDs always win.

## Temporary DeepSeek V4 orphan fallback

vLLM issue/PR
[#49117](https://github.com/vllm-project/vllm/pull/49117) describes V4
occasionally emitting a canonical `<｜DSML｜invoke ...>` without its outer
`tool_calls` wrapper. That shape cannot enter the general raw-block parser,
because accepting wrapperless dialect fragments globally would turn quoted
markup and prose into tool execution.

The YAML-only `deepseek_v4` model profile enables a request-aware fallback. It
scans `content` only, requires declared function tools and `tool_choice !=
"none"`, accepts only the fullwidth-bar V4 dialect before any non-whitespace
content, and requires an exact declared-tool name. Partial, malformed,
oversized, undeclared, quoted/prose, reasoning-field, and foreign-dialect candidates stay text
byte-for-byte. Trailing prose remains in `content`.

Remove `recover_orphan_invokes: true` after the pinned vLLM image passes the
[live capability probe](../runbooks/deepseek-v4.md). The profile may remain with
the flag false as an explicit record of the upstream contract.
