# Documentation Update Log

## 2026-08-10

* **Resolved**: The DeepSeek-V4-Flash "duplicated fragment" quirk is a pi TUI
repaint bug — confirmed, not inferred. A third sighting was caught with stream
capture enabled: both sides of the turn are byte-identical and hold the text
once, and the repeated unit was a terminal soft-wrap line with no newline in
the bytes, which only the renderer can produce. The prior hypothesis (token
repetition under vLLM sampling) is withdrawn; no sampling change is needed.
Closed in the [DeepSeek V4 runbook](runbooks/deepseek-v4.md), with the method
in [diagnose duplicated output](runbooks/diagnose-duplicated-output.md).
* **Feature**: Added opt-in stream capture (`CAPTURE_STREAM_DIR`,
`CAPTURE_STREAM_MAX_BYTES`, `CAPTURE_STREAM_INCLUDE_REQUEST`), which records
upstream SSE frames alongside the bytes sent to the client, plus
`tools/analyze_capture.py`, which reconstructs both sides and attributes a
duplicated span to a layer. Built to chase the above; kept because it turns any
future output anomaly into an attribution without needing a reproduction. Off
by default: it writes model output, and optionally prompts, to disk in the
clear. See [diagnose duplicated output](runbooks/diagnose-duplicated-output.md).

## 2026-08-03 (5)

* **Todo**: Documented a known DeepSeek-V4-Flash output quirk — occasional
duplicated sentence fragments (observed: a "One flag: ..." sentence emitted
twice, first copy cut mid-word). Hypothesis is model-side token repetition
under vLLM sampling, not proxy re-emission; confirmation steps and tracking
are in the [DeepSeek V4 runbook](runbooks/deepseek-v4.md).

## 2026-08-03 (4)

* **Fix**: A raw tool block whose close tag splits so `</` lands mid-frame is
  no longer stranded as pending text. The streaming guard now re-parses
  whenever the held buffer contains any close-tag start, not only at a frame
  boundary — vLLM streams the close tag as `</|DSML|tool_c` then `alls>`.
  Previously the completed block was flushed as visible markup at the end of
  the turn, so the client rendered DSML prose instead of executing the call.
  Found and validated against the live vLLM host; see
  [tool-call repair](architecture/tool-call-repair.md).

## 2026-08-03 (3)

* **Feature**: Added an explicit YAML `deepseek_v4` compatibility profile and a
  request-aware fallback for canonical orphan DSML invokes associated with vLLM
  #49117. Recovery requires declared tools, an enabled tool choice, an exact
  declared name, and `content`; rejected candidates remain unchanged.
* **Compatibility**: Otherwise valid native OpenAI tool calls missing IDs now
  receive distinct synthetic IDs. Streamed IDs are stable per choice/tool index,
  while valid upstream IDs remain authoritative.
* **Observability**: Added `GET /metrics` with bounded proxy counters for raw and
  orphan repair, synthesized IDs, argument repair outcomes, retries, idle
  termination, and empty turns. vLLM model/cache metrics remain a separate
  scrape target.
* **Operations**: Added a DeepSeek V4 serving runbook and an environment-gated
  direct-vLLM/proxy capability probe. Removing the temporary fallback requires
  the pinned image, full local quality gates, and this live probe.

## 2026-08-03 (2)

* **Documentation**: Caught up `streaming-contract.md` and `tool-call-repair.md` to the fixes committed earlier today (`0bdec5d`) — both still described the pre-session mechanism (`stop`/`tool_calls` only, no argument repair, no empty-turn notice, the two-entry degraded-marker table). Added `EMPTY_TURN_NOTICE` to `configuration.md`, which was missing entirely. Corrected an overclaim in `turn-usability.md`: it referenced "the repair's counter" as an observability signal, but no counter exists yet — only log lines.
* **Audit fix**: Duplicate or post-terminal choice events are ignored; append-only argument repairs are emitted before terminal chunks even when `delta` is missing; repair accumulation is bounded by `MAX_TOOL_CALLS` and `MAX_TOOL_ARGUMENT_CHARS`; and clean empty `[DONE]` turns can receive the configured notice. Added regressions for multi-tool indexes, invalid fragments, and disabled notices.
* **Audit fix**: Raw Qwen blocks with opener attributes or trailing whitespace are now normalized and parsed consistently with block detection. End-to-end streaming regressions exercise those forms through the production guard buffer.

## 2026-08-03

* **Fix**: Truncated streamed tool-call `arguments` are now completed before the turn closes, and turns that produce no content and no tool calls are logged and optionally annotated before their terminal chunk. Both were found by driving a real agentic tool loop against the deployed vLLM; both produce a perfectly well-formed stream that nonetheless strands an agent client, which is why the earlier termination work did not catch them. See [a terminated turn must also be a usable turn](decisions/turn-usability.md).
* **Fix**: An exception of any type during SSE rewriting now terminates the turn instead of truncating the body. Once the 200/SSE headers are out the status is committed, so re-raising cannot produce an HTTP error — it only strands the client, which was the exact symptom being chased.
* **Fix**: `RAW_TOOL_START_MARKERS` gained the two degraded openers the block grammar already accepted (`<DSML:tool_calls>`, `<DSML tool_calls>`). The marker table and complete block grammar now agree for their common forms; the production stream path is covered separately through its fixed-size guard buffer.

## 2026-08-02

* **Fix**: An upstream `httpx.TransportError` or EOF without `[DONE]` now ends as `finish_reason: "length"` plus `[DONE]`, including before the first upstream choice. This closes agent turns without presenting partial output as a successful `stop`; local proxy failures still propagate. The observed Pi incident remains attributed to upstream silence, bounded by `UPSTREAM_STREAM_IDLE_TIMEOUT`. See [always terminate a streamed turn](decisions/stream-termination.md).
* **Fix**: Hardened DSML repair against two more tokeniser degradations: a close tag that drops the U+FF5C delimiters (`</DSML>tool_calls>`) and whitespace around `=` in `name=`/`string=` attributes. Both previously passed the whole block through as raw text. Verified against DeepSeek-V4-Flash-0731's `encoding/encoding_dsv4.py` reference; the canonical delimiter and `string="true|false"` parameter attribute were already handled. See [tool-call repair](architecture/tool-call-repair.md).

## 2026-08-01

* **Documentation**: Added this OKF bundle — architecture (request pipeline, tool-call repair, streaming contract), reference (configuration, API surface), decisions (modality routing, stream termination, retry policy), and runbooks (deploy, diagnose a stalled stream). Recorded the deployed topology, which is the inverse of what the README implies: clients reach LiteLLM, which calls the proxy, whose upstream is vLLM at `192.168.10.221:8080`.
* **Fix**: Streamed turns now always terminate. Two defects were found behind clients showing a turn as in progress after the model had finished: a terminal `finish_reason` was only synthesised when the proxy had repaired raw tool calls, and with `UPSTREAM_READ_TIMEOUT=0` an upstream that stopped sending without closing the socket held the client indefinitely. Added `UPSTREAM_STREAM_IDLE_TIMEOUT` (default 120s) and unconditional terminal-chunk synthesis. See [always terminate a streamed turn](decisions/stream-termination.md).
* **Feature**: Added an optional YAML config file (`PROXY_CONFIG_FILE`) carrying `models:` aliases and `routes:` modality routes, loaded as the lowest-priority pydantic-settings source so environment variables still win. A configured file that is missing or malformed fails startup rather than running with partial routing.
* **Feature**: Added modality routing. Chat requests carrying image or audio parts — including Ollama `images`, which are translated to `image_url` parts before detection — are forwarded to a configured alternate upstream with that route's model substituted. A route's `api_key` replaces the caller's `Authorization` so a credential for the primary upstream never crosses to another host. See [modality routing over a vision tool loop](decisions/modality-routing.md).
* **Reliability**: Adopted three ideas from [VisionBridge](https://github.com/thomasunise/visionbridge) (MIT) after reviewing it. SSE keepalive comments every `SSE_KEEPALIVE_INTERVAL` seconds of upstream silence; `Cache-Control: no-cache` and `X-Accel-Buffering: no` on streamed responses so intermediaries do not buffer; and retries (`UPSTREAM_MAX_RETRIES`, default 2) on transport errors and upstream 429/5xx, strictly before the first response byte reaches the client. See [retry only before the first byte](decisions/retry-policy.md).
* **Review**: Evaluated VisionBridge's tool-loop architecture as an alternative to modality routing and recorded why it was not adopted, along with the middle path — a `describe` route mode reusing only its scene-priming step — which remains open.
