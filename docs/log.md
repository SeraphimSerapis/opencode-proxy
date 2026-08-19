# Documentation Update Log

## 2026-08-19

* **Feature**: Model discovery now advertises a built-in `primary` alias when the upstream does not define one. Requests for `primary` resolve against the first model returned by upstream `/v1/models`; discovery failures return a bounded `502` instead of forwarding an invalid model name.

## 2026-08-16

* **Feature**: Keepalives can be requested as empty-delta chunks instead of SSE comments, per request, via `X-Opencode-Proxy-Keepalive: chunk`. Comments are discarded by SSE parsers, so a client that extends its own deadline on forward progress saw nothing across a multi-minute prefill and cut the turn off early; the comment path is unchanged for callers that do not opt in, and the header is not forwarded upstream.
* **Fix**: The streaming contract claimed keepalive comments gave the client evidence of life. They do not -- an SSE parser drops them before application code -- and the document now says what they actually do.
* **Reliability**: The first-frame stream guard was raised from `480s` to `900s` in the deployed compose unit. Measured cold prefill of a 316,536-token prompt (`cached_tokens=0`) moved from 241.7s to 409.8s after a vLLM prefill retune, and `480` began terminating legitimate prefills.

## 2026-08-15

* **Feature**: The proxy now repairs outgoing message shapes before forwarding (`NORMALIZE_REQUESTS`, default on): `null` assistant content becomes `""`, `reasoning_content` is replayed only on tool-call turns and dropped elsewhere, empty tool results carry `(no output)`, and for `deepseek_v4` models `reasoning_effort: off` becomes `thinking: {"type": "disabled"}`. Each of these is a shape the DeepSeek API rejects outright, and because a rejected message sits durably in the caller's session log, one bad turn was enough to break every later turn of that session. Rules taken from [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), the vendor's own client. See [conform to DeepSeek's own client](decisions/deepseek-wire-contract.md).
* **Feature**: A buffered turn that completes with no content and no tool call is re-sent once (`EMPTY_RESPONSE_RETRIES`) before being annotated, and the buffered transport now gets the same `[proxy: ...]` notice streamed turns already had. A `length` truncation is excluded from the retry. See [retry only before the first byte](decisions/retry-policy.md).
* **Fix**: An abnormal upstream terminator (`content_filter`, vLLM's `insufficient_system_resource`) no longer gets the token-budget notice, which was misattributing every one of them. The notice now names the terminator. The value itself is still forwarded unchanged.
* **Reliability**: A `Retry-After` header on a retryable status now paces the retry instead of the proxy's own backoff curve, clamped to 30s. Both header forms are parsed; malformed or absurd values fall back to the curve.
* **Observability**: Four new counters -- `opencode_proxy_request_normalizations{kind}`, `opencode_proxy_upstream_errors{type}` (auth, quota, rate limit, context window, invalid request, server, other 4xx, other status), `opencode_proxy_finish_reasons{reason,transport}`, and `opencode_proxy_usage_tokens{kind}`. Token usage is counted disjointly: DeepSeek reports `prompt_tokens` with cache hits included, so the cached share is subtracted out of `input`. All counters are documented in the [API surface](reference/api-surface.md).
* **Observability**: Concurrency saturation is now visible through `opencode_proxy_upstream_overloads_total` and the `opencode_proxy_upstream_active` gauge, so proxy admission pressure can be correlated with vLLM queue depth.
* **Feature**: The `deepseek_v4` profile gained `thinking_transport`, because the two servers we talk to disagree about how to turn thinking off. Measured against the deployed vLLM: it ignores the DeepSeek API's top-level `thinking` field and reads `chat_template_kwargs: {"thinking": false}` instead, while the Qwen spelling `enable_thinking` does nothing. Default stays the API form; under the vLLM form `reasoning_effort` is translated to a boolean and dropped, since a template argument cannot carry a level.
* **Fix**: DeepSeek request normalization now also accepts the official `thinking` object, maps Ollama `think`, canonicalizes effort aliases, rewrites `developer` to `system`, and translates `max_completion_tokens` to `max_tokens`. Message normalization and buffered empty-turn retries are scoped to configured `deepseek_v4` profiles so unrelated providers remain transparent; the Ollama adapter now gets the same buffered empty-turn retry and usage/finish metrics.
* **Fix**: Adversarial follow-up preserved API effort levels alongside `thinking: enabled`, correlated Ollama tool-result IDs, forwarded streamed Ollama errors without reading them too early, and passed request-aware orphan recovery through both Ollama response modes. Ollama `think` remains protocol-correct when optional message normalization is disabled.
* **Reliability**: The first-frame stream guard was raised from `240s` to `480s` after the live vLLM histogram-estimated p99 TTFT reached about `351s`; the mid-stream guard remains the separate stall detector.
* **Build**: The production image now installs its non-development environment from `uv.lock` with `uv sync --locked`, so stale locks fail the build and container dependency versions cannot drift from the reviewed environment.
* **Review**: Read DeepSeek's `deepseek-harness` against this proxy. `DSML` appears nowhere in it -- the vendor client assumes native `tool_calls` -- confirming that every raw-text repair here compensates for the self-hosted chat template rather than for the model contract. Recorded the V4 Flash serving facts (1M context, 256k default output cap) in the [DeepSeek V4 runbook](runbooks/deepseek-v4.md), and the deliberate divergences (SSE tail flushing, reasoning-only turns counted as empty) in the decision.

## 2026-08-11

* **Fix**: The container reported `healthy` with its model server down. The
healthcheck ran `/healthz`, which returns `ok` from the moment uvicorn binds and
never touches the upstream, so the one signal an operator reads said nothing
about whether the proxy could serve. The healthcheck now runs `/readyz`, in the
Dockerfile and in the compose unit.
* **Fix**: `/readyz` itself was too lenient to catch a broken backend. It
accepted any non-`5xx`, so a `404` from a wrong `UPSTREAM_URL` read as ready, as
did a `200` listing no models. Non-auth `4xx`, unparseable bodies, and empty
model lists are now `not_ready`, each with a named reason and a new
`opencode_proxy_upstream_ready_failures{reason}` counter.
* **Feature**: `UPSTREAM_HEALTH_PATH` (set to `/health` for the vLLM upstream).
vLLM and LiteLLM both serve `/v1/models` from static configuration, so it keeps
answering `200` after the engine behind it dies — a model listing can never
detect that failure on its own. See [API surface](reference/api-surface.md).
* **Fix**: The public Traefik route was unauthenticated (`chain-no-auth`), so
anyone reaching `opencode-proxy.${DOMAIN_COFFEE}` could run inference on the GPU
and read the internal upstream origin from `/healthz/config`. Replaced with the
three-router pattern LiteLLM already uses: a LAN `ClientIP` router at priority
200, an `X-API-KEY` header router at priority 100, and a `chain-tinyauth`
catch-all at priority 1. Plain Tinyauth alone was not an option — the route
carries 448 requests/14d from headless clients (OpenAI/JS, opencode, curl) that
cannot complete a browser SSO flow, and they now match the LAN router unchanged.
The edge credential is stripped by `opencode-proxy-strip-x-api-key` so it never
reaches vLLM.
* **Fix**: Nothing alerted on any of the above. Prometheus scraped the proxy and
vLLM and loaded `rule_files`, but its Alertmanager target was commented out, so
the existing rules notified nobody — the same class of defect as the healthcheck
itself. Deployed Alertmanager with delivery through mailrise (`*@tme.coffee` →
Gmail + Discord), added proxy and vLLM alert rules, and re-enabled the vLLM
recording rules that were parked as `vllm.yml.disabled` "until scrape is
restored" — the scrape has been healthy for some time. Rules and the reload
procedure are in the [deploy runbook](runbooks/deploy.md).
* **Fix**: The stream idle guard used one budget for two different silences.
Because it starts counting at body iteration, the flat `120` had to cover both
prefill and mid-stream gaps. Measured over seven days against the deployed vLLM,
time-to-first-token p99.9 is 160s while inter-token p99.9 is 1.5s — so `120` was
*below* the prefill p99.9, killing legitimate slow starts, and ~80× the
mid-stream p99.9, leaving dead streams hanging for two minutes. Split into
`UPSTREAM_STREAM_FIRST_FRAME_TIMEOUT` (`240`) and a between-frame
`UPSTREAM_STREAM_IDLE_TIMEOUT` (now `30`), with the expired budget recorded in
`opencode_proxy_stream_idle_terminations{phase}`. Rationale and the latency
table are in [stream termination](decisions/stream-termination.md).
* **Fix**: `deploy/prometheus/opencode-proxy.yml` had drifted from the deployed
unit and still described the retired `proxy -> LiteLLM` direction. Redeploying
from it would have pointed `UPSTREAM_URL` at `litellm:4000` and looped requests
back through LiteLLM. It now mirrors the live topology recorded in
[request pipeline](architecture/request-pipeline.md): `pi -> LiteLLM -> proxy ->
vLLM`. Its `depends_on: litellm` is dropped in the same pass — it coupled proxy
startup to a service that is downstream of it, not upstream.

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
