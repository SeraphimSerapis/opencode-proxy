---
type: decision
title: Modality routing over a vision tool loop
description: Image and audio requests are forwarded wholesale to a second multimodal host rather than exposed to the primary model as vision tools.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/routing.py
tags: [vision, audio, routing, multimodal, trade-off]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
sources:
  - id: visionbridge
    title: VisionBridge — give text-only LLMs vision through a tool loop
    url: https://github.com/thomasunise/visionbridge
---

# Modality routing over a vision tool loop

## Context

The primary model (DeepSeek V4 Flash on vLLM) is text-only. Pasting a screenshot
into a coding agent fails at the model, not the proxy. A second host runs
Gemma 4 E4B with vision and audio support.

## Decision

Detect non-text content parts in chat requests and forward the whole request to
a configured alternate upstream, substituting that route's model.

Detection covers `image_url`, `image`, `input_image` (vision) and `input_audio`,
`audio`, `audio_url` (audio), plus Ollama `images` fields. Ollama messages are
translated to `image_url` parts before routing runs, so one detector serves both
protocols.

## Alternative considered

[VisionBridge](https://github.com/thomasunise/visionbridge) solves the same
problem differently: it keeps the reasoning model in charge and gives it vision
as tools — `look(image_id, question)`, `ocr`, `crop_and_look`, `compare` — with
the vision model as a subordinate answering targeted questions. Images are
stripped from the conversation and replaced with placeholders.

That design is better on the axis that matters most for an agentic client. Our
routing hands the **entire turn** to the vision model, including the tool
calling, the repository context, and the reasoning. A small multimodal model will
do worse at the agent work than the model it displaced, even though it is the
only one that can see.

It was not adopted because the cost is an 800-line orchestrator with its own tool
protocol, fallback mode, and failure surface — a different class of component
from a transparent proxy. Routing is roughly 200 lines and correct for the common
case, which is a one-off "what is in this screenshot".

The middle path, not yet built: borrow only VisionBridge's *scene priming* step.
Ask the vision model once for a description of each image, replace the image
parts with that text, and let the primary model handle the turn. Around 120
lines, no tool loop, and the agent model stays in charge. Tracked as a
`mode: route | describe` option on each route.

## Rules

**Audio wins when a request carries both.** Audio-capable endpoints are the
scarcer capability and generally accept images too.

**An unrouted modality is a warning, not an error.** The request goes to the
primary upstream unchanged. The proxy cannot know whether the primary model
handles images, and failing closed would break every deployment whose primary
model does.

**A route's `api_key` replaces the caller's `Authorization`.** Routing crosses a
host boundary; a credential issued for the primary upstream must not follow the
request to a different machine. When a route sets no `api_key`, the caller's
header is forwarded — appropriate for hosts on the same trusted LAN.

**Routed models are absent from `/v1/models`.** Advertising them would offer
clients a model id that fails when selected directly, since nothing routes by
name.

## Consequences

* A vision turn loses the primary model's agentic quality. Acceptable for
  one-shot image questions; poor for an agent mid-task. This is the main reason
  the `describe` mode remains open.
* Configuration for an unreachable route surfaces as a typed `502` rather than a
  confusing model-side error, because the proxy never sends image parts to a
  text-only model once a route matches.
* Detection is structural, not semantic: a request is "vision" because it
  carries an image part, regardless of whether the question needs sight.
