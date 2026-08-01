# Architecture

* [Request pipeline](request-pipeline.md) - Route order, rewrite stages, deployed topology, and concurrency limiting.
* [Tool-call repair](tool-call-repair.md) - Detecting raw DSML and Qwen tool markup mid-stream and converting it to OpenAI `tool_calls`.
* [Streaming contract](streaming-contract.md) - What a streaming client is promised about turn termination, and the four mechanisms behind it.
