# Decisions

* [Modality routing over a vision tool loop](modality-routing.md) - Why image and audio requests are forwarded wholesale to a second host rather than given to the primary model as tools.
* [Always terminate a streamed turn](stream-termination.md) - Why the proxy synthesises terminal chunks and bounds upstream silence instead of relying on the read timeout.
* [Retry only before the first byte](retry-policy.md) - Which upstream failures are retried, and why a started stream never is.
* [A terminated turn must also be a usable turn](turn-usability.md) - Why the proxy completes truncated tool-call arguments and annotates turns that produce no output.
