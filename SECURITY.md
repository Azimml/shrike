# Security Policy

## Reporting a Vulnerability

shrike is a research/educational inference engine. If you find a security issue
(e.g. a request that can crash or hang the server, or an input-validation gap in
the OpenAI-compatible API), please open a private security advisory on GitHub or
open an issue describing the impact and a reproduction. We aim to respond within
a few days.

## Scope

- The HTTP/SSE server (`shrike/server/`) and request validation
- Model-loading paths that read files from disk

Out of scope: performance characteristics, and behavior under adversarial GPU
resource exhaustion (this is a single-node research engine).
