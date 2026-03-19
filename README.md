# Pydantic Stack Demos

A collection of small, focused examples showing how to use PydanticAI, Logfire, and related tools together.

## Demos

### `distributed-frontend-example/`

React + FastAPI image generator with distributed tracing from browser to backend via Logfire. Shows how to proxy client traces safely and trace calls into OpenAI and file I/O.

### `durable-exec/`

Deep‑research and “Twenty Questions” agents implemented with plain asyncio, DBOS workflows, and Temporal workflows. Includes an evaluation script using pydantic‑evals to compare different base models.

### `fastapi-example/`

Minimal FastAPI app that generates images with OpenAI, stores them, and serves a simple HTML page. Demonstrates FastAPI + Logfire + HTTPX instrumentation.

### `logfire-hello-world/`

Smallest possible Logfire example: configure a service and emit a single structured log line.

### `pai-hello/`

“Hello world” for PydanticAI: a single agent call with a short system prompt, optionally instrumented with Logfire.

### `pai-mcp-sampling/`

MCP sampling demo with PydanticAI acting as both MCP client and MCP server. The client calls an MCP tool that runs another agent via `MCPSamplingModel` to generate SVG images.

### `pai-memory/`

Two Postgres‑backed memory patterns:

- storing full message history per user, and  
- storing and retrieving explicit “facts” through agent tools.

### `pai-pydantic/`

Structured extraction into Pydantic models. Shows how `output_type` works and how validation errors trigger retries when domain rules are enforced.

### `pai-weather/`

Tool‑using agent that resolves locations and fetches (randomised) weather data via HTTP endpoints, with typed tool outputs, dependency injection, and concurrent HTTP requests.
