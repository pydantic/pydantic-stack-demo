# MCP sampling

## What it shows

Pydantic AI on both sides of the Model Context Protocol. The client agent uses an MCP
server as a toolset; the server's tool is itself a Pydantic AI agent, but it has **no model
credentials**: with MCP sampling its model requests travel back over the MCP connection and
the client answers them with its own model. It answers: *"can our MCP servers use an LLM
without us handing every server an API key?"* and *"what does an MCP round trip look like
in Logfire?"*

```
client agent ──tool call──▶ svg-generator (MCP server, stdio)
     ▲                            │
     └──── sampling request ──────┘   (server borrows the client's model)
```

## Run

```bash
uv run mcp-sampling-client
```

The client starts `generate_svg.py` as a subprocess, asks it for "a robot in a punk style,
it should be pink", and the server writes the SVG into `apps/mcp-sampling/output/`.

## What to look at in Logfire

- Two services: `mcp-sampling-client` and `mcp-sampling-server`, linked into one trace by
  `logfire.instrument_mcp()`.
- In the client trace: `svg_client_agent run` → model request → `running tool:
  image_generator` (the MCP tool call).
- Nested under it, from the server process: `svg_agent run` whose model request is a
  `sampling/createMessage` back to the client, answered by the client's model.

## Docs

- [MCP client](https://pydantic.dev/docs/ai/mcp/client/) and
  [MCP sampling](https://pydantic.dev/docs/ai/mcp/client/#mcp-sampling)
- [Pydantic AI inside an MCP server](https://pydantic.dev/docs/ai/mcp/server/)
- [`MCPSamplingModel`](https://pydantic.dev/docs/ai/api/models/mcp-sampling/)
- [Logfire MCP instrumentation](https://pydantic.dev/docs/logfire/integrations/llms/mcp/)

## Files

- `src/mcp_sampling/generate_svg.py` — the MCP server (`mcp.server.fastmcp.FastMCP`) with the sampling agent.
- `src/mcp_sampling/client.py` — the client agent with an `MCPToolset` over stdio.
- `pyproject.toml` pins `mcp<2` and `fastmcp-slim<4`: Logfire's MCP instrumentation and
  server-initiated sampling both need the stateful MCP 1.x session. (FastMCP 4's default
  stateless mode has no connection for the server to send sampling requests over.)
- `tests/test_sampling.py` — full stdio round trip with `TestModel` on both ends.
