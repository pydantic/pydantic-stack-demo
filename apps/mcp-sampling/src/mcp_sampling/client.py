"""An agent that uses the SVG MCP server as a toolset and lends it a model via sampling."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import logfire
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from demo_core import configure_logfire, settings

SERVER = Path(__file__).with_name('generate_svg.py')


def build_agent(output_dir: Path | None = None) -> Agent:
    # stdio transport: the client starts the server as a subprocess with the same interpreter.
    # The subprocess gets a copy of our environment (so it sees the same .env-derived settings).
    env = dict(os.environ)
    if output_dir is not None:
        env['MCP_SAMPLING_OUTPUT_DIR'] = str(output_dir)
    transport = StdioTransport(command=sys.executable, args=[str(SERVER)], env=env)
    toolset = MCPToolset(transport, id='svg-generator')
    agent = Agent(settings().model, name='svg_client_agent', toolsets=[toolset])
    # Answer the server's sampling requests with this agent's own model.
    agent.set_mcp_sampling_model()
    return agent


async def run(prompt: str = 'Create an image of a robot in a punk style, it should be pink.') -> str:
    agent = build_agent()
    async with agent:  # starts the server subprocess, stops it afterwards
        result = await agent.run(prompt)
    return result.output


def main() -> None:
    configure_logfire('mcp-sampling-client')
    logfire.instrument_mcp()
    settings().require_model_credentials()
    print(asyncio.run(run()))


if __name__ == '__main__':
    main()
