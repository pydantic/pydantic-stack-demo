"""An MCP server (stdio) whose tool runs a Pydantic AI agent through MCP sampling.

The server has no model credentials at all. `MCPSamplingModel` sends the agent's model
requests back over the MCP connection, and the client answers them with its own model.
Run it via the client (`uv run mcp-sampling-client`); it is not meant to be started by hand.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import logfire
from mcp.server.fastmcp import Context, FastMCP
from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

from demo_core import configure_logfire

OUTPUT_DIR = Path(os.environ.get('MCP_SAMPLING_OUTPUT_DIR') or Path(__file__).resolve().parents[2] / 'output')

# console=False: stdout is the MCP transport, so nothing else may write to it.
configure_logfire('mcp-sampling-server', console=False)
logfire.instrument_mcp()

server = FastMCP('svg-generator', log_level='WARNING')

svg_agent = Agent(
    name='svg_agent',
    instructions='Generate an SVG image as per the user input. Return the SVG data only as a string.',
)


@server.tool()
async def image_generator(ctx: Context, subject: str, style: str) -> str:
    """Generate an SVG image of `subject` in the given `style` and save it to a file."""
    # No model is configured on the agent: every request goes to the client via sampling.
    result = await svg_agent.run(f'{subject=} {style=}', model=MCPSamplingModel(session=ctx.session))

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f'{slugify(subject)}_{slugify(style)}.svg'
    # Strip a ```svg fence if the model wrapped its answer in markdown.
    svg = result.output
    if m := re.search(r'^```\w*$(.+?)```$', svg, re.S | re.M):
        svg = m.group(1)
    path.write_text(svg.strip() + '\n')
    logfire.info('wrote {path}', path=str(path))
    return f'Saved the image to {path}'


def slugify(text: str) -> str:
    return re.sub(r'\W+', '-', text.lower()).strip('-')


if __name__ == '__main__':
    server.run()  # stdio
