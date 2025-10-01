import re
from pathlib import Path

import logfire
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSessionT
from mcp.shared.context import LifespanContextT, RequestT
from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

logfire.configure(service_name='mcp-server', environment='mcp-sampling', console=False)
logfire.instrument_mcp()

app = FastMCP(log_level='WARNING')

svg_agent = Agent(instructions='Generate an SVG image as per the user input. Return the SVG data only as a string.')


@app.tool()
async def image_generator(ctx: Context[ServerSessionT, LifespanContextT, RequestT], subject: str, style: str) -> str:
    # run the agent, using MCPSamplingModel to proxy the LLM call through the client.
    svg_result = await svg_agent.run(f'{subject=} {style=}', model=MCPSamplingModel(ctx.session))

    path = Path(f'{slugify(subject)}_{slugify(style)}.svg')
    logfire.info(f'writing file to {path}')
    # remove triple backticks if the svg was returned within markdown
    if m := re.search(r'^```\w*$(.+?)```$', svg_result.output, re.S | re.M):
        path.write_text(m.group(1))
    else:
        path.write_text(svg_result.output)
    return f'See {path}'


def slugify(text: str) -> str:
    return re.sub(r'\W+', '-', text.lower())


if __name__ == '__main__':
    # run the server via stdio
    app.run()
