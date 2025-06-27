import re
from pathlib import Path

from mcp import SamplingMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSessionT
from mcp.shared.context import LifespanContextT, RequestT
from mcp.types import TextContent

app = FastMCP(log_level='WARNING')

import logfire

logfire.configure(service_name='mcp-sampling-server', console=False)
logfire.instrument_mcp()


@app.tool()
async def image_generator(ctx: Context[ServerSessionT, LifespanContextT, RequestT], subject: str, style: str) -> str:
    prompt = f'{subject=} {style=}'
    # `ctx.session.create_message` is the sampling call
    result = await ctx.session.create_message(
        [SamplingMessage(role='user', content=TextContent(type='text', text=prompt))],
        max_tokens=1_024,
        system_prompt='Generate an SVG image as per the user input',
    )
    assert isinstance(result.content, TextContent)

    path = Path(f'{subject}_{style}.svg')
    # remove triple backticks if the svg was returned within markdown
    if m := re.search(r'^```\w*$(.+?)```$', result.content.text, re.S | re.M):
        path.write_text(m.group(1))
    else:
        path.write_text(result.content.text)
    return f'See {path}'


if __name__ == '__main__':
    # run the server via stdio
    app.run()
