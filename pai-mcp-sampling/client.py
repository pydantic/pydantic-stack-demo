from pathlib import Path

import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

logfire.configure(service_name='mcp-client', environment='mcp-sampling')
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()

server = MCPServerStdio(command='python', args=[str(Path(__file__).parent / 'generate_svg.py')])
agent = Agent('openai:gpt-4.1-mini', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('Create an image of a robot in a punk style, it should be pink.')
    print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
