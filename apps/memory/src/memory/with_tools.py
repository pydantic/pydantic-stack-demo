"""Memory as tools.

The agent gets `record_memory` and `retrieve_memories` and decides for itself what is
worth keeping and when to look it up. Each run starts with an empty context; only the
facts the agent chose to store come back. Smaller context, but the model has to be
prompted well enough to actually use the tools.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import logfire
from pydantic_ai import Agent, RunContext

from demo_core import configure_logfire, settings
from memory.db import DbConn, connect

SCHEMA = """
create table if not exists memory(
    id serial primary key,
    user_id integer not null,
    value text not null,
    unique(user_id, value)
)
"""


@dataclass
class Deps:
    user_id: int
    conn: DbConn


agent = Agent(
    settings().model,
    name='memory_tools_agent',
    deps_type=Deps,
    instructions=(
        'You are a helpful assistant. Store facts the user tells you about themselves with `record_memory`, '
        'and call `retrieve_memories` before answering questions about the user.'
    ),
)


@agent.tool
async def record_memory(ctx: RunContext[Deps], value: str) -> str:
    """Store a fact about the user in long-term memory."""
    await ctx.deps.conn.execute(
        'insert into memory(user_id, value) values($1, $2) on conflict do nothing', ctx.deps.user_id, value
    )
    return 'Value added to memory.'


@agent.tool
async def retrieve_memories(ctx: RunContext[Deps], memory_contains: str) -> str:
    """Find stored facts about the user containing the given text."""
    rows = await ctx.deps.conn.fetch(
        'select value from memory where user_id = $1 and value ilike $2', ctx.deps.user_id, f'%{memory_contains}%'
    )
    return '\n'.join(row[0] for row in rows) or 'No memories found.'


async def conversation(user_id: int = 123) -> list[str]:
    outputs: list[str] = []
    async with connect(SCHEMA) as conn:
        await conn.execute('delete from memory where user_id = $1', user_id)
        result = await agent.run('My name is Samuel.', deps=Deps(user_id, conn))
        print(result.output)
        outputs.append(result.output)

    # Time goes by: a fresh connection and a fresh run with no message history.
    async with connect(SCHEMA) as conn:
        result = await agent.run('What is my name?', deps=Deps(user_id, conn))
        print(result.output)
        outputs.append(result.output)
    return outputs


def main() -> None:
    configure_logfire('memory')
    logfire.instrument_asyncpg()
    settings().require_model_credentials()
    asyncio.run(conversation())


if __name__ == '__main__':
    main()
