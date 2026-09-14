"""Memory as stored message history.

Every run's new messages are saved as JSON keyed by user; the next run gets the whole
conversation back as `message_history`. The agent "remembers" because it sees everything
it saw before. Simple and complete, but the context grows with every turn.
"""

from __future__ import annotations

import asyncio

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from demo_core import configure_logfire, settings
from memory.db import DbConn, connect

SCHEMA = """
create table if not exists messages(
    id serial primary key,
    ts timestamp not null default now(),
    user_id integer not null,
    messages json not null
)
"""

agent = Agent(settings().model, name='memory_messages_agent', instructions='You are a helpful assistant.')


@logfire.instrument('run agent for user {user_id}')
async def run_agent(conn: DbConn, prompt: str, user_id: int) -> str:
    with logfire.span('retrieve messages'):
        history: list[ModelMessage] = []
        for row in await conn.fetch('SELECT messages FROM messages WHERE user_id = $1 ORDER BY ts', user_id):
            history += ModelMessagesTypeAdapter.validate_json(row[0])

    result = await agent.run(prompt, message_history=history)

    with logfire.span('record messages'):
        await conn.execute(
            'INSERT INTO messages(user_id, messages) VALUES($1, $2)', user_id, result.new_messages_json().decode()
        )
    return result.output


async def conversation(user_id: int = 123) -> list[str]:
    async with connect(SCHEMA) as conn:
        first = await run_agent(conn, 'My name is Samuel.', user_id)
        print(first)
        second = await run_agent(conn, 'What is my name?', user_id)
        print(second)
        return [first, second]


def main() -> None:
    configure_logfire('memory')
    logfire.instrument_asyncpg()
    settings().require_model_credentials()
    asyncio.run(conversation())


if __name__ == '__main__':
    main()
