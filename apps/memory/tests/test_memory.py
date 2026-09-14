from typing import Any

from pydantic_ai.models.test import TestModel

from memory import with_messages, with_tools


class FakeConn:
    """Just enough of asyncpg.Connection for the two demos."""

    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[tuple[Any, ...]]:
        return self.rows

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        if sql.lstrip().lower().startswith('insert'):
            self.rows.append(tuple(args[1:]))
        return 'OK'


async def test_messages_are_stored_and_replayed():
    conn = FakeConn()
    with with_messages.agent.override(model=TestModel(custom_output_text='Hi Samuel')):
        out = await with_messages.run_agent(conn, 'My name is Samuel.', 1)  # type: ignore[arg-type]
    assert out == 'Hi Samuel'
    assert len(conn.rows) == 1
    with with_messages.agent.override(model=TestModel(custom_output_text='Samuel')):
        await with_messages.run_agent(conn, 'What is my name?', 1)  # type: ignore[arg-type]
    assert len(conn.rows) == 2


async def test_tools_are_wired():
    conn = FakeConn()
    with with_tools.agent.override(model=TestModel()):
        result = await with_tools.agent.run('My name is Samuel.', deps=with_tools.Deps(1, conn))  # type: ignore[arg-type]
    names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
    assert 'record_memory' in names and 'retrieve_memories' in names
