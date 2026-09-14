# Memory

## What it shows

Two ways to give an agent memory that survives between runs, both backed by Postgres,
and how each looks in Logfire. It answers: *"how do I make my agent remember the user?"*

| Script | Approach | Trade-off |
| --- | --- | --- |
| `memory-messages` | Store every run's new messages as JSON; replay them as `message_history` next time. | Complete recall, zero prompt engineering, context grows every turn. |
| `memory-tools` | Give the agent `record_memory` / `retrieve_memories` tools; it decides what to keep. | Small context, selective recall, depends on the model using the tools. |

## Run

You need a Postgres reachable at `DATABASE_URL` (default `postgresql://postgres@localhost:5433/postgres`).
From the repo root:

```bash
docker compose --profile memory up -d postgres     # throwaway Postgres on port POSTGRES_PORT (5433)
uv run memory-messages
uv run memory-tools
```

## What to look at in Logfire

- `memory-messages`: the `run agent for user 123` span contains `retrieve messages`, the
  `memory_messages_agent run` (whose second run carries the first run's messages in
  `gen_ai.input.messages`), and `record messages`. The `INSERT`/`SELECT` spans come from
  `logfire.instrument_asyncpg()`.
- `memory-tools`: two separate `memory_tools_agent run` traces. The first shows a
  `record_memory` tool call, the second a `retrieve_memories` call whose result is the
  stored fact. No message history is passed between them.
- Compare `gen_ai.usage.input_tokens` on the second model request of each approach.

## Docs

- [Message history](https://pydantic.dev/docs/ai/core-concepts/message-history/)
- [Function tools](https://pydantic.dev/docs/ai/tools-toolsets/tools/)
- [Harness `Memory` capability](https://pydantic.dev/docs/ai/harness/memory/) for a
  ready-made, namespaced version of the tool approach
- [asyncpg integration](https://pydantic.dev/docs/logfire/integrations/databases/asyncpg/)

## Files

- `src/memory/db.py` — connection helper; creates the table if needed.
- `src/memory/with_messages.py` — message-history approach.
- `src/memory/with_tools.py` — tool-based approach.
- `tests/test_memory.py` — offline tests with `TestModel` and a fake connection.
