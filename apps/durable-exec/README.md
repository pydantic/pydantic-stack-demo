# durable-exec

**What it shows.** The two multi-agent workflows from `apps/multi-agent` made durable, once
with [DBOS](https://pydantic.dev/docs/ai/durable-execution/dbos/) and once with
[Temporal](https://pydantic.dev/docs/ai/durable-execution/temporal/). The agents are the same
objects, built with a `DBOSDurability()` or `TemporalDurability()` capability attached; the
workflow body is the same Python. Kill the process mid-run and start it again with the
workflow id: the run resumes from the last completed model request instead of paying for the
whole thing again. This is the answer to "what happens to a 10-minute agent run when the pod
restarts?".

| Script | Engine | Workflow |
|---|---|---|
| `durable-exec-dbos-twenty-questions` | DBOS (in-process, SQLite by default) | one agent run |
| `durable-exec-dbos-research` | DBOS | plan → child workflows fan out → analyse |
| `durable-exec-temporal-twenty-questions` | Temporal | one agent run, with a deliberately flaky tool |
| `durable-exec-temporal-research` | Temporal | plan → `asyncio.TaskGroup` fan-out → analyse |

## Run

DBOS needs nothing else; it checkpoints to `durable_exec_dbos.sqlite` in the current directory
(set `DBOS_SYSTEM_DATABASE_URL` to a Postgres URL for production; `docker compose --profile
postgres up` provides one on port 5433).

```bash
uv run durable-exec-dbos-twenty-questions          # prints "starting workflow twenty-questions-<id>"
uv run durable-exec-dbos-research ["your question"]
```

Temporal needs a server. Either `docker compose --profile durable-exec up temporal` (UI on
http://localhost:8233) or `temporal server start-dev`; the address comes from
`TEMPORAL_ADDRESS` (default `localhost:7233`). The script runs a worker in-process and
executes the workflow on it.

```bash
uv run durable-exec-temporal-twenty-questions      # prints "starting workflow twenty-questions-<id>"
uv run durable-exec-temporal-research ["your question"]
```

### Resume walkthrough

1. Start a game: `uv run durable-exec-temporal-twenty-questions`. Copy the workflow id from
   the first line of output.
2. After three or four questions, press `Ctrl-C`.
3. Run `uv run durable-exec-temporal-twenty-questions twenty-questions-<id>`. The worker
   picks the workflow back up: Temporal replays the event history, the already-answered
   questions are not sent to the model again, and the game continues. The same works for DBOS
   with `uv run durable-exec-dbos-twenty-questions twenty-questions-<id>`, and for the research
   scripts with `--resume <id>`.

The Temporal twenty-questions tool fails on purpose 10% of the time
(`tool_failure_rate=0.1` in `temporal_twenty_questions.py`). Watch the worker log a
`Completing activity as failed` line and Temporal retry the activity; the game never notices.

## What to look at in Logfire

- Each run is one trace: DBOS adds `play_workflow` / `search_workflow` spans around the
  agent runs (`enable_otlp=True`); Temporal's `LogfirePlugin` adds `RunWorkflow` /
  `RunActivity` spans, so the retried activity appears twice under one workflow span.
- On a resumed run, notice which `chat ...` spans are missing: those are the requests that
  were replayed from the checkpoint rather than re-sent to the model.
- Cost per workflow: `SELECT attributes->>'temporal.workflow_id', sum((attributes->'operation.cost')::numeric) ...`
  on the Temporal traces, or group by trace id for DBOS.

## Docs

- [Durable execution overview](https://pydantic.dev/docs/ai/durable-execution/overview/)
- [DBOS integration](https://pydantic.dev/docs/ai/durable-execution/dbos/): `DBOSDurability`, step config, streaming
- [Temporal integration](https://pydantic.dev/docs/ai/durable-execution/temporal/): `TemporalDurability`, activity config, payload limits
- [Temporal dev server](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally)

## Files

- `src/durable_exec/dbos_twenty_questions.py` — `@DBOS.workflow` around the agent run, resume by id
- `src/durable_exec/dbos_research.py` — child workflows started with `DBOS.start_workflow_async` for the fan-out
- `src/durable_exec/temporal_twenty_questions.py` — `PydanticAIWorkflow` subclass, in-process worker, flaky tool
- `src/durable_exec/temporal_research.py` — the fan-out inside a Temporal workflow with a longer activity timeout
- `tests/` — the agents run with `TestModel` outside any workflow (the capabilities are transparent there)
