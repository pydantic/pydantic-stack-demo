# multi-agent

**What it shows.** The three ways teams compose agents in Pydantic AI, and how to evaluate
the result. "How do I build a multi-agent system?" almost always means one of these:

| Script | Pattern | When to use it |
|---|---|---|
| `multi-agent-twenty-questions` | **Agent as a tool**: the questioner's `ask_question` tool runs the answerer agent, charging its usage to the parent with `usage=ctx.usage` | Two agents with distinct roles and a fixed protocol between them |
| `multi-agent-research` | **Plan → fan out → analyse**: a plan agent produces a structured plan, search agents run concurrently in an `asyncio.TaskGroup`, an analysis agent writes the report (and may search again through a tool) | Pipelines where plain Python should own the control flow |
| `multi-agent-subagents` | **Harness `SubAgents`**: one `delegate_task` tool, delegates listed in the system prompt, shared usage limits, delegation events | Delegation without writing the plumbing yourself |
| `multi-agent-evals` | **Evaluate the whole system** with Pydantic Evals: the game is the task, `Success` and `QuestionCount` are the evaluators, run once per model with `agent.override(model=...)` | Comparing models or prompts on an agentic task, not a single completion |

The search agent uses the provider's native web search (`WebSearch()` capability) through
the Gateway; it works on OpenAI Responses and Anthropic models. For a model without native
search, pass `WebSearch(local='duckduckgo')` (the `duckduckgo` extra is already installed).

## Run

```bash
uv run multi-agent-twenty-questions      # ~9 questions, ~$0.02
uv run multi-agent-research              # default question about Python free-threading; pass your own as arguments
uv run multi-agent-research "Compare DBOS and Temporal for durable agents"
uv run multi-agent-subagents
uv run multi-agent-evals                 # 3 games × 2 models, ~2 minutes, ~$0.20
```

Standard `.env` keys only (`PYDANTIC_AI_GATEWAY_API_KEY`, `LOGFIRE_TOKEN`, `DEMO_MODEL`,
`DEMO_FAST_MODEL`). The questioner, plan and analysis agents use `DEMO_MODEL`; the answerer
and search agents use `DEMO_FAST_MODEL`.

## What to look at in Logfire

- **Twenty questions**: open the `questioner_agent run` trace. Each `running tool: ask_question`
  span contains a nested `answerer_agent run`; the run span's aggregated usage covers both
  agents because the tool passed `usage=ctx.usage`.
- **Research**: the `deep_research` span holds `plan_agent run`, three overlapping
  `search_agent run` spans (the fan-out) and `analysis_agent run`. The search runs show the
  native `web_search` tool call and its citations.
- **SubAgents**: `orchestrator run` → `running tool: delegate_task` → `researcher run` /
  `editor run`. The `usage` printed at the end is the whole tree.
- **Evals**: each `dataset.evaluate` prints a Logfire experiment URL. Open two experiments in
  *Evals → Compare* to see which model wins on `Success` and how many questions it needed.

Useful SQL (Explore) to see cost per agent across all of this:

```sql
SELECT attributes->>'gen_ai.agent.name' AS agent,
       count(*) AS runs,
       round(sum((attributes->'operation.cost')::numeric), 4) AS cost_usd
FROM records
WHERE span_name LIKE 'invoke_agent %' OR span_name LIKE '% run'
GROUP BY 1 ORDER BY 3 DESC
```

## Docs

- [Multi-agent applications](https://pydantic.dev/docs/ai/multi-agent-applications/)
- [Web Search capability](https://pydantic.dev/docs/ai/capabilities/web-search/)
- [Harness SubAgents](https://pydantic.dev/docs/ai/harness/subagents/)
- [Usage limits](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits)
- [Pydantic Evals](https://pydantic.dev/docs/ai/evals/) and [Logfire experiments](https://pydantic.dev/docs/logfire/evaluate/)

## Files

- `src/multi_agent/twenty_questions.py` — the two agents, `build_agents()` (reused by `apps/durable-exec`), `play()`
- `src/multi_agent/research.py` — plan / search / analysis agents and the `deep_research` pipeline
- `src/multi_agent/subagents.py` — `SubAgents` orchestrator with a delegation-event listener
- `src/multi_agent/evals.py` — dataset, evaluators and the per-model comparison loop
- `tests/` — offline tests with `TestModel` / `FunctionModel`
