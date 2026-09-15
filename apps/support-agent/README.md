# Support agent: the reference agent and its eval loop

**What it shows.** The question this answers is "how do I know my agent is still good after
I change the prompt, the model, or the tools?" It is a small support-desk agent (three
policies behind a lookup tool, one escalation tool, structured output) wrapped in the
three-layer eval loop from
[Three-layer evals with Logfire](https://pydantic.dev/articles/three-layer-evals-logfire):

1. **Deterministic checks** that run on every call and cost nothing
   (`NoSensitiveDataLeaked`, `ResolutionShape`, `EscalationMatchesExpectation`).
2. **One narrow LLM judge** (`support_resolution_feasible`) with a written rubric in
   `eval_assets/judge_rubric.md`, asked a single yes/no question.
3. **Human review** in Logfire, which calibrates the judge and feeds confirmed failures
   back into the regression dataset.

The same evaluators run **offline** against `eval_assets/dataset.jsonl` (an experiment you
compare across runs) and **online** on live traffic through the `OnlineEvaluation`
capability (sampled, bounded, in the background). One rubric, two places.

## Run

All commands run from the repo root with a filled-in `.env` (`LOGFIRE_TOKEN`,
`PYDANTIC_AI_GATEWAY_API_KEY`; see `.env.example`).

```bash
# Offline: run the 7-case regression dataset, print the report, link the experiment
uv run support-agent-evals-offline

# Online: send paced traffic through the agent; --judge-all forces the 10% judge to 100%
uv run support-agent-traffic --count 6 --interval 2 --judge-all

# Service: the same agent behind POST /support/resolve (online evaluation on)
uv run support-agent-serve            # http://127.0.0.1:8000, docs at /docs
docker compose --profile support-agent up --build   # same thing in Docker, port 8002

# Optional: publish the dataset to Logfire's managed datasets (needs LOGFIRE_API_KEY
# with dataset read/write on the project)
uv run support-agent-evals-publish
```

Extra `.env` keys: `JUDGE_SAMPLE_RATE` (service only, default `0.1`), `LOGFIRE_API_KEY`
and `LOGFIRE_API_BASE_URL` for publishing.

## What to look at in Logfire

- **Evals -> Experiments** after `support-agent-evals-offline`. The script prints the
  experiment URL. Each case row shows the assertions, the judge's reason, tokens and
  cost; click a case to open the agent run with its tool calls. Run it twice after a
  prompt change and use *Compare* to see what moved.
- **Evals -> Live Monitoring** after `support-agent-traffic` or while the service runs.
  Target `support_resolution_agent`; evaluators `support_resolution_feasible`,
  `NoSensitiveDataLeaked`, `has_next_step`, `escalation_flag_consistent`. Each event links
  to the trace that produced it.
- **Live view**: filter by `service_name = 'support-agent'`. A traffic run produces one
  `POST /support/resolve` span per request with the agent run, the model requests, the
  `execute_tool` spans and the `gen_ai.evaluation.result` events nested under it.
- The `environment` picker separates `offline-eval`, `online-eval` and `local` traffic.

Pass rate per evaluator over the last day, straight from the evaluation events:

```sql
SELECT
  attributes->>'gen_ai.evaluation.name' AS evaluator,
  avg((attributes->'gen_ai.evaluation.score.value')::float) AS pass_rate,
  count(*) AS events
FROM records
WHERE attributes->>'gen_ai.evaluation.target' = 'support_resolution_agent'
  AND start_timestamp > now() - interval '1 day'
GROUP BY 1 ORDER BY 1
```

## Adding a regression case from a real trace

1. In Live Monitoring or Live view, find a run a human has judged wrong and note the
   request text.
2. Append a line to `eval_assets/dataset.jsonl` with a stable `case_id`, the
   `failure_mode` it guards against, the `input`, the `expected` resolution and whether
   it `expects_escalation`.
3. `uv run support-agent-evals-offline` and confirm the new case is the one that fails;
   fix the prompt or tools; run again. The policy for what qualifies is in
   `eval_assets/eval_policy.md`.

## Docs

- [Pydantic Evals quick start](https://pydantic.dev/docs/ai/evals/quick-start/)
- [Online evaluation](https://pydantic.dev/docs/ai/evals/online-evaluation/) and the
  [`OnlineEvaluation` capability](https://pydantic.dev/docs/ai/evals/online-evaluation/#agent-integration)
- [LLM judge](https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/) and
  [span-based evaluators](https://pydantic.dev/docs/ai/evals/evaluators/span-based/)
- [Logfire: experiments](https://pydantic.dev/docs/logfire/evaluate/datasets-and-experiments/),
  [live evals](https://pydantic.dev/docs/logfire/evaluate/live-evals/),
  [human review](https://pydantic.dev/docs/logfire/evaluate/human-review/),
  [datasets SDK](https://pydantic.dev/docs/logfire/evaluate/datasets-sdk/)

## Files

- `src/support_agent/agent.py` — the agent, its tools and `build_agent()` which attaches online evaluation.
- `src/support_agent/models.py` — `SupportRequest`, `SupportResolution`, `CaseMetadata`.
- `src/support_agent/main.py` — FastAPI service (`support-agent-serve`).
- `src/support_agent/evals/evaluators.py` — the deterministic evaluators and the judge builders.
- `src/support_agent/evals/dataset.py` — loads `eval_assets/dataset.jsonl` into a typed `Dataset`.
- `src/support_agent/evals/offline.py` — `support-agent-evals-offline`.
- `src/support_agent/evals/online.py` — `support-agent-traffic`.
- `src/support_agent/evals/publish.py` — `support-agent-evals-publish`.
- `eval_assets/` — `dataset.jsonl`, `judge_rubric.md`, `eval_policy.md`.
- `Dockerfile` — service image, built from the repo root.
- `tests/` — offline tests with `TestModel`; no model calls.
