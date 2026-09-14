# prompt-optimization

**What it shows.** Automated prompt optimization with [GEPA](https://github.com/gepa-ai/gepa),
Pydantic AI and Pydantic Evals. A contact-extraction agent starts with a one-line prompt; a
dataset of eight cases and a field-accuracy evaluator score it; GEPA proposes better prompts
(using another Pydantic AI agent as the "prompt engineer"), keeps the ones that score higher,
and returns the best. The answer to "can we stop hand-tuning prompts?": yes, once you have an
eval dataset, which is the real prerequisite.

The mechanics worth copying:

- `Agent.override(instructions=...)` swaps candidate prompts in without rebuilding the agent.
- `Dataset.evaluate()` reports become GEPA's scores *and* its reflection feedback
  (`adapter.py`), so the optimizer sees the same numbers you see in Logfire.
- Every evaluation run is a Logfire experiment; `compare` prints two URLs you can diff.

Blog post with the full walkthrough:
[Automated Prompt Optimization with GEPA, Pydantic AI, and Pydantic Evals](https://pydantic.dev/articles/prompt-optimization-with-gepa).

## Run

```bash
uv run prompt-optimization eval             # score the initial prompt (8 cases)
uv run prompt-optimization eval --expert    # score the hand-written expert prompt
uv run prompt-optimization compare          # both, back to back
uv run prompt-optimization optimize --max-calls 20 --output optimized_prompt.txt
```

`--max-calls` bounds the number of evaluation calls GEPA may spend; 20 is a few minutes and
well under a dollar on the default models. Standard `.env` keys only: the task agent runs on
`DEMO_FAST_MODEL`, the proposer on `DEMO_MODEL`.

## What to look at in Logfire

- **Evals → Experiments**: one experiment per `eval` / `compare` run named
  `contact extraction: <label>`, with per-case `accuracy` scores and one assertion per field.
  Compare the initial and expert experiments to see which cases the weak prompt loses.
- During `optimize`, every GEPA candidate evaluation is its own experiment, and the
  `prompt_proposer run` spans show the feedback the optimizer sent and the prompt it got back.

## Docs

- [Pydantic Evals](https://pydantic.dev/docs/ai/evals/): datasets, evaluators, reports
- [Agent.override](https://pydantic.dev/docs/ai/testing/#overriding-model-via-pytest-fixtures)
- [GEPA](https://github.com/gepa-ai/gepa) and the [blog post](https://pydantic.dev/articles/prompt-optimization-with-gepa)

## Files

- `src/prompt_optimization/task.py` — `ContactInfo` schema, the agent, the initial and expert prompts
- `src/prompt_optimization/evals.py` — eight cases and `FieldAccuracyEvaluator`
- `src/prompt_optimization/adapter.py` — the GEPA adapter built on `Dataset.evaluate()`
- `src/prompt_optimization/cli.py` — `eval`, `compare`, `optimize`
- `tests/` — evaluator and adapter tests with `TestModel`
