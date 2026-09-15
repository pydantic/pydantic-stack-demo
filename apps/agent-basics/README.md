# agent-basics

**What it shows.** The four things every team asks on day one with Pydantic AI, each as a
script you can read in one screen: a one-line agent; an agent with tools that share
dependencies (an HTTP client); structured output validated by Pydantic; and validation-driven
retries, where a validator rejects the model's first answer and Pydantic AI asks again. All
four are instrumented, so the question "what did the model actually do?" is answered by
opening the trace.

## Run

```bash
uv run agent-basics-hello        # model + instructions + one question
uv run agent-basics-weather      # tools, dependencies, outbound HTTP traced
uv run agent-basics-structured   # output_type=Person
uv run agent-basics-retry        # a validator forces a second model request
```

Models come from `.env`: `DEMO_MODEL` (default `gateway/openai:gpt-5.2`) and
`DEMO_FAST_MODEL` (default `gateway/anthropic:claude-haiku-4-5`).

## What to look at in Logfire

- `hello_agent run` → one `chat` span with `gen_ai.usage.*` tokens and `operation.cost`.
- `weather_agent run` → tool spans `get_lat_lng` / `get_weather` with their arguments and
  return values, and under each the `GET demo-endpoints.pydantic.workers.dev/...` HTTP spans
  captured by `logfire.instrument_httpx(capture_all=True)`.
- `extract_person run` → the structured output on the run span (`final_result`).
- `extract_person_strict run` → **two** `chat` spans. Open the second request: the
  `retry-prompt` part contains the validation error the model was shown.

## Docs

- [Agents](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Function tools and dependencies](https://pydantic.dev/docs/ai/tools-toolsets/tools/)
- [Structured output](https://pydantic.dev/docs/ai/core-concepts/output/)
- [Retries and validation](https://pydantic.dev/docs/ai/core-concepts/output/#output-validator-functions)
- [Logfire with Pydantic AI](https://pydantic.dev/docs/logfire/ai/pydantic-ai/)

## Files

- `src/agent_basics/hello.py` — one-line agent.
- `src/agent_basics/weather.py` — tools, `deps_type`, instrumented httpx client.
- `src/agent_basics/structured_output.py` — `output_type=Person`.
- `src/agent_basics/validation_retry.py` — `field_validator` that triggers a retry.
- `tests/test_agents.py` — offline tests with `TestModel` and a mocked HTTP transport.
