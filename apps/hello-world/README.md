# hello-world

**What it shows.** The smallest Logfire program: configure once, then emit a span and a log
line. Every other demo in this repo starts from these two calls. Use it to check that a fresh
environment can reach your Logfire project before running anything larger.

## Run

```bash
uv run hello-world
```

With `LOGFIRE_TOKEN` set in `.env` the console prints the project URL and the span appears
in Logfire within a second. Without a token the same lines print to the console and nothing
is sent (`send_to_logfire='if-token-present'`).

## What to look at in Logfire

- **Live view**: one trace named `greeting world` with a child log `hello world`.
- Click the span: `place` is a queryable attribute, and `service.name` is `hello-world`.
- The environment picker shows the value of `LOGFIRE_ENVIRONMENT` (default `local`).

## Docs

- [Logfire: first trace](https://pydantic.dev/docs/logfire/)
- [Manual tracing: spans and logs](https://pydantic.dev/docs/logfire/guides/onboarding-checklist/add-manual-tracing/)
- [Write tokens](https://pydantic.dev/docs/logfire/how-to-guides/create-write-tokens/)

## Files

- `src/hello_world/main.py` — the whole demo.
- `tests/test_hello.py` — runs it offline.
