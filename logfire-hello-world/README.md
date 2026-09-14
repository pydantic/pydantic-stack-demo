# Logfire Hello World

Minimal demo showing how to configure Logfire and emit a single log message. `main.py` calls
`logfire.configure()` and logs `hello world`.

## Running

From the repo root:

```bash
uv run python logfire-hello-world/main.py
```

## Environment variables

- `LOGFIRE_TOKEN` - write token for your Logfire project. If unset, `logfire.configure()` will
  prompt you to authenticate interactively (or run `uv run logfire auth` first).

## Where to look in Logfire

Open your project's live view - you should see a single `hello world` log entry from the
`hello-world` service.
