---
name: add-demo
description: Use when adding a new demo to pydantic-stack-demo — scaffolding apps/<name> as a uv workspace member that shares demo_core, wiring its scripts, tests, README and (if it is a service) its Dockerfile and docker-compose profile.
---

# Add a demo

## Checklist

1. **Pick a name** that says what the demo answers (`gateway-routing`, not `demo7`).
   Directory `apps/<name>/`, package `src/<name_with_underscores>/`.
2. **`pyproject.toml`** — copy `apps/agent-basics/pyproject.toml`. Keep `demo-core` as a
   workspace source, list every direct import, add one `[project.scripts]` entry per
   runnable module named `<name>-<script>`.
3. **Code** — one module per entry point with a sync `main()`. First lines of `main()`:
   ```python
   configure_logfire('<name>')
   settings().require_model_credentials()
   ```
   Use `settings().model` / `settings().fast_model`; name every `Agent`; v2 idioms only
   (`capabilities=[...]`, `MCPToolset`, `DBOSDurability()`).
4. **Tests** — `tests/conftest.py` sets `LOGFIRE_SEND_TO_LOGFIRE=false`,
   `PYDANTIC_AI_NO_BANNER=1`, `DEMO_MODEL=test`, `DEMO_FAST_MODEL=test`. Test agent
   wiring with `agent.override(model=TestModel())`; test FastAPI apps with `TestClient`.
   No `tests/__init__.py`.
5. **README.md** — follow the *App README contract* in `AGENTS.md` (What it shows / Run /
   What to look at in Logfire / Docs / Files).
6. **Service?** Add `Dockerfile` (copy `apps/instrumentation/Dockerfile`, build context is
   the repo root) and a `docker-compose.yml` service with `profiles: ["<name>", "all"]`,
   `env_file: [{path: .env, required: false}, {path: apps/<name>/.env, required: false}]`
   and a `${<NAME>_PORT:-<default>}` host port. Add the port to `.env.example`.
7. **Catalog** — add a row to the table in the root `README.md` and, if it is a script
   demo, to `scripts/smoke.py`.
8. **Verify** — `uv sync --all-packages`, `make check`, run every script for real, and
   `docker compose --profile <name> config` if you touched compose.

## Common mistakes

- Hard-coding a model or provider key instead of reading `settings()`.
- Forgetting `name=` on an agent (Logfire then labels it `agent`).
- Using v1 APIs (`builtin_tools=`, `instrument=True`, `MCPServerStdio`, `DBOSAgent`).
- A README command that only works from inside the app directory. Every command in this
  repo is run from the repo root.
- Putting anything customer-specific in this public repo.
