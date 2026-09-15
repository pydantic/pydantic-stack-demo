# AGENTS.md

Instructions for coding agents (and humans) working in this repo. `README.md` is the
customer-facing catalog; this file is the contract behind it.

## What this repo is

Public, runnable reference code for the Pydantic stack: Pydantic AI, Pydantic Evals,
Pydantic AI Gateway and Logfire. Customers copy from it, workshops run it, and Solutions
Engineers use it to answer "how do I do X" with a link instead of a paragraph. Every demo
must therefore be **small, honest and runnable by a stranger in under five minutes** once
`.env` is filled in.

## Layout

- `packages/demo_core/` — the only shared code. `settings` (one `.env` for everything,
  typed `DemoSettings`), `configure_logfire()` (the standard two-line Logfire setup plus
  our flags), `gateway_model()` (explicit Gateway provider for routing/fallback demos),
  `create_app()` (FastAPI with `/health` and instrumentation), and two eval helpers.
  Nothing else goes in here until two apps need the same thing.
- `apps/<name>/` — one demo per directory, each a uv workspace member:
  - `pyproject.toml` — `name = "<name>"` (kebab-case), depends on `demo-core` via
    `[tool.uv.sources] demo-core = { workspace = true }`, declares everything it imports,
    and exposes each runnable entry point under `[project.scripts]` as `<name>-<script>`
    (e.g. `agent-basics-weather`). After `uv sync --all-packages` every script is on the
    path: `uv run agent-basics-weather`.
  - `src/<name_with_underscores>/` — the code. Each entry point module has a sync `main()`
    and an `if __name__ == '__main__': main()` guard.
  - `README.md` — see *App README contract* below.
  - `tests/` — offline tests using `TestModel`/`FunctionModel` via `agent.override(...)`;
    `conftest.py` sets `LOGFIRE_SEND_TO_LOGFIRE=false` and `DEMO_MODEL=test`. **No
    `tests/__init__.py`** (importlib mode keeps same-named test modules apart; a package
    would make them collide silently).
  - `Dockerfile` — only for long-running services (FastAPI apps, workers). Build context
    is the **repo root** because of the `demo-core` path dependency. Copy the pattern from
    `apps/instrumentation/Dockerfile`.
- `docker-compose.yml` — one Compose profile per app that needs infrastructure or is a
  service; `postgres` and `temporal` services are shared by profile. Host ports are
  `${NAME_PORT:-default}` substitutions with defaults listed in `.env.example`.
- `.env.example` — every variable any demo reads, with a comment. Real `.env` is gitignored.
- `scripts/smoke.py` — runs every script demo once against real credentials.

## Conventions every app follows

1. **Configure Logfire first.** The entry point calls
   `configure_logfire('<app-name>')` before constructing agents or apps, then
   `settings().require_model_credentials()` so a missing key fails with a sentence, not
   a stack trace. Short scripts end with `logfire.force_flush()` (or rely on
   `configure_logfire` + normal exit).
2. **Models come from settings.** `settings().model` (default `gateway/openai:gpt-5.2`)
   and `settings().fast_model` (default `gateway/anthropic:claude-haiku-4-5`). Never
   hard-code a provider key; never hard-code a model string unless the demo is *about*
   that model (say so in a comment).
3. **Name your agents** (`Agent(..., name='weather_agent')`). It is what Logfire shows.
4. **Pydantic AI v2 idioms only.** `capabilities=[...]` (not `builtin_tools=`,
   `instrument=`), `MCPToolset`, `openai:` means Responses API, `result.usage` is a
   property, `Dataset(name=...)` is required, durable execution via
   `DBOSDurability()` / `TemporalDurability()` capabilities. See
   `pydantic-ai/docs/migration.md` when unsure.
5. **Keep the code copyable.** Prefer explicit Pydantic AI / Logfire calls over helper
   indirection; comments explain *why*, not *what*. No customer names, account data or
   internal URLs anywhere in this repo — it is public.
6. **Attributes people will query.** When a demo is about attribution (tenant, user,
   feature, environment), put those on spans and agent `metadata=` so the README can show
   the SQL that groups by them.
7. **Every number in a README has a source** (a docs link or the code that prints it).

## App README contract

Each `apps/<name>/README.md` has, in this order:

1. A one-paragraph **What it shows** naming the customer question it answers.
2. **Run** — the exact commands (`uv run <script>`, and `docker compose --profile <name>
   up` if applicable), plus any extra `.env` keys beyond the standard ones.
3. **What to look at in Logfire** — the spans/attributes/views to open, and a SQL query
   where that is the point.
4. **Docs** — links to the pydantic.dev docs pages the demo relies on.
5. **Files** — one line per file.

## Verification before a PR

```bash
make install      # uv sync --all-packages
make check        # ruff, basedpyright, pytest (offline)
uv run <script>   # each new entry point, against a real .env, traces visible in Logfire
docker compose --profile <name> config   # if you touched compose
```

State in the PR which Logfire project the traces went to.

## Adding a demo

Use the `add-demo` skill (`.claude/skills/add-demo/SKILL.md`): it has the file-by-file
checklist and the README template.
