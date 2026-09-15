# Pydantic stack demos

Small, runnable reference code for the Pydantic stack: **Pydantic AI** (agents),
**Pydantic Evals** (offline and online evaluation), **Pydantic AI Gateway** (one key for
every model provider, with spend controls) and **Pydantic Logfire** (observability for all
of it). Each demo answers one question a team building with LLMs asks, and every one sends
traces to Logfire so you can see what happened, not just read what should have.

Use it as a workshop repo, as a source of copy-paste code, or as the answer to "how do I…".

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.12+. Docker is optional.

```bash
git clone https://github.com/pydantic/pydantic-stack-demo
cd pydantic-stack-demo
uv sync --all-packages          # one virtualenv with every demo installed
cp .env.example .env            # then add one key, see Authentication below
uv run hello-world              # one span in Logfire
uv run agent-basics-weather     # an agent with tools, traced end to end
```

### Authentication

The demos need two things: somewhere to send traces (a Logfire project) and a way to call
models (the Pydantic AI Gateway). One key can do both.

**Option 1: one project API key (simplest).** In Logfire open your project, then
**Settings → API Keys → New API Key**, tick **Send telemetry** and the Gateway access
capability, and put the key in `.env`:

```bash
LOGFIRE_API_KEY=pylf_v1_...
```

`demo_core` uses it as the write token *and* as the Gateway key. Gateway calls are billed
to the organization's Gateway balance (or your own provider credentials if you configured
BYOK providers), and the same key works for the Platform API demos that only need
project scopes.

**Option 2: least privilege.** A write token for traces (**Settings → Write tokens**) and
a Gateway key for models (**Gateway → API Keys**, where you can also set daily, weekly
and monthly spend caps on the key):

```bash
LOGFIRE_TOKEN=pylf_v1_...
PYDANTIC_AI_GATEWAY_API_KEY=pylf_v1_...
```

**Local development without a token.** `uv run logfire auth` (opens the browser), then
`uv run logfire projects use <project>` writes `.logfire/logfire_credentials.json`, which
the SDK picks up automatically. Docker containers do not see that file, so use one of the
options above for `docker compose`.

**Direct to a provider instead of the Gateway.** Set `DEMO_MODEL=openai:gpt-5.2` (or any
Pydantic AI model string) plus that provider's own key, for example `OPENAI_API_KEY`.

Without any Logfire credential the demos still run and print spans to the console; without
a model credential the agent demos stop with a one-line message saying what to set.

Docs: [API keys](https://pydantic.dev/docs/logfire/reference/advanced/use-api-keys/) ·
[Write tokens](https://pydantic.dev/docs/logfire/how-to-guides/create-write-tokens/) ·
[AI Gateway](https://pydantic.dev/docs/logfire/reference/advanced/gateway/)

Every command in this README runs from the repo root. `uv run <script>` works because
`uv sync --all-packages` installs each app's `[project.scripts]` into the shared virtualenv.

## The demos

Each app lives in `apps/<name>/` with its own README (what it shows, how to run it, what to
look at in Logfire, which docs it is based on).

### Start here

| App | What it answers | Run |
| --- | --- | --- |
| [hello-world](apps/hello-world) | What is the smallest possible Logfire program? | `uv run hello-world` |
| [agent-basics](apps/agent-basics) | How do I build an agent with tools, structured output and validation retries, and see it in Logfire? | `uv run agent-basics-hello`, `-weather`, `-structured`, `-retry` |

### Instrument

| App | What it answers | Run |
| --- | --- | --- |
| [instrumentation](apps/instrumentation) | How do I instrument a FastAPI service: request spans, outbound HTTP, manual spans, custom metrics, scrubbing, sampling, service metadata? | `uv run instrumentation-serve`, then `uv run instrumentation-traffic` |
| [distributed-frontend](apps/distributed-frontend) | How do browser traces join backend traces without exposing a write token to the browser? | see app README (React + FastAPI) |
| [memory](apps/memory) | How do I give an agent persistent memory (message history vs memory tools) in Postgres, with the queries traced? | `docker compose --profile memory up -d postgres`, then `uv run memory-messages` / `memory-tools` |

### Govern: routing, cost and access

| App | What it answers | Run |
| --- | --- | --- |
| [gateway-routing](apps/gateway-routing) | How do I use one key for many providers, fail over between them, cap spend per run / per tenant / per key, and keep PII out of traces? | `uv run gateway-routing-providers`, `-fallback`, `-budgets`, `-privacy` |
| [rbac](apps/rbac) | Who can see and send what in Logfire? Roles, tokens vs API keys, projects vs environments, and an audit script for an organization. | `uv run rbac-audit` (needs `LOGFIRE_API_KEY`), `uv run rbac-tokens` |

### Build agents that do more

| App | What it answers | Run |
| --- | --- | --- |
| [multi-agent](apps/multi-agent) | How do agents call other agents: agent-as-tool, plan → parallel fan-out → analysis, and the Harness `SubAgents` capability? Plus evals of the pattern across models. | `uv run multi-agent-twenty-questions`, `-research`, `-subagents`, `-evals` |
| [mcp-sampling](apps/mcp-sampling) | How does an MCP server make LLM calls through the client's model (MCP sampling), with both sides traced? | `uv run mcp-sampling-client` |
| [durable-exec](apps/durable-exec) | How do I make a multi-agent workflow survive crashes and restarts with DBOS or Temporal, and resume it by workflow id? | `uv run durable-exec-dbos-twenty-questions` (SQLite, no infra); `docker compose --profile temporal up -d`, then `uv run durable-exec-temporal-research` |

### Evaluate and improve

| App | What it answers | Run |
| --- | --- | --- |
| [support-agent](apps/support-agent) | What does a production-shaped agent with an eval loop look like: offline dataset + narrow LLM judge, online evaluation sampled on live traffic, results in Logfire? | `uv run support-agent-evals-offline`, `uv run support-agent-traffic --judge-all`, `uv run support-agent-serve` |
| [prompt-optimization](apps/prompt-optimization) | How do I optimise a prompt automatically with GEPA against a Pydantic Evals dataset? | `uv run prompt-optimization eval`, `compare`, `optimize --max-calls 20` |

## Running in Docker

Every service demo has a Dockerfile and a Compose profile; shared infrastructure (Postgres,
Temporal) is a profile too, and the `cli` service runs any script demo inside a container.

```bash
docker compose --profile instrumentation up --build      # a service, on http://localhost:8001
docker compose --profile memory up -d postgres           # infrastructure for a script demo
docker compose run --rm cli agent-basics-weather         # any script, in Docker
docker compose --profile all up --build                  # everything
```

Host ports default to `8001+` and `5433`/`7233` and can be changed in `.env` (see
`.env.example`). Containers read the same `.env` as the host runs.

## Repo layout

```
packages/demo_core/   settings (one .env), Logfire setup, Gateway model helper, FastAPI factory, eval helpers
apps/<name>/          one demo: pyproject.toml, src/<name>/, tests/, README.md, optional Dockerfile
docker-compose.yml    one profile per app plus postgres/temporal
scripts/smoke.py      runs every script demo once against .env
AGENTS.md             conventions; .claude/skills/add-demo has the checklist for a new demo
```

Verify a clone the way CI does:

```bash
make check                        # ruff, basedpyright, pytest (offline, no model calls)
uv run python scripts/smoke.py    # every script demo for real (a few cents of model usage)
```

## Workshops

The demos are grouped to back an *instrument → evaluate → operate* workshop arc:

| Workshop | Apps |
| --- | --- |
| Instrumentation | hello-world, agent-basics, instrumentation, distributed-frontend |
| Agents and multi-agent | agent-basics, multi-agent, mcp-sampling, memory, durable-exec |
| Evals | support-agent, prompt-optimization, multi-agent (evals) |
| Governance: Gateway, cost and access | gateway-routing, rbac |

## Docs

- Logfire: <https://pydantic.dev/docs/logfire/>
- Pydantic AI: <https://pydantic.dev/docs/ai/>
- Pydantic Evals: <https://pydantic.dev/docs/ai/evals/>
- Pydantic AI Gateway: <https://pydantic.dev/docs/logfire/reference/advanced/gateway/>
- Pydantic AI Harness: <https://pydantic.dev/docs/ai/harness/>

## Contributing

Read [AGENTS.md](AGENTS.md) first. Add a demo with the `add-demo` skill checklist, keep it
runnable by a stranger in five minutes, and open a PR. This repo is public: no customer
names, internal URLs or credentials.

Licensed under the [MIT license](LICENSE).
