# gateway-routing

## What it shows

One Pydantic AI Gateway key in front of every model provider, and what that buys a team
that asked for per-key spend control, LLM cost visibility per team, and a way to keep
personal data out of traces. Four scripts, each answering one question:

| Script | Question it answers |
| --- | --- |
| `gateway-routing-providers` | "Can we call OpenAI, Anthropic and Google with one key, and route through a failover endpoint?" |
| `gateway-routing-fallback` | "What happens when a model is down, and should we fail over in code or in the Gateway?" |
| `gateway-routing-budgets` | "How do we cap spend per run, per tenant per day, and per API key, and see cost per team?" |
| `gateway-routing-privacy` | "How do we stop emails and phone numbers reaching the model or Logfire?" |

## Run

From the repo root, with `PYDANTIC_AI_GATEWAY_API_KEY` and `LOGFIRE_TOKEN` in `.env`:

```bash
uv run gateway-routing-providers
uv run gateway-routing-fallback
uv run gateway-routing-budgets
uv run gateway-routing-privacy
```

Optional, for the endpoint leg of the providers demo: create an endpoint in Logfire under
**Gateway -> Endpoints**, add two providers with different priorities, then

```bash
GATEWAY_ROUTE=<endpoint-slug> GATEWAY_ROUTE_FORMAT=anthropic GATEWAY_ROUTE_MODEL=claude-haiku-4-5 \
  uv run gateway-routing-providers
```

Each script spends well under one US cent on the default models.

## What to look at in Logfire

**Providers.** The `gateway providers probe` span holds one `routing_probe run` per provider.
Open a `chat <model>` span: `gen_ai.provider.name`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens` / `output_tokens` and `operation.cost` are the fields the Gateway
priced the call with. The **LLMs** view in the sidebar gives the same per-model table without SQL.

**Fallback.** The `fallback_agent run` span shows a model request named
`chat fallback:no-such-model-for-demo,claude-haiku-4-5`; the response's `gen_ai.response.model`
is the model that actually answered. The all-broken run records the `FallbackExceptionGroup`
on the span.

**Budgets.** Every model request span carries `tenant`, `user` and `feature` (from
`logfire.set_baggage`) next to `operation.cost`, and the `invoke_agent budgeted_agent` span
carries the same values in its `metadata` attribute. Cost per tenant per model per day, in
**Explore**:

```sql
SELECT
    date_trunc('day', start_timestamp) AS day,
    attributes->>'tenant' AS tenant,
    attributes->>'gen_ai.request.model' AS model,
    SUM((attributes->'operation.cost')::numeric) AS cost_usd,
    SUM((attributes->'gen_ai.usage.input_tokens')::numeric) AS input_tokens,
    SUM((attributes->'gen_ai.usage.output_tokens')::numeric) AS output_tokens
FROM records
WHERE attributes ? 'operation.cost' AND attributes ? 'tenant'
GROUP BY day, tenant, model
ORDER BY day DESC, cost_usd DESC
```

Swap `tenant` for `user` or `feature` to slice the other way. The refused call shows up as a
`spend budget exhausted` span under `budgeted_agent run` with no model request beneath it.
`gen_ai.aggregated_usage.*` on the agent-run span is the run total; the per-request numbers
live on the `chat` spans, which is why the query filters on `operation.cost`.

**Privacy.** The `privacy_agent run` span contains a `guardrail redacted input` span and a
`chat` span with tokens and cost but no `gen_ai.input.messages` / `gen_ai.output.messages`
attributes, because the SDK was configured with `include_content=False`.

### Which control runs where

| Control | Enforced | Scope | Configure |
| --- | --- | --- | --- |
| `UsageLimits(cost_limit=, request_limit=)` | client, per run | one agent run | code (`budgets.py`) |
| `SpendLimits(budgets=[Budget(usd=, window=, scope=)])` | client, across runs | per tenant/user/whatever `scope` returns; store decides persistence | code (`budgets.py`); Redis store for multi-process |
| Gateway per-key limits: daily, weekly, monthly, total | server, every client using the key | one API key | Logfire: Gateway -> API Keys |
| Gateway per-member limits: daily, weekly, monthly | server | one organization member | Logfire: Gateway -> Spending |
| Gateway endpoint priority / weight | server | one route | Logfire: Gateway -> Endpoints |

Client-side limits stop a runaway loop before the bill arrives and give you per-tenant
semantics the Gateway does not know about. Gateway limits are the backstop that holds even
when an app is misconfigured, written in another language, or bypasses Pydantic AI.

`cost_limit` and `SpendLimits` price calls with the open-source `genai-prices` dataset; a model
without price data raises (`on_unpriced='raise'`) rather than counting as free. Pair a cost
limit with `request_limit`: the request that crosses the line still completes.

## Docs

- Gateway overview, providers, API keys, endpoints, spending limits: https://pydantic.dev/docs/logfire/reference/advanced/gateway/
- Pydantic AI Gateway from code (`gateway/` model strings, `gateway_provider(route=...)`): https://pydantic.dev/docs/ai/gateway/
- `FallbackModel`: https://pydantic.dev/docs/ai/models/overview/#fallback-model
- Usage limits and `cost_limit`: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
- Harness `SpendLimits`: https://pydantic.dev/docs/ai/harness/spend/
- Harness guardrails and detectors: https://pydantic.dev/docs/ai/harness/guardrails/
- Track and alert on LLM cost (the SQL above is adapted from it): https://pydantic.dev/docs/logfire/cookbook/track-llm-cost/
- Baggage as span attributes: https://pydantic.dev/docs/logfire/reference/advanced/baggage/
- Scrubbing, and why LLM messages are not scrubbed: https://pydantic.dev/docs/logfire/how-to-guides/scrubbing/
- Excluding prompts and completions (`include_content=False`): https://pydantic.dev/docs/ai/logfire/#excluding-prompts-and-completions

## Files

- `src/gateway_routing/providers.py` — same prompt through three providers, optional gateway endpoint.
- `src/gateway_routing/fallback.py` — `FallbackModel` with a broken primary; all-broken `FallbackExceptionGroup`.
- `src/gateway_routing/budgets.py` — `UsageLimits`, per-tenant `SpendLimits`, baggage and metadata for attribution.
- `src/gateway_routing/privacy.py` — input/output PII guardrails plus `include_content=False`.
- `tests/test_gateway_routing.py` — offline tests with `TestModel` / `FunctionModel` (fallback, limits, budgets, span attributes, redaction).
