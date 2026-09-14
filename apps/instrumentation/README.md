# Instrumentation

## What it shows

How to instrument a Python web service with Logfire, from the two-line setup to the
details customers ask about in the first week: service metadata and environments, manual
spans and structured attributes, custom metrics, scrubbing sensitive fields, sampling, and
what an agent call looks like nested inside an HTTP request. It answers: *"we've added
`logfire.configure()`, what next?"*

The service is a tiny order-support API:

- `GET /orders/{id}` — auto-instrumented request span, then a manual `order lookup` span,
  an `@logfire.instrument`-wrapped function, a structured log line, and two custom metrics.
- `POST /ask` — a named Pydantic AI agent whose tool makes an outbound HTTP call. The
  trace is request → agent run → model request → tool → HTTP call.

## Run

```bash
uv run instrumentation-serve                      # http://127.0.0.1:8000
uv run instrumentation-traffic --count 10         # in another shell: hits /orders and /ask
uv run instrumentation-traffic --no-ask           # no model calls, just spans and metrics
```

Or in Docker from the repo root: `docker compose --profile instrumentation up --build`
(host port `INSTRUMENTATION_PORT`, default 8001; then `uv run instrumentation-traffic
--base-url http://127.0.0.1:8001`).

Optional: `LOGFIRE_SAMPLE_RATE=0.25 uv run instrumentation-serve` keeps a quarter of
traces (head sampling).

## What to look at in Logfire

- **Live view**: filter `service_name = 'instrumentation'`. Open a `GET /orders/{order_id}`
  trace to see the nested `order lookup A100` → `load order A100` → `enrich order` spans.
- **Scrubbing**: the `order A100 status=shipped` log has `customer_ref` and `password`
  replaced with `[Scrubbed due to 'customer_ref']` / `[Scrubbed due to 'password']`.
  `customer_ref` is scrubbed because of the `extra_patterns` passed to `ScrubbingOptions`;
  `password` matches the built-in patterns.
- **Metrics explorer**: `orders.lookups` (counter, by `outcome`) and `orders.value`
  (histogram). System metrics (`system.cpu.*`, `process.*`) come from
  `instrument_system_metrics()`.
- **Environments picker**: traces are tagged with `LOGFIRE_ENVIRONMENT` from `.env`.
- **`POST /ask`**: the `order_support_agent run` span contains the model call and the
  `lookup_delivery_estimate` tool span, with the httpx request (headers captured) inside it.

Lookups per order, from the manual span's attribute:

```sql
SELECT attributes->>'order_id' AS order_id, count(*) AS lookups
FROM records
WHERE span_name = 'order lookup {order_id}'
GROUP BY 1 ORDER BY 2 DESC
```

## Docs

- [Python instrumentation](https://pydantic.dev/docs/logfire/instrumentation/python/)
- [Manual tracing](https://pydantic.dev/docs/logfire/instrumentation/python/manual-tracing/)
- [FastAPI](https://pydantic.dev/docs/logfire/integrations/web-frameworks/fastapi/) and
  [HTTPX](https://pydantic.dev/docs/logfire/integrations/http-clients/httpx/) integrations
- [Metrics](https://pydantic.dev/docs/logfire/instrumentation/python/metrics/)
- [Scrubbing sensitive data](https://pydantic.dev/docs/logfire/how-to-guides/scrubbing/)
- [Sampling](https://pydantic.dev/docs/logfire/how-to-guides/sampling/)
- [Environments](https://pydantic.dev/docs/logfire/how-to-guides/environments/)
- [Pydantic AI instrumentation](https://pydantic.dev/docs/logfire/integrations/llms/pydanticai/)

## Files

- `src/instrumentation/main.py` — the service: configuration, agent, endpoints, metrics.
- `src/instrumentation/traffic.py` — traffic generator for live demos.
- `Dockerfile` — image for `docker compose --profile instrumentation`.
- `tests/test_service.py` — offline tests with `TestClient` and `TestModel`.
