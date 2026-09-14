# Distributed frontend tracing

## What it shows

One trace from a button click in the browser, through a FastAPI backend, into a Pydantic AI
agent's model call. The React app is instrumented with `@pydantic/logfire-browser`; the
backend runs with `distributed_tracing=True` and proxies the browser's spans to Logfire so
the write token never leaves the server. It answers: *"can we see our frontend and backend
in the same trace?"* and *"how do we send browser telemetry without exposing a token?"*

```
browser (React, logfire-browser) ──fetch /ask + traceparent──▶ FastAPI ──▶ agent ──▶ model
        └──── OTLP export ──▶ /client-traces (proxy, adds LOGFIRE_TOKEN) ──▶ Logfire
```

## Run

Requires Node.js 22+ for the frontend.

```bash
uv run distributed-frontend-backend                 # http://127.0.0.1:8000
cd apps/distributed-frontend/frontend && npm install && npm run dev   # http://localhost:5173
```

Open http://localhost:5173, ask a question, then open Logfire. To point the frontend at a
different backend, create `frontend/.env.local` with `VITE_BACKEND_BASE_URL=...`.

The backend alone can run in Docker: `docker compose --profile distributed-frontend up
--build` (host port `DISTRIBUTED_BACKEND_PORT`, default 8003; set `VITE_BACKEND_BASE_URL`
to match). `LOGFIRE_BASE_URL` selects the region the proxy forwards to (default
`https://logfire-us.pydantic.dev`; use `https://logfire-eu.pydantic.dev` for EU projects).

## What to look at in Logfire

- Filter `service_name = 'frontend'`: the browser's `HTTP POST` fetch span is the root.
- Expand it: the backend's `POST /ask` span (service `distributed-frontend-backend`) is a
  child, because the browser sent a `traceparent` header and the backend was configured
  with `distributed_tracing=True`. Under it: `browser_qa_agent run` and the model request.
- The `/client-traces` route produces no spans of its own: it is the export path, wrapped
  in `logfire.suppress_instrumentation()`.
- In production, protect `/client-traces` (auth, rate limits) and lock `allow_origins` to
  your real frontend origin.

## Docs

- [Distributed tracing](https://pydantic.dev/docs/logfire/how-to-guides/distributed-tracing/)
- [Browser instrumentation (`@pydantic/logfire-browser`)](https://pydantic.dev/docs/logfire/languages/javascript/browser/)
- [FastAPI integration](https://pydantic.dev/docs/logfire/integrations/web-frameworks/fastapi/)

## Files

- `src/distributed_frontend/main.py` — FastAPI backend: `/ask` (agent), `/client-traces` (proxy), CORS.
- `frontend/src/ClientInstrumentationProvider.tsx` — browser SDK setup and fetch instrumentation.
- `frontend/src/App.tsx` — the question form.
- `frontend/src/config.ts` — backend URL (`VITE_BACKEND_BASE_URL`).
- `Dockerfile` — backend image for `docker compose --profile distributed-frontend`.
- `tests/test_backend.py` — offline tests: agent endpoint, CORS preflight with `traceparent`, proxy auth header.
