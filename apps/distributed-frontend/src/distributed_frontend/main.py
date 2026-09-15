"""Backend for the distributed tracing demo.

Two things make browser -> backend -> agent traces join up:

1. `distributed_tracing=True` so the backend continues the `traceparent` the browser sends
   (and CORS must allow that header).
2. `/client-traces` proxies the browser's OTLP export to Logfire using the server-side
   write token, so the token never ships in the JavaScript bundle. Protect this route in
   production; here it is open for the demo.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent

from demo_core import configure_logfire, create_app, settings

configure_logfire('distributed-frontend-backend', distributed_tracing=True)

LOGFIRE_BASE_URL = os.environ.get('LOGFIRE_BASE_URL', 'https://logfire-us.pydantic.dev').rstrip('/')
FRONTEND_ORIGIN = os.environ.get('FRONTEND_ORIGIN', 'http://localhost:5173')

agent = Agent(
    settings().model,
    name='browser_qa_agent',
    instructions='Answer the question in two sentences at most.',
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncClient(timeout=30) as client:
        app.state.client = client
        yield


# The proxy endpoint itself is excluded from tracing so it doesn't create a span per export.
app = create_app('Distributed tracing demo', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=['*'],
    # `traceparent` must be allowed, or the browser strips it and the traces stay separate.
    allow_headers=['*'],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post('/ask')
async def ask(body: AskRequest) -> AskResponse:
    result = await agent.run(body.question)
    return AskResponse(answer=result.output)


@app.post('/client-traces')
async def client_traces(request: Request) -> JSONResponse:
    """Forward the browser's OTLP/JSON payload to Logfire with the server's write token."""
    token = settings().logfire_token
    if not token:
        return JSONResponse({'detail': 'LOGFIRE_TOKEN not set; browser spans dropped'}, status_code=202)
    with logfire.suppress_instrumentation():
        response = await request.app.state.client.post(
            f'{LOGFIRE_BASE_URL}/v1/traces',
            headers={'Authorization': token, 'Content-Type': 'application/json'},
            content=await request.body(),
        )
    # Pass Logfire's verdict straight back so a malformed export is visible in the browser console.
    return JSONResponse({'proxied_to': f'{LOGFIRE_BASE_URL}/v1/traces'}, status_code=response.status_code)


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', '8000')))


if __name__ == '__main__':
    serve()
