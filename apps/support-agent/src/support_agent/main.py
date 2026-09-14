"""The support agent as an HTTP service, with online evaluation on by default."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic_ai import Agent

from demo_core import configure_logfire, create_app, settings
from support_agent.agent import build_agent
from support_agent.models import SupportRequest, SupportResolution

_FORM = """<!doctype html><title>Support agent</title>
<h1>Support agent</h1>
<p>POST JSON <code>{"case_id": "...", "request": "..."}</code> to <code>/support/resolve</code>,
or try the docs at <a href="/docs">/docs</a>.</p>"""


def create_support_app(agent: Agent[None, SupportResolution] | None = None) -> FastAPI:
    app = create_app('Support agent')
    resolver = agent or build_agent(sample_rate=float(os.environ.get('JUDGE_SAMPLE_RATE', '0.1')))

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return _FORM

    @app.post('/support/resolve')
    async def resolve(payload: SupportRequest) -> SupportResolution:
        result = await resolver.run(f'Case ID: {payload.case_id}\nSupport request: {payload.request}')
        return result.output

    return app


# Scrubbing stays on for the service: it is the production-shaped entry point.
configure_logfire('support-agent')
app = create_support_app()


def main() -> None:
    import uvicorn

    settings().require_model_credentials()
    uvicorn.run(
        'support_agent.main:app', host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '8000'))
    )


if __name__ == '__main__':
    main()
