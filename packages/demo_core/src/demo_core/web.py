"""FastAPI factory with the instrumentation every demo service wants."""

from __future__ import annotations

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def create_app(title: str, *, capture_headers: bool = False, **fastapi_kwargs: object) -> FastAPI:
    """Build a FastAPI app with a `/health` route, Logfire request spans and an error handler.

    Call `configure_logfire()` first so `instrument_fastapi()` has a configured SDK.
    """
    app = FastAPI(title=title, **fastapi_kwargs)  # type: ignore[arg-type]
    logfire.instrument_fastapi(app, capture_headers=capture_headers)

    @app.get('/health', include_in_schema=False)
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logfire.exception('Unhandled error on {method} {path}', method=request.method, path=request.url.path)
        return JSONResponse(status_code=500, content={'error': 'internal_server_error'})

    return app
