"""Configure Logfire the same way in every demo.

This is the two-line setup from the Logfire docs, plus the flags that make demos pleasant:
send only when a token is present, tag traces with an environment, and record system
metrics so the Logfire host panels have something to show.
"""

from __future__ import annotations

import os
from pathlib import Path

import logfire

from demo_core.settings import load_env, settings


def configure_logfire(
    service_name: str,
    *,
    app_dir: Path | str | None = None,
    environment: str | None = None,
    instrument_pydantic_ai: bool = True,
    system_metrics: bool = False,
    include_content: bool | None = None,
    console: bool = True,
    **configure_kwargs: object,
) -> logfire.Logfire:
    """Load `.env`, call `logfire.configure()` and turn on the standard instrumentation.

    Call it once, at the top of the entry point, before building any Agent or FastAPI app,
    so `logfire.instrument_*()` registers against a configured SDK.

    `include_content=False` keeps prompts and completions out of Logfire while still
    recording timings, tokens and cost: the recommended setting when LLM traffic may carry
    personal data (see apps/gateway-routing).
    """
    load_env(app_dir)
    s = settings()
    instance = logfire.configure(
        service_name=service_name,
        environment=environment or s.logfire_environment,
        # An explicit LOGFIRE_SEND_TO_LOGFIRE (tests set it to false) wins over our default.
        send_to_logfire=None if 'LOGFIRE_SEND_TO_LOGFIRE' in os.environ else 'if-token-present',
        console=False if not console else None,
        **configure_kwargs,  # type: ignore[arg-type]
    )
    if instrument_pydantic_ai:
        logfire.instrument_pydantic_ai(include_content=include_content)
    if system_metrics:
        logfire.instrument_system_metrics()
    return instance


def flush() -> None:
    """Flush pending spans. Short-lived scripts should call this before exiting."""
    logfire.force_flush()
