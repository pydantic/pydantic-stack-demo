"""Credentials are scoped: a write token sends data and does nothing else.

Step 1 sends one span to the project with `LOGFIRE_TOKEN` (a write token or a project API key
with "Send telemetry"). Step 2 presents the very same token to the Platform API to list
projects. The API answers 403 and names the missing scope (`project:read`): the credential
that can write telemetry cannot read settings, list members, or mint other tokens. That is
the property you want in every deployed service. Management operations need an API key with
exactly the scopes they use; see `audit.py` and the README.
"""

from __future__ import annotations

import os
import sys

import httpx
import logfire

from demo_core import configure_logfire
from rbac.audit import DEFAULT_BASE_URL


def send_one_span(token: str) -> None:
    logfire.configure(token=token, send_to_logfire=True, console=False)
    with logfire.span('rbac token separation demo'):
        logfire.info('sent with a write token; this is all a write token can do')
    logfire.force_flush()
    print('1) write token: sent one span to the project  -> OK')


def try_platform_read(token: str, base_url: str) -> None:
    response = httpx.get(
        f'{base_url.rstrip("/")}/api/v1/projects/', headers={'Authorization': f'Bearer {token}'}, timeout=30
    )
    if response.status_code == 200:
        print(
            f'2) same token on GET /v1/projects/       -> 200 ({len(response.json())} projects): this token has project:read'
        )
        print(
            '   That means LOGFIRE_TOKEN is an API key with more than "Send telemetry". Prefer a narrower key in services.'
        )
        return
    detail = ''
    try:
        detail = response.json().get('detail', '')
    except ValueError:
        detail = response.text[:120]
    print(f'2) same token on GET /v1/projects/       -> {response.status_code}: {detail}')
    print(
        '   Expected. Write tokens carry only project:write_otlp; management calls need an API key with project:read.'
    )


def main() -> int:
    configure_logfire('rbac', instrument_pydantic_ai=False, console=False)
    token = os.environ.get('LOGFIRE_TOKEN')
    if not token:
        print('LOGFIRE_TOKEN is not set; add a write token for your project to .env', file=sys.stderr)
        return 2
    base_url = os.environ.get('LOGFIRE_API_BASE_URL', DEFAULT_BASE_URL)
    send_one_span(token)
    try_platform_read(token, base_url)
    return 0


if __name__ == '__main__':
    sys.exit(main())
