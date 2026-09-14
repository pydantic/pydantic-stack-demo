"""Postgres connection helper shared by both memory approaches."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import asyncpg

from demo_core import settings

# asyncpg's Connection is generic only under type checking.
if TYPE_CHECKING:
    DbConn = asyncpg.Connection[asyncpg.Record]
else:
    DbConn = asyncpg.Connection


@asynccontextmanager
async def connect(schema_sql: str) -> AsyncIterator[DbConn]:
    """Connect to `DATABASE_URL`, make sure the demo table exists, yield the connection."""
    try:
        conn = await asyncpg.connect(settings().database_url)
    except OSError as exc:
        raise SystemExit(
            f'Could not connect to Postgres at {settings().database_url}: {exc}. '
            'Start one with `docker compose --profile memory up postgres` (or point DATABASE_URL elsewhere).'
        ) from exc
    try:
        await conn.execute(schema_sql)
        yield conn
    finally:
        await conn.close()
