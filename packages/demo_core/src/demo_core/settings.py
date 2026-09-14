"""Environment loading and typed settings shared by every demo.

One `.env` at the repo root configures every app (see `.env.example`). An app can add
`apps/<name>/.env` on top for anything specific to it. Real environment variables always
win over both files, which is what Docker Compose and CI rely on.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: this file) to the directory holding the uv workspace."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        pyproject = candidate / 'pyproject.toml'
        if pyproject.is_file() and '[tool.uv.workspace]' in pyproject.read_text():
            return candidate
    return None


def load_env(app_dir: Path | str | None = None) -> None:
    """Load `<repo>/.env`, then `<app_dir>/.env`, without overriding real environment variables."""
    root = find_repo_root()
    if root is not None:
        load_dotenv(root / '.env', override=False)
    if app_dir is not None:
        load_dotenv(Path(app_dir) / '.env', override=False)


class DemoSettings(BaseSettings):
    """Settings every demo can rely on. Field names map to the variables in `.env.example`."""

    model_config = SettingsConfigDict(extra='ignore', populate_by_name=True)

    logfire_token: str | None = Field(default=None, validation_alias='LOGFIRE_TOKEN')
    logfire_environment: str = Field(default='local', validation_alias='LOGFIRE_ENVIRONMENT')
    gateway_api_key: str | None = Field(default=None, validation_alias='PYDANTIC_AI_GATEWAY_API_KEY')
    model: str = Field(default='gateway/openai:gpt-5.2', validation_alias='DEMO_MODEL')
    fast_model: str = Field(default='gateway/anthropic:claude-haiku-4-5', validation_alias='DEMO_FAST_MODEL')
    database_url: str = Field(default='postgresql://postgres@localhost:5433/postgres', validation_alias='DATABASE_URL')
    temporal_address: str = Field(default='localhost:7233', validation_alias='TEMPORAL_ADDRESS')

    @property
    def uses_gateway(self) -> bool:
        return self.model.startswith('gateway/') or self.fast_model.startswith('gateway/')

    def require_model_credentials(self) -> None:
        """Fail early with a helpful message instead of a deep provider error."""
        if self.uses_gateway and not self.gateway_api_key:
            raise SystemExit(
                'PYDANTIC_AI_GATEWAY_API_KEY is not set. Create a key under Gateway -> API Keys in Logfire '
                'and put it in .env, or set DEMO_MODEL to a direct provider model (e.g. openai:gpt-5.2) '
                "plus that provider's own API key."
            )


@lru_cache
def settings(app_dir: Path | str | None = None) -> DemoSettings:
    """Load `.env` files and return the settings. Cached, so call it freely."""
    load_env(app_dir)
    return DemoSettings()


def is_ci_or_test() -> bool:
    return bool(os.environ.get('CI')) or bool(os.environ.get('PYTEST_CURRENT_TEST'))
