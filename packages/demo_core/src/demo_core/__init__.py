"""Shared helpers for the pydantic-stack-demo apps.

Everything here is deliberately small. A demo should read like the code a customer
would write themselves; `demo_core` only removes the boilerplate that is identical in
every app (finding `.env`, picking a model, configuring Logfire the same way).
"""

from demo_core.logfire_setup import configure_logfire
from demo_core.models import gateway_model
from demo_core.settings import DemoSettings, find_repo_root, load_env, settings
from demo_core.web import create_app

__all__ = [
    'DemoSettings',
    'configure_logfire',
    'create_app',
    'find_repo_root',
    'gateway_model',
    'load_env',
    'settings',
]
