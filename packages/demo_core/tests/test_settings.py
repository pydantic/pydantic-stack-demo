from pathlib import Path

from demo_core import find_repo_root
from demo_core.settings import DemoSettings


def test_find_repo_root_points_at_workspace():
    root = find_repo_root()
    assert root is not None
    assert (root / 'pyproject.toml').is_file()
    assert (root / 'apps').is_dir()


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv('DEMO_MODEL', 'test')
    monkeypatch.setenv('DEMO_FAST_MODEL', 'test')
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'x')
    s = DemoSettings()
    assert s.model == 'test'
    assert not s.uses_gateway


def test_repo_root_from_app_dir():
    assert find_repo_root(Path(__file__)) == find_repo_root()


def test_one_api_key_covers_token_and_gateway(monkeypatch):
    import demo_core.settings as settings_module

    monkeypatch.setattr(settings_module, 'find_repo_root', lambda start=None: None)  # ignore a developer .env
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
    monkeypatch.delenv('PYDANTIC_AI_GATEWAY_API_KEY', raising=False)
    monkeypatch.setenv('LOGFIRE_API_KEY', 'pylf_v1_us_key')
    settings_module.load_env()
    assert DemoSettings().gateway_api_key == 'pylf_v1_us_key'
    assert DemoSettings().logfire_token == 'pylf_v1_us_key'
