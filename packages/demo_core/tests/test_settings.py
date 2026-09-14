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
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'x')
    s = DemoSettings()
    assert s.model == 'test'
    assert not s.uses_gateway


def test_repo_root_from_app_dir():
    assert find_repo_root(Path(__file__)) == find_repo_root()
