import pytest
from support_agent.evals import publish


class FakeClient:
    def __init__(self, api_key=None, base_url=None):
        self.pushed = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def push_dataset(self, dataset, *, description=None):
        self.pushed = dataset
        return {'name': dataset.name, 'id': 'ds-1', 'cases': len(dataset.cases)}


def test_publish_pushes_cases_without_code_evaluators(monkeypatch):
    monkeypatch.setattr(publish, 'LogfireAPIClient', FakeClient)
    detail = publish.publish('key')
    assert detail['name'] == 'support_resolution_failures'
    assert detail['cases'] == 7


def test_main_requires_api_key(monkeypatch):
    monkeypatch.delenv('LOGFIRE_API_KEY', raising=False)
    with pytest.raises(SystemExit):
        publish.main()
