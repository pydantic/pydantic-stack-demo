from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from support_agent.agent import build_agent
from support_agent.main import app, create_support_app


def test_default_app_has_health_and_index():
    client = TestClient(app)
    assert client.get('/health').json() == {'status': 'ok'}
    assert 'Support agent' in client.get('/').text


def test_resolve_endpoint_returns_structured_output():
    agent = build_agent(online_evaluation=False)
    test_app = create_support_app(agent)
    with agent.override(model=TestModel(custom_output_args={'resolution': 'Use the reset flow.', 'escalated': False})):
        r = TestClient(test_app).post('/support/resolve', json={'case_id': 'c1', 'request': 'forgot password'})
    assert r.status_code == 200
    assert r.json() == {'resolution': 'Use the reset flow.', 'escalated': False}
