import httpx
from distributed_frontend import main
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel


def test_ask_and_cors_preflight():
    with TestClient(main.app) as client:
        with main.agent.override(model=TestModel(custom_output_text='Rayleigh scattering.')):
            r = client.post('/ask', json={'question': 'Why is the sky blue?'})
        assert r.json() == {'answer': 'Rayleigh scattering.'}

        preflight = client.options(
            '/ask',
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'traceparent,content-type',
            },
        )
        assert preflight.status_code == 200
        assert 'traceparent' in preflight.headers['access-control-allow-headers'].lower()


def test_client_traces_proxy_uses_server_token(monkeypatch):
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen['auth'] = request.headers.get('Authorization')
        seen['url'] = str(request.url)
        return httpx.Response(200, json={})

    monkeypatch.setattr(main, 'settings', lambda: type('S', (), {'logfire_token': 'pylf_v1_us_test'})())
    with TestClient(main.app) as client:
        main.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        r = client.post('/client-traces', json={'resourceSpans': []})
    assert r.status_code == 200
    assert seen['auth'] == 'pylf_v1_us_test'
    assert str(seen['url']).endswith('/v1/traces')
