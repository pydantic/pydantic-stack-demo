import httpx
from fastapi.testclient import TestClient
from instrumentation import main
from pydantic_ai.models.test import TestModel


def test_health_and_orders():
    with TestClient(main.app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        r = client.get('/orders/A100')
        assert r.status_code == 200
        assert r.json()['status'] == 'shipped'
        assert client.get('/orders/nope').status_code == 404


def test_ask_uses_agent_and_tool():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='3')

    with TestClient(main.app) as client:
        # Swap the outbound client for one that never leaves the process.
        main.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with main.support_agent.override(model=TestModel()):
            r = client.post('/ask', json={'question': 'When does A100 arrive?'})
    assert r.status_code == 200
    assert 'lookup_delivery_estimate' in r.json()['answer']
