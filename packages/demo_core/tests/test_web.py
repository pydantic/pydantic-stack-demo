import logfire
from fastapi.testclient import TestClient

from demo_core import create_app

logfire.configure(send_to_logfire=False, console=False)


def test_health_and_error_handler():
    app = create_app('t')

    @app.get('/boom')
    async def boom():
        raise RuntimeError('x')

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/boom').status_code == 500
