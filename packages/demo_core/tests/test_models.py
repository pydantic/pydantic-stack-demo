import pytest

from demo_core import gateway_model


def test_gateway_model_rejects_unknown_format():
    with pytest.raises(ValueError):
        gateway_model('nope', 'x', api_key='pylf_v1_us_testkey')


def test_gateway_model_builds_with_route():
    m = gateway_model('anthropic', 'claude-haiku-4-5', route='my-endpoint', api_key='pylf_v1_us_testkey')
    assert m.model_name == 'claude-haiku-4-5'
