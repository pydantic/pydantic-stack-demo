import httpx
from agent_basics import structured_output, weather
from agent_basics.hello import agent as hello_agent
from pydantic_ai.models.test import TestModel


def test_hello_agent():
    with hello_agent.override(model=TestModel(custom_output_text='From a 1974 C textbook.')):
        assert hello_agent.run_sync('hi').output == 'From a 1974 C textbook.'


def test_structured_output_validates():
    with structured_output.agent.override(model=TestModel()):
        out = structured_output.agent.run_sync('x').output
    assert isinstance(out, structured_output.Person)


async def test_weather_tools_called():
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith('/latlng'):
            return httpx.Response(200, json={'lat': 51.5, 'lng': -0.1})
        if path.endswith('/number'):
            return httpx.Response(200, text='21')
        return httpx.Response(200, text='Sunny')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with weather.weather_agent.override(model=TestModel()):
            result = await weather.weather_agent.run('weather?', deps=weather.Deps(client=client))
    tool_names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
    assert 'get_lat_lng' in tool_names and 'get_weather' in tool_names
