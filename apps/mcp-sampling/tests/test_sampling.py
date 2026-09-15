from pydantic_ai.models.test import TestModel

from mcp_sampling import client, generate_svg


def test_slugify():
    assert generate_svg.slugify('A Robot!') == 'a-robot'


async def test_client_calls_server_tool_and_server_samples_from_client(tmp_path):
    """End to end over stdio, with TestModel on the client side answering the sampling requests."""
    agent = client.build_agent(output_dir=tmp_path)
    # TestModel calls every tool once, then answers; the server's sampling request is
    # answered by the same TestModel through `set_mcp_sampling_model()`.
    with agent.override(model=TestModel()):
        agent.set_mcp_sampling_model(TestModel(custom_output_text='<svg></svg>'))
        async with agent:
            result = await agent.run('draw a cat')
    names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
    assert names == ['image_generator']
    assert [p.name for p in tmp_path.glob('*.svg')] == ['a_a.svg']
