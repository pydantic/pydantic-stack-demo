"""The smallest agent: a model, instructions, one question."""

from pydantic_ai import Agent

from demo_core import configure_logfire, settings

agent = Agent(
    settings().model,
    name='hello_agent',
    instructions='Be concise, reply with one sentence.',
)


def main() -> None:
    configure_logfire('agent-basics')
    settings().require_model_credentials()

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.output)
    print(f'usage: {result.usage}')


if __name__ == '__main__':
    main()
