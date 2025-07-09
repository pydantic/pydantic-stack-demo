from pydantic_ai import Agent

# import logfire
# logfire.configure(service_name='pai-hello')
# logfire.instrument_pydantic_ai()

agent = Agent(
    'openai:gpt-4o',
    system_prompt='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
