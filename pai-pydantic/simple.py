from datetime import date

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent

logfire.configure(service_name='pai-pydantic-simple')
logfire.instrument_pydantic_ai()


class Person(BaseModel):
    name: str
    dob: date
    city: str


agent = Agent(
    'openai:gpt-4o',
    output_type=Person,
    instructions='Extract information about the person',
)
result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
print(repr(result.output))
