"""Structured output: the agent returns a validated Pydantic model, not a string."""

from datetime import date

from pydantic import BaseModel
from pydantic_ai import Agent

from demo_core import configure_logfire, settings


class Person(BaseModel):
    name: str
    dob: date
    city: str


agent = Agent(
    settings().fast_model,
    name='extract_person',
    output_type=Person,
    instructions='Extract information about the person.',
)


def main() -> None:
    configure_logfire('agent-basics')
    settings().require_model_credentials()

    result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
    print(repr(result.output))


if __name__ == '__main__':
    main()
