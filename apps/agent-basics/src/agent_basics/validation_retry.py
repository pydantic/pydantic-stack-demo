"""Validation-driven retries.

A Pydantic validator rejects the first answer; Pydantic AI feeds the validation error back
to the model and asks again. In Logfire the retry shows up as a second model request inside
the same agent run, with the error message the model was given.
"""

from datetime import date

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

from demo_core import configure_logfire, settings


class Person(BaseModel):
    name: str
    dob: date
    city: str

    @field_validator('dob')
    @classmethod
    def validate_dob(cls, v: date) -> date:
        if v >= date(1900, 1, 1):
            raise ValueError('The person must be born in the 19th century')
        return v


agent = Agent(
    settings().model,
    name='extract_person_strict',
    output_type=Person,
    instructions='Extract information about the person.',
)


def main() -> None:
    configure_logfire('agent-basics')
    settings().require_model_credentials()

    result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
    print(repr(result.output))
    print(f'requests made: {result.usage.requests}')


if __name__ == '__main__':
    main()
