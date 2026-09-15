"""The task being optimized: extracting contact information from free text.

`contact_agent` starts with deliberately weak instructions. `Agent.override(instructions=...)`
is how the optimizer (and the eval runner) swaps in candidate prompts without rebuilding the
agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from demo_core import settings

INITIAL_INSTRUCTIONS = 'Extract contact information from the provided text.'

# What a good prompt looks like: used by `prompt-optimization compare` as a reference point.
EXPERT_INSTRUCTIONS = """Extract contact information from the provided text with high precision.

Guidelines:
1. NAME: Look for full names (first + last). Ignore titles like Dr., Mr., etc. in the extracted name.
2. EMAIL: Extract any valid email address format (user@domain.tld).
3. PHONE: Extract phone numbers in any format (with or without country codes, parentheses, dashes).
4. COMPANY: Look for organization names, often near titles or after "at" or "from".
5. TITLE: Extract job titles/roles, usually appearing near names or before company names.

Important:
- If multiple contacts appear, focus on the PRIMARY contact being introduced or highlighted.
- If information is missing, leave the field as null rather than guessing.
- Normalize phone numbers by preserving the original format.
- For names, extract just the name without titles or credentials (Ph.D., Jr., etc.)."""


class ContactInfo(BaseModel):
    """Extracted contact information from text."""

    name: str | None = Field(default=None, description="The person's full name")
    email: str | None = Field(default=None, description='Email address')
    phone: str | None = Field(default=None, description='Phone number')
    company: str | None = Field(default=None, description='Company or organization name')
    title: str | None = Field(default=None, description='Job title or role')


contact_agent = Agent(
    settings().fast_model,
    name='contact_agent',
    output_type=ContactInfo,
    instructions=INITIAL_INSTRUCTIONS,
)


@dataclass
class TaskInput:
    """Input to the contact extraction task."""

    text: str


async def extract_contact_info(input: TaskInput) -> ContactInfo:
    """The task function under evaluation and optimization."""
    result = await contact_agent.run(input.text)
    return result.output
