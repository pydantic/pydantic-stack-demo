from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import Agent, RunContext

from .models import TimeRangeInputs, TimeRangeResponse
from .self_improving_agent import Coach, SelfImprovingAgentModel


@dataclass
class TimeRangeDeps:
    now: datetime


system_prompt = "Convert the user's request into a structured time range."
time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    'anthropic:claude-sonnet-4-0',
    output_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    system_prompt=system_prompt,
    retries=1,
)


def get_coach() -> Coach:
    logfire_read_token = os.environ['LOGFIRE_READ_TOKEN']
    return Coach('time_range_agent', logfire_read_token)


@time_range_agent.tool
def inject_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Add the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST' to context."""
    return f"The user's current time is {ctx.deps.now:%A, %B %d, %Y %H:%M:%S %Z}."


async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    model = SelfImprovingAgentModel('anthropic:claude-sonnet-4-0')
    result = await time_range_agent.run(inputs.prompt, deps=TimeRangeDeps(now=inputs.now), model=model)
    return result.output
