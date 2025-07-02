from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import Agent, RunContext

from .models import TimeRangeInputs, TimeRangeResponse


@dataclass
class TimeRangeDeps:
    now: datetime


instrunctions = "Convert the user's request into a structured time range."
time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    'anthropic:claude-sonnet-4-0',
    output_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    instructions=instrunctions,
    retries=1,
)


@time_range_agent.instructions
def inject_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Add the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST' to context."""
    return f"The user's current time is {ctx.deps.now:%A, %B %d, %Y %H:%M:%S %Z}."


async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    result = await time_range_agent.run(inputs.prompt, deps=TimeRangeDeps(now=inputs.now))
    return result.output
