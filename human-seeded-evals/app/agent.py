from __future__ import annotations as _annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

from cloudkv import AsyncCloudKV
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from .models import TimeRangeInputs, TimeRangeResponse
from .self_improving_agent import SelfImprovingAgentModel
from .self_improving_agent_storage import CloudKVStorage


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


@asynccontextmanager
async def self_improving_model() -> AsyncIterator[SelfImprovingAgentModel]:
    cloudkv_read_token, cloudkv_write_token = os.environ['CLOUDKV_TOKEN'].split('.')
    logfire_read_token = os.environ['LOGFIRE_READ_TOKEN']
    async with AsyncCloudKV(cloudkv_read_token, cloudkv_write_token) as cloudkv:
        storage = CloudKVStorage(cloudkv)
        m = SelfImprovingAgentModel('anthropic:claude-sonnet-4-0', storage, logfire_read_token, 'time_range_agent')
        yield m
        await m.wait_for_coach()


@time_range_agent.instructions
def inject_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Add the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST' to context."""
    return f"The user's current time is {ctx.deps.now:%A, %B %d, %Y %H:%M:%S %Z}."


async def infer_time_range(inputs: TimeRangeInputs, *, model: Model | None = None) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    result = await time_range_agent.run(inputs.prompt, deps=TimeRangeDeps(now=inputs.now), model=model)
    return result.output
