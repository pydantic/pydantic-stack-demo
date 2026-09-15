"""An agent with tools and dependencies.

The agent decides which tools to call and in what order; the tools get their HTTP client
from the run's dependencies. With `logfire.instrument_httpx()` every outbound request
shows up nested under the tool span that made it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from random import randint
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from demo_core import configure_logfire, settings


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    settings().model,
    name='weather_agent',
    # Some models call the tools below with just 'Be concise'; others need a nudge.
    instructions='Be concise, reply with one sentence. Use the tools to look up locations and weather.',
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    # NOTE: this public demo endpoint returns random coordinates; the point is the trace shape.
    r = await ctx.deps.client.get(
        'https://demo-endpoints.pydantic.workers.dev/latlng',
        params={'location': location_description, 'sleep': randint(200, 1200)},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/number',
            params={'min': 10, 'max': 30, 'sleep': randint(200, 1200)},
        ),
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/weather',
            params={'lat': lat, 'lng': lng, 'sleep': randint(200, 1200)},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {'temperature': f'{temp_response.text} °C', 'description': descr_response.text}


async def run(question: str = 'What is the weather like in London and in Wiltshire?') -> str:
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        result = await weather_agent.run(question, deps=Deps(client=client))
        return result.output


def main() -> None:
    configure_logfire('agent-basics')
    settings().require_model_credentials()
    print(asyncio.run(run()))


if __name__ == '__main__':
    main()
