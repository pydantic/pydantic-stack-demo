import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncIterator

import logfire
from cloudkv import AsyncCloudKV
from pydantic import BaseModel
from pydantic_ai import Agent

from app.self_improving_agent import ModelContextPatch, SelfImprovingAgentModel, SelfImprovingAgentStorage

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


class City(BaseModel, use_attribute_docstrings=True):
    """Details about a city."""

    city_name: str
    """The city name."""
    country: str
    """The country name."""


cloudkv_read_token = 'r2WqFgs0tQBv4jUH9basLcjT'
cloudkv_write_token = 'Cpnj8XazGk9oPDHREA76680UPaV8juHfZ5eWDJadiijkQQnz'


@dataclass
class CloudKVStorage(SelfImprovingAgentStorage):
    cloud_kv: AsyncCloudKV

    async def get_patch(self, agent_name: str) -> ModelContextPatch | None:
        return await self.cloud_kv.get_as(agent_name, ModelContextPatch)

    async def set_patch(self, agent_name: str, patch: ModelContextPatch, expires: timedelta) -> None:
        await self.cloud_kv.set(agent_name, patch, expires=expires)

    @asynccontextmanager
    async def lock(self, agent_name: str) -> AsyncIterator[bool]:
        key = f'lock:{agent_name}'
        r = await self.cloud_kv.get(key)
        if r is None:
            await self.cloud_kv.set(key, True, expires=3600)
            try:
                yield True
            finally:
                await self.cloud_kv.delete(key)
        else:
            yield False


city_agent = Agent(output_type=City)


async def main():
    async with AsyncCloudKV(cloudkv_read_token, cloudkv_write_token) as cloudkv:
        storage = CloudKVStorage(cloudkv)
        model = SelfImprovingAgentModel('openai:gpt-4o', storage, os.environ['LOGFIRE_READ_TOKEN'], 'city_agent')
        # with model.blocking_context():
        result = await city_agent.run('The windy city in the US of A.', model=model)
        debug(result.output)
        await model.wait_for_coach()


if __name__ == '__main__':
    asyncio.run(main())
