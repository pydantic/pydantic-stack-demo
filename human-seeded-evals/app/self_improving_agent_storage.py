from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncIterator

from cloudkv import AsyncCloudKV

from .self_improving_agent import ModelContextPatch, SelfImprovingAgentStorage


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
