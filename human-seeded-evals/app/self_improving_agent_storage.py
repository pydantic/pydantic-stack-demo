import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import AsyncIterator, Callable, ParamSpec, TypeVar

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


@dataclass
class LocalStorage(SelfImprovingAgentStorage):
    directory: Path = Path('.self-improving-agent')

    def __post_init__(self):
        self.directory.mkdir(exist_ok=True)

    async def get_patch(self, agent_name: str) -> ModelContextPatch | None:
        file = self.directory / f'{agent_name}.json'
        if file.exists():
            content = await asyncify(file.read_bytes)
            return ModelContextPatch.model_validate_json(content)

    async def set_patch(self, agent_name: str, patch: ModelContextPatch, expires: timedelta) -> None:
        # note we're ignoring expiry here
        file = self.directory / f'{agent_name}.json'
        content = patch.model_dump_json(indent=2)
        await asyncify(file.write_text, content)

    @asynccontextmanager
    async def lock(self, agent_name: str) -> AsyncIterator[bool]:
        file = self.directory / f'lock:{agent_name}'
        if not await asyncify(file.exists):
            await asyncify(file.touch)
            try:
                yield True
            finally:
                await asyncify(file.unlink)
        else:
            yield False


P = ParamSpec('P')
R = TypeVar('R')


async def asyncify(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return await asyncio.get_event_loop().run_in_executor(None, partial(func, *args, **kwargs))
