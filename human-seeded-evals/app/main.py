from contextlib import asynccontextmanager
from typing import cast

import logfire
from fastapi import FastAPI, Request

from .agent import infer_time_range, self_improving_model
from .models import TimeRangeInputs, TimeRangeResponse
from .self_improving_agent import SelfImprovingAgentModel

logfire.configure(environment='dev')

logfire.instrument_pydantic_ai()
logfire.instrument_httpx()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with self_improving_model() as model:
        app.state.model = model
        yield


app = FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.post('/api/timerange')
async def convert_time_range(request: Request, time_range_inputs: TimeRangeInputs) -> TimeRangeResponse:
    model = cast(SelfImprovingAgentModel, request.app.state.model)
    return await infer_time_range(time_range_inputs, model=model)
