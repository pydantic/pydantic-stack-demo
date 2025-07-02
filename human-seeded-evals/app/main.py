import logfire
from fastapi import FastAPI

from .agent import infer_time_range
from .models import TimeRangeInputs, TimeRangeResponse

logfire.configure(environment='dev')
logfire.instrument_pydantic_ai()

app = FastAPI()
logfire.instrument_fastapi(app)


@app.post('/api/timerange')
async def convert_time_range(time_range_inputs: TimeRangeInputs) -> TimeRangeResponse:
    return await infer_time_range(time_range_inputs)
