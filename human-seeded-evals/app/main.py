from datetime import datetime, timezone

import logfire
from fastapi import FastAPI
from pydantic import BaseModel

from .agent import get_coach, infer_time_range
from .models import TimeRangeInputs, TimeRangeResponse
from .self_improving_agent import ModelContextPatch

logfire.configure(environment='dev')

logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
coach = get_coach()


app = FastAPI()
logfire.instrument_fastapi(app)


@app.post('/api/timerange')
async def convert_time_range(time_range_inputs: TimeRangeInputs) -> TimeRangeResponse:
    return await infer_time_range(time_range_inputs)


class Field(BaseModel):
    id: str
    text: str


@app.get('/api/context')
def get_agent_context() -> list[Field]:
    coach_fields = coach.get_fields() or []
    fields = [Field(id=f.key, text=f.current_prompt or '') for f in coach_fields]

    if patch := coach.get_patch():
        for field in fields:
            if new_text := patch.context_patch.get(field.id):
                field.text = new_text

    return fields


class PostFields(BaseModel):
    fields: list[Field]


@app.post('/api/context')
def post_agent_context(m: PostFields):
    context_patch = {f.id: f.text for f in m.fields if f.text}
    coach.update_patch(ModelContextPatch(context_patch=context_patch, timestamp=datetime.now(tz=timezone.utc)))


@app.post('/api/context/update')
async def post_update_agent_context():
    await coach.run()
