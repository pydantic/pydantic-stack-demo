import asyncio

import logfire
from app.agent import infer_time_range, self_improving_model
from app.models import TimeRangeInputs

logfire.configure(environment='evals')

logfire.instrument_pydantic_ai()


async def main():
    async with self_improving_model() as model:
        with model.blocking_context():
            with logfire.span('running infer_time_range with blocking coach'):
                await infer_time_range(TimeRangeInputs(prompt='yesterday'), model=model)


if __name__ == '__main__':
    asyncio.run(main())
