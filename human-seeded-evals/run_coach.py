import asyncio

import logfire
from app.agent import get_coach

logfire.configure(environment='evals')

logfire.instrument_pydantic_ai()


if __name__ == '__main__':
    asyncio.run(get_coach().run())
