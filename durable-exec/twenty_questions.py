import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext

secret = 'potato'

answerer_agent = Agent(
    # 'anthropic:claude-3-5-haiku-latest',
    'groq:openai/gpt-oss-120b',
    instructions=f"""
You are playing a question and answer game.
Your job is to answer yes/no questions about the secret object truthfully.
ALWAYS answer with only true for 'yes' and false for 'no'.

THE SECRET OBJECT IS: {secret}.
""",
    output_type=bool,
)


@dataclass
class GameState:
    step: int = 0


# Agent that asks questions to guess the object
questioner_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    deps_type=GameState,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask yes/no questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Literal['yes', 'no']:
    print(f'{ctx.deps.step:>2}: {question}:', end=' ', flush=True)
    ctx.deps.step += 1
    result = await answerer_agent.run(question)
    ans = 'yes' if result.output else 'no'
    print(ans)
    return ans


async def play() -> None:
    state = GameState()
    result = await questioner_agent.run('start', deps=state)
    print(f'After {state.step}, the answer is: {result.output}')


if __name__ == '__main__':
    asyncio.run(play())
