import asyncio
from dataclasses import dataclass
from enum import StrEnum

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits

logfire.configure(console=False)
logfire.instrument_pydantic_ai()


class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'


answerer_agent = Agent(
    'gateway/anthropic:claude-haiku-4-5',
    # 'groq:openai/gpt-oss-120b',
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
)


@answerer_agent.instructions
def add_answer(ctx: RunContext[str]) -> str:
    return f'THE SECRET OBJECT IS: "{ctx.deps}".'


@dataclass
class GameState:
    answer: str


class GameResult(BaseModel, use_attribute_docstrings=True):
    answer: str
    """The exact object the other player is thinking of."""
    explanation: str
    """The explanation for the object the player is thinking of."""


# Agent that asks questions to guess the object
questioner_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    deps_type=GameState,
    output_type=GameResult,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask quantitative questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
    result = await answerer_agent.run(question, deps=ctx.deps.answer)
    print(f'{ctx.run_step:>2}: {question}: {result.output}')
    return result.output


async def play(answer: str) -> AgentRunResult[GameResult]:
    state = GameState(answer=answer)
    result = await questioner_agent.run('start', deps=state, usage_limits=UsageLimits(request_limit=25))
    print(f'After {len(result.all_messages()) / 2}, the answer is: {result.output}')
    return result


if __name__ == '__main__':
    asyncio.run(play('potato'))
