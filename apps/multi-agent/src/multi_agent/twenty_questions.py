"""Agent-as-a-tool: one agent calls another from inside a tool.

Two agents play twenty questions. The *questioner* asks yes/no questions through its
`ask_question` tool; that tool runs the *answerer* agent, which knows the secret object.
The parent passes `usage=ctx.usage` so the child's tokens and cost count against the
parent's `UsageLimits`, and Logfire shows the child run nested under the tool span.

`build_agents()` exists so `apps/durable-exec` can build the same pair with a durability
capability attached; plain scripts use the module-level `agents`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import random

from pydantic import BaseModel
from pydantic_ai import Agent, AgentCapability, AgentRunResult, RunContext, UsageLimitExceeded, UsageLimits

from demo_core import configure_logfire, settings


class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    completely_wrong = 'completely wrong'


@dataclass
class GameState:
    answer: str


class GameResult(BaseModel, use_attribute_docstrings=True):
    answer: str
    """The exact object the other player is thinking of."""
    explanation: str
    """Why you think it is that object."""


@dataclass
class TwentyQuestions:
    questioner: Agent[GameState, GameResult]
    answerer: Agent[str, Answer]


def build_agents(
    *,
    capabilities: Sequence[AgentCapability] = (),
    questioner_model: str | None = None,
    answerer_model: str | None = None,
    tool_failure_rate: float = 0.0,
) -> TwentyQuestions:
    """Build the questioner/answerer pair.

    `tool_failure_rate` makes `ask_question` raise randomly; the durable-execution demos use
    it to show a workflow surviving a flaky tool.
    """
    answerer: Agent[str, Answer] = Agent(
        answerer_model or settings().fast_model,
        name='answerer_agent',
        deps_type=str,
        output_type=Answer,
        instructions='You are playing twenty questions. Answer questions about the secret object truthfully; '
        'use "kind of" or "not really" when the honest answer is partial.',
        capabilities=list(capabilities),
    )

    @answerer.instructions
    def add_secret(ctx: RunContext[str]) -> str:
        return f'THE SECRET OBJECT IS: "{ctx.deps}".'

    questioner: Agent[GameState, GameResult] = Agent(
        questioner_model or settings().model,
        name='questioner_agent',
        deps_type=GameState,
        output_type=GameResult,
        instructions="""
You are playing twenty questions. You need to guess the object the other player is thinking of.
Ask yes/no questions with the `ask_question` tool to narrow down the possibilities.

Start broad: first establish whether it is an animal, a plant or food, or a man-made object, then
whether it is bigger than a breadbox, and only then get specific. You have at most 20 questions.
When you are confident, ask "Is it a <object>?"; once the answer is yes, return that object.
""",
        capabilities=list(capabilities),
    )

    @questioner.tool
    async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
        """Ask the other player a yes/no style question about the secret object."""
        if tool_failure_rate and random() < tool_failure_rate:
            raise RuntimeError('simulated transient failure in ask_question')
        # `usage=ctx.usage` charges the child run to the parent's usage limits.
        result = await answerer.run(question, deps=ctx.deps.answer, usage=ctx.usage)
        print(f'{ctx.run_step:>2}: {question} -> {result.output}')
        return result.output

    return TwentyQuestions(questioner=questioner, answerer=answerer)


agents = build_agents()


async def play(answer: str, *, game: TwentyQuestions = agents, request_limit: int = 40) -> AgentRunResult[GameResult]:
    result = await game.questioner.run(
        'start', deps=GameState(answer=answer), usage_limits=UsageLimits(request_limit=request_limit)
    )
    print(f'After {result.usage.requests} model requests the guess is: {result.output.answer!r}')
    print(f'usage: {result.usage}')
    return result


def main() -> None:
    configure_logfire('multi-agent')
    settings().require_model_credentials()
    try:
        asyncio.run(play('potato'))
    except UsageLimitExceeded as e:
        print(f'The questioner ran out of questions: {e}')


if __name__ == '__main__':
    main()
