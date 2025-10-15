import asyncio
import sys
from dataclasses import dataclass
from random import random
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalAgent
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

secret = 'potato'

answerer_agent = Agent(
    'anthropic:claude-3-5-haiku-latest',
    instructions=f"""
You are playing a question and answer game.
Your job is to answer yes/no questions about the secret object truthfully.
ALWAYS answer with only true for 'yes' and false for 'no'.

THE SECRET OBJECT IS: {secret}.
""",
    output_type=bool,
    name='answerer_agent',
)
temporal_answerer_agent = TemporalAgent(answerer_agent)


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
    name='questioner_agent',
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Literal['yes', 'no']:
    if random() > 0.9:
        raise RuntimeError('broken')
    print(f'{ctx.deps.step:>2}: {question}:', end=' ', flush=True)
    ctx.deps.step += 1
    result = await temporal_answerer_agent.run(question)
    ans = 'yes' if result.output else 'no'
    print(ans)
    return ans


temporal_questioner_agent = TemporalAgent(questioner_agent)


@workflow.defn
class TwentyQuestionsWorkflow:
    @workflow.run
    async def run(self) -> None:
        state = GameState()
        result = await temporal_questioner_agent.run('start', deps=state)
        print(f'After {state.step}, the answer is: {result.output}')


async def play(resume: bool):
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])

    async with Worker(
        client,
        task_queue='twenty_questions',
        workflows=[TwentyQuestionsWorkflow],
        plugins=[AgentPlugin(temporal_answerer_agent), AgentPlugin(temporal_questioner_agent)],
    ):
        workflow_id = 'twenty_questions'
        if resume:
            await client.get_workflow_handle(workflow_id).result()  # type: ignore[ReportUnknownMemberType]
        else:
            await client.execute_workflow(  # type: ignore[ReportUnknownMemberType]
                TwentyQuestionsWorkflow.run,
                id=workflow_id,
                task_queue='twenty_questions',
            )


if __name__ == '__main__':
    asyncio.run(play('resume' in sys.argv))
