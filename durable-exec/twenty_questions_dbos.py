import asyncio
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum

import logfire
from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandleAsync
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits
from pydantic_ai.durable_exec.dbos import DBOSAgent

logfire.configure(console=False)
logfire.instrument_pydantic_ai()


class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'


answerer_agent = Agent(
    'anthropic:claude-3-5-haiku-latest',
    # 'groq:openai/gpt-oss-120b',
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
    name='answerer_agent',
)

dbos_answerer_agent = DBOSAgent(answerer_agent)


@answerer_agent.instructions
def add_answer(ctx: RunContext[str]) -> str:
    return f'THE SECRET OBJECT IS: "{ctx.deps}".'


@dataclass
class GameState:
    answer: str


# Agent that asks questions to guess the object
questioner_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=GameState,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask quantitative questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
    name='questioner_agent',
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
    result = await dbos_answerer_agent.run(question, deps=ctx.deps.answer)
    print(f'{ctx.run_step:>2}: {question}: {result.output}')
    return result.output


dbos_questioner_agent = DBOSAgent(questioner_agent)


async def play(resume_id: str | None, answer: str) -> AgentRunResult[str]:
    config: DBOSConfig = {
        'name': 'twenty_questions_durable',
        'enable_otlp': True,
        # run the server with
        # docker run -e POSTGRES_HOST_AUTH_METHOD=trust --rm -it --name pg -p 5432:5432 -d postgres
        'system_database_url': 'postgresql://postgres@localhost:5432/dbos',
        "application_version": "0.1.0",
    }
    DBOS(config=config)
    DBOS.launch()
    if resume_id is not None:
        print('resuming existing workflow', resume_id)
        # Get the workflow handle and wait for the result
        wf_handle: WorkflowHandleAsync[AgentRunResult] = await DBOS.retrieve_workflow_async(resume_id)
        result = await wf_handle.get_result()
    else:
        wf_id = f'twenty-questions-{uuid.uuid4()}'
        print('starting new workflow', wf_id)
        state = GameState(answer=answer)
        with SetWorkflowID(wf_id):
            result = await dbos_questioner_agent.run('start', deps=state, usage_limits=UsageLimits(request_limit=25))

    print(f'After {len(result.all_messages()) / 2}, the answer is: {result.output}')

    return result


if __name__ == '__main__':
    asyncio.run(play(sys.argv[1] if len(sys.argv) > 1 else None, 'potato'))
