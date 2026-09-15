"""Twenty questions as a Temporal workflow.

Same agents as `apps/multi-agent`, built with `TemporalDurability()`. Inside the workflow,
model requests and the `ask_question` tool run as Temporal activities. The tool is built
with a deliberate 10% failure rate: watch Temporal retry the activity and the game carry on
as if nothing happened. Kill the process mid-game and re-run with the workflow id to see the
run resume from Temporal's event history.

Needs a Temporal server (`docker compose --profile durable-exec up temporal`, or
`temporal server start-dev`); the address comes from TEMPORAL_ADDRESS.

    uv run durable-exec-temporal-twenty-questions            # start a new game
    uv run durable-exec-temporal-twenty-questions <wf-id>    # wait for / resume that game
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from pydantic_ai import UsageLimitExceeded, UsageLimits
from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from demo_core import configure_logfire, settings
    from multi_agent.twenty_questions import GameResult, GameState, build_agents

TASK_QUEUE = 'twenty_questions'

game = build_agents(capabilities=[TemporalDurability(deps_type=GameState)], tool_failure_rate=0.1)


@workflow.defn
class TwentyQuestionsWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [game.questioner, game.answerer]

    @workflow.run
    async def run(self, answer: str) -> GameResult:
        result = await game.questioner.run(
            'start', deps=GameState(answer=answer), usage_limits=UsageLimits(request_limit=40)
        )
        return result.output


async def run(resume_id: str | None, answer: str = 'potato') -> GameResult:
    client = await Client.connect(settings().temporal_address, plugins=[PydanticAIPlugin(), LogfirePlugin()])

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[TwentyQuestionsWorkflow]):
        if resume_id is not None:
            print(f'waiting for workflow {resume_id}')
            output: GameResult = await client.get_workflow_handle(resume_id).result()
        else:
            wf_id = f'twenty-questions-{uuid.uuid4()}'
            print(f'starting workflow {wf_id}')
            output = await client.execute_workflow(
                TwentyQuestionsWorkflow.run, args=[answer], id=wf_id, task_queue=TASK_QUEUE
            )
    print(f'The guess is: {output.answer!r} ({output.explanation})')
    return output


def main() -> None:
    configure_logfire('durable-exec')
    settings().require_model_credentials()
    try:
        asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
    except (UsageLimitExceeded, WorkflowFailureError) as e:
        # Running out of the request budget before guessing is a normal outcome of the game.
        print(f'The questioner ran out of questions: {e}')


if __name__ == '__main__':
    main()
