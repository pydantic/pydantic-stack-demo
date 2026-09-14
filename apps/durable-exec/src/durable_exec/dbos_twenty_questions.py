"""Twenty questions as a durable DBOS workflow.

The agents are the ones from `apps/multi-agent`, built with `DBOSDurability()` attached.
Inside the `@DBOS.workflow` every model request becomes a checkpointed step, so if the
process dies mid-game, re-running with the same workflow id resumes from the last
completed request instead of starting over.

    uv run durable-exec-dbos-twenty-questions            # start a new game, prints its workflow id
    uv run durable-exec-dbos-twenty-questions <wf-id>    # resume / fetch the result of that game
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from dbos import DBOS, DBOSConfig, SetWorkflowID
from pydantic_ai import UsageLimits
from pydantic_ai.durable_exec.dbos import DBOSDurability

from demo_core import configure_logfire, settings
from multi_agent.twenty_questions import GameResult, GameState, build_agents

# SQLite keeps the demo self-contained; point this at Postgres (see .env.example) for production.
DBOS_SYSTEM_DATABASE_URL = os.environ.get('DBOS_SYSTEM_DATABASE_URL', 'sqlite:///durable_exec_dbos.sqlite')

game = build_agents(capabilities=[DBOSDurability()])


@DBOS.workflow()
async def play_workflow(answer: str) -> GameResult:
    result = await game.questioner.run(
        'start', deps=GameState(answer=answer), usage_limits=UsageLimits(request_limit=40)
    )
    return result.output


async def run(resume_id: str | None, answer: str = 'potato') -> GameResult:
    config: DBOSConfig = {
        'name': 'durable_exec_twenty_questions',
        'system_database_url': DBOS_SYSTEM_DATABASE_URL,
        'enable_otlp': True,  # DBOS workflow/step spans join the Pydantic AI spans in Logfire
    }
    DBOS(config=config)
    DBOS.launch()
    try:
        if resume_id is not None:
            print(f'resuming workflow {resume_id}')
            handle = await DBOS.retrieve_workflow_async(resume_id)
            output: GameResult = await handle.get_result()
        else:
            wf_id = f'twenty-questions-{uuid.uuid4()}'
            print(f'starting workflow {wf_id}')
            with SetWorkflowID(wf_id):
                output = await play_workflow(answer)
        print(f'The guess is: {output.answer!r} ({output.explanation})')
        return output
    finally:
        DBOS.destroy()


def main() -> None:
    configure_logfire('durable-exec')
    settings().require_model_credentials()
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))


if __name__ == '__main__':
    main()
