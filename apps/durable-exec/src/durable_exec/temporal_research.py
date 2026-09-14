"""Plan / fan-out / analyse as a Temporal workflow.

The workflow body is the same Python as `multi_agent.research.deep_research`; with
`TemporalDurability()` on the agents, each model request is an activity and the
`asyncio.TaskGroup` fan-out is replayed deterministically from the event history.

    uv run durable-exec-temporal-research [question...]
    uv run durable-exec-temporal-research --resume <wf-id>
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import timedelta

from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import format_as_xml

    from demo_core import configure_logfire, settings
    from multi_agent.research import DEFAULT_QUERY, build_agents

TASK_QUEUE = 'deep_research'

# The analysis step can take a while; give its activities a longer timeout than the 60s default.
research = build_agents(
    capabilities=[
        TemporalDurability(activity_config=workflow.ActivityConfig(start_to_close_timeout=timedelta(minutes=5)))
    ]
)


@workflow.defn
class DeepResearchWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [research.plan, research.search, research.analysis]

    @workflow.run
    async def run(self, query: str) -> str:
        plan = (await research.plan.run(query)).output

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(research.search.run(step.search_terms)) for step in plan.web_search_steps]
        search_results = [task.result().output for task in tasks]

        analysis = await research.analysis.run(
            format_as_xml(
                {'query': query, 'search_results': search_results, 'instructions': plan.analysis_instructions}
            )
        )
        return analysis.output


async def run(query: str, resume_id: str | None) -> str:
    client = await Client.connect(settings().temporal_address, plugins=[PydanticAIPlugin(), LogfirePlugin()])

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[DeepResearchWorkflow]):
        if resume_id is not None:
            print(f'waiting for workflow {resume_id}')
            report: str = await client.get_workflow_handle(resume_id).result()
        else:
            wf_id = f'deep-research-{uuid.uuid4()}'
            print(f'starting workflow {wf_id}')
            report = await client.execute_workflow(
                DeepResearchWorkflow.run, args=[query], id=wf_id, task_queue=TASK_QUEUE
            )
    print(report)
    return report


def main() -> None:
    configure_logfire('durable-exec')
    settings().require_model_credentials()
    args = sys.argv[1:]
    resume_id = args[args.index('--resume') + 1] if '--resume' in args else None
    query = ' '.join(a for a in args if a != '--resume' and a != resume_id) or DEFAULT_QUERY
    asyncio.run(run(query, resume_id))


if __name__ == '__main__':
    main()
