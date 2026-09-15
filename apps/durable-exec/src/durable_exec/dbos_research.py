"""Plan / fan-out / analyse as durable DBOS workflows.

Each search runs in its own child workflow started with `DBOS.start_workflow_async`, so the
fan-out survives a crash too: on resume, finished searches are read back from the
database and only the unfinished ones run again.

    uv run durable-exec-dbos-research [question...]
    uv run durable-exec-dbos-research --resume <wf-id>
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandleAsync
from pydantic_ai import format_as_xml
from pydantic_ai.durable_exec.dbos import DBOSDurability

from demo_core import configure_logfire, settings
from multi_agent.research import DEFAULT_QUERY, build_agents

DBOS_SYSTEM_DATABASE_URL = os.environ.get('DBOS_SYSTEM_DATABASE_URL', 'sqlite:///durable_exec_dbos.sqlite')

research = build_agents(capabilities=[DBOSDurability()])


@DBOS.workflow()
async def search_workflow(search_terms: str) -> str:
    return (await research.search.run(search_terms)).output


@DBOS.workflow()
async def deep_research_workflow(query: str) -> str:
    plan = (await research.plan.run(query)).output

    handles: list[WorkflowHandleAsync[str]] = []
    for step in plan.web_search_steps:
        handles.append(await DBOS.start_workflow_async(search_workflow, step.search_terms))
    search_results = [await handle.get_result() for handle in handles]

    analysis = await research.analysis.run(
        format_as_xml({'query': query, 'search_results': search_results, 'instructions': plan.analysis_instructions})
    )
    return analysis.output


async def run(query: str, resume_id: str | None) -> str:
    config: DBOSConfig = {
        'name': 'durable_exec_research',
        'system_database_url': DBOS_SYSTEM_DATABASE_URL,
        'enable_otlp': True,
    }
    DBOS(config=config)
    DBOS.launch()
    try:
        if resume_id is not None:
            print(f'resuming workflow {resume_id}')
            handle = await DBOS.retrieve_workflow_async(resume_id)
            report: str = await handle.get_result()
        else:
            wf_id = f'deep-research-{uuid.uuid4()}'
            print(f'starting workflow {wf_id}')
            with SetWorkflowID(wf_id):
                report = await deep_research_workflow(query)
        print(report)
        return report
    finally:
        DBOS.destroy()


def main() -> None:
    configure_logfire('durable-exec')
    settings().require_model_credentials()
    args = sys.argv[1:]
    resume_id = args[args.index('--resume') + 1] if '--resume' in args else None
    query = ' '.join(a for a in args if a != '--resume' and a != resume_id) or DEFAULT_QUERY
    asyncio.run(run(query, resume_id))


if __name__ == '__main__':
    main()
