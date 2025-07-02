import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import logfire
from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, format_as_xml

sys.path.append(str(Path(__file__).parent.parent))

from app import agent

read_token = os.environ['LOGFIRE_READ_TOKEN']
logfire.configure(environment='evals')
logfire.instrument_pydantic_ai()

auto_annotation_agent = Agent(
    'anthropic:claude-opus-4-0',
    instructions="""
Your task is to build a system prompt for an agent (the evals agent) which will evaluate the performance of another
agent and provide feedback on its performance.

You should return the system prompt for the evals agent ONLY.
""",
)


class RunFeedback(BaseModel):
    reaction: Literal['positive', 'negative'] | None
    comment: str | None


class AgentRunSummary(BaseModel):
    prompt: str
    context: Any
    output: Any
    feedback: RunFeedback | None = None


count_runs_query = "select count(*) from records where message = 'time_range_agent run'"
runs_query = """
select
    trace_id,
    span_id,
    'time timestamp: ' || created_at as context,
    attributes->'all_messages_events'->1->>'content' as prompt,
    attributes->'final_result' as output
from records
where message = 'time_range_agent run'
"""
feedback_query = """
select
    trace_id,
    parent_span_id,
    attributes->>'Annotation' as reaction,
    attributes->>'logfire.feedback.comment' as comment
from records
where kind='annotation' and attributes->>'logfire.feedback.name'='Annotation'
"""
min_count = 1


async def get_runs() -> None | list[AgentRunSummary]:
    min_timestamp = datetime(2025, 7, 2)
    async with AsyncLogfireQueryClient(read_token) as client:
        c = await client.query_json(sql=count_runs_query, min_timestamp=min_timestamp)
        count = c['columns'][0]['values'][0]
        if count < min_count:
            print(f'Insufficient runs ({count})')
            return

        r = await client.query_json_rows(sql=feedback_query, min_timestamp=min_timestamp)
        feedback_lookup: dict[str, Any] = {
            f'{row["trace_id"]}-{row["parent_span_id"]}': RunFeedback(**row) for row in r['rows']
        }

        r = await client.query_json_rows(sql=runs_query, min_timestamp=min_timestamp)
        runs: list[AgentRunSummary] = []
        with_feedback = 0
        for row in r['rows']:
            key = f'{row["trace_id"]}-{row["span_id"]}'
            if feedback := feedback_lookup.get(key):
                row['feedback'] = feedback
                with_feedback += 1
            runs.append(AgentRunSummary(**row))

        logfire.info(f'Found {len(runs)} runs, {with_feedback} with feedback')
        return runs


async def generate_evals_prompt(
    name: str, instrunctions: str, output_type: type[Any] | None, runs: list[AgentRunSummary]
) -> str:
    data: dict[str, Any] = {'agent_name': name, 'agent_instructions': instrunctions}
    if output_type is not None:
        data['output_schema'] = json.dumps(TypeAdapter(output_type).json_schema(), indent=2)
    data['agent_runs'] = [run.model_dump(exclude_none=True) for run in runs]
    prompt = format_as_xml(data, include_root_tag=False)
    r = await auto_annotation_agent.run(prompt)
    return r.output


async def main():
    runs = await get_runs()
    if runs:
        prompt = await generate_evals_prompt(
            'time_range_agent',
            agent.instrunctions,
            agent.TimeRangeResponse,  # type: ignore
            runs,
        )
        prompt_path = Path(__file__).parent / 'eval_agent_prompt.txt'
        prompt_path.write_text(prompt)
        print(f'prompt written to {prompt_path}')


if __name__ == '__main__':
    asyncio.run(main())
