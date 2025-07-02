import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import logfire
from logfire.experimental import annotations
from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, format_as_xml

read_token = os.environ['LOGFIRE_READ_TOKEN']
logfire.configure(environment='evals')
logfire.instrument_pydantic_ai()


class EvalFeedback(BaseModel, use_attribute_docstrings=True):
    reaction: Literal['positive', 'negative']
    comment: str | None = None
    """Very concise comment for the evaluation"""


prompt_path = Path(__file__).parent / 'eval_agent_prompt.txt'
evals_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    instructions=prompt_path.read_text(),
    output_type=EvalFeedback,
)
runs_query = """
select
    created_at,
    trace_id,
    span_id,
    attributes->'all_messages_events'->1->>'content' as prompt,
    attributes->'final_result' as output
from records
where otel_scope_name = 'pydantic-ai' and message = 'time_range_agent run'
"""

with_annotations_query = """
select
    '00-' || trace_id || '-' || parent_span_id || '-01' as trace_parent
from records
where kind='annotation'
"""


class RunData(BaseModel):
    created_at: datetime
    trace_id: str
    span_id: str
    prompt: str
    output: Any

    @property
    def trace_parent(self):
        return f'00-{self.trace_id}-{self.span_id}-01'


run_data_list_schema = TypeAdapter(list[RunData])


async def apply_feedback(run: RunData):
    if run.output is None:
        return
    r = await evals_agent.run(
        format_as_xml({'run_timestamp': run.created_at, 'prompt': run.prompt, 'output': run.output})
    )
    print(f'Adding feedback to {run.trace_parent}: {r.output}')
    annotations.record_feedback(
        run.trace_parent,
        'AI Annotation',
        value=r.output.reaction,
        comment=r.output.comment,
        extra={'path': ''},
    )


async def main():
    min_timestamp = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
    async with AsyncLogfireQueryClient(read_token) as client:
        while True:
            response = await client.query_json_rows(runs_query, min_timestamp=min_timestamp)
            runs = run_data_list_schema.validate_python(response['rows'])
            if runs:
                response = await client.query_json_rows(with_annotations_query, min_timestamp=min_timestamp)
                annotated_spans: set[str] = {r['trace_parent'] for r in response['rows']}
                runs = [run for run in runs if run.trace_parent not in annotated_spans]
                if runs:
                    print('')
                    logfire.info('found {runs} new runs to evaluate', runs=len(runs))
                    min_timestamp = min(runs, key=lambda run: run.created_at).created_at.astimezone(timezone.utc)
                    await asyncio.gather(*[apply_feedback(run) for run in runs])
                    await asyncio.sleep(2)
                    continue

            min_timestamp = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
            print('.', end='', flush=True)

            await asyncio.sleep(2)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('stopping')
