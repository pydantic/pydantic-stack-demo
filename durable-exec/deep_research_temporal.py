import asyncio
import os
import sys
import uuid
from datetime import timedelta
from typing import Annotated

import logfire
from annotated_types import MaxLen
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.durable_exec.temporal import AgentPlugin, LogfirePlugin, PydanticAIPlugin, TemporalAgent
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

logfire.configure()
logfire.instrument_pydantic_ai()


class WebSearchStep(BaseModel):
    """A step that performs a web search.

    And returns a summary of the search results.
    """

    search_terms: str


class DeepResearchPlan(BaseModel, **ConfigDict(use_attribute_docstrings=True)):
    """A structured plan for deep research."""

    summary: str
    """A summary of the research plan."""

    web_search_steps: Annotated[list[WebSearchStep], MaxLen(5)]
    """A list of web search steps to perform to gather raw information."""

    analysis_instructions: str
    """The analysis step to perform after all web search steps are completed."""


plan_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Analyze the users query and design a plan for deep research to answer their query.',
    output_type=DeepResearchPlan,
    name='plan_agent',
)


search_agent = Agent(
    'openai-responses:gpt-4.1-mini',
    instructions="""
Perform a web search for the given terms and return a concise summary of the results.

Include links to original sources whenever possible.
""",
    tools=[tavily_search_tool(os.environ['TAVILY_API_KEY'])],
    name='search_agent',
)

analysis_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions="""
Analyze the research from the previous steps and generate a report on the given subject.

If the search results do not contain enough information, you may perform further searches using the
`extra_search` tool.

Your report should start with an executive summary of the results, then a concise analysis of the findings.

Include links to original sources whenever possible.
""",
    name='analysis_agent',
)


@analysis_agent.tool_plain
async def extra_search(query: str) -> str:
    """Perform an extra search for the given query."""
    result = await search_agent.run(query)
    return result.output


temporal_plan_agent = TemporalAgent(plan_agent)
temporal_search_agent = TemporalAgent(search_agent)
temporal_analysis_agent = TemporalAgent(
    analysis_agent,
    activity_config=workflow.ActivityConfig(start_to_close_timeout=timedelta(hours=1)),
)


@workflow.defn
class DeepResearchWorkflow:
    @workflow.run
    async def run(self, query: str) -> str:
        result = await temporal_plan_agent.run(query)
        plan = result.output
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(temporal_search_agent.run(step.search_terms)) for step in plan.web_search_steps]

        search_results = [task.result().output for task in tasks]

        analysis_result = await temporal_analysis_agent.run(
            format_as_xml(
                {
                    'query': query,
                    'search_results': search_results,
                    'instructions': plan.analysis_instructions,
                }
            ),
        )
        return analysis_result.output


async def deep_research_durable(query: str):
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin(), LogfirePlugin()])

    async with Worker(
        client,
        task_queue='deep_research',
        workflows=[DeepResearchWorkflow],
        plugins=[
            AgentPlugin(temporal_plan_agent),
            AgentPlugin(temporal_search_agent),
            AgentPlugin(temporal_analysis_agent),
        ],
    ):
        resume_id = sys.argv[1] if len(sys.argv) > 1 else None
        if resume_id is not None:
            print('resuming existing workflow', resume_id)
            summary = await client.get_workflow_handle(resume_id).result()  # type: ignore[ReportUnknownMemberType]
        else:
            summary = await client.execute_workflow(
                DeepResearchWorkflow.run,
                args=[query],
                id=f'deep_research-{uuid.uuid4()}',
                task_queue='deep_research',
            )
        print(summary)


if __name__ == '__main__':
    asyncio.run(
        deep_research_durable(
            'Whats the best Python agent framework to use if I care about durable execution and type safety?'
        )
    )
