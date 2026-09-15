"""Plan, fan out, analyse: three agents orchestrated by plain Python.

1. `plan_agent` turns the question into a structured `DeepResearchPlan` (up to 3 searches).
2. One `search_agent` run per step, concurrently in an `asyncio.TaskGroup`, using the
   provider's native web search through the Gateway (`WebSearch()` works on OpenAI
   Responses and Anthropic models; add `local='duckduckgo'` for a model without it).
3. `analysis_agent` writes the report, and can call `extra_search` if the results are thin.

No framework orchestrates this; the control flow is the code you see. In Logfire the
`deep_research` span holds three agent runs, with the search runs overlapping in time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import logfire
from annotated_types import MaxLen
from pydantic import BaseModel
from pydantic_ai import Agent, AgentCapability, format_as_xml
from pydantic_ai.capabilities import WebSearch

from demo_core import configure_logfire, settings


class WebSearchStep(BaseModel):
    """A web search to run, and what we hope to learn from it."""

    search_terms: str


class DeepResearchPlan(BaseModel, use_attribute_docstrings=True):
    """A structured plan for deep research."""

    summary: str
    """A one-paragraph summary of the research plan."""
    web_search_steps: Annotated[list[WebSearchStep], MaxLen(3)]
    """Web searches to run to gather raw information (at most three)."""
    analysis_instructions: str
    """What the analysis step should focus on once the searches are done."""


@dataclass
class ResearchAgents:
    plan: Agent[None, DeepResearchPlan]
    search: Agent[None, str]
    analysis: Agent[None, str]


def build_agents(*, capabilities: Sequence[AgentCapability] = ()) -> ResearchAgents:
    plan: Agent[None, DeepResearchPlan] = Agent(
        settings().model,
        name='plan_agent',
        output_type=DeepResearchPlan,
        instructions='Analyse the user query and design a plan for deep research to answer it.',
        capabilities=list(capabilities),
    )
    search: Agent[None, str] = Agent(
        settings().fast_model,
        name='search_agent',
        instructions='Perform a web search for the given terms and return a concise report with source links.',
        capabilities=[WebSearch(max_uses=3), *capabilities],
    )
    analysis: Agent[None, str] = Agent(
        settings().model,
        name='analysis_agent',
        instructions="""
Analyse the research from the previous steps and write a report on the subject.

If the search results do not contain enough information, run further searches with the
`extra_search` tool. Start with an executive summary, then a concise analysis, and include
links to original sources whenever possible.
""",
        capabilities=list(capabilities),
    )

    @analysis.tool_plain
    async def extra_search(query: str) -> str:
        """Run one more web search when the gathered results are not enough."""
        result = await search.run(query)
        return result.output

    return ResearchAgents(plan=plan, search=search, analysis=analysis)


agents = build_agents()


@logfire.instrument('deep_research {query!r}')
async def deep_research(query: str, *, research: ResearchAgents = agents) -> str:
    plan = (await research.plan.run(query)).output
    logfire.info('plan: {summary}', summary=plan.summary, steps=len(plan.web_search_steps))

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(research.search.run(step.search_terms)) for step in plan.web_search_steps]
    search_results = [task.result().output for task in tasks]

    analysis = await research.analysis.run(
        format_as_xml({'query': query, 'search_results': search_results, 'instructions': plan.analysis_instructions})
    )
    return analysis.output


DEFAULT_QUERY = 'What changed in Python 3.13 free-threading (no-GIL) support, and which libraries have adopted it?'


def main() -> None:
    import sys

    configure_logfire('multi-agent')
    settings().require_model_credentials()
    query = ' '.join(sys.argv[1:]) or DEFAULT_QUERY
    print(asyncio.run(deep_research(query)))


if __name__ == '__main__':
    main()
