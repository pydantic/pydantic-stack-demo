"""Delegation with the Harness `SubAgents` capability.

Same idea as `twenty_questions.py` (a parent agent calling child agents) with none of the
plumbing: `SubAgents` exposes one `delegate_task(agent_name, task)` tool, lists the
delegates in the system prompt, forwards `deps`, and shares usage so one `UsageLimits`
bounds the whole tree. Each delegation is an isolated run; the child never sees the parent
conversation. A tiny capability listens for the delegation events and prints them.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai_harness.subagents import DelegationEndEvent, DelegationStartEvent, SubAgent, SubAgents

from demo_core import configure_logfire, settings

researcher = Agent(
    settings().fast_model,
    name='researcher',
    description='Collects the key facts about a topic as terse bullet points.',
    instructions='Reply with at most five bullet points of facts. No prose.',
)

editor = Agent(
    settings().fast_model,
    name='editor',
    description='Turns bullet points into two polished paragraphs for a general audience.',
    instructions='Rewrite the notes you are given as two short, friendly paragraphs.',
)


class PrintDelegations(AbstractCapability[Any]):
    """Observe delegations without touching the agent logic."""

    @on_event(DelegationStartEvent)
    async def started(self, ctx: RunContext[Any], event: DelegationStartEvent) -> None:
        print(f'-> delegating to {event.agent_name}: {event.task[:60]!r}')

    @on_event(DelegationEndEvent)
    async def ended(self, ctx: RunContext[Any], event: DelegationEndEvent) -> None:
        print(f'<- {event.agent_name}: {event.outcome} in {event.duration_seconds:.1f}s')


orchestrator = Agent(
    settings().model,
    name='orchestrator',
    instructions="Use the researcher first, then hand its notes to the editor, and return the editor's text verbatim.",
    capabilities=[
        SubAgents(
            agents=[
                SubAgent(researcher, usage_limits=UsageLimits(request_limit=3)),
                SubAgent(editor, usage_limits=UsageLimits(request_limit=3)),
            ]
        ),
        PrintDelegations(),
    ],
)


def main() -> None:
    configure_logfire('multi-agent')
    settings().require_model_credentials()

    result = orchestrator.run_sync(
        'Explain why Pydantic validates data at runtime, for someone new to Python.',
        usage_limits=UsageLimits(request_limit=12),
    )
    print()
    print(result.output)
    print(f'\nusage across the whole tree: {result.usage}')


if __name__ == '__main__':
    main()
