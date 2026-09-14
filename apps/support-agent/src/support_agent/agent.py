"""The reference support agent.

Deliberately small: one model, three policies behind a lookup tool, one escalation tool,
and a structured output. The interesting part is what wraps it: the offline dataset in
`evals/` and the `OnlineEvaluation` capability attached below, which scores a sample of
real runs in the background and sends the results to Logfire's Live Monitoring view.
"""

from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent
from pydantic_evals.online import OnlineEvalConfig, OnlineEvaluator
from pydantic_evals.online_capability import OnlineEvaluation

from demo_core import settings
from support_agent.evals.evaluators import NoSensitiveDataLeaked, ResolutionShape, online_judge
from support_agent.models import SupportResolution

AGENT_NAME = 'support_resolution_agent'

SUPPORT_INSTRUCTIONS = """\
You resolve support requests using only the supplied policy tools.

Always look up the relevant policy before answering. Give a concrete, safe next
step that an authorized support operator can realistically perform. Never expose
credentials, account secrets, or private billing data. Escalate urgent cases when
the policy calls for it, but never bypass identity or permission checks. Do not
claim an action completed unless a tool confirms it. Set `escalated` to true only
if you called queue_urgent_escalation.
"""

PolicyArea = Literal['password_reset', 'billing_export', 'account_lockout']

SUPPORT_POLICIES: dict[str, str] = {
    'password_reset': (
        'Use the official reset flow and require normal identity verification. '
        'Never reveal, set, or request a password. For a time-sensitive VIP '
        'launch blocker, queue an urgent identity-support escalation while the '
        'user follows the safe reset path; do not bypass verification.'
    ),
    'billing_export': (
        'A verified finance admin may export invoices from Billing > Invoices '
        'after selecting the requested date range. Confirm the account and '
        'quarter, return the export through the authorized workspace, and never '
        'request or expose payment credentials or account secrets. Users without '
        'the finance admin role must ask a finance admin; do not export on their behalf.'
    ),
    'account_lockout': (
        'An MFA lockout is resolved through the identity team after verification. '
        'If the locked-out user is on call during an active incident, queue an urgent '
        'escalation to the identity team; never disable MFA or issue bypass codes yourself.'
    ),
}


def build_agent(*, online_evaluation: bool = True, sample_rate: float = 0.1) -> Agent[None, SupportResolution]:
    """Build the agent, optionally with online evaluation attached.

    The offline eval script builds it with `online_evaluation=False` so an experiment run
    is scored once (by the dataset), not twice.
    """
    capabilities = []
    if online_evaluation:
        capabilities.append(
            OnlineEvaluation(
                evaluators=[
                    # Cheap, deterministic checks run on every call.
                    OnlineEvaluator(evaluator=NoSensitiveDataLeaked(), sample_rate=1.0),
                    OnlineEvaluator(evaluator=ResolutionShape(), sample_rate=1.0),
                    # The LLM judge costs a model call, so it is sampled and bounded.
                    OnlineEvaluator(evaluator=online_judge(), sample_rate=sample_rate, max_concurrency=3),
                ],
                config=OnlineEvalConfig(metadata={'app': 'support-agent', 'judge_sample_rate': sample_rate}),
            )
        )

    agent = Agent(
        settings().model,
        name=AGENT_NAME,
        output_type=SupportResolution,
        instructions=SUPPORT_INSTRUCTIONS,
        capabilities=capabilities,
    )

    @agent.tool_plain
    def lookup_support_policy(policy_area: PolicyArea) -> str:
        """Return the approved policy for the support request category."""
        return SUPPORT_POLICIES[policy_area]

    @agent.tool_plain
    def queue_urgent_escalation(case_id: str, reason: str) -> str:
        """Queue an urgent escalation without bypassing identity checks."""
        return f'Urgent escalation queued for {case_id}: {reason}'

    return agent


support_resolution_agent = build_agent()
