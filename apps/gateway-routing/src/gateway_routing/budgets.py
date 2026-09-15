"""Budgets and cost attribution.

Three layers of cost control, from the model call outwards:

1. `UsageLimits(cost_limit=..., request_limit=...)` on a single run: stops a runaway tool loop
   in one request. Enforced client-side by Pydantic AI, priced with `genai-prices`.
2. `SpendLimits` (Pydantic AI Harness) across runs: a per-tenant daily budget that refuses to
   *start* a request once the window is exhausted. Client-side; the store decides whether it
   survives restarts (in-memory here, Redis in production).
3. Gateway spending limits: daily / weekly / monthly / total caps per API key and per member,
   set in Logfire under Gateway -> API Keys and Gateway -> Spending. Server-side, so they hold
   for every client that uses the key, whatever framework it is written in.

Every run below is tagged with tenant / user / feature, both as agent-run metadata and as
OpenTelemetry baggage, so the cost of each model call can be grouped in Logfire SQL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal

import logfire
from pydantic_ai import Agent, RunContext, UsageLimitExceeded, UsageLimits
from pydantic_ai_harness.spend import Budget, SpendLimitExceeded, SpendLimits

from demo_core import configure_logfire, settings


@dataclass
class Deps:
    tenant: str
    user: str
    feature: str = 'summaries'


# --- 1. Per-run limits -----------------------------------------------------------------------

looping_agent = Agent(
    settings().fast_model,
    name='looping_agent',
    deps_type=Deps,
    instructions=(
        'You are paginating through a report. Call `next_page` and keep calling it until it '
        "returns 'done'. Only then summarise what you read in one sentence."
    ),
)


@looping_agent.tool
def next_page(ctx: RunContext[Deps], page: int = 1) -> str:
    """Return the next page of the report, or 'done' when there are no more pages."""
    if page >= 12:
        return 'done'
    return f'Page {page}: revenue for tenant {ctx.deps.tenant} grew again. Call next_page with page={page + 1}.'


def per_run_limits(deps: Deps) -> None:
    limits = UsageLimits(cost_limit=Decimal('0.0005'), request_limit=6)
    print(f'per-run limits: {limits}')
    try:
        result = looping_agent.run_sync('Read the report.', deps=deps, usage_limits=limits, metadata=deps.__dict__)
        print(f'  finished within limits: {result.usage}')
    except UsageLimitExceeded as exc:
        print(f'  stopped by UsageLimitExceeded: {exc}')


# --- 2. Cross-run, per-tenant budgets --------------------------------------------------------

DAILY_TENANT_BUDGET = Decimal('0.00005')  # tiny on purpose: the first call spends it

budgeted_agent = Agent(
    settings().fast_model,
    name='budgeted_agent',
    deps_type=Deps,
    instructions='Reply in one short sentence.',
    capabilities=[
        SpendLimits(
            budgets=[
                Budget(usd=DAILY_TENANT_BUDGET, window='day', scope=lambda ctx: ctx.deps.tenant, name='tenant-daily'),
                Budget(window='month', scope=lambda ctx: ctx.deps.tenant, name='tenant-monthly-accounting'),
            ],
            on_unpriced='raise',  # a model without pricing data would otherwise count as free
        )
    ],
)


async def tenant_call(deps: Deps, prompt: str) -> None:
    # Baggage copies tenant/user/feature onto every span in this context, including the model
    # request spans that carry `operation.cost`. Run metadata lands on the agent-run span.
    with logfire.set_baggage(tenant=deps.tenant, user=deps.user, feature=deps.feature):
        try:
            result = await budgeted_agent.run(prompt, deps=deps, metadata=deps.__dict__)
            print(f'  [{deps.tenant}/{deps.user}] ok, cost={result.usage.cost}: {result.output[:60]}')
        except SpendLimitExceeded as exc:
            print(f'  [{deps.tenant}/{deps.user}] refused, SpendLimitExceeded: {exc}')


async def per_tenant_budgets() -> None:
    print(f'per-tenant daily budget: ${DAILY_TENANT_BUDGET}')
    await tenant_call(Deps('acme', 'ada'), 'Name one benefit of budgets.')
    await tenant_call(Deps('acme', 'bob'), 'Name another benefit of budgets.')  # same tenant: refused
    await tenant_call(Deps('globex', 'cy'), 'Name one benefit of budgets.')  # other tenant: still runs


def main() -> None:
    configure_logfire('gateway-routing')
    settings().require_model_credentials()

    with logfire.span('budgets demo'):
        print('1) per-run UsageLimits')
        per_run_limits(Deps('acme', 'ada'))
        print('\n2) per-tenant SpendLimits')
        asyncio.run(per_tenant_budgets())
    print('\n3) Gateway limits are set in Logfire (Gateway -> API Keys / Spending); see README.')
    logfire.force_flush()


if __name__ == '__main__':
    main()
