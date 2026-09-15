from decimal import Decimal

import logfire
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, FallbackExceptionGroup, ModelAPIError, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.spend import Budget, SpendLimitExceeded, SpendLimits

from gateway_routing import budgets, privacy


class BrokenModel(TestModel):
    async def request(self, *args, **kwargs):
        raise ModelAPIError(self.model_name, 'is unavailable')


def test_fallback_uses_secondary_when_primary_fails():
    fallback = FallbackModel(BrokenModel(), TestModel(custom_output_text='from secondary'))
    agent = Agent(fallback, name='t')
    result = agent.run_sync('hi')
    assert result.output == 'from secondary'


def test_fallback_raises_group_when_all_fail():
    agent = Agent(FallbackModel(BrokenModel(), BrokenModel()), name='t')
    with pytest.raises(FallbackExceptionGroup) as info:
        agent.run_sync('hi')
    assert len(info.value.exceptions) == 2


def test_per_run_request_limit_stops_tool_loop():
    def keep_paging(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('next_page', {'page': 1})])

    with budgets.looping_agent.override(model=FunctionModel(keep_paging)):
        with pytest.raises(UsageLimitExceeded):
            budgets.looping_agent.run_sync('go', deps=budgets.Deps('t', 'u'), usage_limits=UsageLimits(request_limit=3))


def test_per_run_cost_limit():
    def priced(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('next_page', {'page': 1})],
            usage=RequestUsage(input_tokens=10, output_tokens=10, cost=Decimal('0.001')),
        )

    with budgets.looping_agent.override(model=FunctionModel(priced)):
        with pytest.raises(UsageLimitExceeded, match='cost_limit'):
            budgets.looping_agent.run_sync(
                'go', deps=budgets.Deps('t', 'u'), usage_limits=UsageLimits(cost_limit=Decimal('0.0005'))
            )


def priced_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('ok')], usage=RequestUsage(input_tokens=5, output_tokens=5, cost=Decimal('0.0001'))
    )


async def test_tenant_budget_blocks_second_call_only_for_that_tenant():
    agent = Agent(
        FunctionModel(priced_answer),
        name='budgeted_test',
        deps_type=budgets.Deps,
        capabilities=[
            SpendLimits(
                budgets=[Budget(usd=Decimal('0.00005'), window='day', scope=lambda ctx: ctx.deps.tenant)],
                price=lambda response: response.usage.cost,
            )
        ],
    )
    await agent.run('a', deps=budgets.Deps('acme', 'ada'))
    with pytest.raises(SpendLimitExceeded):
        await agent.run('b', deps=budgets.Deps('acme', 'bob'))
    result = await agent.run('c', deps=budgets.Deps('globex', 'cy'))
    assert result.output == 'ok'


async def test_baggage_and_metadata_land_on_spans():
    exporter = InMemorySpanExporter()
    logfire.configure(send_to_logfire=False, console=False, additional_span_processors=[SimpleSpanProcessor(exporter)])
    logfire.instrument_pydantic_ai()
    agent = Agent(FunctionModel(priced_answer), name='attributed_agent', deps_type=budgets.Deps)
    deps = budgets.Deps('acme', 'ada')
    with logfire.set_baggage(tenant=deps.tenant, user=deps.user, feature=deps.feature):
        await agent.run('hi', deps=deps, metadata=deps.__dict__)
    spans = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}
    model_span = next(a for n, a in spans.items() if n.startswith('chat'))
    assert model_span['tenant'] == 'acme'
    assert model_span['user'] == 'ada' and model_span['feature'] == 'summaries'
    agent_span = next(a for n, a in spans.items() if 'attributed_agent' in n)
    assert 'acme' in str(agent_span['metadata'])


def test_input_guardrail_redacts_email_and_phone():
    with privacy.agent.override(model=TestModel(custom_output_text='Done.')):
        result = privacy.agent.run_sync(privacy.PROMPT)
    sent = privacy.prompt_as_sent(result.all_messages())
    assert 'jane.doe@example.com' not in sent
    assert '7946' not in sent
    assert '[redacted:email]' in sent and '[redacted:phone]' in sent


def test_output_guardrail_redacts_model_output():
    with privacy.agent.override(model=TestModel(custom_output_text='Reach me at leak@example.com')):
        result = privacy.agent.run_sync('hi')
    assert result.output == 'Reach me at [redacted:email]'
