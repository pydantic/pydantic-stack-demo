from pydantic_ai.models.test import TestModel

from support_agent.agent import SUPPORT_POLICIES, build_agent, support_resolution_agent
from support_agent.models import SupportResolution


def test_agent_uses_tools_and_returns_structured_output():
    with support_resolution_agent.override(model=TestModel()):
        result = support_resolution_agent.run_sync('Case ID: x\nSupport request: locked out')
    assert isinstance(result.output, SupportResolution)
    tool_names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
    assert 'lookup_support_policy' in tool_names
    assert 'queue_urgent_escalation' in tool_names


def test_build_agent_without_online_evaluation():
    agent = build_agent(online_evaluation=False)
    with agent.override(model=TestModel()):
        assert isinstance(agent.run_sync('x').output, SupportResolution)


def test_policies_cover_tool_literal():
    assert set(SUPPORT_POLICIES) == {'password_reset', 'billing_export', 'account_lockout'}
