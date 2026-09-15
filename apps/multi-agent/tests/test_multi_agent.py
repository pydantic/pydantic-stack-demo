from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from multi_agent import research, subagents, twenty_questions
from multi_agent.evals import QuestionCount, Success, dataset, play_case


async def test_twenty_questions_tool_delegates_to_answerer():
    game = twenty_questions.build_agents(questioner_model='test', answerer_model='test')
    result = await twenty_questions.play('potato', game=game, request_limit=5)
    tool_calls = [p for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
    assert tool_calls and tool_calls[0].tool_name == 'ask_question'
    assert result.usage.requests > 1  # the answerer's request was charged to the parent


async def test_research_pipeline_runs_offline():
    from pydantic_ai import ModelMessage, ModelResponse, TextPart

    def plan_then_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # `TestModel` cannot use the native web search tool, so drive the pipeline with a
        # FunctionModel that returns the structured plan then plain text.
        return ModelResponse(parts=[TextPart('a report')])

    agents = research.build_agents()
    with (
        agents.plan.override(model=TestModel()),
        agents.search.override(model=FunctionModel(plan_then_text)),
        agents.analysis.override(model=FunctionModel(plan_then_text)),
    ):
        out = await research.deep_research('anything', research=agents)
    assert out == 'a report'


def test_subagents_delegate_task_routes_to_named_agent():
    import json
    from collections.abc import AsyncIterator

    from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls

    delegate_args = {'agent_name': 'researcher', 'task': 'facts'}

    def orchestrate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # First turn: delegate to the researcher; second turn: answer.
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('delegate_task', delegate_args)])
        return ModelResponse(parts=[TextPart('done')])

    async def orchestrate_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='delegate_task', json_args=json.dumps(delegate_args))}
        else:
            yield 'done'

    with (
        subagents.orchestrator.override(model=FunctionModel(orchestrate, stream_function=orchestrate_stream)),
        subagents.researcher.override(model=TestModel(custom_output_text='notes')),
        subagents.editor.override(model=TestModel(custom_output_text='prose')),
    ):
        result = subagents.orchestrator.run_sync('go')
    returns = [p for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-return']
    assert result.output == 'done'
    assert returns and 'notes' in str(returns[0].content)


async def test_evaluators_score_play_result():
    with twenty_questions.agents.questioner.override(model=TestModel()):
        out = await play_case('potato')
    assert set(out) == {'questions', 'guess', 'success'}
    report = await dataset.evaluate(play_case, max_concurrency=1, progress=False)
    names = {r.name for c in report.cases for r in [*c.scores.values(), *c.assertions.values()]}
    assert names == {QuestionCount.__name__, Success.__name__}
