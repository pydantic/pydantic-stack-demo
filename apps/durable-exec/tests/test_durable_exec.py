"""Outside a workflow the durability capabilities are transparent, so the agents can be
exercised with `TestModel` without a DBOS database or a Temporal server."""

from pydantic_ai.models.test import TestModel

from durable_exec import dbos_twenty_questions, temporal_research, temporal_twenty_questions
from multi_agent.twenty_questions import GameState


async def test_dbos_agents_run_outside_workflow():
    game = dbos_twenty_questions.game
    with game.questioner.override(model=TestModel()), game.answerer.override(model=TestModel()):
        result = await game.questioner.run('start', deps=GameState(answer='potato'))
    assert result.output.answer


async def test_temporal_agents_run_outside_workflow():
    game = temporal_twenty_questions.game
    with game.questioner.override(model=TestModel()), game.answerer.override(model=TestModel()):
        result = await game.questioner.run('start', deps=GameState(answer='potato'))
    assert result.output.answer


def test_workflows_declare_their_agents():
    assert temporal_twenty_questions.TwentyQuestionsWorkflow.__pydantic_ai_agents__ == [
        temporal_twenty_questions.game.questioner,
        temporal_twenty_questions.game.answerer,
    ]
    assert len(temporal_research.DeepResearchWorkflow.__pydantic_ai_agents__) == 3
