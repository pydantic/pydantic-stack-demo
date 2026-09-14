import json

from prompt_optimization.adapter import EvalsGEPAAdapter
from prompt_optimization.evals import FieldAccuracyEvaluator, contact_dataset
from prompt_optimization.task import ContactInfo, TaskInput, contact_agent, extract_contact_info
from pydantic_ai.models.test import TestModel
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel.span_tree import SpanTree


def test_field_accuracy_scores_partial_matches():
    ctx = EvaluatorContext(
        name='c',
        inputs=TaskInput(text='x'),
        output=ContactInfo(name='john smith', email='wrong@example.com'),
        expected_output=ContactInfo(name='John Smith', email='john@example.com'),
        metadata=None,
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )
    result = FieldAccuracyEvaluator().evaluate(ctx)
    assert result['accuracy'] == 0.5
    assert result['name_correct'] and not result['email_correct']


def test_adapter_evaluates_and_reflects_offline():
    adapter = EvalsGEPAAdapter(dataset=contact_dataset, task=extract_contact_info, agent=contact_agent)
    with contact_agent.override(model=TestModel()), adapter._proposer.override(model=TestModel()):
        batch = adapter.evaluate(contact_dataset.cases[:2], {'instructions': json.dumps('x')}, capture_traces=True)
        assert len(batch.scores) == 2 and batch.trajectories is not None
        reflective = adapter.make_reflective_dataset({'instructions': json.dumps('x')}, batch, ['instructions'])
        assert len(reflective['instructions']) == 2
        proposal = adapter._propose_new_texts({'instructions': json.dumps('x')}, reflective, ['instructions'])
    assert isinstance(json.loads(proposal['instructions']), str)
