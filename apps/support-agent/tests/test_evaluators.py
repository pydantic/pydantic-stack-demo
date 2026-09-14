from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from support_agent.evals.evaluators import EscalationMatchesExpectation, NoSensitiveDataLeaked, ResolutionShape
from support_agent.models import CaseMetadata, SupportRequest, SupportResolution


def ctx(output: SupportResolution, metadata: CaseMetadata | None = None) -> EvaluatorContext:
    return EvaluatorContext(
        name='t',
        inputs=SupportRequest(case_id='t', request='x'),
        metadata=metadata,
        expected_output=None,
        output=output,
        duration=0.1,
        _span_tree=None,  # pyright: ignore[reportArgumentType]
        attributes={},
        metrics={},
    )


def test_no_secrets_leaked():
    ok = NoSensitiveDataLeaked().evaluate(
        ctx(SupportResolution(resolution='Use the official reset flow after verification.'))
    )
    assert ok.value is True
    bad = NoSensitiveDataLeaked().evaluate(ctx(SupportResolution(resolution='Your password is hunter2, log in now.')))
    safe = NoSensitiveDataLeaked().evaluate(ctx(SupportResolution(resolution='Never issue bypass codes; escalate.')))
    assert safe.value is True
    assert bad.value is False and 'password' in (bad.reason or '')


def test_resolution_shape():
    out = ResolutionShape().evaluate(ctx(SupportResolution(resolution='short')))
    has_next_step = out['has_next_step']
    assert isinstance(has_next_step, EvaluationReason) and has_next_step.value is False
    assert out['resolution_chars'] == 5
    assert 'escalation_flag_consistent' not in out  # no span tree available offline


def test_escalation_matches_expectation_falls_back_to_output_flag():
    ev = EscalationMatchesExpectation()
    meta = CaseMetadata(failure_mode='missed_urgency', expects_escalation=True)
    assert ev.evaluate(ctx(SupportResolution(resolution='x', escalated=True), meta)).value is True
    assert ev.evaluate(ctx(SupportResolution(resolution='x', escalated=False), meta)).value is False
