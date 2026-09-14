from pydantic_evals.evaluators import LLMJudge
from support_agent.evals.dataset import DATASET_NAME, build_dataset, load_cases
from support_agent.evals.evaluators import JUDGE_NAME


def test_dataset_loads_all_cases():
    cases = load_cases()
    assert len(cases) == 7
    assert len({c.name for c in cases}) == 7
    assert {c.metadata.failure_mode for c in cases if c.metadata} >= {'missed_urgency', 'credential_exposure'}


def test_dataset_has_judge_by_default():
    ds = build_dataset()
    assert ds.name == DATASET_NAME
    judges = [e for e in ds.evaluators if isinstance(e, LLMJudge)]
    assert len(judges) == 1
    assert judges[0].assertion and judges[0].assertion.get('evaluation_name') == JUDGE_NAME
    assert not [e for e in build_dataset(include_judge=False).evaluators if isinstance(e, LLMJudge)]
