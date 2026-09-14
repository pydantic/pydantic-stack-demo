"""The regression dataset: `eval_assets/dataset.jsonl` loaded into a typed `Dataset`.

The JSONL file is the source of truth so a case can be added with a text editor (or by
`support-agent-evals-publish` round-tripping through Logfire's managed datasets).
"""

from __future__ import annotations

from pydantic_evals import Case, Dataset

from demo_core.evals import read_jsonl
from support_agent.evals.evaluators import (
    EVAL_ASSETS,
    EscalationMatchesExpectation,
    NoSensitiveDataLeaked,
    ResolutionShape,
    offline_judge,
)
from support_agent.models import CaseMetadata, SupportRequest, SupportResolution

DATASET_NAME = 'support_resolution_failures'
DATASET_PATH = EVAL_ASSETS / 'dataset.jsonl'

SupportDataset = Dataset[SupportRequest, SupportResolution, CaseMetadata]


def load_cases() -> list[Case[SupportRequest, SupportResolution, CaseMetadata]]:
    cases: list[Case[SupportRequest, SupportResolution, CaseMetadata]] = []
    for row in read_jsonl(DATASET_PATH):
        expects_escalation = bool(row.get('expects_escalation', False))
        cases.append(
            Case(
                name=row['case_id'],
                inputs=SupportRequest(case_id=row['case_id'], request=row['input']),
                expected_output=SupportResolution(resolution=row['expected'], escalated=expects_escalation),
                metadata=CaseMetadata(
                    failure_mode=row['failure_mode'],
                    expects_escalation=expects_escalation,
                    human_label=row.get('human_label', 'unreviewed'),
                    regression_candidate=row.get('regression_candidate', True),
                ),
            )
        )
    return cases


def build_dataset(*, include_judge: bool = True, judge_model: str | None = None) -> SupportDataset:
    evaluators = [NoSensitiveDataLeaked(), ResolutionShape(), EscalationMatchesExpectation()]
    if include_judge:
        evaluators.append(offline_judge(judge_model))
    return SupportDataset(name=DATASET_NAME, cases=load_cases(), evaluators=evaluators)
