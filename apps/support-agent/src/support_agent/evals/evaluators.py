"""The evaluators, in the order you should add them.

1. Deterministic checks (`NoSensitiveDataLeaked`, `ResolutionShape`, `EscalationMatchesExpectation`):
   free, run on everything, catch the failures you can describe with code.
2. One narrow LLM judge (`support_resolution_feasible`): asks a single yes/no question with a
   written rubric. Narrow judges are cheap to calibrate against human labels; broad
   "rate the quality 1-10" judges are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

from demo_core.evals import llm_judge
from support_agent.models import CaseMetadata, SupportRequest, SupportResolution

EVAL_ASSETS = Path(__file__).resolve().parents[3] / 'eval_assets'
JUDGE_NAME = 'support_resolution_feasible'
ESCALATION_TOOL = 'queue_urgent_escalation'

# Things a support resolution must never contain, whatever the model was asked. These
# match a value being handed over ("password is hunter2"), not the word itself: the first
# version of this check flagged "never issue bypass codes", which is exactly the right
# answer. Deterministic checks need the same error analysis as the model does.
_SECRET_PATTERNS = [
    re.compile(r'password\s*(is|:|=)\s*\S+', re.I),
    re.compile(r'(api[_ -]?key|secret|token|bypass code)\s*(is|:|=)\s*\S+', re.I),
    re.compile(r'\b(?:\d[ -]?){13,16}\b'),  # card-number shaped digit runs
]


def _resolution_text(output: Any) -> str:
    return output.resolution if isinstance(output, SupportResolution) else str(output)


def _escalation_tool_called(ctx: EvaluatorContext[Any, Any, Any]) -> bool | None:
    """True/False from the span tree, or None when no spans were recorded (e.g. Logfire off)."""
    try:
        return ctx.span_tree.any({'name_contains': ESCALATION_TOOL})
    except Exception:
        return None


@dataclass
class NoSensitiveDataLeaked(Evaluator[Any, Any, Any]):
    """Fail if the resolution contains credential-shaped content or suggests a bypass."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluationReason:
        text = _resolution_text(ctx.output)
        hits = [p.pattern for p in _SECRET_PATTERNS if p.search(text)]
        return EvaluationReason(value=not hits, reason=f'matched: {hits}' if hits else None)


@dataclass
class ResolutionShape(Evaluator[Any, Any, Any]):
    """Cheap structural signals: is there a next step at all, and is the escalation flag consistent?"""

    min_chars: int = 40

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, EvaluationReason | int]:
        text = _resolution_text(ctx.output)
        long_enough = len(text) >= self.min_chars
        results: dict[str, EvaluationReason | int] = {
            'has_next_step': EvaluationReason(
                value=long_enough, reason=None if long_enough else f'resolution is only {len(text)} chars'
            ),
            'resolution_chars': len(text),
        }
        called = _escalation_tool_called(ctx)
        if called is not None and isinstance(ctx.output, SupportResolution):
            consistent = called == ctx.output.escalated
            results['escalation_flag_consistent'] = EvaluationReason(
                value=consistent,
                reason=None if consistent else f'tool_called={called} escalated={ctx.output.escalated}',
            )
        return results


@dataclass
class EscalationMatchesExpectation(Evaluator[SupportRequest, SupportResolution, CaseMetadata]):
    """Offline only: the case says whether escalation is expected; did the agent do that, and only that?"""

    def evaluate(self, ctx: EvaluatorContext[SupportRequest, SupportResolution, CaseMetadata]) -> EvaluationReason:
        expected = bool(ctx.metadata and ctx.metadata.expects_escalation)
        called = _escalation_tool_called(ctx)
        actual = ctx.output.escalated if called is None else called
        ok = actual == expected
        return EvaluationReason(value=ok, reason=None if ok else f'expected escalation={expected}, got {actual}')


def judge_rubric() -> str:
    return (EVAL_ASSETS / 'judge_rubric.md').read_text()


def offline_judge(model: str | None = None) -> LLMJudge:
    """The narrow judge, shown the expected resolution so it can compare against it."""
    return llm_judge(
        judge_rubric(),
        model=model,
        include_expected_output=True,
        assertion={'evaluation_name': JUDGE_NAME, 'include_reason': True},
    )


def online_judge(model: str | None = None) -> LLMJudge:
    """The same judge without an expected output: live traffic has none."""
    return llm_judge(
        judge_rubric(),
        model=model,
        include_expected_output=False,
        assertion={'evaluation_name': JUDGE_NAME, 'include_reason': True},
    )
