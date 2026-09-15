"""GEPA adapter that evaluates candidate prompts with pydantic-evals.

GEPA (Genetic-Pareto prompt evolution) needs three things from an adapter: score a candidate
on a batch of cases, turn the results into feedback, and propose new candidates from that
feedback. Scoring and feedback come straight from a pydantic-evals report; the proposal is
one Pydantic AI agent acting as a prompt engineer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python
from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import ReportCase, ReportCaseFailure

from demo_core import settings

PROPOSER_INSTRUCTIONS = """You are an expert prompt engineer. Your task is to improve system prompts
for AI agents based on evaluation feedback.

You will receive the current instructions and examples of inputs, outputs and feedback from
evaluation. Analyse what went wrong and propose improved instructions that increase accuracy,
handle edge cases better, and are clear and specific.

Return ONLY the improved instructions text, nothing else."""


@dataclass
class EvalTrajectory[InputsT, OutputT, MetadataT]:
    """What GEPA reflects on: the pydantic-evals report for one case."""

    report_case: ReportCase[InputsT, OutputT, MetadataT] | ReportCaseFailure[InputsT, OutputT, MetadataT]


@dataclass
class EvalsGEPAAdapter[InputsT, OutputT, MetadataT](
    GEPAAdapter[Case[InputsT, OutputT, MetadataT], EvalTrajectory[InputsT, OutputT, MetadataT], OutputT | None]
):
    """Bridge between GEPA and a pydantic-evals `Dataset`.

    The candidate dict has a single key, `instructions`, holding the JSON-encoded prompt.
    """

    dataset: Dataset[InputsT, OutputT, MetadataT]
    task: Callable[[InputsT], Awaitable[OutputT]]
    agent: Agent[Any, Any]
    score_key: str = 'accuracy'
    max_concurrency: int = 5
    proposer_model: str | None = None
    _proposer: Agent[None, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._proposer = Agent(
            self.proposer_model or settings().model,
            name='prompt_proposer',
            output_type=str,
            instructions=PROPOSER_INSTRUCTIONS,
        )
        # GEPA looks for this attribute rather than a method.
        self.propose_new_texts = self._propose_new_texts

    def evaluate(
        self,
        batch: list[Case[InputsT, OutputT, MetadataT]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EvalTrajectory[InputsT, OutputT, MetadataT], OutputT | None]:
        instructions = json.loads(candidate['instructions'])
        batch_dataset: Dataset[InputsT, OutputT, MetadataT] = Dataset(
            name=self.dataset.name or 'gepa_batch', cases=batch, evaluators=list(self.dataset.evaluators)
        )

        with self.agent.override(instructions=instructions):
            report = asyncio.run(
                batch_dataset.evaluate(self.task, max_concurrency=self.max_concurrency, progress=False)
            )

        outputs: list[OutputT | None] = []
        scores: list[float] = []
        trajectories: list[EvalTrajectory[InputsT, OutputT, MetadataT]] | None = [] if capture_traces else None

        for case in report.cases:
            outputs.append(case.output)
            score = case.scores.get(self.score_key) or next(iter(case.scores.values()), None)
            scores.append(float(score.value) if score is not None else 0.0)
            if trajectories is not None:
                trajectories.append(EvalTrajectory(report_case=case))

        for failure in report.failures:
            outputs.append(None)
            scores.append(0.0)
            if trajectories is not None:
                trajectories.append(EvalTrajectory(report_case=failure))

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EvalTrajectory[InputsT, OutputT, MetadataT], OutputT | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {}

        examples: list[dict[str, Any]] = []
        for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores, strict=True):
            case = trajectory.report_case
            record: dict[str, Any] = {
                'case_name': case.name,
                'inputs': to_jsonable_python(case.inputs),
                'expected_output': to_jsonable_python(case.expected_output),
                'score': score,
            }
            if isinstance(case, ReportCase):
                record['actual_output'] = to_jsonable_python(case.output)
                record['scores'] = {k: v.value for k, v in case.scores.items()}
                record['assertions'] = [
                    {'name': a.name, 'passed': a.value, 'reason': a.reason} for a in case.assertions.values()
                ]
            else:
                record['error'] = case.error_message
            examples.append(record)

        return {'instructions': examples}

    def _propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        current = json.loads(candidate['instructions'])
        examples = reflective_dataset.get('instructions', [])
        if not examples:
            return candidate

        examples_text = '\n\n'.join(
            f'Example {i + 1}:\n'
            f'  Input: {json.dumps(ex.get("inputs"))}\n'
            f'  Expected: {json.dumps(ex.get("expected_output"))}\n'
            f'  Actual: {json.dumps(ex.get("actual_output"))}\n'
            f'  Score: {ex.get("score", 0):.2f}\n'
            f'  Feedback: {json.dumps(ex.get("scores", {}))}'
            for i, ex in enumerate(examples[:10])
        )
        prompt = f"""Current Instructions:
{current}

Evaluation Results:
{examples_text}

Based on this feedback, propose improved instructions that will increase accuracy. Focus on the
patterns that led to incorrect outputs, on making the instructions clearer and more specific, and
on the edge cases that need handling. Respond with ONLY the new instructions text."""

        result = self._proposer.run_sync(prompt)
        return {'instructions': json.dumps(result.output)}
