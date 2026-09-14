"""Evaluate a multi-agent system with Pydantic Evals.

The task under test is the whole twenty-questions game. Two evaluators score each case:
did the questioner get it right, and how many questions did it take. Running the same
dataset with `questioner_agent.override(model=...)` gives a like-for-like comparison of
models; each run lands in Logfire as an experiment you can diff against the others.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

import logfire
from pydantic_ai import UsageLimitExceeded
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from demo_core import configure_logfire, settings
from multi_agent.twenty_questions import agents, play

REQUEST_LIMIT = 30


class PlayResult(TypedDict):
    questions: int
    guess: str
    success: bool


@dataclass
class QuestionCount(Evaluator[str, PlayResult]):
    def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> int:
        return ctx.output['questions']


@dataclass
class Success(Evaluator[str, PlayResult]):
    def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> bool:
        return ctx.output['success']


dataset: Dataset[str, PlayResult] = Dataset(
    name='twenty_questions',
    cases=[
        Case(name='Potato', inputs='potato'),
        Case(name='Bicycle', inputs='bicycle'),
        Case(name='Moon', inputs='moon'),
    ],
    evaluators=[QuestionCount(), Success()],
)


async def play_case(answer: str) -> PlayResult:
    try:
        result = await play(answer, request_limit=REQUEST_LIMIT)
    except UsageLimitExceeded:
        logfire.warn('gave up: usage limit exceeded for {answer}', answer=answer)
        return {'questions': REQUEST_LIMIT, 'guess': '', 'success': False}
    return {
        'questions': result.usage.tool_calls,
        'guess': result.output.answer,
        'success': result.output.answer.strip().lower() == answer.lower(),
    }


async def run_evals(models: list[str]) -> None:
    for model in models:
        with agents.questioner.override(model=model):
            report = await dataset.evaluate(play_case, name=f'twenty questions: {model}', max_concurrency=3)
        report.print(include_input=False, include_output=False)
        if url := logfire.url_from_eval(report):
            print(f'Logfire experiment: {url}')


def main() -> None:
    configure_logfire('multi-agent')
    settings().require_model_credentials()
    asyncio.run(run_evals([settings().model, settings().fast_model]))


if __name__ == '__main__':
    main()
