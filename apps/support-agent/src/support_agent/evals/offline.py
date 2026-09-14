"""Run the regression dataset against the agent and record an experiment in Logfire.

Every case becomes a span tree in Logfire (Evals -> Experiments), so a failing assertion
links straight to the model calls and tool calls that produced it.
"""

from __future__ import annotations

import asyncio

import logfire
from pydantic_evals import set_eval_attribute
from pydantic_evals.reporting import EvaluationReport

from demo_core import configure_logfire, settings
from support_agent.agent import build_agent
from support_agent.evals.dataset import DATASET_NAME, build_dataset
from support_agent.evals.evaluators import JUDGE_NAME
from support_agent.models import CaseMetadata, SupportRequest, SupportResolution


async def run_offline_eval() -> EvaluationReport[SupportRequest, SupportResolution, CaseMetadata]:
    dataset = build_dataset(include_judge=True)
    # No online evaluation here: the dataset's evaluators score this run.
    agent = build_agent(online_evaluation=False)
    metadata_by_case = {c.inputs.case_id: c.metadata for c in dataset.cases if c.metadata is not None}

    async def resolve_support_request(request: SupportRequest) -> SupportResolution:
        meta = metadata_by_case[request.case_id]
        set_eval_attribute('case_id', request.case_id)
        set_eval_attribute('failure_mode', meta.failure_mode)
        set_eval_attribute('human_label', meta.human_label)
        result = await agent.run(f'Case ID: {request.case_id}\nSupport request: {request.request}')
        set_eval_attribute('requests', result.usage.requests)
        return result.output

    return await dataset.evaluate(
        resolve_support_request,
        task_name='resolve_support_request',
        max_concurrency=2,
        metadata={'dataset': DATASET_NAME, 'judge': JUDGE_NAME, 'model': settings().model},
    )


def main() -> None:
    # The dataset is synthetic and "password" is its subject, so Logfire's default scrubbing
    # would redact case names and inputs in the experiment view. Keep scrubbing on in production.
    configure_logfire('support-agent', environment='offline-eval', scrubbing=False)
    settings().require_model_credentials()

    report = asyncio.run(run_offline_eval())
    report.print(width=120, include_reasons=True)
    if url := logfire.url_from_eval(report):
        print(f'\nExperiment in Logfire: {url}')
    logfire.force_flush()


if __name__ == '__main__':
    main()
