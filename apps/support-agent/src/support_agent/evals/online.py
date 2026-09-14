"""Send production-shaped traffic through the agent and watch the online evaluators fire.

Each request is wrapped in a span that looks like a real service call, so the trace in
Logfire is the same shape you would see from `support-agent-serve`. The judge normally
samples 10% of runs; `--judge-all` forces 100% so a live demo shows results immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

import logfire
from pydantic_evals.online import wait_for_evaluations

from demo_core import configure_logfire, settings
from support_agent.agent import AGENT_NAME, build_agent
from support_agent.evals.dataset import load_cases
from support_agent.evals.evaluators import JUDGE_NAME


async def send_traffic(*, count: int, interval: float, sample_rate: float) -> int:
    agent = build_agent(sample_rate=sample_rate)
    cases = load_cases()
    for i in range(count):
        case = cases[i % len(cases)]
        request_id = uuid.uuid4().hex
        attributes = {
            'request_id': request_id,
            'case_id': case.inputs.case_id,
            'http.request.method': 'POST',
            'http.route': '/support/resolve',
            'http.response.status_code': 200,
        }
        with logfire.span('POST /support/resolve', **attributes):
            result = await agent.run(
                f'Case ID: {case.inputs.case_id}\nSupport request: {case.inputs.request}',
                metadata=attributes,
            )
            logfire.info('resolved {case_id}', case_id=case.inputs.case_id, escalated=result.output.escalated)
        print(
            f'[{i + 1}/{count}] request_id={request_id[:8]} case={case.inputs.case_id} '
            f'escalated={result.output.escalated}'
        )
        if interval and i + 1 < count:
            await asyncio.sleep(interval)

    # Evaluators run in the background; give them time to finish before the process exits.
    await wait_for_evaluations(timeout=90)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate paced traffic for the online evaluators.')
    parser.add_argument('--count', type=int, default=6, help='requests to send (default 6)')
    parser.add_argument('--interval', type=float, default=2.0, help='seconds between requests (default 2)')
    parser.add_argument('--judge-all', action='store_true', help='judge every request instead of a 10%% sample')
    args = parser.parse_args()

    # Synthetic data whose subject is "password": see offline.py for why scrubbing is off here.
    configure_logfire('support-agent', environment='online-eval', scrubbing=False)
    settings().require_model_credentials()

    sample_rate = 1.0 if args.judge_all else 0.1
    print(f'Sending {args.count} requests every {args.interval:g}s; judge sample rate {sample_rate:.0%}.')
    sent = asyncio.run(send_traffic(count=args.count, interval=args.interval, sample_rate=sample_rate))
    print(
        f'Sent {sent} requests. In Logfire open Evals -> Live Monitoring, '
        f'target {AGENT_NAME!r}, evaluator {JUDGE_NAME!r}.'
    )
    logfire.force_flush()


if __name__ == '__main__':
    main()
