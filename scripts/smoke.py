"""Run every script demo once against the credentials in .env and report pass/fail.

This is the "does a fresh clone still work" check. It makes real model calls (a few cents)
and sends traces to the Logfire project in .env. Demos that need extra infrastructure
(Postgres, Temporal, a browser) are listed separately and only run with --infra.

    uv run python scripts/smoke.py            # script demos only
    uv run python scripts/smoke.py --infra    # also the ones needing docker compose services
    uv run python scripts/smoke.py --only agent-basics
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Demo:
    app: str
    command: list[str]
    needs_infra: bool = False
    timeout: int = 240


DEMOS: list[Demo] = [
    Demo('hello-world', ['hello-world']),
    Demo('agent-basics', ['agent-basics-hello']),
    Demo('agent-basics', ['agent-basics-weather']),
    Demo('agent-basics', ['agent-basics-structured']),
    Demo('agent-basics', ['agent-basics-retry']),
    Demo('instrumentation', ['instrumentation-traffic', '--count', '3', '--serve']),
    Demo('gateway-routing', ['gateway-routing-providers']),
    Demo('gateway-routing', ['gateway-routing-fallback']),
    Demo('gateway-routing', ['gateway-routing-budgets']),
    Demo('gateway-routing', ['gateway-routing-privacy']),
    Demo('rbac', ['rbac-tokens']),
    Demo('multi-agent', ['multi-agent-twenty-questions']),
    Demo('multi-agent', ['multi-agent-research'], timeout=600),
    Demo('multi-agent', ['multi-agent-subagents']),
    Demo('mcp-sampling', ['mcp-sampling-client']),
    Demo('support-agent', ['support-agent-evals-offline'], timeout=600),
    Demo('support-agent', ['support-agent-traffic', '--count', '2', '--interval', '0', '--judge-all']),
    Demo('prompt-optimization', ['prompt-optimization', 'eval']),
    Demo('memory', ['memory-messages'], needs_infra=True),
    Demo('memory', ['memory-tools'], needs_infra=True),
    Demo('durable-exec', ['durable-exec-dbos-twenty-questions'], needs_infra=True, timeout=600),
    Demo('durable-exec', ['durable-exec-temporal-twenty-questions'], needs_infra=True, timeout=600),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--infra', action='store_true', help='also run demos that need docker compose services')
    parser.add_argument('--only', help='run only demos whose app name contains this string')
    args = parser.parse_args()

    selected = [d for d in DEMOS if (args.infra or not d.needs_infra) and (not args.only or args.only in d.app)]
    results: list[tuple[Demo, bool, float, str]] = []
    for demo in selected:
        name = ' '.join(demo.command)
        print(f'\n=== {name}', flush=True)
        start = time.monotonic()
        try:
            proc = subprocess.run(['uv', 'run', *demo.command], timeout=demo.timeout, capture_output=True, text=True)
            ok, tail = proc.returncode == 0, (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            ok, tail = False, ['timed out']
        elapsed = time.monotonic() - start
        print('\n'.join(tail))
        results.append((demo, ok, elapsed, tail[-1] if tail else ''))

    print('\n' + '=' * 72)
    width = max(len(' '.join(d.command)) for d, *_ in results) if results else 10
    for demo, ok, elapsed, last in results:
        print(f'{"PASS" if ok else "FAIL"}  {" ".join(demo.command):<{width}}  {elapsed:6.1f}s  {last[:60]}')
    failed = [d for d, ok, *_ in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
