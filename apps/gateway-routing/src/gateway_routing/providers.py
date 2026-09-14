"""One Gateway key, many providers.

The same prompt goes to OpenAI, Anthropic and Google through the Pydantic AI Gateway using a
single `PYDANTIC_AI_GATEWAY_API_KEY`. No provider keys live in this process; the Gateway holds
them (or bills built-in providers to the organization's balance), enforces the key's spending
limits, and records every request in Logfire.

If `GATEWAY_ROUTE` is set to the slug of a gateway *endpoint* (Gateway -> Endpoints in Logfire),
the last call goes through that endpoint instead of a provider slug. Endpoints are where
failover (priority) and load balancing (weight) between providers are configured, server-side.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import logfire
from pydantic_ai import Agent

from demo_core import configure_logfire, gateway_model, settings

PROMPT = 'In one sentence: what is an AI gateway for?'

# (api_format, model). The api_format is the wire format the Gateway speaks for that route.
PROVIDERS: list[tuple[str, str]] = [
    ('openai', 'gpt-5.2'),
    ('anthropic', 'claude-haiku-4-5'),
    ('google-cloud', 'gemini-3-flash-preview'),
]


@dataclass
class Probe:
    route: str
    model: str
    latency_s: float
    cost_usd: str
    answer: str


async def ask(agent: Agent, *, route: str, model_label: str) -> Probe:
    started = time.perf_counter()
    result = await agent.run(PROMPT)
    latency = time.perf_counter() - started
    return Probe(
        route=route,
        model=result.response.model_name or model_label,
        latency_s=latency,
        cost_usd=f'{result.usage.cost:.6f}' if result.usage.cost is not None else 'n/a',
        answer=result.output.strip(),
    )


async def run() -> list[Probe]:
    probes: list[Probe] = []
    with logfire.span('gateway providers probe'):
        for api_format, model_name in PROVIDERS:
            # The `gateway/<api_format>:<model>` string is all it takes; the key comes from the environment.
            agent = Agent(f'gateway/{api_format}:{model_name}', name='routing_probe', instructions='Be brief.')
            probes.append(await ask(agent, route=api_format, model_label=model_name))

        if route := os.environ.get('GATEWAY_ROUTE'):
            api_format = os.environ.get('GATEWAY_ROUTE_FORMAT', 'anthropic')
            model_name = os.environ.get('GATEWAY_ROUTE_MODEL', 'claude-haiku-4-5')
            agent = Agent(
                gateway_model(api_format, model_name, route=route),
                name='routing_probe_endpoint',
                instructions='Be brief.',
            )
            probes.append(await ask(agent, route=f'endpoint:{route}', model_label=model_name))
    return probes


def main() -> None:
    configure_logfire('gateway-routing')
    settings().require_model_credentials()

    probes = asyncio.run(run())
    print(f'{"route":<28}{"model":<28}{"latency":>9}{"cost USD":>11}  answer')
    for p in probes:
        print(f'{p.route:<28}{p.model:<28}{p.latency_s:>8.2f}s{p.cost_usd:>11}  {p.answer[:70]}')

    if not os.environ.get('GATEWAY_ROUTE'):
        print(
            '\nTo route through a gateway endpoint (failover / load balancing configured in Logfire):\n'
            '  Logfire -> Gateway -> Endpoints -> New Endpoint, add providers with priority and weight,\n'
            '  then: GATEWAY_ROUTE=<endpoint-slug> GATEWAY_ROUTE_FORMAT=anthropic '
            'GATEWAY_ROUTE_MODEL=claude-haiku-4-5 uv run gateway-routing-providers'
        )
    logfire.force_flush()


if __name__ == '__main__':
    main()
