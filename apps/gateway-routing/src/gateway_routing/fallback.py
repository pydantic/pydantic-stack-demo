"""Client-side failover with `FallbackModel`.

`FallbackModel(primary, secondary, ...)` tries each model in order and moves on when the
current one raises a `ModelAPIError` (4xx/5xx). Here the primary is deliberately broken (a
model name the Gateway does not know), so every run is answered by the secondary. When *all*
candidates fail, a `FallbackExceptionGroup` carries every underlying error.

Compare with a Gateway endpoint (see `providers.py`): an endpoint fails over server-side, for
every client of that key, with no code change. `FallbackModel` is the right tool when the
fallback crosses providers the endpoint cannot group, or when the app itself must decide
(different prompts per model, different output types, cost-aware ordering).
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent, FallbackExceptionGroup
from pydantic_ai.models.fallback import FallbackModel

from demo_core import configure_logfire, gateway_model, settings


def build_agent() -> Agent:
    broken_primary = gateway_model('openai', 'no-such-model-for-demo')
    # The secondary is a plain model string; FallbackModel accepts models or model strings.
    fallback = FallbackModel(broken_primary, settings().fast_model)
    return Agent(fallback, name='fallback_agent', instructions='Answer in one short sentence.')


def build_all_broken_agent() -> Agent:
    fallback = FallbackModel(
        gateway_model('openai', 'no-such-model-for-demo'),
        gateway_model('anthropic', 'also-not-a-model'),
    )
    return Agent(fallback, name='fallback_agent_all_broken', instructions='Answer in one short sentence.')


def main() -> None:
    configure_logfire('gateway-routing')
    settings().require_model_credentials()

    with logfire.span('fallback demo'):
        result = build_agent().run_sync('Why would an app keep a fallback model?')
        print(f'answered by: {result.response.model_name}')
        print(result.output)

        print('\nNow with every candidate broken:')
        try:
            build_all_broken_agent().run_sync('Anyone there?')
        except FallbackExceptionGroup as group:
            print(f'FallbackExceptionGroup with {len(group.exceptions)} errors:')
            for exc in group.exceptions:
                print(f'  - {type(exc).__name__}: {str(exc)[:120]}')
    logfire.force_flush()


if __name__ == '__main__':
    main()
