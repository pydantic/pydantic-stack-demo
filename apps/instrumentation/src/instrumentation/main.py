"""A small order-support service that shows every Logfire instrumentation technique in one place.

- `configure_logfire()` runs first, with service metadata, system metrics, extra scrubbing
  patterns and optional head sampling.
- FastAPI and httpx are auto-instrumented: every request and outbound call becomes a span.
- `/orders/{id}` adds manual spans, structured log attributes and custom metrics.
- `/ask` runs a named Pydantic AI agent whose tool makes an outbound HTTP call, so the
  trace shows request -> agent run -> model call -> tool -> HTTP, nested.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import logfire
from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from demo_core import configure_logfire, create_app, settings

SERVICE_VERSION = '0.1.0'

# Head sampling: keep only this fraction of traces (1.0 = everything). Set LOGFIRE_SAMPLE_RATE=0.25 to try it.
_sample_rate = float(os.environ.get('LOGFIRE_SAMPLE_RATE', '1.0'))

configure_logfire(
    'instrumentation',
    service_version=SERVICE_VERSION,
    system_metrics=True,
    # `customer_ref` is a field name specific to this service that should never reach Logfire.
    # The default patterns (password, secret, api_key, ...) still apply on top of this.
    scrubbing=logfire.ScrubbingOptions(extra_patterns=['customer_ref']),
    sampling=logfire.SamplingOptions(head=_sample_rate) if _sample_rate < 1.0 else None,
)

# Custom metrics: a counter for business events and a histogram for a measurement.
orders_looked_up = logfire.metric_counter('orders.lookups', unit='1', description='Order lookups by outcome')
order_value = logfire.metric_histogram('orders.value', unit='USD', description='Value of orders looked up')

ORDERS: dict[str, dict[str, object]] = {
    'A100': {'status': 'shipped', 'total': 129.0, 'customer_ref': 'cust_8842'},
    'A101': {'status': 'processing', 'total': 42.5, 'customer_ref': 'cust_1190'},
    'A102': {'status': 'delivered', 'total': 310.0, 'customer_ref': 'cust_5521'},
}


@dataclass
class Deps:
    client: AsyncClient


support_agent = Agent(
    settings().model,
    name='order_support_agent',
    deps_type=Deps,
    instructions='You answer questions about orders. Use the lookup tool; reply in one sentence.',
)


@support_agent.tool
async def lookup_delivery_estimate(ctx: RunContext[Deps], order_id: str) -> str:
    """Fetch the delivery estimate for an order from the (demo) carrier API."""
    # A public endpoint that returns a random number after a delay: the point is the
    # nested httpx span under the tool span, not the answer.
    r = await ctx.deps.client.get(
        'https://demo-endpoints.pydantic.workers.dev/number', params={'min': 1, 'max': 5, 'sleep': 300}
    )
    r.raise_for_status()
    return f'Order {order_id} arrives in about {r.text} days.'


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_headers=True)
        app.state.client = client
        yield


app = create_app('Instrumentation demo', lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post('/ask')
async def ask(body: AskRequest) -> AskResponse:
    result = await support_agent.run(body.question, deps=Deps(client=app.state.client))
    return AskResponse(answer=result.output)


@logfire.instrument('load order {order_id}')
async def load_order(order_id: str) -> dict[str, object]:
    """`@logfire.instrument` wraps the call in a span named from the arguments."""
    order = ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail='order not found')
    return order


@app.get('/orders/{order_id}')
async def get_order(order_id: str) -> dict[str, object]:
    with logfire.span('order lookup {order_id}', order_id=order_id):
        order = await load_order(order_id)
        with logfire.span('enrich order'):
            # Structured attributes are queryable in Logfire; `customer_ref` is scrubbed
            # by the extra pattern above, `password` by the defaults.
            logfire.info(
                'order {order_id} status={status}',
                order_id=order_id,
                status=order['status'],
                total=order['total'],
                customer_ref=order['customer_ref'],
                password='hunter2',
            )
        orders_looked_up.add(1, {'outcome': 'found'})
        order_value.record(float(order['total']))  # type: ignore[arg-type]
    return {'order_id': order_id, 'status': order['status'], 'total': order['total']}


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', '8000')))


if __name__ == '__main__':
    serve()
