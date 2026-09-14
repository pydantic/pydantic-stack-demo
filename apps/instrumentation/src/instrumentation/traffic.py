"""Generate traffic against a running instrumentation service, for live demos."""

from __future__ import annotations

import argparse
import asyncio
import random

import httpx

QUESTIONS = ['When will order A100 arrive?', 'Is order A101 shipped yet?', 'Status of A102?']


async def run(base_url: str, count: int, interval: float, ask: bool) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        for i in range(count):
            order_id = random.choice(['A100', 'A101', 'A102', 'A100', 'A999'])
            r = await client.get(f'/orders/{order_id}')
            print(f'[{i + 1}/{count}] GET /orders/{order_id} -> {r.status_code}')
            if ask and i % 3 == 0:
                q = random.choice(QUESTIONS)
                r = await client.post('/ask', json={'question': q})
                answer = r.json().get('answer', '') if r.status_code == 200 else ''
                print(f'          POST /ask {q!r} -> {r.status_code} {answer[:80]}')
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description='Send demo traffic to the instrumentation service.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--no-ask', action='store_true', help='skip the /ask endpoint (no model calls)')
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.count, args.interval, not args.no_ask))


if __name__ == '__main__':
    main()
