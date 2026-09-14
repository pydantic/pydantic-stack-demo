"""Publish the regression dataset to Logfire's managed datasets.

Once hosted, cases can be edited in the Logfire UI and new ones added straight from a
production trace. Re-running this script upserts by case name, so it is safe to repeat.
Needs an API key with dataset read/write on the project (LOGFIRE_API_KEY).
"""

from __future__ import annotations

import os

from logfire.experimental.api_client import LogfireAPIClient

from demo_core import load_env
from support_agent.evals.dataset import build_dataset


def publish(api_key: str, base_url: str | None = None) -> dict[str, object]:
    # Custom evaluators are not pushed: the hosted copy holds cases, the repo holds the code.
    dataset = build_dataset(include_judge=False)
    dataset.evaluators = []
    with LogfireAPIClient(api_key=api_key, base_url=base_url) as client:
        detail = client.push_dataset(dataset, description='pydantic-stack-demo support-agent regression cases')
    return dict(detail)


def main() -> None:
    load_env()
    api_key = os.environ.get('LOGFIRE_API_KEY')
    if not api_key:
        raise SystemExit('Set LOGFIRE_API_KEY (project API key with dataset read/write) to publish the dataset.')
    detail = publish(api_key, os.environ.get('LOGFIRE_API_BASE_URL'))
    print(f'Published dataset {detail.get("name")!r} (id {detail.get("id")})')


if __name__ == '__main__':
    main()
