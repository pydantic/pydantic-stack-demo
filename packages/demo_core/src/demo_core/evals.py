"""Small helpers around pydantic-evals used by the eval-focused demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators import LLMJudge

from demo_core.settings import settings


def llm_judge(rubric: str, *, model: str | None = None, include_input: bool = True, **kwargs: Any) -> LLMJudge:
    """An `LLMJudge` on the demo's default model, with the settings we use everywhere."""
    return LLMJudge(rubric=rubric, model=model or settings().model, include_input=include_input, **kwargs)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line; blank lines are skipped."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
