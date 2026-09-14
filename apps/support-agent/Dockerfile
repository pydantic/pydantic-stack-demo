# Build context is the repo root (the app depends on packages/demo_core).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/demo_core ./packages/demo_core
COPY apps/support-agent ./apps/support-agent

RUN uv sync --frozen --no-dev --package support-agent
ENV PATH="/app/.venv/bin:$PATH" PYDANTIC_AI_NO_BANNER=1

EXPOSE 8000
CMD ["uvicorn", "support_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
