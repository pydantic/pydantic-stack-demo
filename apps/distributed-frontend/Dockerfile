# Backend only. Build context is the repo root (demo-core is a workspace path dependency).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/demo_core ./packages/demo_core
COPY apps/distributed-frontend/pyproject.toml ./apps/distributed-frontend/pyproject.toml
COPY apps/distributed-frontend/src ./apps/distributed-frontend/src

RUN uv sync --frozen --no-dev --package distributed-frontend
ENV PATH="/app/.venv/bin:$PATH" PYDANTIC_AI_NO_BANNER=1

EXPOSE 8000
CMD ["uvicorn", "distributed_frontend.main:app", "--host", "0.0.0.0", "--port", "8000"]
