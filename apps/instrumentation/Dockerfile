# Build context is the repo root (demo-core is a workspace path dependency).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/demo_core ./packages/demo_core
COPY apps/instrumentation ./apps/instrumentation

RUN uv sync --frozen --package instrumentation

EXPOSE 8000
CMD ["uv", "run", "--package", "instrumentation", "uvicorn", "instrumentation.main:app", "--host", "0.0.0.0", "--port", "8000"]
