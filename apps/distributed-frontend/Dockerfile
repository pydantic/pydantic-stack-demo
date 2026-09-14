# Backend only. Build context is the repo root (demo-core is a workspace path dependency).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/demo_core ./packages/demo_core
COPY apps/distributed-frontend/pyproject.toml ./apps/distributed-frontend/pyproject.toml
COPY apps/distributed-frontend/src ./apps/distributed-frontend/src

RUN uv sync --frozen --package distributed-frontend

EXPOSE 8000
CMD ["uv", "run", "--package", "distributed-frontend", "uvicorn", "distributed_frontend.main:app", "--host", "0.0.0.0", "--port", "8000"]
