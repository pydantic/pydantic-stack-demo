# One image with every app installed, for running any script demo in Docker:
#   docker compose run --rm cli agent-basics-weather
FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY apps ./apps
RUN uv sync --frozen --all-packages --no-dev
ENV PATH="/app/.venv/bin:$PATH" PYDANTIC_AI_NO_BANNER=1
ENTRYPOINT ["/bin/sh", "-c", "exec \"$@\"", "--"]
CMD ["hello-world"]
