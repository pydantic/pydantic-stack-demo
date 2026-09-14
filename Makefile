.DEFAULT_GOAL := help

.PHONY: install
install:  ## Install every app and package into one virtualenv, plus pre-commit hooks
	uv sync --all-packages
	uv run pre-commit install --install-hooks 2>/dev/null || true

.PHONY: format
format:  ## Format and auto-fix
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint:  ## Lint without changing files
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck
typecheck:  ## Type-check every package and app
	uv run basedpyright

.PHONY: test
test:  ## Run the offline test suite (no model calls, no network)
	uv run pytest -q

.PHONY: check
check: lint typecheck test  ## Everything CI runs

.PHONY: smoke
smoke:  ## Run every script demo once against real credentials from .env (costs a few cents)
	uv run python scripts/smoke.py

.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
