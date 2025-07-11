.PHONY: install
install:
	cd frontend && npm install
	uv sync

.PHONY: frontend-dev
frontend-dev: 
	cd frontend && npm run dev

.PHONY: backend-dev
backend-dev: 
	uv run main.py
