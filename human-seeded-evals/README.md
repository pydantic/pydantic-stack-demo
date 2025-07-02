# Human Seeded Evals Demo

Like evals ... but without all the hard work.

Panacea or pipedream?

# Usage

Run the frontend:

```bash
cd frontend
npm run dev
```

Run the backend:

```bash
uv run uvicorn app.main:app --port 5000
```

Run the eval generator:

```bash
uv run evals/eval_prompt_generator.py
```

Run the live evals agent:

```bash
uv run evals/auto_evals.py
```
