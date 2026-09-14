import os

os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
os.environ.setdefault('DEMO_MODEL', 'test')
os.environ.setdefault('DEMO_FAST_MODEL', 'test')

from pydantic_evals.online import configure

# Never let a sampled LLM judge fire in the background during unit tests.
configure(enabled=False)
