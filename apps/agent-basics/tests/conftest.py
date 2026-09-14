import os

os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
os.environ.setdefault('DEMO_MODEL', 'test')
os.environ.setdefault('DEMO_FAST_MODEL', 'test')
