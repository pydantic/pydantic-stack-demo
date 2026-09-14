import os

os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
os.environ.setdefault('DEMO_MODEL', 'test')
os.environ.setdefault('DEMO_FAST_MODEL', 'test')
os.environ.setdefault('PYDANTIC_AI_GATEWAY_API_KEY', 'pylf_v1_us_testkey')
