import os

# Keep every test offline and independent of a developer's real .env.
os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
