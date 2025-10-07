import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logfire.configure(service_name='backend', distributed_tracing=True)

# Enable this if you experience issues with the proxy endpoint and want to capture all outgoing HTTP requests
# logfire.instrument_httpx(capture_all=True)
http_client: AsyncClient
openai_client = AsyncOpenAI()
logfire.instrument_openai(openai_client)

logfire_token = os.getenv('LOGFIRE_TOKEN')
logfire_base_url = os.getenv('LOGFIRE_BASE_URL')

assert logfire_token is not None, 'LOGFIRE_TOKEN is not set'
assert logfire_base_url is not None, 'LOGFIRE_BASE_URL is not set'


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client, openai_client
    async with AsyncClient() as _http_client:
        http_client = _http_client
        logfire.instrument_httpx(http_client, capture_headers=True)
        yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],  # Our frontend in dev mode
    allow_credentials=True,
    allow_methods=['*'],  # Allow all methods
    # Important: if you don't want to allow all headers and you're using distributed tracing, you need to explicitly list the 'traceparent' header
    allow_headers=['*'],  # Allow all headers
)

# We don't want telemetry on the logfire proxy endpoint itself, although it could be turned on for troubleshooting purposes
logfire.instrument_fastapi(app, capture_headers=True, excluded_urls='(.+)client-traces$')

this_dir = Path(__file__).parent
image_dir = Path(__file__).parent / 'images'
image_dir.mkdir(exist_ok=True)

app.mount('/static', StaticFiles(directory=image_dir), name='static')


class GenerateResponse(BaseModel):
    next_url: str = Field(serialization_alias='nextUrl')


@app.post('/generate')
async def generate_image(prompt: str) -> GenerateResponse:
    response = await openai_client.images.generate(prompt=prompt, model='dall-e-3')

    assert response.data, 'No image in response'

    image_url = response.data[0].url
    assert image_url, 'No image URL in response'
    r = await http_client.get(image_url)
    r.raise_for_status()
    path = f'{uuid4().hex}.jpg'
    (image_dir / path).write_bytes(r.content)
    return GenerateResponse(next_url=f'/static/{path}')


# Proxy to Logfire for client traces from the browser
@app.post('/client-traces')
async def client_traces(request: Request):
    assert logfire_token is not None, 'Logfire token is not set'
    response = await http_client.request(
        method=request.method,
        url=f'{logfire_base_url}v1/traces',
        headers={'Authorization': logfire_token},
        json=await request.json(),
    )
    response.raise_for_status()

    return {
        'status_code': response.status_code,
        'body': response.text,
        'proxied_to': f'{logfire_base_url}v1/traces',
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
