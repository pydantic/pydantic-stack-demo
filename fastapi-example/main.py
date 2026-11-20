from base64 import b64decode
from pathlib import Path
from uuid import uuid4

import logfire
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

logfire.configure(service_name='fastapi-example')
openai_client = AsyncOpenAI()
logfire.instrument_openai(openai_client)


app = FastAPI()
logfire.instrument_fastapi(app, capture_headers=True)
this_dir = Path(__file__).parent
image_dir = Path(__file__).parent / 'images'
image_dir.mkdir(exist_ok=True)
app.mount('/static', StaticFiles(directory=image_dir), name='static')


@app.get('/')
@app.get('/display/{image:path}')
async def main() -> FileResponse:
    return FileResponse(this_dir / 'page.html')


class GenerateResponse(BaseModel):
    next_url: str = Field(serialization_alias='nextUrl')


@app.post('/generate')
async def generate_image(prompt: str) -> GenerateResponse:
    response = await openai_client.images.generate(prompt=prompt, model='gpt-image-2.6-flare', output_format='png')

    assert response.data, 'No image in response'

    image_data = response.data[0].b64_json
    assert image_data, 'No image data in response'
    path = f'{uuid4().hex}.png'
    (image_dir / path).write_bytes(b64decode(image_data, validate=True))
    return GenerateResponse(next_url=f'/display/{path}')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
