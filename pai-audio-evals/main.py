import random
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import logfire
from nltk import edit_distance
from pydantic import TypeAdapter
from pydantic_ai import Agent, BinaryContent, AudioUrl
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput


@dataclass
class EditSimilarity(Evaluator[object, str, object]):
    def evaluate(self, ctx: EvaluatorContext[object, str, object]) -> EvaluatorOutput:
        if ctx.expected_output is None:
            return {}  # no metric
        actual_tokens = re.sub(r'[^a-z0-9\s]', '', ctx.output.lower()).split()
        expected_tokens = re.sub(r'[^a-z0-9\s]', '', ctx.expected_output.lower()).split()
        distance = edit_distance(actual_tokens, expected_tokens)
        normalized_distance = distance / max(len(actual_tokens), len(expected_tokens))
        return 1 - normalized_distance


logfire.configure(service_name='pai-audio-evals', scrubbing=False, console=False)
logfire.instrument_pydantic_ai()

this_dir = Path(__file__).parent
assets = this_dir / 'assets'


@dataclass
class AudioFile:
    file: str
    text: str

    def audio_url(self) -> AudioUrl:
        return AudioUrl(f'https://smokeshow.helpmanual.io/4l1l1s0s6q4741012x1w/{self.file}')

    def binary_content(self) -> BinaryContent:
        path = assets / self.file
        return BinaryContent(data=path.read_bytes(), media_type='audio/mpeg')


n_files = 10
files_schema = TypeAdapter(list[AudioFile])
files = files_schema.validate_json((this_dir / 'assets.json').read_bytes())[:n_files]
# random.seed(42)
random.shuffle(files)

audio_agent = Agent(instructions='return the transcription only, no prefix or quotes')
dataset = Dataset(
    cases=[Case(name=file.file, inputs=file.audio_url(), expected_output=file.text) for file in files],
    evaluators=[EditSimilarity()],
)


async def task(audio_url: AudioUrl, model: str) -> str:
    return (await audio_agent.run(['transcribe', audio_url], model=model)).output


with logfire.span('Compare models'):
    for model in 'gpt-4o-audio-preview', 'gpt-4o-mini-audio-preview', 'google-vertex:gemini-2.0-flash':
        dataset.evaluate_sync(partial(task, model=model), name=model, max_concurrency=10)
