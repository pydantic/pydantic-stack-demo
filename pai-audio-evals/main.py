from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_ai import Agent, BinaryContent

this_dir = Path(__file__).parent
assets = this_dir / 'assets'


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


@dataclass
class AudioFile:
    file: str
    text: str

    def binary_content(self) -> BinaryContent:
        path = assets / self.file
        return BinaryContent(data=path.read_bytes(), media_type='audio/mpeg')


files_schema = TypeAdapter(list[AudioFile])
files = files_schema.validate_json((this_dir / 'assets.json').read_bytes())

for audio_file in files[:10]:
    model_distances: list[tuple[str, int]] = []
    for model in 'gpt-4o-audio-preview', 'gpt-4o-mini-audio-preview', 'google-vertex:gemini-2.0-flash':
        agent = Agent(model='gpt-4o-audio-preview', instructions='return the transcription only, no prefix or quotes')
        result = agent.run_sync(['transcribe', audio_file.binary_content()])
        model_distances.append((model, levenshtein_distance(audio_file.text, result.output)))
    print(audio_file.text)
    print('  ', model_distances)
