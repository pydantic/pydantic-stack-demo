"""
[
  {
    "timestamp": "2025-05-01T05:41:30.311402Z",
    "input": "Last 2 hour",
    "final_result": {
      "min_timestamp_with_offset": "2025-05-01T03:41:30Z",
      "max_timestamp_with_offset": "2025-05-01T05:41:30Z",
      "explanation": "",
      "error_message": null
    }
  },
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter


class InputData(BaseModel):
    timestamp: str
    input: str
    final_result: dict[str, Any]


"""

- name: Single day mention
  inputs:
    prompt: I want to see logs from 2021-05-08
    now: '2023-10-28T09:30:00Z'
  expected_output:
    min_timestamp_with_offset: '2021-05-08T00:00:00Z'
    max_timestamp_with_offset: '2021-05-08T23:59:59Z'
    explanation: You mentioned a single day (2021-05-08). The entire day is used.
  evaluators:
  - IsInstance: TimeRangeBuilderSuccess
"""


class Case(BaseModel):
    name: str
    inputs: dict[str, Any]
    expected_output: dict[str, Any]
    evaluators: list[dict[str, Any]]


def map_data(input_data: InputData) -> Case:
    expected_output = {k: v for k, v in input_data.final_result.items() if v is not None}
    if 'error_message' in expected_output:
        evaluators = [{'IsInstance': 'TimeRangeBuilderError'}]
    else:
        evaluators = [{'IsInstance': 'TimeRangeBuilderSuccess'}]
    return Case(
        name=input_data.input,
        inputs={'prompt': input_data.input, 'now': input_data.timestamp},
        expected_output=expected_output,
        evaluators=evaluators,
    )


input_schema = TypeAdapter(list[InputData])


class Output(BaseModel):
    cases: list[Case]


this_dir = Path(__file__).parent
input_data = input_schema.validate_json(Path(this_dir / 'scratch' / 'real_world.json').read_bytes())

output = Output(cases=[map_data(data) for data in input_data])
(this_dir / 'evals' / 'dataset.json').write_text(output.model_dump_json(indent=2))
