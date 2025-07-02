import sys
from pathlib import Path
from types import NoneType

import logfire
from pydantic_evals import Dataset

sys.path.append(str(Path(__file__).parent.parent))

from custom_evaluators import CUSTOM_EVALUATOR_TYPES

from app.agent import infer_time_range
from app.models import TimeRangeInputs, TimeRangeResponse

logfire.configure(environment='evals')
logfire.instrument_pydantic_ai()


def evaluate_dataset():
    dataset_path = Path(__file__).parent / 'dataset.json'
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
        dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
    )
    report = dataset.evaluate_sync(infer_time_range)
    print(report)


if __name__ == '__main__':
    evaluate_dataset()
