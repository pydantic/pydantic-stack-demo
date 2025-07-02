from dataclasses import dataclass
from datetime import timedelta

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from app.models import TimeRangeBuilderSuccess, TimeRangeInputs, TimeRangeResponse


@dataclass
class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    def evaluate(self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            window_end = ctx.output.end_timestamp
            window_size = window_end - ctx.output.start_timestamp
            return {
                'window_is_not_too_long': window_size <= timedelta(days=30),
                'window_is_not_in_the_future': window_end <= ctx.inputs['now'],
            }

        return {}  # No evaluation needed for errors


@dataclass
class UserMessageIsConcise(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    async def evaluate(
        self,
        ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse],
    ) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            user_facing_message = ctx.output.explanation
        else:
            user_facing_message = ctx.output.error

        if user_facing_message is not None:
            return len(user_facing_message.split()) < 50
        else:
            return {}


CUSTOM_EVALUATOR_TYPES = (ValidateTimeRange, UserMessageIsConcise)
