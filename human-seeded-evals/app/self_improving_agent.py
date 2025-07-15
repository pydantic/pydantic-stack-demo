from __future__ import annotations as _annotations

import csv
import io
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Protocol, TypeAlias, TypedDict, TypeGuard, cast

from annotated_types import Ge, Le
from logfire import Logfire
from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart
from pydantic_ai.models import KnownModelName, Model, ModelRequestParameters, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ObjectJsonSchema, ToolDefinition

logfire = Logfire(otel_scope='self-improving-agent')
FieldsPatch = dict[str, str]


@dataclass
class FieldDetails:
    key: str
    description: str
    current_prompt: str | None = None


fields_path = Path('agent_context_fields.json')
fields_schema = TypeAdapter(list[FieldDetails])


class ModelContextPatch(BaseModel):
    context_patch: FieldsPatch
    timestamp: AwareDatetime


class AbstractCoachOutput(Protocol):
    context_patch: FieldsPatch
    developer_suggestions: str | None
    overall_context_score: int


patch_path = Path('agent_context_patches.json')


@dataclass(init=False)
class SelfImprovingAgentModel(WrapperModel):
    wrapped_model: Model

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        fields = list(context_patch_fields(messages, model_request_parameters))
        fields_path.write_bytes(fields_schema.dump_json(fields, indent=2))

        if patch := Coach.get_patch():
            messages, model_request_parameters = apply_patch(messages, model_request_parameters, patch.context_patch)

        return await super().request(messages, model_settings, model_request_parameters)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self.inner_model.model_name

    @property
    def system(self) -> str:
        """The system prompt."""
        return self.inner_model.system


@dataclass
class Coach:
    agent_name: str
    logfire_read_token: str
    logfire_environment: str | None = None
    """Name of the environment in logfire where the main agent is running, improves query performance"""
    logfire_filter: str | None = None
    """Additional logfire filter when looking for agent run traces to improve performance"""

    coach_model: Model | KnownModelName = 'anthropic:claude-opus-4-0'
    """Model used for the coach agent"""

    async def run(self):
        fields = self.get_fields()
        last_path = self.get_patch()
        runs, last_run = await self._get_runs(last_path and last_path.timestamp)

        prompt_data: dict[str, Any] = {
            'default_model_context': {f.key: f.current_prompt for f in fields if f.current_prompt}
        }
        if last_path:
            prompt_data['previous_context_patch'] = last_path.context_patch
        if runs:
            prompt_data['recent_agent_runs'] = runs
        prompt = format_as_xml(prompt_data, include_root_tag=False)
        coach_agent = self._get_coach_agent(fields)
        r = await coach_agent.run(prompt)
        run_count = len(runs) if runs is not None else None
        if r.output.overall_context_score < 5:
            logfire.warning(
                'Coach run with quality warning, score={output.overall_context_score} {run_count=}',
                output=r.output,
                run_count=run_count,
            )
        else:
            logfire.info(
                'Coach run with quality ok, score={output.overall_context_score} {run_count=}',
                output=r.output,
                run_count=run_count,
            )
        patch = ModelContextPatch(context_patch=r.output.context_patch, timestamp=last_run)
        self.update_patch(patch)

    def get_fields(self) -> list[FieldDetails]:
        return fields_schema.validate_json(fields_path.read_bytes())

    @staticmethod
    def get_patch() -> ModelContextPatch | None:
        try:
            return ModelContextPatch.model_validate_json(patch_path.read_bytes())
        except FileNotFoundError:
            return None

    def update_patch(self, patch: ModelContextPatch) -> None:
        patch_path.write_text(patch.model_dump_json(indent=2))

    async def _get_runs(self, min_timestamp: datetime | None) -> tuple[list[AgentRunSummary] | None, datetime]:
        async with AsyncLogfireQueryClient(self.logfire_read_token) as client:
            runs_where = ["otel_scope_name='pydantic-ai'", f"message='{self.agent_name} run'"]
            if self.logfire_environment:
                runs_where.append(f"deployment_environment='{self.logfire_environment}'")
            if self.logfire_filter:
                runs_where.append(self.logfire_filter)
            sql = runs_query.format(where=' AND '.join(runs_where))
            min_timestamp = datetime.now(tz=timezone.utc) - timedelta(hours=2)
            r = await client.query_json_rows(sql=sql, min_timestamp=min_timestamp)
            runs_rows = r['rows']
            count = len(runs_rows)
            if not count:
                logfire.info('Found {run_count} runs', run_count=count)
                return None, datetime.now(tz=timezone.utc) - timedelta(seconds=30)

            created_ats = datetime_list_schema.validate_python([row['created_at'] for row in runs_rows])
            last_run = max(created_ats)

            r = await client.query_json_rows(sql=feedback_query, min_timestamp=min_timestamp)
            feedback_lookup: dict[str, Any] = {
                '{trace_id}-{parent_span_id}'.format(**row): RunFeedback(**row) for row in r['rows']
            }

            runs: list[AgentRunSummary] = []
            feedback_count = 0
            for row in runs_rows:
                if feedback := feedback_lookup.get('{trace_id}-{span_id}'.format(**row)):
                    row['feedback'] = feedback
                    feedback_count += 1
                run = AgentRunSummary.model_validate(row)
                if run.prompt is not None:
                    runs.append(run)

            logfire.info(
                'Found {run_count} runs, {feedback_count} with feedback, running coach',
                run_count=count,
                feedback_count=feedback_count,
            )
            return runs, last_run

    def _get_coach_agent(self, fields: list[FieldDetails]) -> Agent[None, AbstractCoachOutput]:
        fields_dict = {f.key: Annotated[str, Field(description=f.description)] for f in fields}
        ModelRequestFields = TypedDict(
            'ModelRequestFields',
            fields_dict,  # type: ignore
            total=False,
        )

        class CoachOutput(BaseModel, use_attribute_docstrings=True):
            context_patch: ModelRequestFields
            """Patch to update context fields to improve the agent's performance."""
            developer_suggestions: str | None = None
            """Suggestions to the developer about how to improve the agent code."""
            overall_context_score: Annotated[int, Ge(0), Le(10)]
            """Overall quality of the context, on a scale from zero to ten, zero being the worst, ten being the best.

            Any value below 5 will trigger a warning to the agent maintainers.
            """

        coach_model = infer_model(self.coach_model)
        self._coach_agent = agent = cast(
            Agent[None, AbstractCoachOutput],
            Agent(coach_model, output_type=CoachOutput, instructions=coach_instrunctions),
        )
        return agent


class RunFeedback(BaseModel):
    reaction: Literal['positive', 'negative'] | None
    comment: str | None


class AgentRunSummary(BaseModel):
    prompt: str | None
    output: Any
    feedback: RunFeedback | None = None


datetime_list_schema = TypeAdapter(list[AwareDatetime])

runs_query = """
select
    created_at,
    trace_id,
    span_id,
    attributes->'all_messages_events'->1->>'content' as prompt,
    attributes->'final_result' as output
from records
where {where}
order by created_at desc
limit 20
"""
feedback_query = """
select
    trace_id,
    parent_span_id,
    attributes->>'Annotation' as reaction,
    attributes->>'logfire.feedback.comment' as comment
from records
where
    kind='annotation' and
    attributes->>'logfire.feedback.name'='Annotation'
order by created_at desc
-- bigger limit to get all feedback linked to relevant runs
limit 200
"""
# this is a rough prompt, can almost certainly be improved
coach_instrunctions = """\
Your job is to improve the performance of an AI agent by analyzing the context provided to the model
and the agent's behavior (inputs, outputs and feedback where available),
then rewriting context where you are confident that it will improve the agent's performance.
To do this return a patch of model context prompts and descriptions where appropriate.

Pay special attention to the `instructions` or `system_prompt` fields as they have the most significant impact on the agent's behavior.

Be concise and clear: increasing text length will increase token usage and thereby cost.

If you identify shortcomings in the context provided to the model that cannot be solved by adjusting the instructions
and tool descriptions, please suggest improvements the developer should make.

YOU SHOULD ONLY INCLUDE SUGGESTIONS IF THERE ARE SHORTCOMINGS IN THE CONTEXT PROVIDED TO THE MODEL
THAT CANNOT BE SOLVED BY ADJUSTING THE INSTRUCTIONS AND TOOL DESCRIPTIONS.
"""


def context_patch_fields(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
) -> Iterable[FieldDetails]:
    found_sys_prompt = False
    if system_prompt := get_system_prompt(messages):
        found_sys_prompt = True
        yield FieldDetails('system_prompt', 'System prompt', system_prompt)

    instructions = get_instrunctions(messages)
    if instructions or not found_sys_prompt:
        yield FieldDetails('instructions', 'Instructions', instructions)

    yield from get_tools_fields(model_request_parameters.function_tools, 'function_tools', 'Function tool description')

    yield from get_tools_fields(model_request_parameters.output_tools, 'output_tools', 'Output tool description')


def get_system_prompt(messages: list[ModelMessage]) -> str | None:
    """Get the first system prompt from messages, other system prompts are ignored."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    return part.content


def get_instrunctions(messages: list[ModelMessage]) -> str | None:
    """Get the first instruction from messages, other instructions are ignored."""
    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions:
                return message.instructions


def get_tools_fields(tools: list[ToolDefinition], prefix: str, description: str) -> Iterable[FieldDetails]:
    for t in tools:
        tool_prefix = f'{prefix}.{escape_key(t.name)}'
        yield FieldDetails(f'{tool_prefix}.description', description, t.description)
        yield from json_schema_fields(t.parameters_json_schema, f'{tool_prefix}.parameters')


JsonSchema = dict[str, Any]


def json_schema_fields(schema: JsonSchema, prefix: str) -> Iterable[FieldDetails]:
    yield FieldDetails(f'{prefix}.description', 'JSON schema field description', schema.get('description'))

    type_ = schema.get('type')
    if type_ == 'object':
        yield from _js_object(schema, prefix)
    elif type_ == 'array':
        yield from _js_array(schema, prefix)
    elif type_ is None:
        yield from _js_union(schema, prefix, 'anyOf')
        yield from _js_union(schema, prefix, 'oneOf')


def _js_object(schema: ObjectJsonSchema, prefix: str) -> Iterable[FieldDetails]:
    if properties := schema.get('properties'):
        for key, value in properties.items():
            yield from json_schema_fields(value, f'{prefix}.properties.{escape_key(key)}')

    if additional_properties := schema.get('additionalProperties'):
        if _is_json_schema(additional_properties):
            yield from json_schema_fields(additional_properties, f'{prefix}.additionalProperties')

    if pattern_properties := schema.get('patternProperties'):
        for key, value in pattern_properties.items():
            yield from json_schema_fields(value, f'{prefix}.patternProperties.{escape_key(key)}')


def _js_array(schema: ObjectJsonSchema, prefix: str) -> Iterable[FieldDetails]:
    if prefix_items := schema.get('prefixItems'):
        assert isinstance(prefix_items, list), f'Expected list for prefixItems, got {type(prefix_items)}'
        for i, item in enumerate(cast(list[Any], prefix_items)):
            if _is_json_schema(item):
                yield from json_schema_fields(item, f'{prefix}.prefixItems.{i}')

    if items := schema.get('items'):
        if _is_json_schema(items):
            yield from json_schema_fields(items, f'{prefix}.items')


def _js_union(schema: JsonSchema, prefix: str, union_kind: Literal['anyOf', 'oneOf']) -> Iterable[FieldDetails]:
    members = schema.get(union_kind)
    if not members:
        return

    for member in members:
        if _is_json_schema(member):
            yield from json_schema_fields(member, f'{prefix}.{union_kind}')


def escape_key(s: str) -> str:
    if '.' in s:
        # double double quotes matches how the csv module parses strings
        return '"' + s.replace('"', '""') + '"'
    else:
        return s


def apply_patch(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters, patch: FieldsPatch
) -> tuple[list[ModelMessage], ModelRequestParameters]:
    if not patch:
        return messages, model_request_parameters

    messages = deepcopy(messages)
    model_request_parameters = deepcopy(model_request_parameters)
    changes = 0

    nested_patch = unflatten(patch)
    if system_prompt := nested_patch.get('system_prompt'):
        assert isinstance(system_prompt, str), f'Expected str for system_prompt, got {type(system_prompt)}'
        if set_system_prompt(messages, system_prompt):
            changes += 1
        else:
            logfire.warning('No system prompt found to replace')

    if instructions := nested_patch.get('instructions'):
        assert isinstance(instructions, str), f'Expected str for instructions, got {type(instructions)}'
        if set_instructions(messages, instructions):
            changes += 1
        else:
            logfire.warning('No instructions found to replace')

    changes += set_tools_fields(model_request_parameters.function_tools, 'function_tools', nested_patch)
    changes += set_tools_fields(model_request_parameters.output_tools, 'output_tools', nested_patch)

    logfire.info('updated {changes} fields in messages and model request parameters', changes=changes)
    return messages, model_request_parameters


UnflattenedPatch: TypeAlias = 'dict[str, UnflattenedPatch | str]'


def unflatten(patch: FieldsPatch) -> UnflattenedPatch:
    d: UnflattenedPatch = {}
    for key, value in patch.items():
        local_d = d
        *parts, last = split_key(key)
        for part in parts:
            local_d = local_d.setdefault(part, {})
            assert isinstance(local_d, dict), f'Expected dict at {part}, got {type(local_d)}'
        local_d[last] = value
    return d


def set_system_prompt(messages: list[ModelMessage], system_prompt: str) -> bool:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    part.content = system_prompt
                    return True
    return False


def set_instructions(messages: list[ModelMessage], instructions: str) -> bool:
    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions is not None:
                message.instructions = instructions
                return True

    # if we didn't find existing instructions to replace, set instructions on the first model request
    for message in messages:
        if isinstance(message, ModelRequest):
            message.instructions = instructions
            return True

    return False


def set_tools_fields(tools: list[ToolDefinition], key: str, patch: UnflattenedPatch) -> int:
    tools_patch = patch.get(key)
    changes = 0
    if not tools_patch:
        return changes
    assert isinstance(tools_patch, dict), f'Expected dict at {key}, got {type(tools_patch)}'
    for tool_name, tool_patch in tools_patch.items():
        assert isinstance(tool_name, str), f'Expected str at {key}.{tool_name}, got {type(tool_name)}'
        assert isinstance(tool_patch, dict), f'Expected dict at {key}.{tool_name}, got {type(tool_patch)}'
        tool = next((t for t in tools if t.name == tool_name), None)
        assert tool is not None, f'Unable to find tool {key}.{tool_name}'

        if description := tool_patch.get('description'):
            assert isinstance(description, str), (
                f'Expected str at {key}.{tool_name}.description, got {type(description)}'
            )
            tool.description = description
            changes += 1

        if parameters := tool_patch.get('parameters'):
            assert isinstance(parameters, dict), (
                f'Expected dict at {key}.{tool_name}.parameters, got {type(parameters)}'
            )
            changes += update_json_schema(tool.parameters_json_schema, parameters, [key, tool_name, 'parameters'])
    return changes


def update_json_schema(schema: JsonSchema, patch: UnflattenedPatch, path: list[str]) -> int:
    changes = 0
    patch_copy = patch.copy()
    if description := patch_copy.pop('description', None):
        assert isinstance(description, str), f'Expected str at {".".join(path)}.description, got {type(description)}'
        schema['description'] = description
        changes += 1

    for k, v in patch_copy.items():
        sub_path = path + [k]
        assert isinstance(v, dict), f'Expected dict at {".".join(sub_path)}, got {type(v)}'

        sub_schema = schema.get(k)
        if not sub_schema:
            print('WARNING: Schema key not found')

        if _is_json_schema(sub_schema):
            changes += update_json_schema(sub_schema, v, sub_path)
        else:
            assert isinstance(sub_schema, list), (
                f'Expected dict or list at {".".join(sub_path)}, got {type(sub_schema)}'
            )

            for k2, v2 in patch_copy.items():
                sub_sub_path = sub_path + [k2]
                array_schema = cast(JsonSchema, sub_schema[int(k2)])
                assert isinstance(v2, dict), f'Expected dict at {".".join(sub_sub_path)}, got {type(v)}'
                changes += update_json_schema(array_schema, v2, sub_sub_path)

    return changes


def split_key(s: str) -> list[str]:
    if '"' in s:
        # quotes in the string means we have to parse it properly
        return next(csv.reader(io.StringIO(s), delimiter='.'))
    else:
        return s.split('.')


def _is_json_schema(obj: Any) -> TypeGuard[JsonSchema]:
    return isinstance(obj, dict)
