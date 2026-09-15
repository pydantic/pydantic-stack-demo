"""Keeping personal data out of the model and out of the traces.

Two independent controls, used together:

- `InputGuardrail(guard=redact_pii)` rewrites the prompt before the model sees it and replaces
  the original in the run's message history, so the email/phone never reach the provider or
  the trace. An `OutputGuardrail` does the same on the way back.
- `configure_logfire(..., include_content=False)` tells the Pydantic AI instrumentation to
  omit prompts, completions and tool arguments from spans entirely. Logfire still records
  timings, token counts and cost.

Why both? Logfire's SDK scrubbing (`ScrubbingOptions`) redacts attribute values that match
sensitive patterns, but it deliberately skips LLM message attributes: model text is full of
false positives ("your password was reset") and personal data rarely announces itself with a
keyword. The recommended approach is to not send content at all, and to redact at the input.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai_harness.guardrails import InputGuardrail, OutputGuardrail, detectors

from demo_core import configure_logfire, settings

# The built-in patterns cover email, IBAN, credit card and US SSN; add a phone pattern.
redact_pii = detectors.personal_data(extra={'phone': r'(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)'})

agent = Agent(
    settings().fast_model,
    name='privacy_agent',
    instructions='You are a support assistant. Confirm what the user asked for in one sentence.',
    capabilities=[InputGuardrail(guard=redact_pii), OutputGuardrail(guard=detectors.for_text(redact_pii))],
)

PROMPT = (
    'Please update my contact details: my new email is jane.doe@example.com and my phone is +44 20 7946 0958. '
    'Confirm by repeating them back to me.'
)


def prompt_as_sent(messages: list) -> str:
    """The user prompt as it exists in the run's history after the input guardrail ran."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ''


def main() -> None:
    configure_logfire('gateway-routing', include_content=False)
    settings().require_model_credentials()

    result = agent.run_sync(PROMPT)
    print('original prompt:  ', PROMPT)
    print('prompt as sent:   ', prompt_as_sent(result.all_messages()))
    print('model output:     ', result.output)
    print(
        '\nIn Logfire this run shows the agent span, model request span, tokens and cost, '
        'but no prompt or completion text (include_content=False).'
    )


if __name__ == '__main__':
    main()
