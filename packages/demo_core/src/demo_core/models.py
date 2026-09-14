"""Model helpers.

Most demos just pass `settings().model` (a model string such as `gateway/openai:gpt-5.2`)
straight to `Agent(...)` and let Pydantic AI infer the provider. `gateway_model()` is for
the cases that need an explicit Gateway provider object: routing through a named gateway
endpoint, or building a `FallbackModel` chain (see apps/gateway-routing).
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.gateway import gateway_provider

from demo_core.settings import settings

# api_format -> the Pydantic AI model class that speaks that wire format.
_MODEL_CLASSES: dict[str, type[Model]] = {
    'openai': OpenAIResponsesModel,
    'openai-responses': OpenAIResponsesModel,
    'openai-chat': OpenAIChatModel,
    'anthropic': AnthropicModel,
    'google-cloud': GoogleModel,
}


def gateway_model(api_format: str, model_name: str, *, route: str | None = None, api_key: str | None = None) -> Model:
    """Build a model routed through the Pydantic AI Gateway.

    `api_format` selects the wire format and model class (`openai`, `openai-chat`, `anthropic`,
    `google-cloud`). `route` overrides the provider slug with a gateway endpoint slug, which is
    how failover and load balancing configured in Logfire get used from code.
    """
    try:
        model_cls = _MODEL_CLASSES[api_format]
    except KeyError:
        raise ValueError(f'Unsupported api_format {api_format!r}; expected one of {sorted(_MODEL_CLASSES)}') from None
    provider = gateway_provider(api_format, route=route, api_key=api_key or settings().gateway_api_key)  # type: ignore[arg-type]
    return model_cls(model_name, provider=provider)  # type: ignore[call-arg]
