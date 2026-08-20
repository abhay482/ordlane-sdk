from __future__ import annotations

from typing import Any

from ordlane.exceptions import ConfigError
from ordlane.models.providers.anthropic import AnthropicProvider
from ordlane.models.providers.base import LLMProvider
from ordlane.models.providers.bedrock import BedrockProvider
from ordlane.models.providers.fake import FakeProvider
from ordlane.models.providers.langchain import LangChainProvider
from ordlane.models.providers.openai import OpenAIProvider
from ordlane.types import CategorizerConfig, ModelConfig

_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "bedrock": BedrockProvider,
    "langchain": LangChainProvider,
    "fake": FakeProvider,
}


def build_provider(
    provider: str,
    *,
    api_key: str | None = None,
    region: str | None = None,
    extra: dict[str, Any] | None = None,
    model: str | None = None,
) -> LLMProvider:
    extra = dict(extra or {})
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ConfigError(
            f"Unsupported provider '{provider}'. "
            "Use openai, anthropic, bedrock, langchain, or fake."
        )
    kwargs: dict[str, Any] = {**extra}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if region is not None:
        kwargs["region"] = region
    if provider == "langchain" and model is not None:
        kwargs.setdefault("model", model)
    return cls(**kwargs)


def provider_for_model(cfg: ModelConfig) -> LLMProvider:
    return build_provider(
        cfg.provider,
        api_key=cfg.api_key,
        region=cfg.region,
        extra=cfg.extra,
        model=cfg.model,
    )


def provider_for_categorizer(cfg: CategorizerConfig) -> LLMProvider:
    return build_provider(
        cfg.provider,
        api_key=cfg.api_key,
        region=cfg.region,
        extra=cfg.extra,
        model=cfg.model,
    )
