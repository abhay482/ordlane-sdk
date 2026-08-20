from __future__ import annotations

import os
from typing import Any

from ordlane.exceptions import ProviderError
from ordlane.models.providers.base import ChatMessage, LLMProvider


class AnthropicProvider(LLMProvider):
    """Claude Messages API (official Anthropic SDK)."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, **extra: Any) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.extra = extra

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("Install Claude support: pip install 'ordlane[anthropic]'") from exc
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=self.api_key)
        system = " ".join(m.content for m in messages if m.role == "system") or None
        user_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if not user_messages:
            user_messages = [{"role": "user", "content": ""}]
        resp = client.messages.create(
            model=model,
            max_tokens=int(kwargs.get("max_tokens") or 1024),
            system=system or anthropic.NOT_GIVEN,
            messages=user_messages,
            temperature=kwargs.get("temperature", 0.2),
        )
        return "".join(getattr(block, "text", "") for block in resp.content)
