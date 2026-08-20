from __future__ import annotations

import os
from typing import Any

from ordlane.exceptions import ProviderError
from ordlane.models.providers.base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, **extra: Any) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.extra = extra

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install OpenAI support: pip install 'ordlane[openai]'") from exc
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=self.api_key, **{k: v for k, v in self.extra.items() if k in {"base_url", "organization"}})
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens"),
        )
        return resp.choices[0].message.content or ""
