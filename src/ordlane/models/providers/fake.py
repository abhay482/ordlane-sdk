from __future__ import annotations

from typing import Any

from ordlane.models.providers.base import ChatMessage, LLMProvider


class FakeProvider(LLMProvider):
    """Deterministic provider for tests and dry-run demos (no network)."""

    name = "fake"

    def __init__(self, canned: str | None = None, **_extra: Any) -> None:
        self.canned = canned

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        if self.canned:
            return self.canned
        last = messages[-1].content if messages else ""
        return f"[fake:{model}] {last[:400]}"
