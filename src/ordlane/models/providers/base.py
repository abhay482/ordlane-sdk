from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        raise NotImplementedError
