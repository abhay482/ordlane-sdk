from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ConversionResult:
    content: bytes
    source_mime: str
    target_mime: str
    text: str
    tokens_before: int
    tokens_after: int
    skipped: bool = False
    warning: str = ""

    @property
    def savings_ratio(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.tokens_after / self.tokens_before))


class BaseConverter(ABC):
    source_mime: str
    target_mime: str

    @abstractmethod
    def convert(self, data: bytes, filename: str = "") -> ConversionResult:
        raise NotImplementedError
