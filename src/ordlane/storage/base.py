from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

StorageKind = Literal["local", "s3"]


@dataclass
class StoredObject:
    uri: str
    kind: StorageKind
    key: str
    bytes_written: int
    content_type: str = ""
    metadata: dict[str, Any] | None = None


class StorageBackend(ABC):
    """Write converted (or original) bytes to a destination."""

    kind: StorageKind

    @abstractmethod
    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        raise NotImplementedError
