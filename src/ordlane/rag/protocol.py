from __future__ import annotations

from typing import Protocol

from ordlane.types import RetrievedChunk


class RAGBackend(Protocol):
    """Any retriever can plug in: in-memory, vector DB, LangChain store, or a custom callable."""

    name: str

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> int: ...

    def retrieve(self, query: str, *, k: int = 6) -> list[RetrievedChunk]: ...

    def delete(self, doc_id: str) -> None: ...
