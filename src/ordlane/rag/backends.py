from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ordlane.convert.tokens import estimate_tokens, section_count
from ordlane.types import RetrievedChunk, StorageMode


@dataclass
class IndexedDoc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    storage_mode: StorageMode = "RAG_STORE"


def decide_storage_mode(filename: str, text: str) -> StorageMode:
    tokens = estimate_tokens(text)
    sections = section_count(text)
    name = filename.lower()
    if tokens <= 6000 and sections <= 3:
        return "READ_AS_IS"
    if tokens >= 20000 or sections >= 8:
        return "RAG_STORE"
    if any(h in name for h in ("readme", "policy", "sla", "faq", "schema")):
        return "HYBRID"
    return "HYBRID"


def chunk_text(text: str, doc_id: str, *, chunk_size: int = 800, overlap: int = 100, metadata: dict | None = None) -> list[RetrievedChunk]:
    metadata = metadata or {}
    words = text.split()
    if not words:
        return []
    chunks: list[RetrievedChunk] = []
    step = max(1, chunk_size - overlap)
    idx = 0
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size])
        if not piece.strip():
            continue
        digest = hashlib.sha1(f"{doc_id}:{idx}:{piece[:64]}".encode()).hexdigest()[:12]
        chunks.append(
            RetrievedChunk(
                id=f"{doc_id}-{idx}-{digest}",
                text=piece,
                score=0.0,
                metadata={**metadata, "doc_id": doc_id, "index": idx},
            )
        )
        idx += 1
        if start + chunk_size >= len(words):
            break
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class InMemoryRAG:
    """Embedding-free overlap retriever. Drop-in default with no extra services."""

    name = "in_memory"

    def __init__(self) -> None:
        self._chunks: dict[str, list[RetrievedChunk]] = {}
        self.docs: dict[str, IndexedDoc] = {}

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        metadata = metadata or {}
        filename = str(metadata.get("filename") or doc_id)
        mode = decide_storage_mode(filename, text)
        self.docs[doc_id] = IndexedDoc(id=doc_id, text=text, metadata=metadata, storage_mode=mode)
        chunks = chunk_text(text, doc_id, metadata=metadata)
        self._chunks[doc_id] = chunks
        return len(chunks)

    def delete(self, doc_id: str) -> None:
        self._chunks.pop(doc_id, None)
        self.docs.pop(doc_id, None)

    def retrieve(self, query: str, *, k: int = 6) -> list[RetrievedChunk]:
        query_tokens = set(_tokenize(query))
        scored: list[RetrievedChunk] = []
        for chunks in self._chunks.values():
            for chunk in chunks:
                chunk_tokens = set(_tokenize(chunk.text))
                if not query_tokens or not chunk_tokens:
                    score = 0.0
                else:
                    score = len(query_tokens & chunk_tokens) / len(query_tokens)
                if score > 0:
                    scored.append(
                        RetrievedChunk(id=chunk.id, text=chunk.text, score=score, metadata=chunk.metadata)
                    )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


class CallableRAG:
    """Wrap any `retrieve(query, k) -> list[RetrievedChunk]` function as a RAG backend."""

    name = "custom"

    def __init__(self, retrieve_fn, ingest_fn=None, delete_fn=None) -> None:
        self._retrieve_fn = retrieve_fn
        self._ingest_fn = ingest_fn
        self._delete_fn = delete_fn

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        if self._ingest_fn is None:
            return 0
        return int(self._ingest_fn(doc_id, text, metadata or {}) or 0)

    def retrieve(self, query: str, *, k: int = 6) -> list[RetrievedChunk]:
        return list(self._retrieve_fn(query, k=k))

    def delete(self, doc_id: str) -> None:
        if self._delete_fn:
            self._delete_fn(doc_id)


class LangChainVectorRAG:
    """Adapter for LangChain-style vector stores (`similarity_search` / `add_texts`)."""

    name = "langchain_vector"

    def __init__(self, vectorstore) -> None:
        self.vectorstore = vectorstore

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        chunks = chunk_text(text, doc_id, metadata=metadata)
        if hasattr(self.vectorstore, "add_texts"):
            self.vectorstore.add_texts(
                [c.text for c in chunks],
                metadatas=[{**c.metadata, "chunk_id": c.id} for c in chunks],
                ids=[c.id for c in chunks],
            )
        return len(chunks)

    def retrieve(self, query: str, *, k: int = 6) -> list[RetrievedChunk]:
        docs = self.vectorstore.similarity_search(query, k=k)
        out: list[RetrievedChunk] = []
        for i, doc in enumerate(docs):
            meta = dict(getattr(doc, "metadata", {}) or {})
            out.append(
                RetrievedChunk(
                    id=str(meta.get("chunk_id") or f"lc-{i}"),
                    text=getattr(doc, "page_content", str(doc)),
                    score=float(meta.get("score") or 1.0 - i * 0.05),
                    metadata=meta,
                )
            )
        return out

    def delete(self, doc_id: str) -> None:
        if hasattr(self.vectorstore, "delete"):
            self.vectorstore.delete(ids=None, filter={"doc_id": doc_id})
