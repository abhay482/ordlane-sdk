from __future__ import annotations

from typing import Any, TypedDict

from ordlane.exceptions import RAGError
from ordlane.rag.protocol import RAGBackend
from ordlane.types import RAGKind, RetrievedChunk


class GraphState(TypedDict, total=False):
    question: str
    needs_rag: bool
    rag_kind: str
    context: str
    chunks: list[dict[str, Any]]
    context_mode: str


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[chunk {c.id} score={c.score:.2f}]\n{c.text}" for c in chunks)


def assemble_context(
    question: str,
    backend: RAGBackend,
    *,
    kind: RAGKind = "hybrid",
    k: int = 6,
    needs_rag: bool = True,
) -> tuple[str, str, list[RetrievedChunk]]:
    if kind == "none" or not needs_rag:
        return "", "none", []

    chunks = backend.retrieve(question, k=k)
    if not chunks:
        return "", "none", []

    if kind == "naive":
        return format_chunks(chunks), "rag_chunks", chunks

    if kind == "map_reduce":
        summaries = [c.text[:500] for c in chunks]
        text = "Summarized evidence:\n\n" + "\n\n".join(f"- {s}" for s in summaries)
        return text, "map_reduce", chunks

    # hybrid / custom: chunks plus any full docs the backend exposes
    full_parts: list[str] = []
    docs = getattr(backend, "docs", None)
    if isinstance(docs, dict):
        q = question.lower()
        for doc in docs.values():
            name = str((doc.metadata or {}).get("filename") or doc.id)
            mode = getattr(doc, "storage_mode", "RAG_STORE")
            if name.lower() in q or any(part in q for part in name.lower().replace("_", " ").split(".")):
                if mode in {"READ_AS_IS", "HYBRID"}:
                    full_parts.append(f"### File: {name}\n\n{doc.text}")
    if full_parts:
        text = "\n\n".join(full_parts + [format_chunks(chunks)])
        return text, "hybrid", chunks
    return format_chunks(chunks), "rag_chunks", chunks


def build_langgraph_rag(backend: RAGBackend, *, kind: RAGKind = "hybrid", k: int = 6):
    """Compile a LangGraph retrieve ? assemble graph around any RAGBackend."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RAGError("Install LangGraph support: pip install 'ordlane[langgraph]'") from exc

    def retrieve_node(state: GraphState) -> GraphState:
        question = state.get("question") or ""
        needs_rag = bool(state.get("needs_rag", True))
        rag_kind = state.get("rag_kind") or kind
        context, mode, chunks = assemble_context(
            question,
            backend,
            kind=rag_kind,  # type: ignore[arg-type]
            k=k,
            needs_rag=needs_rag,
        )
        return {
            **state,
            "context": context,
            "context_mode": mode,
            "chunks": [{"id": c.id, "text": c.text, "score": c.score, "metadata": c.metadata} for c in chunks],
        }

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()
