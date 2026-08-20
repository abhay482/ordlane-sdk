"""Architectural boundary tests for RAG assembly and context optimization."""

from __future__ import annotations

from ordlane.optimizer import optimize_context
from ordlane.rag import InMemoryRAG, assemble_context
from ordlane.rag.backends import decide_storage_mode


def test_retrieval_with_no_results_returns_empty_context():
    rag = InMemoryRAG()
    context, mode, chunks = assemble_context(
        "anything about chargebacks",
        rag,
        kind="hybrid",
        needs_rag=True,
    )
    assert context == ""
    assert mode == "none"
    assert chunks == []


def test_map_reduce_summarizes_retrieved_chunks():
    rag = InMemoryRAG()
    rag.ingest(
        "policy",
        "Visa CNP chargeback window is 120 days. Mid-market settlement SLA is 24 hours.",
        metadata={"filename": "settlement_policy.md"},
    )
    context, mode, chunks = assemble_context(
        "chargeback window",
        rag,
        kind="map_reduce",
        needs_rag=True,
    )
    assert mode == "map_reduce"
    assert chunks
    assert context.startswith("Summarized evidence:")
    assert "120 days" in context


def test_hybrid_injects_named_file_plus_chunks():
    rag = InMemoryRAG()
    # Small policy files land in READ_AS_IS / HYBRID and can be injected in full.
    policy = "# Settlement policy\n\nVisa CNP chargeback window is 120 days.\n"
    assert decide_storage_mode("settlement_policy.md", policy) in {"READ_AS_IS", "HYBRID"}
    rag.ingest("policy", policy, metadata={"filename": "settlement_policy.md"})
    rag.ingest(
        "other",
        "Unrelated merchant onboarding notes about KYC refresh cycles.",
        metadata={"filename": "onboarding.md"},
    )
    context, mode, chunks = assemble_context(
        "What is the Visa chargeback window in settlement_policy.md?",
        rag,
        kind="hybrid",
        needs_rag=True,
    )
    assert mode == "hybrid"
    assert "### File: settlement_policy.md" in context
    assert chunks


def test_optimizer_truncates_and_preserves_original_order():
    # High-relevance chunk is last in source order; after budget cut, relative order is kept.
    text = "\n\n".join(
        [
            "[chunk a score=0.1]\n" + ("weather forecast update " * 80),
            "[chunk b score=0.2]\n" + ("office snacks menu " * 80),
            "[chunk c score=0.9]\nchargeback visa window 120 days " + ("policy detail " * 40),
            "[chunk d score=0.8]\nsettlement sla mid-market 24 hours " + ("ops note " * 40),
        ]
    )
    optimized = optimize_context(text, "visa chargeback window", max_tokens=120)
    assert optimized.compressed is True
    assert optimized.tokens_after <= optimized.tokens_before
    assert "chargeback" in optimized.text.lower()

    positions = []
    for marker in ("[chunk a", "[chunk b", "[chunk c", "[chunk d"):
        idx = optimized.text.find(marker)
        if idx >= 0:
            positions.append((marker, idx))
    assert positions, "expected at least one kept chunk"
    # Kept markers must remain in original relative order.
    assert positions == sorted(positions, key=lambda item: item[1])


def test_needs_rag_false_skips_retrieval():
    rag = InMemoryRAG()
    rag.ingest("policy", "Visa chargeback window is 120 days.", metadata={"filename": "policy.md"})
    context, mode, chunks = assemble_context("hello", rag, kind="hybrid", needs_rag=False)
    assert context == ""
    assert mode == "none"
    assert chunks == []
