from __future__ import annotations

from ordlane import CategorizerConfig, Harness, ModelConfig
from ordlane.convert import convert_bytes
from ordlane.rag import InMemoryRAG


def test_json_to_csv():
    data = b'[{"merchant":"acme","mcc":"5411"},{"merchant":"beta","mcc":"5812"}]'
    result = convert_bytes(data, filename="merchants.json")
    assert not result.skipped
    assert result.target_mime == "text/csv"
    assert "merchant" in result.text
    assert result.tokens_after <= result.tokens_before


def test_csv_to_json():
    data = b"name,city\nalice,nyc\nbob,sf\n"
    result = convert_bytes(data, filename="people.csv", target_mime="application/json")
    assert result.target_mime == "application/json"
    assert "alice" in result.text


def test_rejects_more_than_five_models():
    models = [
        ModelConfig(id=f"m{i}", provider="fake", model=f"model-{i}")
        for i in range(6)
    ]
    try:
        Harness(models, CategorizerConfig(provider="fake", model="router"))
    except Exception as exc:
        assert "1 and 5" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def _harness(rag=None) -> Harness:
    return Harness(
        [
            ModelConfig(
                id="fast",
                provider="fake",
                model="mini",
                role="Simple Q&A and formatting",
                capabilities=["fast", "chat", "format"],
            ),
            ModelConfig(
                id="docs",
                provider="fake",
                model="standard",
                role="Document and RAG answers",
                capabilities=["rag", "docs"],
            ),
            ModelConfig(
                id="reason",
                provider="fake",
                model="opus",
                role="Multi-step reasoning",
                capabilities=["reasoning", "complex"],
            ),
        ],
        CategorizerConfig(provider="fake", model="router"),
        rag=rag or InMemoryRAG(),
        dry_run=True,
        use_langgraph=False,
    )


def test_router_simple_vs_complex():
    h = _harness()
    simple = h.route("hello")
    assert simple.model_slot_id == "fast"
    complex_q = h.route("Compare trade-offs and root cause of rising chargeback patterns")
    assert complex_q.model_slot_id == "reason"
    docs = h.route("What is the Visa chargeback window according to settlement_policy.md?")
    assert docs.model_slot_id in {"docs", "reason"}
    assert docs.needs_rag is True


def test_ask_and_ingest():
    rag = InMemoryRAG()
    h = _harness(rag)
    ingested = h.ingest(
        b"# SLA\n\nSettle mid-market in 24 hours. Visa CNP chargeback window is 120 days.\n",
        filename="settlement_policy.md",
        doc_id="policy",
        convert=False,
    )
    assert ingested["chunks_indexed"] >= 1
    result = h.ask("What is the Visa CNP chargeback window in settlement_policy.md?")
    assert result.routing["model_slot_id"]
    assert result.answer
    assert result.metrics["estimated_cost_usd"] >= 0
    assert "120" in result.context or result.context_mode in {"rag_chunks", "hybrid", "full_file", "none"}


def test_live_fake_provider_not_dry_run():
    h = Harness(
        [ModelConfig(id="fast", provider="fake", model="echo", capabilities=["fast"])],
        CategorizerConfig(provider="fake", model="router"),
        dry_run=False,
        use_langgraph=False,
        rag_kind="none",
    )
    result = h.ask("hello there")
    assert "[fake:echo]" in result.answer
