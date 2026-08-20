"""File conversion through Harness, then index and ask a specialist model.

Shows the same convert + ingest path used inside Harness:
  convert()  - JSON->CSV, HTML->Markdown, CSV->JSON, PDF->Markdown
  ingest()   - converts first (convert=True), then stores text in RAG
  ask()      - categorizer picks a specialist from the Harness models
"""

from pathlib import Path

from ordlane import CategorizerConfig, Harness, ModelConfig

ROOT = Path(__file__).resolve().parent / "sample_files"
ROOT.mkdir(exist_ok=True)

(ROOT / "merchants.json").write_text(
    """[
  {"merchant": "acme", "mcc": "5411", "country": "US"},
  {"merchant": "beta", "mcc": "5812", "country": "IN"}
]
""",
    encoding="utf-8",
)
(ROOT / "policy.html").write_text(
    """<!doctype html>
<html><body>
<h1>Settlement policy</h1>
<p>Visa CNP chargeback window is 120 days.</p>
<p>Mid-market settlement SLA is 24 hours.</p>
</body></html>
""",
    encoding="utf-8",
)

harness = Harness(
    models=[
        ModelConfig(
            id="fast",
            provider="fake",
            model="mini",
            role="Greetings, formatting, short answers",
            capabilities=["fast", "chat"],
        ),
        ModelConfig(
            id="docs",
            provider="fake",
            model="standard",
            role="Document and converted-file Q&A",
            capabilities=["rag", "docs"],
        ),
        ModelConfig(
            id="reason",
            provider="fake",
            model="opus",
            role="Multi-step analysis",
            capabilities=["reasoning"],
        ),
    ],
    categorizer=CategorizerConfig(provider="fake", model="router"),
    dry_run=True,
)

print("Registered converters:", harness.converters())

# 1) Convert a file on disk (JSON -> CSV by default)
csv_result = harness.convert(path=str(ROOT / "merchants.json"))
print("\nJSON ->", csv_result.target_mime, "skipped=", csv_result.skipped)
print(csv_result.text)

# 2) Convert in-memory bytes and pick the target type
csv_bytes = csv_result.content
json_again = harness.convert(
    data=csv_bytes,
    filename="merchants.csv",
    target_mime="application/json",
)
print("CSV ->", json_again.target_mime)

# 3) Convert HTML -> Markdown
md_result = harness.convert(path=str(ROOT / "policy.html"))
print("\nHTML ->", md_result.target_mime, "skipped=", md_result.skipped)
print(md_result.text)

# 4) ingest() runs the same converter, then indexes converted text for RAG
indexed = harness.ingest(path=str(ROOT / "policy.html"), doc_id="settlement_policy")
print("\nIngest conversion:", indexed["conversion"])
print("Chunks indexed:", indexed["chunks_indexed"])

# Convert + store to local disk (no RAG). See examples/store_converted.py for S3.
stored = harness.convert(
    path=str(ROOT / "merchants.json"),
    store_to=str(ROOT / "out"),
)
print("\nStored converted file:", stored["stored"]["uri"])

# Optional: ingest already-converted bytes under a stable filename
harness.ingest(
    data=csv_result.content,
    filename="merchants.csv",
    doc_id="merchants",
)

# 5) Categorizer picks a Harness model; RAG uses converted knowledge
result = harness.ask("What is the Visa chargeback window according to policy.html?")
print("\nRouted to:", result.routing)
print("Context mode:", result.context_mode)
print("Answer:", result.answer)

# PDF path (needs pip install 'ordlane[pdf]'):
# pdf = harness.convert(path="report.pdf")           # PDF -> Markdown
# harness.ingest(path="report.pdf", doc_id="report") # convert + index
# harness.ask("Summarize report.pdf")
