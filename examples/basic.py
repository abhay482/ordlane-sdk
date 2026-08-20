"""Minimal plug-and-play example (no API keys; fake provider)."""

from ordlane import CategorizerConfig, Harness, ModelConfig

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
            role="Knowledge-base and policy questions",
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

# Convert through Harness (JSON -> CSV). See also examples/convert_with_harness.py
converted = harness.convert(
    data=b'[{"merchant":"acme","mcc":"5411"}]',
    filename="merchants.json",
)
print("converted", converted.source_mime, "->", converted.target_mime)
print(converted.text)

# ingest(..., convert=True) converts first, then indexes for RAG
harness.ingest(
    b"# Settlement policy\n\nVisa CNP chargeback window is 120 days.\n",
    filename="settlement_policy.md",
    convert=True,
)
result = harness.ask("What is the Visa chargeback window according to settlement_policy.md?")
print(result.routing)
print(result.answer)
