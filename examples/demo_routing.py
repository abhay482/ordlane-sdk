"""Terminal demo: routing questions to different specialists (dry_run / heuristic).

Run:
  PYTHONPATH=src python examples/demo_routing.py

Record GIF (requires VHS):
  vhs demos/routing.tape
"""

from __future__ import annotations

from ordlane import CategorizerConfig, Harness, ModelConfig

harness = Harness(
    models=[
        ModelConfig(
            id="fast",
            provider="fake",
            model="gpt-4o-mini",
            role="Greetings and short answers",
            capabilities=["fast", "chat", "format"],
        ),
        ModelConfig(
            id="docs",
            provider="fake",
            model="gpt-4o",
            role="Document and policy Q&A",
            capabilities=["rag", "docs"],
        ),
        ModelConfig(
            id="reason",
            provider="fake",
            model="claude-sonnet",
            role="Multi-step reasoning",
            capabilities=["reasoning", "complex"],
        ),
    ],
    categorizer=CategorizerConfig(provider="fake", model="router"),
    dry_run=True,
    use_langgraph=False,
    rag_kind="none",
)

QUESTIONS = [
    "hello",
    "Compare trade-offs and root cause of rising chargeback patterns",
    "What is the Visa chargeback window in settlement_policy.md?",
]

print("Ordlane routing demo")
print("mode: dry_run=True (heuristic router, zero API calls)")
print("-" * 72)

for question in QUESTIONS:
    decision = harness.route(question)
    print()
    print(f"Q: {question}")
    print(
        f"  specialist : {decision.model_slot_id}"
        f"  ({decision.model_id})"
    )
    print(
        f"  complexity : {decision.complexity}"
        f"  | needs_rag: {decision.needs_rag}"
        f"  | confidence: {decision.confidence:.2f}"
    )
    print(f"  reason     : {decision.reason}")

print()
print("-" * 72)
print("Same Harness API with live categorizer when dry_run=False")
