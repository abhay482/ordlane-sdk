"""Architectural boundary tests for routing / categorizer behavior."""

from __future__ import annotations

from ordlane import CategorizerConfig, Harness, ModelConfig
from ordlane.categorizer import categorize_question
from ordlane.exceptions import ProviderError
from ordlane.models.providers.base import ChatMessage, LLMProvider


class _StubCategorizer(LLMProvider):
    name = "stub"

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs) -> str:
        return self.raw


class _FailingSpecialist(LLMProvider):
    name = "stub"

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs) -> str:
        raise ProviderError("upstream timeout")


def _models() -> list[ModelConfig]:
    return [
        ModelConfig(id="fast", provider="fake", model="mini", capabilities=["fast", "chat"]),
        ModelConfig(id="docs", provider="fake", model="standard", capabilities=["rag", "docs"]),
        ModelConfig(id="reason", provider="fake", model="opus", capabilities=["reasoning"]),
    ]


def test_unknown_categorizer_model_id_falls_back_to_first_specialist():
    decision = categorize_question(
        "What is in the policy?",
        _models(),
        provider=_StubCategorizer('{"model_id":"does-not-exist","needs_rag":true,"complexity":"standard","confidence":0.9,"reason":"bad id"}'),
        model_name="router",
        dry_run=False,
    )
    assert decision.model_slot_id == "fast"
    assert decision.used_llm is True
    assert decision.needs_rag is True


def test_malformed_categorizer_json_falls_back_safely():
    decision = categorize_question(
        "Explain settlement SLA",
        _models(),
        provider=_StubCategorizer("sorry, I cannot follow instructions {{{ not json"),
        model_name="router",
        dry_run=False,
    )
    assert decision.model_slot_id == "fast"
    assert decision.complexity == "standard"
    assert decision.used_llm is True
    assert decision.reason == "categorizer"


def test_invalid_complexity_normalized():
    decision = categorize_question(
        "hello",
        _models(),
        provider=_StubCategorizer('{"model_id":"docs","needs_rag":false,"complexity":"ultra","confidence":0.2,"reason":"ok"}'),
        model_name="router",
        dry_run=False,
    )
    assert decision.model_slot_id == "docs"
    assert decision.complexity == "standard"


def test_provider_failure_falls_back_instead_of_raising():
    class FailingLLM:
        def invoke(self, messages, **kwargs):
            raise RuntimeError("boom")

    h = Harness(
        [
            ModelConfig(
                id="docs",
                provider="langchain",
                model="custom",
                capabilities=["rag", "docs", "fast"],
                extra={"llm": FailingLLM()},
            )
        ],
        CategorizerConfig(provider="fake", model="router"),
        dry_run=False,
        use_langgraph=False,
        rag_kind="none",
    )
    result = h.ask("What is the SLA according to policy.md?")
    assert result.routing["model_slot_id"] == "docs"
    assert result.answer.startswith("[provider_error:docs]")
