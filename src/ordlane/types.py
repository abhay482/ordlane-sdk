from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderName = Literal["openai", "anthropic", "bedrock", "fake"]
RAGKind = Literal["none", "naive", "hybrid", "map_reduce", "custom"]
Complexity = Literal["simple", "standard", "complex"]
StorageMode = Literal["READ_AS_IS", "RAG_STORE", "HYBRID"]


@dataclass
class ModelConfig:
    """One specialist model slot. A harness accepts 1-5 of these."""

    id: str
    provider: ProviderName
    model: str
    role: str = ""
    capabilities: list[str] = field(default_factory=list)
    api_key: str | None = None
    region: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class CategorizerConfig:
    """Dedicated question-classification model that picks among specialist slots."""

    provider: ProviderName
    model: str
    api_key: str | None = None
    region: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    model_slot_id: str
    model_id: str
    provider: str
    needs_rag: bool
    confidence: float
    reason: str
    complexity: Complexity
    used_llm: bool = False


@dataclass
class QueryResult:
    answer: str
    routing: dict[str, Any]
    context_mode: str
    context: str
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
