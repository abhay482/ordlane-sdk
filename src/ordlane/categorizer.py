from __future__ import annotations

import json
import re
from typing import Sequence

from ordlane.exceptions import ConfigError
from ordlane.models.providers.base import ChatMessage, LLMProvider
from ordlane.types import Complexity, ModelConfig, RouteDecision

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def heuristic_route(question: str, models: Sequence[ModelConfig]) -> RouteDecision:
    """Offline fallback when the categorizer model is unavailable."""
    q = question.lower().strip()
    by_id = {m.id: m for m in models}

    simple = any(
        re.search(p, q)
        for p in (r"^(hi|hello|hey)\b", r"\b(classify|format|rewrite|summarize briefly)\b")
    ) and len(q.split()) < 12
    complex_q = any(
        re.search(p, q)
        for p in (
            r"\b(compare|trade-?off|multi-step|analyze|reasoning|strategy|why did|root cause)\b",
        )
    )
    rag_hit = any(
        re.search(p, q)
        for p in (
            r"\b(according to|in the (doc|file|policy)|from our)\b",
            r"\.(json|csv|md|pdf)\b",
            r"\b(what is|explain|list|summarize)\b",
        )
    ) or len(q.split()) > 6

    if simple:
        slot = _pick_by_capability(models, by_id, ("chat", "format", "classify", "fast"), 0)
        complexity: Complexity = "simple"
        needs_rag = False
        reason = "short_simple_query"
        confidence = 0.9
    elif complex_q:
        slot = _pick_by_capability(models, by_id, ("reasoning", "analysis", "complex"), -1)
        complexity = "complex"
        needs_rag = True
        reason = "complex_reasoning_signals"
        confidence = 0.75
    else:
        slot = _pick_by_capability(models, by_id, ("rag", "docs", "standard"), min(1, len(models) - 1))
        complexity = "standard"
        needs_rag = rag_hit
        reason = "factual_or_rag_query"
        confidence = 0.7

    return RouteDecision(
        model_slot_id=slot.id,
        model_id=slot.model_id,
        provider=slot.provider,
        needs_rag=needs_rag,
        confidence=confidence,
        reason=reason,
        complexity=complexity,
        used_llm=False,
    )


def _pick_by_capability(
    models: Sequence[ModelConfig],
    by_id: dict[str, ModelConfig],
    caps: tuple[str, ...],
    fallback_index: int,
) -> ModelConfig:
    lowered = [c.lower() for c in caps]
    for model in models:
        hay = " ".join([model.id, model.role, *model.capabilities]).lower()
        if any(c in hay for c in lowered):
            return model
    return models[fallback_index]


def categorize_question(
    question: str,
    models: Sequence[ModelConfig],
    *,
    provider: LLMProvider,
    model_name: str,
    dry_run: bool = False,
) -> RouteDecision:
    if not models:
        raise ConfigError("At least one specialist model is required")
    if dry_run or getattr(provider, "name", "") == "fake":
        return heuristic_route(question, models)

    catalog = "\n".join(
        f"- id={m.id} provider={m.provider} model={m.model} role={m.role or 'general'} "
        f"capabilities={','.join(m.capabilities) or 'general'}"
        for m in models
    )
    prompt = (
        "You are a question categorizer. Pick the single best specialist model for the user question.\n"
        "Return ONLY JSON with keys: model_id, needs_rag, complexity, confidence, reason.\n"
        "complexity must be one of: simple, standard, complex.\n"
        "model_id must be one of the specialist ids listed.\n\n"
        f"Specialists:\n{catalog}\n\n"
        f"Question:\n{question}"
    )
    raw = provider.complete([ChatMessage(role="user", content=prompt)], model=model_name, temperature=0)
    parsed = _parse_json(raw)
    slot_id = str(parsed.get("model_id") or models[0].id)
    by_id = {m.id: m for m in models}
    slot = by_id.get(slot_id) or models[0]
    complexity = parsed.get("complexity") or "standard"
    if complexity not in {"simple", "standard", "complex"}:
        complexity = "standard"
    return RouteDecision(
        model_slot_id=slot.id,
        model_id=slot.model_id,
        provider=slot.provider,
        needs_rag=bool(parsed.get("needs_rag", True)),
        confidence=float(parsed.get("confidence") or 0.6),
        reason=str(parsed.get("reason") or "categorizer"),
        complexity=complexity,
        used_llm=True,
    )


def _parse_json(raw: str) -> dict:
    match = _JSON_RE.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
