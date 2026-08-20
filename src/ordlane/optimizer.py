from __future__ import annotations

import re
from dataclasses import dataclass

from ordlane.convert.tokens import estimate_tokens


@dataclass
class OptimizedContext:
    text: str
    tokens_before: int
    tokens_after: int
    compressed: bool
    dropped_chunks: int = 0


def _split_chunks(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"\n(?=### File:|\[chunk )", text)
    return [p for p in parts if p.strip()]


def optimize_context(text: str, query: str, *, max_tokens: int = 4000) -> OptimizedContext:
    before = estimate_tokens(text)
    if before <= max_tokens:
        return OptimizedContext(text=text, tokens_before=before, tokens_after=before, compressed=False)

    query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
    parts = _split_chunks(text)
    if not parts:
        words = text.split()
        ratio = max_tokens / max(before, 1)
        keep = max(50, int(len(words) * ratio))
        trimmed = " ".join(words[:keep])
        return OptimizedContext(
            text=trimmed + "\n\n[truncated for token budget]",
            tokens_before=before,
            tokens_after=estimate_tokens(trimmed),
            compressed=True,
        )

    scored: list[tuple[float, str]] = []
    for part in parts:
        tokens = set(re.findall(r"[a-z0-9_]+", part.lower()))
        score = len(query_terms & tokens) / max(len(query_terms), 1)
        if part.startswith("### File:"):
            score += 0.15
        scored.append((score, part))
    scored.sort(key=lambda x: x[0], reverse=True)

    kept: list[str] = []
    dropped = 0
    running = 0
    for _score, part in scored:
        part_tokens = estimate_tokens(part)
        if running + part_tokens > max_tokens and kept:
            dropped += 1
            continue
        kept.append(part)
        running += part_tokens

    kept_set = set(kept)
    ordered = [p for p in parts if p in kept_set]
    out = "\n\n".join(ordered)
    if dropped:
        out += f"\n\n[{dropped} less-relevant chunk(s) dropped for token budget]"
    return OptimizedContext(
        text=out,
        tokens_before=before,
        tokens_after=estimate_tokens(out),
        compressed=True,
        dropped_chunks=dropped,
    )
