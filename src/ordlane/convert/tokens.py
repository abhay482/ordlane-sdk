from __future__ import annotations

import re

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def section_count(text: str) -> int:
    if not text.strip():
        return 0
    headings = re.findall(r"(?m)^#{1,6}\s+\S+", text)
    if headings:
        return len(headings)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    return max(1, len(blocks))
