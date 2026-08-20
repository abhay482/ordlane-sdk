from __future__ import annotations

from ordlane.convert.base import BaseConverter, ConversionResult
from ordlane.convert.tokens import estimate_tokens


class HtmlToMarkdownConverter(BaseConverter):
    source_mime = "text/html"
    target_mime = "text/markdown"

    def convert(self, data: bytes, filename: str = "") -> ConversionResult:
        html = data.decode("utf-8", errors="replace")
        tokens_before = estimate_tokens(html)
        try:
            from markdownify import markdownify as md
        except ImportError:
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text=html,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning="Install HTML support: pip install 'ordlane[html]'",
            )
        markdown = md(html, heading_style="ATX", strip=["script", "style"])
        tokens_after = estimate_tokens(markdown)
        return ConversionResult(
            content=markdown.encode("utf-8"),
            source_mime=self.source_mime,
            target_mime=self.target_mime,
            text=markdown,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
