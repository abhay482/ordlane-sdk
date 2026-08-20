from __future__ import annotations

import io

from ordlane.convert.base import BaseConverter, ConversionResult
from ordlane.convert.tokens import estimate_tokens


class PdfToMarkdownConverter(BaseConverter):
    source_mime = "application/pdf"
    target_mime = "text/markdown"

    def convert(self, data: bytes, filename: str = "") -> ConversionResult:
        tokens_before = max(1, len(data) // 8)
        try:
            from pypdf import PdfReader
        except ImportError:
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text="",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning="Install PDF support: pip install 'ordlane[pdf]'",
            )
        try:
            reader = PdfReader(io.BytesIO(data))
            pages: list[str] = []
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"## Page {i}\n\n{text}")
            md = f"# {filename or 'Document'}\n\n" + "\n\n".join(pages)
            if not pages:
                return ConversionResult(
                    content=data,
                    source_mime=self.source_mime,
                    target_mime=self.source_mime,
                    text="",
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    skipped=True,
                    warning="No extractable text in PDF",
                )
            tokens_after = estimate_tokens(md)
            raw = "\n\n".join(p.split("\n\n", 1)[-1] for p in pages)
            tokens_before = estimate_tokens(raw) or tokens_before
            return ConversionResult(
                content=md.encode("utf-8"),
                source_mime=self.source_mime,
                target_mime=self.target_mime,
                text=md,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
            )
        except Exception as exc:  # pragma: no cover
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text="",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning=f"PDF conversion failed: {exc}",
            )
