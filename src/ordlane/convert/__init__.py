from __future__ import annotations

import mimetypes
from pathlib import Path

from ordlane.convert.base import BaseConverter, ConversionResult
from ordlane.convert.csv_json import CsvToJsonConverter
from ordlane.convert.html_md import HtmlToMarkdownConverter
from ordlane.convert.json_csv import JsonToCsvConverter
from ordlane.convert.pdf_md import PdfToMarkdownConverter
from ordlane.convert.tokens import estimate_tokens

_REGISTRY: dict[tuple[str, str], BaseConverter] = {
    ("application/json", "text/csv"): JsonToCsvConverter(),
    ("text/csv", "application/json"): CsvToJsonConverter(),
    ("application/pdf", "text/markdown"): PdfToMarkdownConverter(),
    ("text/html", "text/markdown"): HtmlToMarkdownConverter(),
}

_DEFAULT_TARGETS: dict[str, str] = {
    "application/json": "text/csv",
    "text/csv": "application/json",
    "application/pdf": "text/markdown",
    "text/html": "text/markdown",
}


def list_converters() -> list[dict[str, str]]:
    return [
        {"source_mime": src, "target_mime": tgt, "converter": type(conv).__name__}
        for (src, tgt), conv in _REGISTRY.items()
    ]


def register_converter(converter: BaseConverter) -> None:
    _REGISTRY[(converter.source_mime, converter.target_mime)] = converter


def get_converter(source_mime: str, target_mime: str) -> BaseConverter | None:
    return _REGISTRY.get((source_mime, target_mime))


def detect_mime(filename: str, data: bytes | None = None) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    if data:
        head = data[:200].lstrip()
        if head.startswith(b"{") or head.startswith(b"["):
            return "application/json"
        if head.startswith(b"%PDF"):
            return "application/pdf"
        if head.lower().startswith(b"<!doctype") or head.lower().startswith(b"<html"):
            return "text/html"
    suffix = Path(filename).suffix.lower()
    return {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".pdf": "application/pdf",
    }.get(suffix, "application/octet-stream")


def suggest_target(source_mime: str) -> str | None:
    return _DEFAULT_TARGETS.get(source_mime)


def convert_bytes(
    data: bytes,
    filename: str = "",
    source_mime: str | None = None,
    target_mime: str | None = None,
) -> ConversionResult:
    source_mime = source_mime or detect_mime(filename, data)
    target = target_mime or suggest_target(source_mime)
    if not target or target == source_mime:
        text = data.decode("utf-8", errors="replace") if not source_mime.endswith("pdf") else ""
        tokens = estimate_tokens(text) if text else max(1, len(data) // 8)
        return ConversionResult(
            content=data,
            source_mime=source_mime,
            target_mime=source_mime,
            text=text,
            tokens_before=tokens,
            tokens_after=tokens,
            skipped=True,
            warning="No conversion available for this type" if target != source_mime else "",
        )

    converter = get_converter(source_mime, target)
    if converter is None:
        text = data.decode("utf-8", errors="replace")
        tokens = estimate_tokens(text)
        return ConversionResult(
            content=data,
            source_mime=source_mime,
            target_mime=source_mime,
            text=text,
            tokens_before=tokens,
            tokens_after=tokens,
            skipped=True,
            warning=f"No converter registered for {source_mime} -> {target}",
        )
    return converter.convert(data, filename=Path(filename).name)


def convert_file(path: str | Path, *, target_mime: str | None = None) -> ConversionResult:
    path = Path(path)
    return convert_bytes(path.read_bytes(), filename=path.name, target_mime=target_mime)
