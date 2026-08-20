from __future__ import annotations

import csv
import io
import json
from typing import Any

from ordlane.convert.base import BaseConverter, ConversionResult
from ordlane.convert.tokens import estimate_tokens


def _flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            _flatten(value, path, out)
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix or "value"] = "|".join(str(x) for x in obj)
        else:
            for idx, value in enumerate(obj):
                path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                _flatten(value, path, out)
    else:
        out[prefix or "value"] = obj
    return out


def _rows_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(item, dict) for item in data):
            return [_flatten(item) for item in data]
        return [{"value": json.dumps(item, ensure_ascii=False)} for item in data]
    if isinstance(data, dict):
        for _key, value in data.items():
            if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
                return [_flatten(item) for item in value]
        return [_flatten(data)]
    return [{"value": str(data)}]


class JsonToCsvConverter(BaseConverter):
    source_mime = "application/json"
    target_mime = "text/csv"

    def convert(self, data: bytes, filename: str = "") -> ConversionResult:
        original_text = data.decode("utf-8", errors="replace")
        tokens_before = estimate_tokens(original_text)
        try:
            parsed = json.loads(original_text)
        except json.JSONDecodeError as exc:
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text=original_text,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning=f"Invalid JSON: {exc}",
            )

        rows = _rows_from_json(parsed)
        if not rows:
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text=original_text,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning="Empty JSON - conversion skipped",
            )

        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        csv_text = buffer.getvalue()
        tokens_after = estimate_tokens(csv_text)

        if tokens_after > tokens_before * 1.05:
            return ConversionResult(
                content=data,
                source_mime=self.source_mime,
                target_mime=self.source_mime,
                text=original_text,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                skipped=True,
                warning="CSV would use more tokens - kept original JSON",
            )

        return ConversionResult(
            content=csv_text.encode("utf-8"),
            source_mime=self.source_mime,
            target_mime=self.target_mime,
            text=csv_text,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
