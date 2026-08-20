from __future__ import annotations

import csv
import io
import json

from ordlane.convert.base import BaseConverter, ConversionResult
from ordlane.convert.tokens import estimate_tokens


class CsvToJsonConverter(BaseConverter):
    source_mime = "text/csv"
    target_mime = "application/json"

    def convert(self, data: bytes, filename: str = "") -> ConversionResult:
        text = data.decode("utf-8", errors="replace")
        tokens_before = estimate_tokens(text)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        json_text = json.dumps(rows, ensure_ascii=False, indent=2)
        return ConversionResult(
            content=json_text.encode("utf-8"),
            source_mime=self.source_mime,
            target_mime=self.target_mime,
            text=json_text,
            tokens_before=tokens_before,
            tokens_after=estimate_tokens(json_text),
        )
