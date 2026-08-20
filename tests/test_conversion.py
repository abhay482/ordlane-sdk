"""Architectural boundary tests for file conversion."""

from __future__ import annotations

import json

from ordlane.convert import convert_bytes


def test_nested_json_flattens_into_csv_columns():
    payload = {
        "merchants": [
            {
                "name": "acme",
                "profile": {"country": "US", "mcc": "5411"},
                "tags": ["retail", "priority"],
            },
            {
                "name": "beta",
                "profile": {"country": "IN", "mcc": "5812"},
                "tags": ["food"],
            },
        ]
    }
    result = convert_bytes(json.dumps(payload).encode(), filename="merchants.json")
    assert not result.skipped
    assert result.target_mime == "text/csv"
    assert "profile.country" in result.text
    assert "tags" in result.text
    assert "acme" in result.text
    assert "priority" in result.text or "retail|priority" in result.text


def test_invalid_json_is_skipped_with_warning():
    result = convert_bytes(b'{"merchant":', filename="broken.json")
    assert result.skipped is True
    assert result.target_mime == "application/json"
    assert "Invalid JSON" in result.warning


def test_empty_json_array_is_skipped():
    result = convert_bytes(b"[]", filename="empty.json")
    assert result.skipped is True
    assert "Empty JSON" in result.warning


def test_invalid_csv_still_round_trips_to_json():
    # DictReader tolerates odd rows; conversion should not raise.
    result = convert_bytes(b"name,city\nalice\nbob,sf,extra\n", filename="messy.csv")
    assert result.target_mime == "application/json"
    assert "alice" in result.text or result.text.startswith("[")


def test_large_json_input_converts_without_raising():
    rows = [{"id": i, "merchant": f"m-{i}", "mcc": "5411"} for i in range(2500)]
    data = json.dumps(rows).encode()
    result = convert_bytes(data, filename="large.json")
    assert result.target_mime in {"text/csv", "application/json"}
    assert len(result.content) > 0
    if not result.skipped:
        assert result.tokens_after <= result.tokens_before * 1.05 + 1
