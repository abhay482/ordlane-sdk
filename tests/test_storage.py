"""Architectural boundary tests for local / S3 storage."""

from __future__ import annotations

import pytest

from ordlane.exceptions import StorageError
from ordlane.storage import LocalStorage, S3Storage, parse_store_to, suggest_output_filename


def test_local_storage_rejects_path_traversal(tmp_path):
    store = LocalStorage(tmp_path)
    with pytest.raises(StorageError, match="Unsafe storage key"):
        store.put("../secret.txt", b"nope")


def test_s3_prefix_and_key_normalization():
    store = S3Storage(bucket="bucket", prefix="/docs/converted/")
    assert store.prefix == "docs/converted"

    calls: list[dict] = []

    class FakeClient:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    store._client = lambda: FakeClient()  # type: ignore[method-assign]
    stored = store.put("/reports/a.csv", b"a,b\n1,2\n", content_type="text/csv")
    assert stored.uri == "s3://bucket/docs/converted/reports/a.csv"
    assert stored.key == "docs/converted/reports/a.csv"
    assert calls[0]["Bucket"] == "bucket"
    assert calls[0]["Key"] == "docs/converted/reports/a.csv"


def test_s3_put_failure_raises_storage_error():
    store = S3Storage(bucket="bucket", prefix="out")

    class FakeClient:
        def put_object(self, **kwargs):
            raise RuntimeError("AccessDenied")

    store._client = lambda: FakeClient()  # type: ignore[method-assign]
    with pytest.raises(StorageError, match="Failed to upload s3://bucket/out/file.csv"):
        store.put("file.csv", b"x")


def test_parse_store_to_rejects_empty_and_unknown_schemes():
    with pytest.raises(Exception):
        parse_store_to("")
    with pytest.raises(StorageError, match="Unsupported store_to URI"):
        parse_store_to("ftp://files/out")


def test_suggest_output_filename_by_mime():
    assert suggest_output_filename("merchants.json", "text/csv") == "merchants.csv"
    assert suggest_output_filename("report.pdf", "text/markdown") == "report.md"
    assert suggest_output_filename("page.html", "text/markdown") == "page.md"
