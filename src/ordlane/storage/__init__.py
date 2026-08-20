from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from ordlane.exceptions import ConfigError, StorageError
from ordlane.storage.base import StorageBackend, StoredObject
from ordlane.storage.local import LocalStorage
from ordlane.storage.s3 import S3Storage

__all__ = [
    "StorageBackend",
    "StoredObject",
    "LocalStorage",
    "S3Storage",
    "parse_store_to",
    "suggest_output_filename",
]


def suggest_output_filename(filename: str, target_mime: str) -> str:
    stem = Path(filename).stem or "converted"
    ext = {
        "text/csv": ".csv",
        "application/json": ".json",
        "text/markdown": ".md",
        "text/html": ".html",
        "text/plain": ".txt",
        "application/pdf": ".pdf",
    }.get(target_mime, Path(filename).suffix or ".bin")
    return f"{stem}{ext}"


def parse_store_to(store_to: str | StorageBackend) -> StorageBackend:
    """Build a storage backend from an instance or URI.

    Supported URIs:
      - local:///absolute/path   or   local://./relative/path
      - file:///absolute/path
      - /absolute/path           or   ./relative/path
      - s3://bucket/optional/prefix
    """
    if isinstance(store_to, StorageBackend):
        return store_to
    if not isinstance(store_to, str) or not store_to.strip():
        raise ConfigError("store_to must be a StorageBackend or URI string")

    value = store_to.strip()
    parsed = urlparse(value)

    if parsed.scheme in {"local", "file"}:
        # local:///tmp/out -> /tmp/out ; local://./out -> ./out
        path = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc not in {".", ""}:
            # local://relative/out
            path = f"{parsed.netloc}{path}"
        if not path:
            raise ConfigError(f"Invalid local store URI: {value}")
        if parsed.scheme == "file" and not path.startswith("/"):
            path = "/" + path
        return LocalStorage(path)

    if parsed.scheme == "s3":
        bucket = parsed.netloc
        if not bucket:
            raise ConfigError(f"Invalid S3 URI (missing bucket): {value}")
        prefix = unquote(parsed.path or "").lstrip("/")
        return S3Storage(bucket=bucket, prefix=prefix)

    if parsed.scheme in {"", None} or value.startswith((".", "/", "~")):
        return LocalStorage(value)

    raise StorageError(
        f"Unsupported store_to URI '{value}'. Use local://path, ./path, /path, or s3://bucket/prefix"
    )
