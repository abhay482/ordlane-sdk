from __future__ import annotations

from pathlib import Path

from ordlane.exceptions import StorageError
from ordlane.storage.base import StorageBackend, StoredObject


class LocalStorage(StorageBackend):
    """Store files under a local directory."""

    kind = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        safe = key.lstrip("/").replace("\\", "/")
        if ".." in Path(safe).parts:
            raise StorageError(f"Unsafe storage key: {key}")
        path = self.root / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(data)
        except OSError as exc:
            raise StorageError(f"Failed to write local file {path}: {exc}") from exc
        return StoredObject(
            uri=path.as_uri() if path.is_absolute() else str(path),
            kind="local",
            key=safe,
            bytes_written=len(data),
            content_type=content_type,
            metadata={"path": str(path), **(metadata or {})},
        )
