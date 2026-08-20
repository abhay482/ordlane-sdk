from __future__ import annotations

import os
from typing import Any

from ordlane.exceptions import StorageError
from ordlane.storage.base import StorageBackend, StoredObject


class S3Storage(StorageBackend):
    """Store files in an S3 (or S3-compatible) bucket."""

    kind = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self.endpoint_url = endpoint_url
        self._client_kwargs = {
            **client_kwargs,
            **({} if aws_access_key_id is None else {"aws_access_key_id": aws_access_key_id}),
            **({} if aws_secret_access_key is None else {"aws_secret_access_key": aws_secret_access_key}),
        }
        if self.region:
            self._client_kwargs.setdefault("region_name", self.region)
        if endpoint_url:
            self._client_kwargs["endpoint_url"] = endpoint_url

    def _client(self):
        try:
            import boto3
        except ImportError as exc:
            raise StorageError("Install S3 support: pip install 'ordlane[s3]' or 'ordlane[bedrock]'") from exc
        return boto3.client("s3", **self._client_kwargs)

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        safe = key.lstrip("/")
        full_key = f"{self.prefix}/{safe}" if self.prefix else safe
        full_key = full_key.lstrip("/")
        try:
            extra: dict[str, Any] = {"ContentType": content_type}
            if metadata:
                extra["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
            self._client().put_object(Bucket=self.bucket, Key=full_key, Body=data, **extra)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to upload s3://{self.bucket}/{full_key}: {exc}") from exc
        return StoredObject(
            uri=f"s3://{self.bucket}/{full_key}",
            kind="s3",
            key=full_key,
            bytes_written=len(data),
            content_type=content_type,
            metadata={"bucket": self.bucket, **(metadata or {})},
        )
