from __future__ import annotations

import os
from typing import Any

from ordlane.exceptions import ProviderError
from ordlane.models.providers.base import ChatMessage, LLMProvider


class BedrockProvider(LLMProvider):
    """Amazon Bedrock Converse API (Claude, Llama, Titan, and other Bedrock models)."""

    name = "bedrock"

    def __init__(self, api_key: str | None = None, region: str | None = None, **extra: Any) -> None:
        self.region = region or extra.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.extra = extra

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        try:
            import boto3
        except ImportError as exc:
            raise ProviderError("Install Bedrock support: pip install 'ordlane[bedrock]'") from exc

        client = boto3.client("bedrock-runtime", region_name=self.region)
        system = [{"text": m.content} for m in messages if m.role == "system"]
        converse_messages = []
        for m in messages:
            if m.role == "system":
                continue
            role = "assistant" if m.role == "assistant" else "user"
            converse_messages.append({"role": role, "content": [{"text": m.content}]})
        if not converse_messages:
            converse_messages = [{"role": "user", "content": [{"text": ""}]}]

        payload: dict[str, Any] = {
            "modelId": model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": int(kwargs.get("max_tokens") or 1024),
                "temperature": float(kwargs.get("temperature", 0.2)),
            },
        }
        if system:
            payload["system"] = system
        try:
            resp = client.converse(**payload)
        except Exception as exc:
            raise ProviderError(f"Bedrock converse failed: {exc}") from exc
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
