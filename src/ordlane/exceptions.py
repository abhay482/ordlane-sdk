class OrdlaneError(Exception):
    """Base error for the Ordlane SDK."""


class ConfigError(OrdlaneError):
    """Invalid harness or model configuration."""


class ProviderError(OrdlaneError):
    """LLM provider call failed."""


class ConversionError(OrdlaneError):
    """File conversion failed."""


class StorageError(OrdlaneError):
    """File storage (local / S3) failed."""


class RAGError(OrdlaneError):
    """RAG ingest or retrieve failed."""
