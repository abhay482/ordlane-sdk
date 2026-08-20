"""Ordlane: plug-and-play model routing, file conversion, and LangGraph RAG."""

from __future__ import annotations

from ordlane.convert import ConversionResult, convert_bytes, convert_file, list_converters, register_converter
from ordlane.exceptions import ConfigError, ConversionError, OrdlaneError, ProviderError, RAGError, StorageError
from ordlane.harness import Harness
from ordlane.rag import CallableRAG, InMemoryRAG, LangChainVectorRAG, build_langgraph_rag
from ordlane.storage import LocalStorage, S3Storage, parse_store_to
from ordlane.types import CategorizerConfig, ModelConfig, QueryResult, RetrievedChunk, RouteDecision

__all__ = [
    "Harness",
    "ModelConfig",
    "CategorizerConfig",
    "QueryResult",
    "RouteDecision",
    "RetrievedChunk",
    "ConversionResult",
    "InMemoryRAG",
    "CallableRAG",
    "LangChainVectorRAG",
    "build_langgraph_rag",
    "LocalStorage",
    "S3Storage",
    "parse_store_to",
    "convert_bytes",
    "convert_file",
    "list_converters",
    "register_converter",
    "OrdlaneError",
    "ConfigError",
    "ProviderError",
    "ConversionError",
    "StorageError",
    "RAGError",
]

__version__ = "0.1.0"
