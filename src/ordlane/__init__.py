"""Ordlane: plug-and-play model routing, file conversion, and LangGraph RAG."""

from __future__ import annotations

from ordlane.convert import ConversionResult, convert_bytes, convert_file, list_converters, register_converter
from ordlane.exceptions import ConfigError, ConversionError, OrdlaneError, ProviderError, RAGError
from ordlane.harness import Harness
from ordlane.rag import CallableRAG, InMemoryRAG, LangChainVectorRAG, build_langgraph_rag
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
    "convert_bytes",
    "convert_file",
    "list_converters",
    "register_converter",
    "OrdlaneError",
    "ConfigError",
    "ProviderError",
    "ConversionError",
    "RAGError",
]

__version__ = "0.1.0"
