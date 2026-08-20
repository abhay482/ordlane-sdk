from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from ordlane.categorizer import categorize_question
from ordlane.convert import ConversionResult, convert_bytes, convert_file, list_converters
from ordlane.convert.tokens import estimate_tokens
from ordlane.exceptions import ConfigError, ProviderError
from ordlane.models.providers import provider_for_categorizer, provider_for_model
from ordlane.models.providers.base import ChatMessage
from ordlane.optimizer import optimize_context
from ordlane.rag import InMemoryRAG, assemble_context, build_langgraph_rag
from ordlane.rag.protocol import RAGBackend
from ordlane.storage import StorageBackend, parse_store_to, suggest_output_filename
from ordlane.types import CategorizerConfig, ModelConfig, QueryResult, RAGKind, RouteDecision

MAX_MODELS = 5


class Harness:
    """Plug-and-play routing layer: 1 categorizer + up to 5 specialist models + optional RAG."""

    def __init__(
        self,
        models: Sequence[ModelConfig],
        categorizer: CategorizerConfig,
        *,
        rag: RAGBackend | None = None,
        rag_kind: RAGKind = "hybrid",
        system_prompt: str = "",
        dry_run: bool = False,
        use_langgraph: bool = True,
        max_context_tokens: int = 4000,
        rag_top_k: int = 6,
        default_store: str | StorageBackend | None = None,
    ) -> None:
        if not 1 <= len(models) <= MAX_MODELS:
            raise ConfigError(f"Provide between 1 and {MAX_MODELS} specialist models")
        ids = [m.id for m in models]
        if len(ids) != len(set(ids)):
            raise ConfigError("Specialist model ids must be unique")
        self.models = list(models)
        self.models_by_id = {m.id: m for m in self.models}
        self.categorizer_cfg = categorizer
        self.rag = rag if rag is not None else InMemoryRAG()
        self.rag_kind = rag_kind
        self.system_prompt = system_prompt
        self.dry_run = dry_run
        self.use_langgraph = use_langgraph
        self.max_context_tokens = max_context_tokens
        self.rag_top_k = rag_top_k
        self.default_store = parse_store_to(default_store) if default_store is not None else None
        self._categorizer = provider_for_categorizer(categorizer)
        self._providers = {m.id: provider_for_model(m) for m in self.models}
        self._rag_graph = None
        if use_langgraph:
            try:
                self._rag_graph = build_langgraph_rag(self.rag, kind=rag_kind, k=rag_top_k)
            except Exception:
                self._rag_graph = None

    def convert(
        self,
        data: bytes | None = None,
        *,
        path: str | None = None,
        filename: str = "",
        source_mime: str | None = None,
        target_mime: str | None = None,
        store_to: str | StorageBackend | None = None,
        store_key: str | None = None,
        store_original: bool = False,
    ) -> ConversionResult | dict:
        """Convert a file. Optionally write converted bytes to local disk or S3.

        store_to examples:
          - "./converted" or "local://./converted"
          - "s3://my-bucket/converted/"
          - LocalStorage(...) / S3Storage(...)
        """
        if path:
            result = convert_file(path, target_mime=target_mime)
            filename = filename or Path(path).name
        elif data is not None:
            result = convert_bytes(data, filename=filename, source_mime=source_mime, target_mime=target_mime)
        else:
            raise ConfigError("Pass file bytes via data= or a filesystem path via path=")

        storage = self._resolve_store(store_to)
        if storage is None:
            return result

        key = store_key or suggest_output_filename(filename or "converted.bin", result.target_mime)
        stored = storage.put(
            key,
            result.content,
            content_type=result.target_mime,
            metadata={"source_filename": filename, "source_mime": result.source_mime},
        )
        payload: dict = {
            "conversion": {
                "from": result.source_mime,
                "to": result.target_mime,
                "skipped": result.skipped,
                "warning": result.warning,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "savings_ratio": round(result.savings_ratio, 4),
                "text_preview": result.text[:2000],
            },
            "stored": {
                "uri": stored.uri,
                "kind": stored.kind,
                "key": stored.key,
                "bytes_written": stored.bytes_written,
                "content_type": stored.content_type,
            },
            "result": result,
        }
        if store_original and data is None and path:
            data = Path(path).read_bytes()
        if store_original and data is not None:
            original_key = f"originals/{Path(filename or key).name}"
            original = storage.put(
                original_key,
                data,
                content_type=result.source_mime,
                metadata={"role": "original"},
            )
            payload["stored_original"] = {
                "uri": original.uri,
                "kind": original.kind,
                "key": original.key,
                "bytes_written": original.bytes_written,
            }
        return payload

    def ingest(
        self,
        data: bytes | None = None,
        *,
        path: str | None = None,
        filename: str = "",
        doc_id: str | None = None,
        convert: bool = True,
        index: bool = True,
        store_to: str | StorageBackend | None = None,
        store_key: str | None = None,
        store_original: bool = False,
    ) -> dict:
        """Convert (optional), optionally store to local/S3, optionally index into RAG.

        Use index=False with store_to=... when you only want convert + persist.
        """
        if path:
            p = Path(path)
            raw = p.read_bytes()
            filename = filename or p.name
        elif data is not None:
            raw = data
        else:
            raise ConfigError("Pass data= or path= to ingest")

        if convert:
            converted = convert_bytes(raw, filename=filename)
            text = converted.text
            content = converted.content
            target_mime = converted.target_mime
            conversion = {
                "from": converted.source_mime,
                "to": converted.target_mime,
                "skipped": converted.skipped,
                "warning": converted.warning,
                "tokens_before": converted.tokens_before,
                "tokens_after": converted.tokens_after,
            }
        else:
            text = raw.decode("utf-8", errors="replace")
            content = raw
            target_mime = "application/octet-stream"
            conversion = {"skipped": True}

        asset_id = doc_id or filename or "doc"
        out: dict = {
            "doc_id": asset_id,
            "conversion": conversion,
            "chunks_indexed": 0,
            "indexed": False,
        }

        storage = self._resolve_store(store_to)
        if storage is not None:
            key = store_key or suggest_output_filename(filename or asset_id, target_mime)
            stored = storage.put(
                key,
                content,
                content_type=target_mime,
                metadata={"doc_id": asset_id, "source_filename": filename},
            )
            out["stored"] = {
                "uri": stored.uri,
                "kind": stored.kind,
                "key": stored.key,
                "bytes_written": stored.bytes_written,
                "content_type": stored.content_type,
            }
            if store_original:
                original = storage.put(
                    f"originals/{Path(filename or asset_id).name}",
                    raw,
                    content_type=conversion.get("from") or "application/octet-stream",
                    metadata={"doc_id": asset_id, "role": "original"},
                )
                out["stored_original"] = {
                    "uri": original.uri,
                    "kind": original.kind,
                    "key": original.key,
                    "bytes_written": original.bytes_written,
                }

        if index:
            chunks = self.rag.ingest(asset_id, text, metadata={"filename": filename})
            out["chunks_indexed"] = chunks
            out["indexed"] = True

        return out

    def route(self, question: str) -> RouteDecision:
        return categorize_question(
            question,
            self.models,
            provider=self._categorizer,
            model_name=self.categorizer_cfg.model,
            dry_run=self.dry_run,
        )

    def ask(self, question: str, *, system_prompt: str | None = None) -> QueryResult:
        started = time.perf_counter()
        decision = self.route(question)
        context, context_mode = self._retrieve(question, decision.needs_rag)
        optimized = optimize_context(context, question, max_tokens=self.max_context_tokens)
        prompt_parts = []
        persona = system_prompt if system_prompt is not None else self.system_prompt
        if persona:
            prompt_parts.append(persona.strip())
        prompt_parts.append(f"You are specialist '{decision.model_slot_id}' ({decision.model_id}).")
        if optimized.text:
            prompt_parts.append("Context:\n" + optimized.text)
        prompt_parts.append("User question:\n" + question)
        prompt_parts.append("Answer clearly. Cite files or chunk ids when used.")
        prompt = "\n\n".join(prompt_parts)

        slot = self.models_by_id[decision.model_slot_id]
        provider = self._providers[slot.id]
        if self.dry_run:
            answer = (
                f"[dry-run:{slot.id}/{slot.model_id}] reason={decision.reason} "
                f"complexity={decision.complexity} needs_rag={decision.needs_rag}. "
                f"Query: {question[:240]}"
            )
        else:
            try:
                answer = provider.complete(
                    [ChatMessage(role="user", content=prompt)],
                    model=slot.model,
                )
            except ProviderError:
                if getattr(provider, "name", "") == "fake":
                    raise
                answer = (
                    f"[provider_error:{slot.id}] falling back to dry-run. "
                    f"Query: {question[:240]}"
                )

        in_tok = estimate_tokens(prompt)
        out_tok = estimate_tokens(answer)
        cost = (in_tok / 1_000_000) * slot.input_cost_per_1m + (out_tok / 1_000_000) * slot.output_cost_per_1m
        latency_ms = int((time.perf_counter() - started) * 1000)
        return QueryResult(
            answer=answer,
            routing={
                "model_slot_id": decision.model_slot_id,
                "model_id": decision.model_id,
                "provider": decision.provider,
                "needs_rag": decision.needs_rag,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "complexity": decision.complexity,
                "used_llm": decision.used_llm,
            },
            context_mode=context_mode,
            context=optimized.text,
            metrics={
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "estimated_cost_usd": round(cost, 6),
                "latency_ms": latency_ms,
                "optimizer": {
                    "compressed": optimized.compressed,
                    "tokens_before": optimized.tokens_before,
                    "tokens_after": optimized.tokens_after,
                    "dropped_chunks": optimized.dropped_chunks,
                },
            },
            metadata={"rag_kind": self.rag_kind, "langgraph": self._rag_graph is not None},
        )

    def converters(self) -> list[dict[str, str]]:
        return list_converters()

    def _resolve_store(self, store_to: str | StorageBackend | None) -> StorageBackend | None:
        if store_to is None:
            return self.default_store
        return parse_store_to(store_to)

    def _retrieve(self, question: str, needs_rag: bool) -> tuple[str, str]:
        if self._rag_graph is not None:
            state = self._rag_graph.invoke(
                {
                    "question": question,
                    "needs_rag": needs_rag,
                    "rag_kind": self.rag_kind,
                }
            )
            return str(state.get("context") or ""), str(state.get("context_mode") or "none")
        context, mode, _chunks = assemble_context(
            question,
            self.rag,
            kind=self.rag_kind,
            k=self.rag_top_k,
            needs_rag=needs_rag,
        )
        return context, mode
