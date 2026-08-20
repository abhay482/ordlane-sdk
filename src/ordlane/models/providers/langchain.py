from __future__ import annotations

from typing import Any

from ordlane.exceptions import ProviderError
from ordlane.models.providers.base import ChatMessage, LLMProvider


class LangChainProvider(LLMProvider):
    """Wrap any LangChain / LangGraph-compatible chat model (Ollama, HF, vLLM, etc.).

    Pass a ready chat model::

        from langchain_ollama import ChatOllama
        ModelConfig(
            id="llama",
            provider="langchain",
            model="llama3.2",
            extra={"llm": ChatOllama(model="llama3.2")},
        )

    Or let Ordlane build a common open-source backend from the model id::

        ModelConfig(id="llama", provider="langchain", model="ollama:llama3.2")
        ModelConfig(id="mistral", provider="langchain", model="huggingface:mistralai/Mistral-7B-Instruct-v0.2")
    """

    name = "langchain"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        llm: Any = None,
        chat_model: Any = None,
        backend: str | None = None,
        model: str | None = None,
        **extra: Any,
    ) -> None:
        self.api_key = api_key
        self._llm = llm or chat_model
        self.backend = backend
        self.default_model = model
        self.extra = extra

    def complete(self, messages: list[ChatMessage], *, model: str, **kwargs: Any) -> str:
        llm = self._llm or self._build_llm(model)
        lc_messages = self._to_lc_messages(messages)
        try:
            if hasattr(llm, "invoke"):
                result = llm.invoke(lc_messages, **{k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens", "stop"}})
            elif callable(llm):
                result = llm(lc_messages)
            else:
                raise ProviderError("LangChain llm must support invoke() or be callable")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"LangChain model call failed: {exc}") from exc
        return self._content_to_text(result)

    def _build_llm(self, model: str) -> Any:
        backend, _, name = model.partition(":")
        if not name:
            name = model
            backend = self.backend or "ollama"
        else:
            backend = backend.lower()

        if backend in {"ollama", "chatollama"}:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                try:
                    from langchain_community.chat_models import ChatOllama  # type: ignore
                except ImportError as exc:
                    raise ProviderError(
                        "Install Ollama LangChain support: pip install langchain-ollama"
                    ) from exc
            opts = {k: v for k, v in self.extra.items() if k not in {"llm", "chat_model", "backend"}}
            return ChatOllama(model=name, **opts)

        if backend in {"huggingface", "hf", "huggingfacehub"}:
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            except ImportError as exc:
                raise ProviderError(
                    "Install HuggingFace LangChain support: pip install langchain-huggingface"
                ) from exc
            endpoint = HuggingFaceEndpoint(
                repo_id=name,
                huggingfacehub_api_token=self.api_key or self.extra.get("huggingfacehub_api_token"),
                **{k: v for k, v in self.extra.items() if k not in {"llm", "chat_model", "backend", "huggingfacehub_api_token"}},
            )
            return ChatHuggingFace(llm=endpoint)

        if backend in {"openai_compat", "vllm", "localai", "lmstudio"}:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ProviderError(
                    "Install OpenAI-compatible LangChain support: pip install langchain-openai"
                ) from exc
            base_url = self.extra.get("base_url") or self.extra.get("api_base")
            return ChatOpenAI(
                model=name,
                api_key=self.api_key or self.extra.get("api_key") or "EMPTY",
                base_url=base_url,
                **{k: v for k, v in self.extra.items() if k not in {"llm", "chat_model", "backend", "base_url", "api_base", "api_key"}},
            )

        raise ProviderError(
            f"Unknown LangChain backend '{backend}'. "
            "Use ollama:<model>, huggingface:<repo>, openai_compat:<model>, "
            "or pass extra={{'llm': your_chat_model}}."
        )

    @staticmethod
    def _to_lc_messages(messages: list[ChatMessage]) -> list[Any]:
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError:
            # Allow custom callables / stubs without installing LangChain.
            return [{"role": m.role, "content": m.content} for m in messages]

        out: list[Any] = []
        for m in messages:
            if m.role == "system":
                out.append(SystemMessage(content=m.content))
            elif m.role == "assistant":
                out.append(AIMessage(content=m.content))
            else:
                out.append(HumanMessage(content=m.content))
        return out

    @staticmethod
    def _content_to_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(getattr(block, "text", block)))
            return "".join(parts)
        return str(result)
