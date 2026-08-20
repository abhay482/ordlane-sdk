"""Open-source models via LangChain / LangGraph (Ollama, HuggingFace, vLLM, etc.).

Requires:
  pip install 'ordlane[langchain]'
  # and one of:
  #   ollama serve + ollama pull llama3.2
  #   HuggingFace token for huggingface: models
  #   a local OpenAI-compatible server for openai_compat:
"""

from ordlane import CategorizerConfig, Harness, ModelConfig

# Option A: pass any LangChain chat model instance
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2")

harness = Harness(
    models=[
        ModelConfig(
            id="fast",
            provider="langchain",
            model="ollama:llama3.2",
            role="Fast local answers",
            capabilities=["fast", "chat"],
            # Or: extra={"llm": ChatOllama(model="llama3.2")},
        ),
        ModelConfig(
            id="docs",
            provider="langchain",
            model="ollama:llama3.2",
            role="RAG / document answers",
            capabilities=["rag", "docs"],
        ),
        ModelConfig(
            id="reason",
            # HuggingFace hosted or local endpoint
            provider="langchain",
            model="huggingface:mistralai/Mistral-7B-Instruct-v0.2",
            role="Deeper reasoning",
            capabilities=["reasoning"],
            # api_key="hf_...",
        ),
        # OpenAI-compatible open-source servers (vLLM, LM Studio, LocalAI):
        # ModelConfig(
        #     id="vllm",
        #     provider="langchain",
        #     model="openai_compat:meta-llama/Meta-Llama-3-8B-Instruct",
        #     capabilities=["rag"],
        #     extra={"base_url": "http://localhost:8000/v1"},
        # ),
    ],
    categorizer=CategorizerConfig(provider="langchain", model="ollama:llama3.2"),
    # dry_run=True for offline demos without a running model server
    dry_run=True,
    use_langgraph=True,
)

print("Routed:", harness.route("hello").model_slot_id)
# result = harness.ask("Summarize our settlement SLA")
# print(result.answer)
