# Ordlane SDK

Plug-and-play Python layer for any AI stack: one question categorizer, up to five specialist models, file conversion, and swappable RAG (including LangGraph).

Providers out of the box: OpenAI, Anthropic (Claude SDK), Amazon Bedrock, and open-source models via LangChain (Ollama, HuggingFace, vLLM / OpenAI-compatible).

## Architecture

```text
                    ????????????????????
                    ?    User Query    ?
                    ????????????????????
                             ?
                    ????????????????????
                    ?   Categorizer    ?
                    ????????????????????
                             ?
             ?????????????????????????????????
             ?               ?               ?
          Fast Model      Docs Model      Reasoning
             ?               ?               ?
             ?????????????????????????????????
                             ?
                    ????????????????????
                    ?       RAG        ?
                    ????????????????????
                             ?
                    ????????????????????
                    ?Context Optimizer ?
                    ????????????????????
                             ?
                    ????????????????????
                    ? Provider Layer   ?
                    ????????????????????
                             ?
                    Answer + Telemetry
```

Convert / ingest can also write files to local disk or S3 before (or instead of) indexing.

## Install

```bash
pip install -e ".[all]"   # from this repo
# or extras only:
pip install -e ".[openai,anthropic,bedrock,langchain,langgraph,s3,pdf,html]"
```

## Quick start

```python
from ordlane import Harness, ModelConfig, CategorizerConfig

harness = Harness(
    models=[
        ModelConfig(id="fast", provider="openai", model="gpt-4o-mini",
                    role="Short answers", capabilities=["fast", "chat"]),
        ModelConfig(id="docs", provider="anthropic", model="claude-sonnet-4-0",
                    role="Document Q&A", capabilities=["rag", "docs"]),
        ModelConfig(id="reason", provider="bedrock",
                    model="anthropic.claude-sonnet-4-20250514-v1:0",
                    region="us-east-1", role="Deep analysis",
                    capabilities=["reasoning"]),
        # up to 5 specialists
    ],
    categorizer=CategorizerConfig(provider="openai", model="gpt-4o-mini"),
)

harness.ingest(path="settlement_policy.md")  # convert + index
result = harness.ask("What is the Visa chargeback window in settlement_policy.md?")
print(result.routing)   # which specialist was chosen
print(result.answer)
```

The categorizer model classifies the question and selects the best of your specialists. `dry_run=True` or `provider="fake"` runs without API keys.

## Open-source models (LangChain / LangGraph)

Use `provider="langchain"` with Ollama, HuggingFace, or any OpenAI-compatible local server. You can also pass a ready LangChain chat model. See `examples/opensource_langchain.py`.

```python
from ordlane import Harness, ModelConfig, CategorizerConfig
# from langchain_ollama import ChatOllama

harness = Harness(
    models=[
        ModelConfig(
            id="llama",
            provider="langchain",
            model="ollama:llama3.2",
            capabilities=["fast", "chat", "rag"],
            # Or plug any LangChain / LangGraph chat model:
            # extra={"llm": ChatOllama(model="llama3.2")},
        ),
        ModelConfig(
            id="hf",
            provider="langchain",
            model="huggingface:mistralai/Mistral-7B-Instruct-v0.2",
            capabilities=["reasoning"],
        ),
        ModelConfig(
            id="vllm",
            provider="langchain",
            model="openai_compat:meta-llama/Meta-Llama-3-8B-Instruct",
            capabilities=["rag"],
            extra={"base_url": "http://localhost:8000/v1"},
        ),
    ],
    categorizer=CategorizerConfig(provider="langchain", model="ollama:llama3.2"),
)
```

## File conversion

Use `Harness.convert` / `Harness.ingest` (same models you registered). Runnable copy: `examples/convert_with_harness.py`.

```python
print(harness.converters())

# Convert on disk or from bytes
csv_file = harness.convert(path="merchants.json")              # JSON -> CSV
md_file = harness.convert(path="page.html")                    # HTML -> Markdown
pdf_file = harness.convert(path="report.pdf")                  # PDF -> Markdown
as_json = harness.convert(
    data=csv_file.content,
    filename="merchants.csv",
    target_mime="application/json",
)

# Convert then index into RAG; categorizer still picks among your models
harness.ingest(path="policy.html", doc_id="policy")            # convert=True by default
result = harness.ask("What is the SLA in policy.html?")
print(result.routing["model_slot_id"], result.answer)
```

Built-in pairs:

| From | To |
|------|-----|
| `application/json` | `text/csv` |
| `text/csv` | `application/json` |
| `application/pdf` | `text/markdown` |
| `text/html` | `text/markdown` |

Register your own converter with `register_converter(...)`.

## Store converted files (local or S3)

Convert and persist without (or with) RAG indexing. See `examples/store_converted.py`.

```python
from ordlane import LocalStorage, S3Storage

# Convert + write to a local folder
out = harness.convert(path="merchants.json", store_to="./converted")
print(out["stored"]["uri"])  # file:///.../merchants.csv

# Convert + store on S3 / MinIO
out = harness.convert(path="report.pdf", store_to="s3://my-bucket/converted/")

# Convert + store + index into RAG
harness.ingest(path="policy.html", store_to=LocalStorage("./out"), index=True)

# Convert + store only (skip RAG)
harness.ingest(path="policy.html", store_to="s3://my-bucket/docs/", index=False)

# Optional harness-wide default destination
harness = Harness(
    models=...,
    categorizer=...,
    default_store="local://./converted",
    # default_store="s3://my-bucket/prefix/",
)
```

`store_to` accepts:

- `./path`, `/path`, `local://./path`, `file:///path`
- `s3://bucket/optional/prefix`
- `LocalStorage(...)` or `S3Storage(bucket=..., prefix=..., endpoint_url=...)`

## RAG + LangGraph

Default backend is in-memory (no vector DB). Plug in others:

```python
from ordlane import InMemoryRAG, CallableRAG, LangChainVectorRAG, Harness

# 1) Default overlap retriever
rag = InMemoryRAG()

# 2) Your existing retriever
rag = CallableRAG(retrieve_fn=lambda query, k=6: my_search(query, k))

# 3) LangChain / Chroma / FAISS / Pinecone vector store
rag = LangChainVectorRAG(vectorstore)

harness = Harness(
    models=...,
    categorizer=...,
    rag=rag,
    rag_kind="hybrid",   # none | naive | hybrid | map_reduce | custom
    use_langgraph=True,  # retrieve node runs as a LangGraph
)
```

`rag_kind`:

- `none` - no retrieval
- `naive` - top-k chunks
- `hybrid` - named files in full + chunks (default)
- `map_reduce` - compress retrieved chunks before the specialist
- `custom` - same as hybrid; use with `CallableRAG`

## Environment

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
HUGGINGFACEHUB_API_TOKEN=
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Boundary coverage includes routing fallbacks, RAG assembly modes, conversion edge cases, and storage failure paths.