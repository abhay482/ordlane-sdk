# Ordlane SDK

Plug-and-play Python layer for any AI stack: one question categorizer, up to five specialist models, file conversion, and swappable RAG (including LangGraph).

Providers out of the box: OpenAI, Anthropic (Claude SDK), Amazon Bedrock.

## Install

```bash
pip install -e ".[all]"   # from this repo
# or extras only:
pip install -e ".[openai,anthropic,bedrock,langgraph,pdf,html]"
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

## File conversion

```python
result = harness.convert(path="merchants.json")  # JSON -> CSV
result = harness.convert(path="report.pdf")      # PDF -> Markdown
result = harness.convert(path="page.html")       # HTML -> Markdown
result = harness.convert(data=raw, filename="file.csv", target_mime="application/json")
```

Built-in pairs:

| From | To |
|------|-----|
| `application/json` | `text/csv` |
| `text/csv` | `application/json` |
| `application/pdf` | `text/markdown` |
| `text/html` | `text/markdown` |

Register your own converter with `register_converter(...)`.

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
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
