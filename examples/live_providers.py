"""Wire OpenAI + Claude + Bedrock specialists and a LangGraph RAG backend.

Requires:
  pip install 'ordlane[all]'
  OPENAI_API_KEY, ANTHROPIC_API_KEY, and AWS credentials for Bedrock.
"""

from ordlane import CategorizerConfig, Harness, InMemoryRAG, ModelConfig

rag = InMemoryRAG()  # swap for CallableRAG(...) or LangChainVectorRAG(vectorstore)

harness = Harness(
    models=[
        ModelConfig(
            id="fast",
            provider="openai",
            model="gpt-4o-mini",
            role="Simple Q&A, classification, formatting",
            capabilities=["fast", "chat"],
            input_cost_per_1m=0.15,
            output_cost_per_1m=0.60,
        ),
        ModelConfig(
            id="standard",
            provider="openai",
            model="gpt-4o",
            role="RAG answers with retrieved context",
            capabilities=["rag", "docs"],
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
        ),
        ModelConfig(
            id="claude",
            provider="anthropic",
            model="claude-sonnet-4-0",
            role="Long-context document analysis",
            capabilities=["rag", "long-context"],
        ),
        ModelConfig(
            id="bedrock_reason",
            provider="bedrock",
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            region="us-east-1",
            role="Complex multi-step reasoning",
            capabilities=["reasoning", "complex"],
        ),
    ],
    categorizer=CategorizerConfig(provider="openai", model="gpt-4o-mini"),
    rag=rag,
    rag_kind="hybrid",  # none | naive | hybrid | map_reduce | custom
    use_langgraph=True,
    system_prompt="You are a precise enterprise assistant. Cite sources.",
)

# Convert + index a knowledge file
# harness.ingest(path="policy.pdf")

print(harness.converters())
print(harness.route("hello").model_slot_id)
