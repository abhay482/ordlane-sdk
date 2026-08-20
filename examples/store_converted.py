"""Convert files and store them locally or on S3 (with or without RAG indexing)."""

from pathlib import Path

from ordlane import CategorizerConfig, Harness, LocalStorage, ModelConfig

ROOT = Path(__file__).resolve().parent / "sample_files"
OUT = Path(__file__).resolve().parent / "converted_out"
ROOT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

(ROOT / "merchants.json").write_text(
    '[{"merchant":"acme","mcc":"5411"},{"merchant":"beta","mcc":"5812"}]\n',
    encoding="utf-8",
)

harness = Harness(
    models=[
        ModelConfig(id="docs", provider="fake", model="standard", capabilities=["rag", "docs"]),
    ],
    categorizer=CategorizerConfig(provider="fake", model="router"),
    dry_run=True,
    use_langgraph=False,
    # Optional default destination for every convert/ingest:
    # default_store="local://./converted_out",
    # default_store="s3://my-bucket/converted/",
)

# 1) Convert only + store to local folder
stored = harness.convert(
    path=str(ROOT / "merchants.json"),
    store_to=str(OUT),  # or "local://./converted_out" or LocalStorage(OUT)
)
print("Local store:", stored["stored"])
print("Preview:\n", stored["conversion"]["text_preview"])

# 2) Convert + store + index into RAG
indexed = harness.ingest(
    path=str(ROOT / "merchants.json"),
    doc_id="merchants",
    store_to=LocalStorage(OUT / "ingest"),
    store_original=True,
    index=True,
)
print("Ingest stored:", indexed["stored"])
print("Chunks:", indexed["chunks_indexed"])

# 3) Convert + store only (no RAG)
persisted = harness.ingest(
    path=str(ROOT / "merchants.json"),
    doc_id="merchants-archive",
    store_to=str(OUT / "archive"),
    index=False,
)
print("Archive only:", persisted["stored"], "indexed=", persisted["indexed"])

# 4) S3 / MinIO (needs pip install 'ordlane[s3]' + credentials)
# from ordlane import S3Storage
# harness.convert(
#     path=str(ROOT / "merchants.json"),
#     store_to="s3://my-bucket/converted/",
# )
# harness.ingest(
#     path=str(ROOT / "merchants.json"),
#     store_to=S3Storage(bucket="my-bucket", prefix="docs", endpoint_url="http://localhost:9000"),
#     index=False,
# )

print("Done. Files under:", OUT)
