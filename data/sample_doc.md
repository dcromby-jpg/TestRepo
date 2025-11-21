# Sample Policy

This sample document explains how to run the demo RAG pipeline.

- Installation requires Python 3.10+ and the dependencies pinned in `pyproject.toml`.
- The ingestion step chunks markdown and stores embeddings in `chroma_db` by default.
- Retrieval uses either BM25 or dense embeddings such as `BAAI/bge-small-en-v1.5`.
- Training uses LoRA with document-level train/val splits to avoid leakage.

Use this file to try the CLI without adding your own documents yet.
