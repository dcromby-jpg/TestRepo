# TestRepo RAG Toolkit

This repository provides a reproducible RAG workflow with ingestion, indexing, retrieval, LoRA/QLoRA-friendly fine-tuning, and evaluation exposed through a Typer CLI. Deterministic seeds and pinned model/tokenizer revisions make it easy to rerun experiments.

## How it works

1. **Ingest & chunk**: HTML/Markdown/PDF/text files are parsed, lowercased (optional), and chunked into overlapping windows.
2. **Index**: Chunks plus metadata (source path, section) are written to a lightweight Chroma store for dense retrieval or kept in-memory for BM25.
3. **Retrieve**: BM25 or dense embeddings (default `BAAI/bge-small-en-v1.5`) return deterministic top-*k* results with metadata filters to avoid cross-split leakage.
4. **Ask**: Retrieved passages are assembled into a grounded prompt for the base model (`meta-llama/Meta-Llama-3-8B-Instruct`, revision `c9d618e4e4c3d71a1cc150b76c7a6c5e97f65851`) to answer with citations.
5. **Train & evaluate**: Synthetic or sampled Q&A derived from chunks are split by document, then fine-tuned with LoRA; automated grounding and exact/partial-match checks provide feedback.

## Installation

### Prerequisites

- Python 3.10+ (pinned in `pyproject.toml`)
- CUDA-capable GPU recommended for training/inference (CPU works for small tests)

### Steps

Create an isolated environment (conda/venv/uv) and install the pinned dependencies:

```bash
pip install -e .
```

Optional: to ensure reproducibility across nodes, export `PYTHONHASHSEED=0` and set deterministic seeds in configs.

## Configuration

`configs/sample_config.yaml` captures everything needed to reproduce runs:

- **model**: base model name, tokenizer, and exact revision.
- **ingest**: data directory, index path, chunk size/overlap, and lowercasing toggle.
- **retrieval**: `retriever_type` (`bm25` or `dense`), embedding model, and `top_k`.
- **training**: LoRA hyperparameters (rank, alpha, dropout), learning rate, max sequence length, and output directory for adapter weights/manifests.
- **eval**: validation split ratio, grounding `k`, and seed.

Adjust the file or provide an alternative via `--config-path` for different experiments.

## Repository layout

- `ragtool/`: CLI commands and supporting modules for ingestion, indexing, retrieval, training, and evaluation.
- `configs/sample_config.yaml`: default deterministic settings (model/tokenizer revision, chunk sizes, retrieval type, LoRA hyperparameters).
- `data/`: drop your source documents here; a starter `sample_doc.md` is provided for smoke tests.
- `requirements.txt` and `pyproject.toml`: pinned dependencies to reproduce environments exactly.

## End-to-end quickstart

1. **Prepare data**: place HTML/MD/PDF/text sources under `data/` (or update `ingest.data_dir`).
   - A tiny sample file at `data/sample_doc.md` is included so you can exercise the CLI immediately after installation.
   - Expected layout after adding your own docs:

     ```
     data/
       sample_doc.md
       your_manual.html
       handbook/section_a.md
     ```
2. **Ingest**: normalize, chunk, and index into Chroma:

   ```bash
   ragtool ingest --config-path configs/sample_config.yaml
   ```

3. **Inspect retrieval**: confirm deterministic results:

   ```bash
   ragtool retrieve "what is the policy?" --config-path configs/sample_config.yaml
   ```

4. **Ask with RAG**: run retrieval + grounded generation with citations:

   ```bash
   ragtool ask "summarize the guidelines" --config-path configs/sample_config.yaml
   ```

5. **Train LoRA adapter**: build document-level splits and train lightweight adapters (artifacts under `artifacts/lora-adapter` by default):

   ```bash
   ragtool train --config-path configs/sample_config.yaml
   ```

6. **Evaluate**: grounding plus exact/partial match checks on held-out slices:

   ```bash
   ragtool evaluate --config-path configs/sample_config.yaml
   ```

Use a different YAML config for alternative models, chunk sizes, or index locations.
