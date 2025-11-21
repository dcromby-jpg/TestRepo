from __future__ import annotations

import json
import pathlib
from typing import Optional

import typer

from .config import AppConfig, load_config
from .ingest import DocumentChunk, chunk_documents, ensure_data_dir
from .index import build_index, get_collection
from .pipeline import RAGPipeline
from .retrieval import bm25_retrieve, dense_retrieve
from .seed import set_deterministic_seed
from .training import build_sft_dataset, train_lora
from .evaluate import evaluate_answers

app = typer.Typer(help="RAG pipeline commands")


@app.command()
def ingest(config_path: str = typer.Option("configs/sample_config.yaml", help="Path to YAML config")):
    cfg: AppConfig = load_config(config_path)
    ensure_data_dir(cfg.ingest.data_dir)
    chunks = chunk_documents(cfg.model, cfg.ingest)
    build_index(chunks, cfg.ingest.index_path, cfg.retrieval.embedding_model)
    typer.echo(f"Ingested {len(chunks)} chunks into {cfg.ingest.index_path}")


@app.command()
def retrieve(query: str, config_path: str = typer.Option("configs/sample_config.yaml")):
    cfg = load_config(config_path)
    if cfg.retrieval.retriever_type == "bm25":
        corpus = chunk_documents(cfg.model, cfg.ingest)
        results = bm25_retrieve(corpus, query, cfg.retrieval.top_k, cfg.retrieval.metadata_filters)
    else:
        collection = get_collection(cfg.ingest.index_path, cfg.retrieval.embedding_model)
        results = dense_retrieve(collection, query, cfg.retrieval.top_k, cfg.retrieval.metadata_filters)
    typer.echo(json.dumps([r.metadata for r in results], indent=2))


@app.command()
def ask(query: str, config_path: str = typer.Option("configs/sample_config.yaml")):
    cfg = load_config(config_path)
    collection = None
    corpus = None
    if cfg.retrieval.retriever_type == "bm25":
        corpus = chunk_documents(cfg.model, cfg.ingest)
    else:
        collection = get_collection(cfg.ingest.index_path, cfg.retrieval.embedding_model)
    pipeline = RAGPipeline(cfg.model, cfg.retrieval, collection=collection, corpus=corpus)
    response = pipeline.ask(query)
    typer.echo(response.answer)
    typer.echo(json.dumps(response.sources, default=lambda x: x.__dict__, indent=2))


@app.command()
def train(config_path: str = typer.Option("configs/sample_config.yaml")):
    cfg = load_config(config_path)
    set_deterministic_seed(cfg.eval.seed)
    chunks = chunk_documents(cfg.model, cfg.ingest)
    train_ds, val_ds = build_sft_dataset(chunks, seed=cfg.eval.seed, val_split=cfg.eval.val_split_ratio)
    train_lora(cfg.model, cfg.training, train_ds, val_ds)
    typer.echo(f"Saved adapter to {cfg.training.output_dir}")


@app.command()
def evaluate(config_path: str = typer.Option("configs/sample_config.yaml")):
    cfg = load_config(config_path)
    if cfg.retrieval.retriever_type == "bm25":
        corpus = chunk_documents(cfg.model, cfg.ingest)
        retrieved = bm25_retrieve(corpus, "sanity", cfg.retrieval.top_k, cfg.retrieval.metadata_filters)
    else:
        collection = get_collection(cfg.ingest.index_path, cfg.retrieval.embedding_model)
        retrieved = dense_retrieve(collection, "sanity", cfg.retrieval.top_k, cfg.retrieval.metadata_filters)
    predictions = [f"answer referencing {c.metadata.get('chunk_id')}" for c in retrieved]
    references = [c.text for c in retrieved]
    result = evaluate_answers(predictions, references, retrieved)
    typer.echo(result)


if __name__ == "__main__":
    app()
