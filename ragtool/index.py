from __future__ import annotations

import pathlib
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from .ingest import DocumentChunk


def _get_client(index_path: pathlib.Path) -> chromadb.Client:
    return chromadb.PersistentClient(path=str(index_path))


def build_index(chunks: Iterable[DocumentChunk], index_path: pathlib.Path, embedding_model: str) -> None:
    client = _get_client(index_path)
    collection = client.get_or_create_collection(
        name="rag-documents",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model),
    )
    ids = []
    texts = []
    metadatas = []
    for chunk in chunks:
        ids.append(chunk.metadata.get("chunk_id"))
        texts.append(chunk.text)
        metadatas.append(chunk.metadata)
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)


def get_collection(index_path: pathlib.Path, embedding_model: str):
    client = _get_client(index_path)
    return client.get_collection(
        name="rag-documents",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model),
    )
