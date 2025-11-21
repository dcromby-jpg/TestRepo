from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from .ingest import DocumentChunk


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, str]
    score: float


def _apply_filters(chunks: Iterable[DocumentChunk], filters: dict[str, str]) -> list[DocumentChunk]:
    if not filters:
        return list(chunks)
    return [c for c in chunks if all(c.metadata.get(k) == v for k, v in filters.items())]


def bm25_retrieve(chunks: Sequence[DocumentChunk], query: str, top_k: int, filters: dict[str, str]) -> list[RetrievedChunk]:
    filtered = _apply_filters(chunks, filters)
    corpus = [c.text for c in filtered]
    bm25 = BM25Okapi([doc.split() for doc in corpus])
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(filtered, scores), key=lambda pair: (pair[1], pair[0].metadata.get("chunk_id")), reverse=True)
    return [RetrievedChunk(text=chunk.text, metadata=chunk.metadata, score=float(score)) for chunk, score in ranked[:top_k]]


def dense_retrieve(collection, query: str, top_k: int, filters: dict[str, str]) -> list[RetrievedChunk]:
    results = collection.query(query_texts=[query], n_results=top_k, where=filters or None, include=["documents", "metadatas", "distances"])
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    bundle = sorted(zip(ids, docs, metas, distances), key=lambda x: (x[3], x[0]))
    return [RetrievedChunk(text=doc, metadata=meta, score=float(dist)) for _, doc, meta, dist in bundle[:top_k]]
