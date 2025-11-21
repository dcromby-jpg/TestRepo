from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig, RetrievalConfig
from .retrieval import RetrievedChunk, bm25_retrieve, dense_retrieve


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self, model_cfg: ModelConfig, retrieval_cfg: RetrievalConfig, collection=None, corpus=None):
        self.model_cfg = model_cfg
        self.retrieval_cfg = retrieval_cfg
        self.collection = collection
        self.corpus = corpus or []
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer or model_cfg.base_model, revision=model_cfg.revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.base_model, revision=model_cfg.revision)

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        if self.retrieval_cfg.retriever_type == "bm25":
            return bm25_retrieve(self.corpus, query, self.retrieval_cfg.top_k, self.retrieval_cfg.metadata_filters)
        return dense_retrieve(self.collection, query, self.retrieval_cfg.top_k, self.retrieval_cfg.metadata_filters)

    def _build_prompt(self, query: str, retrieved: Sequence[RetrievedChunk]) -> str:
        context = "\n\n".join([f"[source={c.metadata.get('chunk_id')}] {c.text}" for c in retrieved])
        return (
            "You are a helpful assistant. Answer the question strictly using the provided sources and cite chunk_ids.\n"
            f"Question: {query}\n\nSources:\n{context}\n\n"
            "Answer with inline citations like (source: chunk_id)."
        )

    def ask(self, query: str) -> RAGResponse:
        retrieved = self._retrieve(query)
        prompt = self._build_prompt(query, retrieved)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return RAGResponse(answer=answer, sources=list(retrieved))
