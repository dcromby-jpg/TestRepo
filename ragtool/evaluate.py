from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .retrieval import RetrievedChunk


@dataclass
class EvaluationResult:
    exact_match: float
    partial_match: float
    grounded_fraction: float


def grounding_check(answers: Iterable[str], retrieved: list[RetrievedChunk]) -> float:
    cited_ids = {c.metadata.get("chunk_id") for c in retrieved}
    hits = 0
    total = 0
    for answer in answers:
        total += 1
        if any(chunk_id and chunk_id in answer for chunk_id in cited_ids):
            hits += 1
    return hits / max(total, 1)


def compute_match_metrics(predictions: list[str], references: list[str]) -> tuple[float, float]:
    exact = 0
    partial = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            exact += 1
            partial += 1
        elif pred.strip() in ref or ref.strip() in pred:
            partial += 1
    return exact / max(total, 1), partial / max(total, 1)


def evaluate_answers(predictions: list[str], references: list[str], retrieved: list[RetrievedChunk]) -> EvaluationResult:
    exact, partial = compute_match_metrics(predictions, references)
    grounded = grounding_check(predictions, retrieved)
    return EvaluationResult(exact_match=exact, partial_match=partial, grounded_fraction=grounded)
