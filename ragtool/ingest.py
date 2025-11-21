from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from pypdf import PdfReader
from transformers import AutoTokenizer

from .config import IngestConfig, ModelConfig


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, str]


CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)


def _normalize_text(payload: str, lowercase: bool) -> str:
    def _lower(match: re.Match[str]) -> str:
        content = match.group(0)
        return content

    normalized = CODE_FENCE_PATTERN.sub(_lower, payload)
    return normalized.lower() if lowercase else normalized


def _read_file(path: pathlib.Path) -> str:
    if path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        return soup.get_text("\n")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() for page in reader.pages)
    return path.read_text(encoding="utf-8")


def chunk_documents(model_config: ModelConfig, ingest_config: IngestConfig) -> list[DocumentChunk]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer or model_config.base_model,
        revision=model_config.revision,
        use_fast=True,
    )
    chunks: list[DocumentChunk] = []
    for path in ingest_config.data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".html", ".htm", ".pdf"}:
            continue
        raw_text = _read_file(path)
        normalized = _normalize_text(raw_text, ingest_config.lowercase)
        tokens = tokenizer.encode(normalized)
        stride = ingest_config.chunk_size - ingest_config.chunk_overlap
        for start in range(0, len(tokens), stride):
            end = min(start + ingest_config.chunk_size, len(tokens))
            text = tokenizer.decode(tokens[start:end])
            chunk_id = f"{path.stem}-{start}-{end}"
            chunks.append(
                DocumentChunk(
                    text=text,
                    metadata={
                        "source": str(path),
                        "chunk_id": chunk_id,
                        "section": path.stem,
                    },
                )
            )
            if end == len(tokens):
                break
    return chunks


def ensure_data_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
