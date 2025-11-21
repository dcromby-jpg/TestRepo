from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional

import yaml


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    revision: Optional[str] = "c9d618e4e4c3d71a1cc150b76c7a6c5e97f65851"
    tokenizer: Optional[str] = None


@dataclass
class IngestConfig:
    data_dir: pathlib.Path = pathlib.Path("data")
    index_path: pathlib.Path = pathlib.Path("chroma_db")
    chunk_size: int = 1024
    chunk_overlap: int = 128
    lowercase: bool = True
    persist_embeddings: bool = True


@dataclass
class RetrievalConfig:
    retriever_type: Literal["bm25", "dense"] = "dense"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    metadata_filters: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    output_dir: pathlib.Path = pathlib.Path("artifacts/lora-adapter")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 1024


@dataclass
class EvalConfig:
    k_for_grounding: int = 3
    val_split_ratio: float = 0.2
    seed: int = 13


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | pathlib.Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}

    def _load(dc_cls, section: dict | None):
        section = section or {}
        defaults = dc_cls()
        values = {}
        for key in dc_cls.__annotations__.keys():
            default_value = getattr(defaults, key)
            incoming = section.get(key, default_value)
            if isinstance(default_value, pathlib.Path) and not isinstance(incoming, pathlib.Path):
                incoming = pathlib.Path(incoming)
            values[key] = incoming
        return dc_cls(**values)

    return AppConfig(
        model=_load(ModelConfig, payload.get("model")),
        ingest=_load(IngestConfig, payload.get("ingest")),
        retrieval=_load(RetrievalConfig, payload.get("retrieval")),
        training=_load(TrainingConfig, payload.get("training")),
        eval=_load(EvalConfig, payload.get("eval")),
    )
