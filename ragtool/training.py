from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass
from typing import Iterable

import datasets
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import ModelConfig, TrainingConfig
from .ingest import DocumentChunk
from .seed import set_deterministic_seed


@dataclass
class QAExample:
    question: str
    answer: str
    source_id: str


def build_sft_dataset(chunks: Iterable[DocumentChunk], seed: int, val_split: float) -> tuple[datasets.Dataset, datasets.Dataset]:
    set_deterministic_seed(seed)
    grouped: dict[str, list[QAExample]] = {}
    for chunk in chunks:
        content = chunk.text.strip().split("\n")[0][:200]
        question = f"What does the document section '{chunk.metadata.get('section')}' cover?"
        answer = f"{content} (source: {chunk.metadata.get('chunk_id')})"
        sample = QAExample(question=question, answer=answer, source_id=chunk.metadata.get("chunk_id", ""))
        grouped.setdefault(chunk.metadata.get("source", "unknown"), []).append(sample)

    doc_ids = list(grouped.keys())
    random.shuffle(doc_ids)
    cutoff = int(len(doc_ids) * (1 - val_split))
    train_ids = set(doc_ids[:cutoff])

    train_records: list[QAExample] = []
    val_records: list[QAExample] = []
    for doc_id, samples in grouped.items():
        (train_records if doc_id in train_ids else val_records).extend(samples)

    def _to_dataset(rows: list[QAExample]) -> datasets.Dataset:
        return datasets.Dataset.from_list([{"question": r.question, "answer": r.answer, "source_id": r.source_id} for r in rows])

    return _to_dataset(train_records), _to_dataset(val_records)


def train_lora(model_cfg: ModelConfig, train_cfg: TrainingConfig, train_ds: datasets.Dataset, val_ds: datasets.Dataset):
    set_deterministic_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer or model_cfg.base_model, revision=model_cfg.revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.base_model, revision=model_cfg.revision, torch_dtype=torch.float16)

    lora_cfg = LoraConfig(
        r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(batch):
        inputs = [f"Question: {q}\nAnswer with citation: {a}" for q, a in zip(batch["question"], batch["answer"])]
        tokenized = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=train_cfg.max_seq_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_train = train_ds.map(preprocess, batched=True)
    tokenized_val = val_ds.map(preprocess, batched=True)

    args = TrainingArguments(
        output_dir=str(train_cfg.output_dir),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )
    trainer.train()

    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)

    manifest = {
        "base_model": model_cfg.base_model,
        "revision": model_cfg.revision,
        "tokenizer": model_cfg.tokenizer or model_cfg.base_model,
        "lora": {
            "r": train_cfg.lora_r,
            "alpha": train_cfg.lora_alpha,
            "dropout": train_cfg.lora_dropout,
        },
    }
    manifest_path = pathlib.Path(train_cfg.output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return model, tokenizer
